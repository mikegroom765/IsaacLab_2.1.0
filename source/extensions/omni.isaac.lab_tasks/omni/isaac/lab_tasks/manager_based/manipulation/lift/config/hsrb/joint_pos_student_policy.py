# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as reach_mdp
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg, MobileReachEnvCfg
from omni.isaac.lab_tasks.manager_based.manipulation.reach.config.hsrb.hsrb_grid_experiment import HSRBGridExperimentEnvCfg
from omni.isaac.lab_tasks.manager_based.manipulation.lift.hsrb_lift_env_cfg import HSRBLiftEnvCfg
from omni.isaac.lab_tasks.manager_based.manipulation.lift.hsrb_lift_env_cfg import RewardsCfg as TeacherRewardsCfg
from omni.isaac.lab.assets import RigidObjectCfg, AssetBaseCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.sensors import FrameTransformerCfg, ContactSensorCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.terrains import TerrainImporterCfg, VariedGridTerrainImporterCfg
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import CurriculumCfg, TerminationsCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab_tasks.manager_based.manipulation.reach import mdp as reach_mdp
from omni.isaac.lab_tasks.manager_based.manipulation.inhand import mdp as in_hand_mdp


##
# Pre-defined configs
##
from omni.isaac.lab_assets.hsrb import HSRB_CFG, HSRB_STUDENT_CFG, HSRB_DEFAULT_CAMERA_INTRINSICS, HSRB_SCANDOTS_CFG, HSRB_TILED_DEPTH_CAMERA_CFG  # isort:skip 
from omni.isaac.lab.terrains.config.hsrb_reach import HSRB_LIFT_CUBE_TERRAINS_CFG  # isort: skip


@configclass
class StudentCurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-3, "num_steps": 10000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-3, "num_steps": 300 * num_steps_per_env}
    # )
    
    # levels = CurrTerm(
    #     func=reach_mdp.lift_cube_curriculum, 
    #     params={"event_terms": ["reset_base", "reset_object_position", "reset_robot_joints"],
    #             "level_intervals": 1,
    #             "successful_envs_term": "is_grasped_object"}
    # )

@configclass
class StudentTerminationsCfg(TerminationsCfg):
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_link_contact = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("base_link_contact_force")})
    base_contact_back = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("base_b_bumper_contact_force", body_names=[".*"])})
    base_contact_front = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("base_f_bumper_contact_force", body_names=[".*"])})
    object_out_of_reach = DoneTerm(
        func=in_hand_mdp.object_away_from_goal, params={"threshold": 1.0, "command_name": "object_goal_region"}
    )

@configclass
class HSRBLiftStudentEnvCfg(HSRBLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # switch robot to HSRB
        self.scene.robot = HSRB_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"], scale=0.1, debug_vis=False
        )
        self.actions.base_action = mdp.HSRBaseVelocityControlCfg( 
            asset_name="robot", joint_names=["base_l_drive_wheel_joint", "base_r_drive_wheel_joint", "base_roll_joint"], scale=0.1, debug_vis=False # use_default_offset=True,
        )
        self.actions.gripper_action = mdp.HSRBBinaryGripperActionCfg(asset_name="robot")

        print(f"[INFO] Number of environments: {self.scene.num_envs}")
        a = int(self.scene.num_envs**0.5)  # Start at the square root of n
        while a > 0:
            if self.scene.num_envs % a == 0:  # Check if a divides num_envs
                b = self.scene.num_envs // a
                num_rows, num_cols = a, b
                break
            a -= 1
        print(f"[INFO] Using {num_rows} x {num_cols} grid for training test!!!.")

        size=(5.0, 5.0)
        lift_cube_terrain = HSRB_LIFT_CUBE_TERRAINS_CFG(
            size=size,
            table_height_range=(0.4, 0.8),  # table height
            table_length=1.0,
            table_width=2.0,
        )

        # ground terrain
        self.scene.terrain = VariedGridTerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=lift_cube_terrain, # HSRB_REACH_CORRIDOR_TERRAINS_CFG
            max_init_terrain_level=None,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )
        
        table_heights_list = [lift_cube_terrain.sub_terrains[f"table_{i}"].table_height for i in range(len(lift_cube_terrain.sub_terrains))]

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25 * size[0] + 0.05, 0.1, 0.8), rot=(1, 0, 0, 0)),
            spawn=UsdFileCfg(
                usd_path=f"/workspace/isaaclab/source/standalone/hsrb/blue_cube.usd",
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0025,
                    max_depenetration_velocity=1000.0,
                ),
            ),
        )
        
        self.commands.object_goal_region = mdp.GoalRegionCommandCfg(
            asset_name="robot",
            frame_name="ee_tcp",
            object_name="object",
            goal_region_size=0.15,
            table_heights=table_heights_list,
            table_centre=(0.3 * size[0], 0.0),
            ranges=mdp.GoalRegionCommandCfg.Ranges(
                pos_x=(-0.6, -0.3), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5)
            ),
            debug_vis=True,
        )
        
        self.rewards.static_vel_reward_when_placed.params["object_goal_distance_threshold"] = 0.15
        self.rewards.success_reward.params["object_goal_distance_threshold"] = 0.15

        
        self.events.reset_object_position.params["set_heights"] = table_heights_list
        self.events.reset_robot_joints.params["table_heights"] = table_heights_list
        
        # set initial_level_zero to False on all curriculum reset events
        self.events.reset_robot_joints.params["initial_level_zero"] = False
        self.events.reset_object_position.params["initial_level_zero"] = False
        self.events.reset_base.params["initial_level_zero"] = False

        # # override the curriculum
        self.curriculum = StudentCurriculumCfg()

        # # override the terminations
        self.terminations = StudentTerminationsCfg()
        
        self.scene.height_scan = HSRB_SCANDOTS_CFG.copy()
        
        self.observations.depth = DepthCfg()

        # priviledged height scan observation
        self.observations.height_scan.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scan")},
            # noise=Unoise(n_min=-0.1, n_max=0.1),
            # clip=(-1.0, 1.0),
        )

        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.filtered_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot"), 
                    "joint_names": ["base_roll_joint",
                                    "arm_lift_joint", 
                                    "arm_flex_joint", 
                                    "arm_roll_joint", 
                                    "wrist_flex_joint", 
                                    "wrist_roll_joint"],}, 
            noise=Unoise(n_min=-0.01, n_max=0.01))
        
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.filtered_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot"),
                    "joint_names": ["base_roll_joint",
                                    "arm_lift_joint",
                                    "arm_flex_joint",
                                    "arm_roll_joint",
                                    "wrist_flex_joint",
                                    "wrist_roll_joint"],},
            noise=Unoise(n_min=-0.01, n_max=0.01))
                    
        self.observations.policy.base_pos = ObsTerm(
            func=mdp.hsr_base_odom_pos,
            params={"base_velocity_action_name": "base_action",},
        )
        
        self.observations.policy.base_vel = ObsTerm(
            func=mdp.hsr_base_vel,
            params={"base_velocity_action_name": "base_action",},
        )   
        
        self.observations.critic_policy.joint_pos = ObsTerm(
            func=mdp.filtered_joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot"), 
                    "joint_names": ["base_roll_joint",
                                    "arm_lift_joint", 
                                    "arm_flex_joint", 
                                    "arm_roll_joint", 
                                    "wrist_flex_joint", 
                                    "wrist_roll_joint"],}, 
            # noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        
        self.observations.critic_policy.joint_vel = ObsTerm(
            func=mdp.filtered_joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot"),
                    "joint_names": ["base_roll_joint",
                                    "arm_lift_joint",
                                    "arm_flex_joint",
                                    "arm_roll_joint",
                                    "wrist_flex_joint",
                                    "wrist_roll_joint"],},
            # noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        
        self.observations.critic_policy.base_pos = ObsTerm(
            func=mdp.hsr_base_odom_pos,
            params={"base_velocity_action_name": "base_action",},
        )
        
        self.observations.critic_policy.base_vel = ObsTerm(
            func=mdp.hsr_base_vel,
            params={"base_velocity_action_name": "base_action",},
        )   

        self.observations.policy.masked_object_position = ObsTerm(
            func=mdp.masked_object_position_robot_root_frame,
            params={"camera_frame_name": "head_rgbd_sensor_link",
                    "camera_intrinsics": HSRB_DEFAULT_CAMERA_INTRINSICS,
                    "noise": Unoise(n_min=-0.03, n_max=0.03)},
        )
        
        self.observations.critic_policy.masked_object_position = ObsTerm(
            func=mdp.masked_object_position_robot_root_frame,
            params={"camera_frame_name": "head_rgbd_sensor_link",
                    "camera_intrinsics": HSRB_DEFAULT_CAMERA_INTRINSICS,
                    "noise": None},
        )
        
        self.observations.critic_policy.current_ee_pose_base_frame.noise = None

        self.scene.depth_camera_tiled = HSRB_TILED_DEPTH_CAMERA_CFG.copy()
        self.observations.depth.depth_camera_tiled = ObsTerm(
            func=mdp.tiled_depth_camera,
            params={"sensor_cfg": SceneEntityCfg("depth_camera_tiled"), "max_distance": 2.5}, # , "asset_cfg": SceneEntityCfg("robot")
            noise=Unoise(n_min=-0.1, n_max=0.1),
            # clip=(-1.0, 1.0),
        )

        # if self.scene.depth_camera_tiled is None:
        #     self.scene.sky_light = AssetBaseCfg(
        #         prim_path="/World/skyLight",
        #         spawn=sim_utils.DomeLightCfg(
        #             intensity=750.0,
        #             texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        #         ),
        #     )

        # self.events.reset_base = EventTerm(
        #     func=mdp.reset_root_state_uniform,
        #     mode="reset",
        #     params={
        #         "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)},
        #         "velocity_range": {
        #             "x": (-0.0, 0.0),
        #             "y": (-0.0, 0.0),
        #             "z": (-0.0, 0.0),
        #             "roll": (-0.0, 0.0),
        #             "pitch": (-0.0, 0.0),
        #             "yaw": (-0.0, 0.0),
        #         },
        #     },
        # )

        #### Contact sensors ####

        robot_base_prim_paths = [
            "{ENV_REGEX_NS}/Robot/base_link",
            "{ENV_REGEX_NS}/Robot/base_b_bumper_link",
            "{ENV_REGEX_NS}/Robot/base_f_bumper_link",
        ]

        robot_arm_prim_paths = [
            "{ENV_REGEX_NS}/Robot/arm_lift_link",
            "{ENV_REGEX_NS}/Robot/arm_flex_link",
            "{ENV_REGEX_NS}/Robot/arm_roll_link",
        ]

        robot_wrist_prim_paths = [
            "{ENV_REGEX_NS}/Robot/wrist_flex_link",
            "{ENV_REGEX_NS}/Robot/wrist_roll_link",
        ]

        robot_gripper_prim_paths = [
            "{ENV_REGEX_NS}/Robot/hand_palm_link",
            "{ENV_REGEX_NS}/Robot/hand_l_spring_proximal_link",
            "{ENV_REGEX_NS}/Robot/hand_r_spring_proximal_link",
            "{ENV_REGEX_NS}/Robot/hand_l_distal_link",
            "{ENV_REGEX_NS}/Robot/hand_r_distal_link",
            "{ENV_REGEX_NS}/Robot/hand_l_finger_vacuum_frame",
        ]

        robot_head_prim_paths = [
            "{ENV_REGEX_NS}/Robot/head_pan_link",
            "{ENV_REGEX_NS}/Robot/head_tilt_link",
            "{ENV_REGEX_NS}/Robot/head_rgbd_sensor_link",
        ]
        

        objects_prim_paths = [
            "{ENV_REGEX_NS}/object1",
            "{ENV_REGEX_NS}/object2",
            "{ENV_REGEX_NS}/object3",
            "{ENV_REGEX_NS}/object4",
            # "{ENV_REGEX_NS}/object5",
        ]


        contact_history_length = 1
        #### Base contact forces ####
        self.scene.base_link_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        )
        self.scene.base_b_bumper_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_b_bumper_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=["/World/ground/terrain/mesh"], # /World/ground/terrain/mesh
        )
        self.scene.base_f_bumper_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_f_bumper_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        )
        self.scene.torso_lift_link_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_lift_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=robot_arm_prim_paths,
        )

        #### Arm contact forces ####
        arm_lift_contact_force_prim_paths = robot_base_prim_paths + robot_gripper_prim_paths + robot_head_prim_paths + ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/torso_lift_link", "{ENV_REGEX_NS}/Robot/wrist_roll_link"]
        self.scene.arm_lift_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/arm_lift_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=arm_lift_contact_force_prim_paths # ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/base.*/collisions/mesh.*", "{ENV_REGEX_NS}/Robot/wrist_roll_link", "{ENV_REGEX_NS}/Robot/hand.*", "{ENV_REGEX_NS}/Robot/torso_lift_link", "{ENV_REGEX_NS}/Robot/head.*"], # {ENV_REGEX_NS}/Robot/base_f_bumper_link
        )
        arm_flex_contact_force_prim_paths = robot_base_prim_paths + robot_gripper_prim_paths + robot_head_prim_paths + ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/torso_lift_link", "{ENV_REGEX_NS}/Robot/wrist_roll_link"]
        self.scene.arm_flex_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/arm_flex_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=arm_flex_contact_force_prim_paths # ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/base.*/collisions/mesh.*", "{ENV_REGEX_NS}/Robot/wrist_roll_link", "{ENV_REGEX_NS}/Robot/hand.*", "{ENV_REGEX_NS}/Robot/torso_lift_link", "{ENV_REGEX_NS}/Robot/head.*"], # {ENV_REGEX_NS}/Robot/base_f_bumper_link
        )
        arm_roll_contact_force_prim_paths = robot_base_prim_paths + robot_gripper_prim_paths + robot_head_prim_paths + ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/torso_lift_link", "{ENV_REGEX_NS}/Robot/wrist_roll_link"]
        self.scene.arm_roll_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/arm_roll_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=arm_roll_contact_force_prim_paths # ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/base.*/collisions/mesh.*", "{ENV_REGEX_NS}/Robot/wrist_roll_link", "{ENV_REGEX_NS}/Robot/hand.*", "{ENV_REGEX_NS}/Robot/torso_lift_link", "{ENV_REGEX_NS}/Robot/head.*"], # {ENV_REGEX_NS}/Robot/base_f_bumper_link
        )

        #### Wrist contact forces ####
        wrist_roll_contact_force_prim_paths = robot_base_prim_paths + robot_head_prim_paths + ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/torso_lift_link"]
        self.scene.wrist_roll_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_roll_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=wrist_roll_contact_force_prim_paths
        )

        #### (Self) Gripper contact forces ####
        self_gripper_contact_force_prim_paths = robot_base_prim_paths + robot_arm_prim_paths + robot_head_prim_paths + ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/torso_lift_link"]
        self.scene.self_gripper_hand_palm_link_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/hand_palm_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=self_gripper_contact_force_prim_paths,
        )
        self.scene.self_gripper_hand_l_spring_proximal_link_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/hand_l_spring_proximal_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=self_gripper_contact_force_prim_paths,
        )
        self.scene.self_gripper_hand_r_spring_proximal_link_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/hand_r_spring_proximal_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=self_gripper_contact_force_prim_paths,
        )
        self.scene.self_gripper_hand_l_distal_link_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/hand_l_distal_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=self_gripper_contact_force_prim_paths,
        )
        self.scene.self_gripper_hand_r_distal_link_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/hand_r_distal_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=self_gripper_contact_force_prim_paths,
        )
        self.scene.self_gripper_hand_l_finger_vacuum_frame_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/hand_l_finger_vacuum_frame", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=self_gripper_contact_force_prim_paths,
        )

        #### Head contact forces ####
        head_pan_contact_force_prim_paths = robot_arm_prim_paths + robot_wrist_prim_paths + robot_gripper_prim_paths + ["/World/ground/terrain/mesh"]
        self.scene.head_pan_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_pan_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=head_pan_contact_force_prim_paths,
        )
        head_tilt_contact_force_prim_paths = robot_arm_prim_paths + robot_wrist_prim_paths + robot_gripper_prim_paths + ["/World/ground/terrain/mesh"]
        self.scene.head_tilt_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_tilt_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=head_tilt_contact_force_prim_paths,
        )
        head_rgbd_sensor_contact_force_prim_paths = robot_arm_prim_paths + robot_wrist_prim_paths + robot_gripper_prim_paths + ["/World/ground/terrain/mesh"]
        self.scene.head_rgbd_sensor_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_rgbd_sensor_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=head_rgbd_sensor_contact_force_prim_paths,
        )
        
        #### Gripper object contact forces ####
        left_gripper_object_contact_force_prim_paths = ["{ENV_REGEX_NS}/Object", "/World/ground/terrain/mesh"]# + robot_gripper_right_finger_prim_paths
        self.scene.left_gripper_object_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/hand_l_distal_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=left_gripper_object_contact_force_prim_paths, track_pose=True,
        )
        right_gripper_object_contact_force_prim_paths = ["{ENV_REGEX_NS}/Object", "/World/ground/terrain/mesh"]# + robot_gripper_left_finger_prim_paths
        self.scene.right_gripper_object_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/hand_r_distal_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=right_gripper_object_contact_force_prim_paths, track_pose=True,
        )

@configclass
class DepthCfg(ObsGroup):
    """Observations for depth group."""

    # depth_camera_tiled = ObsTerm(
    #     func=mdp.tiled_depth_camera,
    #     params={"sensor_cfg": SceneEntityCfg("depth_camera_tiled")}, # , "asset_cfg": SceneEntityCfg("robot")
    #     noise=Unoise(n_min=-0.1, n_max=0.1),
    #     # clip=(-1.0, 1.0),
    # )
    depth_camera_tiled: ObsTerm | None = None

    # scandots = ObsTerm(
    #     func=mdp.masked_scan_dot_points,
    #     params={"sensor_cfg": SceneEntityCfg("scandots"), "camera_frame_name": MISSING, "camera_intrinsics": MISSING},
    # )
    scandots: ObsTerm | None = None

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False

@configclass
class HSRBLiftStudentEnvCfg_PPO(HSRBLiftStudentEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # remove risk sensitivity command
        self.commands.risk_sensitivity = None
        # remove risk sensitivity observation
        self.observations.policy.risk_sensitivity = None
        self.observations.critic_policy.risk_sensitivity = None

@configclass
class HSRBLiftStudentEnvCfg_PLAY(HSRBLiftStudentEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class HSRBLiftStudentEnvCfg_PPO_PLAY(HSRBLiftStudentEnvCfg_PPO):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False