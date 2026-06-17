# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg, MobileReachEnvCfg
from omni.isaac.lab_tasks.manager_based.manipulation.reach.config.hsrb.hsrb_grid_experiment import HSRBGridExperimentEnvCfg
from omni.isaac.lab_tasks.manager_based.manipulation.reach.config.hsrb.hsrb_grid_teacher_policy import TeacherRewardsCfg
from omni.isaac.lab.assets import AssetBaseCfg
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

##
# Pre-defined configs
##
from omni.isaac.lab_assets.hsrb import HSRB_CFG, HSRB_STUDENT_CFG, HSRB_DEFAULT_CAMERA_INTRINSICS, HSRB_SCANDOTS_CFG, HSRB_TILED_DEPTH_CAMERA_CFG, CYLINDER_CFG  # isort:skip 
from omni.isaac.lab.terrains.config.hsrb_reach import HSRB_REACH_TERRAINS_CFG, HSRB_STUDENT_REACH_TERRAINS_CFG, generate_grid_reach_terrains_cfg  # isort: skip
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

##
# Environment configuration
##
illegal_contact_list = ["base_link_contact", "base_contact_back", "base_contact_front", 
                            "torso_lift_link_contact", "arm_contact_lift", "arm_contact_flex", 
                            "arm_contact_roll", "wrist_contact_roll", "self_gripper_contact_hand_palm_link", 
                            "self_gripper_contact_hand_l_spring_proximal_link", 
                            "self_gripper_contact_hand_r_spring_proximal_link", 
                            "self_gripper_contact_hand_l_distal_link", "self_gripper_contact_hand_r_distal_link", 
                            "self_gripper_contact_hand_l_finger_vacuum_frame", 
                            "head_pan_contact", "head_tilt_contact", "head_rgbd_sensor_contact", "cylinder_contact"]

params = {
    "frame_name": "ee_tcp",   # Name of ee tcp link
    "command_name": "ee_pose",  # Name of command
    "asset_cfg": SceneEntityCfg("robot"),  # Asset configuration
    "position_threshold": 0.25, # Position threshold for the end-effector goal success
    "goal_reward": 15.0,        # Reward for reaching the goal position
    "soft_ratio": 1.0,          # Soft ratio for joint velocity limits
    "num_steps_per_env": 96,   # Number of steps per environment
    "home_threshold": 0.2,     # Home threshold for give up action
    "time_threshold": 6.0,     # Time threshold for give up action
    "distance_threshold": 0.7,  # Distance threshold for give up action
    "give_up_reward": 0.5,      # Reward for give up action
    "give_up_penalty": -0.01,    # Penalty for give up action
    "min_x": 0.25,              # Min x value for goal levels
    "max_x": 2.0,               # Max x value for goal levels
    "level_intervals": 6,       # Number of intervals for goal levels
    "risk_sensitivity_sample_range": (-1.0, 1.0),   # Risk sensitivity sampling range
    "test_set_config_filename": "/workspace/isaaclab/source/standalone/useful_scripts/test_set.json", # I know its bad practice to use hardcoded paths, but I too am a bad programmer
}

arm_joints = ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
base_bodies = ["base_link"]

@configclass
class TeacherCurriculumCfg(CurriculumCfg):
    """Curriculum terms for the MDP."""
    ### num_steps is calculated as total_steps_per_env = current_iteration * self._num_steps_per_env ###

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.0001, "num_steps": 200 * params['num_steps_per_env']} # Update after 200 iterations
    )

    arm_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "arm_joint_vel", "weight": -0.0025, "num_steps": 200 * params['num_steps_per_env']}
    )
    
    base_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "base_vel", "weight": -0.005, "num_steps": 200 * params['num_steps_per_env']}
    )

    termination_penalty = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "termination_penalty", "weight": -50.0, "num_steps": 800 * params['num_steps_per_env']} # 300 iterations
    )

    ee_acc = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "ee_acc", "weight": -0.0025, "num_steps": 400 * params['num_steps_per_env']} # 400 iterations
    )

    end_effector_position_tracking = CurrTerm(
        func=mdp.modify_reward_parameters, params={"term_name": "end_effector_position_tracking", "parameters": {"frame_name": params['frame_name'], "command_name": params['command_name'], "position_threshold": 0.15, "goal_reward": params['goal_reward']}, "num_steps": 300 * params['num_steps_per_env']} # Update after 300 iterations
    )

    # goal_levels_range = CurrTerm(
    #     func=mdp.goal_levels_range, params={"command_name": params['command_name'], "min_x": (-0.25, 0.25), "min_y": (-0.25, 0.25), "max_x": (-2.0, 2.0), "max_y": (-2.0, 2.0), "level_intervals": 25} 
    # )

    goal_levels = CurrTerm(
        func=mdp.goal_levels, params={"command_name": params['command_name'], "min_x": params['min_x'], "max_x": params['max_x'], "level_intervals": params['level_intervals'], "successful_envs_term": "end_effector_position_tracking"}
    )

@configclass
class TeacherTerminationsCfg(TerminationsCfg):
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_link_contact = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("base_link_contact_force")})
    base_contact_back = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("base_b_bumper_contact_force", body_names=[".*"])})
    base_contact_front = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("base_f_bumper_contact_force", body_names=[".*"])})
    torso_lift_link_contact = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("torso_lift_link_contact_force")})
    arm_contact_lift = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("arm_lift_contact_force")})
    arm_contact_flex = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("arm_flex_contact_force")})
    arm_contact_roll = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("arm_roll_contact_force")})
    wrist_contact_roll = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("wrist_roll_contact_force")})
    self_gripper_contact_hand_palm_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("self_gripper_hand_palm_link_contact_force")})
    self_gripper_contact_hand_l_spring_proximal_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_spring_proximal_link_contact_force")})
    self_gripper_contact_hand_r_spring_proximal_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("self_gripper_hand_r_spring_proximal_link_contact_force")})
    self_gripper_contact_hand_l_distal_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_distal_link_contact_force")})
    self_gripper_contact_hand_r_distal_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("self_gripper_hand_r_distal_link_contact_force")})
    self_gripper_contact_hand_l_finger_vacuum_frame = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_finger_vacuum_frame_contact_force")})
    head_pan_contact = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("head_pan_contact_force")})
    head_tilt_contact = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("head_tilt_contact_force")})
    head_rgbd_sensor_contact = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("head_rgbd_sensor_contact_force")})
    cylinder_contact = DoneTerm(func=mdp.illegal_contact, params={"threshold": 0.1, "sensor_cfg": SceneEntityCfg("cylinder_contact_force")})
    # goal_reached = DoneTerm(func=mdp.command_resample, params={"command_name": params['command_name'], "num_resamples": 2})# , time_out=True)
    goal_reached = DoneTerm(func=mdp.goal_reached, params={"command_name": params['command_name'], "frame_name": params['frame_name'], "reward_term_name": "end_effector_position_tracking",})
    give_up_action = DoneTerm(func=mdp.give_up_action, params={"command_name": params['command_name'], "frame_name": params['frame_name'], "home_threshold": params["home_threshold"], "time_threshold": params['time_threshold'], "distance_threshold": params['distance_threshold'], "asset_cfg": params["asset_cfg"], "action_name": "give_up_action"})
   

@configclass
class HSRBGridDepthEnvCfg(HSRBGridExperimentEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        size=(6.0, 6.0) # size of the sub-terrain in meters
        # num_rows = 2 # number of rows of sub-terrains
        # num_cols = 2 # number of columns of sub-terrains
        grid_rows = 9 # number of rows of grid cells
        grid_cols = 9 # number of columns of grid cells
        grid_cell_width = 0.5 # width of a grid cell in meters

        max_filled_cells = 8 # maximum number of filled cells in a grid cell
        min_filled_cells = 3 # minimum number of filled cells in a grid cell

        table_height_range = (0.25, 0.6) # height range of the tables

        # switch robot to HSRB
        self.scene.robot = HSRB_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.scene.cylinder = CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/cylinder")

        self.scene.object_1 = None
        self.scene.object_2 = None
        self.scene.object_3 = None
        self.scene.object_4 = None
        self.scene.object_5 = None

        # override actions
        self.actions.arm_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot", joint_names=["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"], scale=1.0, debug_vis=False
        )
        self.actions.base_action = mdp.HSRBaseVelocityControlCfg( 
            asset_name="robot", joint_names=["base_l_drive_wheel_joint", "base_r_drive_wheel_joint", "base_roll_joint"], scale=1.0, debug_vis=True # use_default_offset=True,
        )
        self.actions.give_up_action = mdp.GiveUpActionCfg()
        
        self.commands.risk_sensitivity = mdp.ScalarValueCommandCfg(
            value_range=params['risk_sensitivity_sample_range'],
            resampling_time_range=(1000.0, 1000.0),
        )

        self.rewards = TeacherRewardsCfg() # use teacher rewards

        ###### Terrain generation ######

        print(f"[INFO] Number of environments: {self.scene.num_envs}")
        a = int(self.scene.num_envs**0.5)  # Start at the square root of n
        while a > 0:
            if self.scene.num_envs % a == 0:  # Check if a divides num_envs
                b = self.scene.num_envs // a
                num_rows, num_cols = a, b
                break
            a -= 1
        print(f"[INFO] Using {num_rows} x {num_cols} grid for training test!!!.")


        hsrb_grid_reach_terrain = generate_grid_reach_terrains_cfg(size=size,
                                                                   num_rows=num_rows,
                                                                   num_cols=num_cols,
                                                                   grid_cell_width=grid_cell_width,
                                                                   grid_cols=grid_cols,
                                                                   grid_rows=grid_rows,
                                                                   table_height_range=table_height_range,
                                                                   max_filled_cells=max_filled_cells,
                                                                   min_filled_cells=min_filled_cells,
                                                                   exclude_test_set=True,
                                                                   test_set_config_filename=params["test_set_config_filename"],
                                                                   max_attempts=1000,)

        # ground terrain
        self.scene.terrain = VariedGridTerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=hsrb_grid_reach_terrain,
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

        self.commands.ee_pose = mdp.GridUniformPoseCommandCfg(
            asset_name="robot",
            resampling_time_range=(50.0, 100.0),
            debug_vis=False,
            pos_x=torch.tensor([-0.25, 0.25]),
            pos_y=torch.tensor([-0.25, 0.25]),
            pos_z=torch.tensor([0.2, 1.4]),
            roll=torch.tensor([-math.pi, math.pi]),
            pitch=torch.tensor([math.pi/2, math.pi/2]),  # depends on end-effector axis
            yaw=torch.tensor([-math.pi, -math.pi]),
            goal_level_range_sampling=True if self.curriculum.goal_levels_range is not None else False,
            goal_level_sampling=True if self.curriculum.goal_levels is not None else False,
            table_locations=hsrb_grid_reach_terrain.table_locs,
            max_table_height=table_height_range[1],
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            grid_cell_width=grid_cell_width,
            terrain_size=size,
            return_quat=True,
        )

        # # override the curriculum
        self.curriculum = TeacherCurriculumCfg()

        # # override the terminations
        self.terminations = TeacherTerminationsCfg()

        # priviledged height scan - we don't need this in this env since we are training straight from depth
        self.scene.height_scan = None
        self.observations.height_scan = None

        self.observations.depth = DepthCfg()
        
        self.observations.policy.joint_pos = ObsTerm(
            func=mdp.filtered_joint_pos,
            params={"asset_cfg": params["asset_cfg"], 
                    "joint_names": ["base_roll_joint",
                                    "arm_lift_joint", 
                                    "arm_flex_joint", 
                                    "arm_roll_joint", 
                                    "wrist_flex_joint", 
                                    "wrist_roll_joint"],}, 
            noise=Unoise(n_min=-0.01, n_max=0.01))
        
        self.observations.policy.joint_vel = ObsTerm(
            func=mdp.filtered_joint_vel,
            params={"asset_cfg": params["asset_cfg"],
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
            params={"asset_cfg": params["asset_cfg"], 
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
            params={"asset_cfg": params["asset_cfg"],
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

        self.scene.depth_camera_tiled = HSRB_TILED_DEPTH_CAMERA_CFG.copy()
        self.observations.depth.depth_camera_tiled = ObsTerm(
            func=mdp.tiled_depth_camera,
            params={"sensor_cfg": SceneEntityCfg("depth_camera_tiled"), "max_distance": 2.5}, # , "asset_cfg": SceneEntityCfg("robot")
            noise=Unoise(n_min=-0.1, n_max=0.1),
            # clip=(-1.0, 1.0),
        )

        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {
                    "x": (-0.0, 0.0),
                    "y": (-0.0, 0.0),
                    "z": (-0.0, 0.0),
                    "roll": (-0.0, 0.0),
                    "pitch": (-0.0, 0.0),
                    "yaw": (-0.0, 0.0),
                },
            },
        )

        self.events.reset_cylinder = EventTerm(
            func=mdp.reset_cylinder_joints_by_position,
            mode="reset",
            params={
                "position": {
                    "x": 1.5,
                    "y": 0.0,
                    "z": 0.75,
                },
                "asset_cfg": SceneEntityCfg("cylinder"),
            },
        )

        self.events.reset_object_1 = None
        self.events.reset_object_2 = None
        self.events.reset_object_3 = None
        self.events.reset_object_4 = None
        self.events.reset_object_5 = None

        self.events.resample_cylinder_velocity = EventTerm(
                func=mdp.randomise_cylinder_velocities,
                mode="interval",
                interval_range_s=(1.0, 3.0),
                params={
                    "velocity_range": (-0.5, 0.5),
                    "grid_size": (grid_rows*grid_cell_width, grid_cols*grid_cell_width),
                    "asset_cfg": SceneEntityCfg("cylinder"),
                },
        )

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
        
        cylinder_prim_path = "{ENV_REGEX_NS}/cylinder"

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
            prim_path="{ENV_REGEX_NS}/Robot/base_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        )
        self.scene.base_b_bumper_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_b_bumper_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=["/World/ground/terrain/mesh"], track_air_time=True, # /World/ground/terrain/mesh
        )
        self.scene.base_f_bumper_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_f_bumper_link", update_period=0.0, history_length=contact_history_length, debug_vis=False, filter_prim_paths_expr=["/World/ground/terrain/mesh"], track_air_time=True,
        )
        self.scene.torso_lift_link_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_lift_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=robot_arm_prim_paths,
        )

        #### Arm contact forces ####
        arm_lift_contact_force_prim_paths = robot_base_prim_paths + robot_gripper_prim_paths + robot_head_prim_paths + ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/torso_lift_link", "{ENV_REGEX_NS}/Robot/wrist_roll_link"]
        self.scene.arm_lift_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/arm_lift_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=arm_lift_contact_force_prim_paths # ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/base.*/collisions/mesh.*", "{ENV_REGEX_NS}/Robot/wrist_roll_link", "{ENV_REGEX_NS}/Robot/hand.*", "{ENV_REGEX_NS}/Robot/torso_lift_link", "{ENV_REGEX_NS}/Robot/head.*"], # {ENV_REGEX_NS}/Robot/base_f_bumper_link
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
            prim_path="{ENV_REGEX_NS}/Robot/head_pan_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=head_pan_contact_force_prim_paths,
        )
        head_tilt_contact_force_prim_paths = robot_arm_prim_paths + robot_wrist_prim_paths + robot_gripper_prim_paths + ["/World/ground/terrain/mesh"]
        self.scene.head_tilt_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_tilt_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=head_tilt_contact_force_prim_paths,
        )
        head_rgbd_sensor_contact_force_prim_paths = robot_arm_prim_paths + robot_wrist_prim_paths + robot_gripper_prim_paths + ["/World/ground/terrain/mesh"]
        self.scene.head_rgbd_sensor_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/head_rgbd_sensor_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=head_rgbd_sensor_contact_force_prim_paths,
        )

        #### Cylinder contact forces ####
        # cylinder_contact_force_prim_paths = ["/World/ground/terrain/mesh"] # objects_prim_paths + ["/World/ground/terrain/mesh"]
        # # consider only robot prim paths
        cylinder_contact_force_prim_paths = robot_base_prim_paths + robot_arm_prim_paths + robot_wrist_prim_paths + robot_gripper_prim_paths + robot_head_prim_paths # + ["/World/ground/terrain/mesh"]
        self.scene.cylinder_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/cylinder/cylinder/base_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=cylinder_contact_force_prim_paths,
        )
        self.scene.cylinder_world_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/cylinder/cylinder/base_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        )


@configclass
class DepthCfg(ObsGroup):
    """Observations for depth group."""

    depth_camera_tiled: ObsTerm | None = None

    scandots: ObsTerm | None = None

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False
        
@configclass
class HSRBGridDepthEnvCfg_PPO(HSRBGridDepthEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

@configclass
class HSRBGridDepthEnvCfg_PLAY(HSRBGridDepthEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        
@configclass
class HSRBGridDepthEnvCfg_TEST(HSRBGridDepthEnvCfg_PPO):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
