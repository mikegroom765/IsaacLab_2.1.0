# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg, MobileReachEnvCfg
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.sensors import FrameTransformerCfg, ContactSensorCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import CurriculumCfg, TerminationsCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.terrains.config.hsrb_reach import HSRB_REACH_TERRAINS_CFG, HSRB_REACH_CORRIDOR_TERRAINS_CFG, HSRB_REACH_L_CORRIDOR_TERRAINS_CFG # isort: skip
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import EventTermCfg as EventTerm




##
# Pre-defined configs
##
from omni.isaac.lab_assets.hsrb import HSRB_CFG, HSRB_DEFAULT_CAMERA_INTRINSICS, HSRB_SCANDOTS_CFG, HSRB_TILED_DEPTH_CAMERA_CFG  # isort:skip 
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

##
# Environment configuration
##

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
class ExperimentCurriculumCfg(CurriculumCfg):
    """Curriculum terms for the MDP."""
    ### 2000 steps triggers during 22nd iteration - I think this is the start of the 3rd episode ###
    ### num_steps is calculated as total_steps_per_env = current_iteration * self._num_steps_per_env ###

    end_effector_position_tracking = CurrTerm(
        func=mdp.modify_reward_parameters, params={"term_name": "end_effector_position_tracking", "parameters": {"frame_name": "ee_frame", "command_name": "ee_pose", "position_threshold": 0.10}, "num_steps": 72000} # Update after 750 iterations
    )

    end_effector_orientation_tracking = CurrTerm(
        func=mdp.modify_reward_parameters, params={"term_name": "end_effector_orientation_tracking", "parameters": {"asset_cfg": SceneEntityCfg("robot"), "command_name": "ee_pose", "frame_name": "ee_frame", "orientation_threshold": math.pi/8}, "num_steps": 72000}
    )

@configclass
class ExperimentTerminationsCfg(TerminationsCfg):
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact_back = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_b_bumper_contact_force", body_names=[".*"])})
    base_contact_front = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("base_f_bumper_contact_force", body_names=[".*"])})
    arm_contact_flex = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_flex_contact_force")})
    arm_contact_roll = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("arm_roll_contact_force")})
    wrist_contact_roll = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("wrist_roll_contact_force")})
    self_gripper_contact_hand_palm_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_palm_link_contact_force")})
    self_gripper_contact_hand_l_spring_proximal_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_spring_proximal_link_contact_force")})
    self_gripper_contact_hand_r_spring_proximal_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_r_spring_proximal_link_contact_force")})
    self_gripper_contact_hand_l_distal_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_distal_link_contact_force")})
    self_gripper_contact_hand_r_distal_link = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_r_distal_link_contact_force")})
    self_gripper_contact_hand_l_finger_vacuum_frame = DoneTerm(func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("self_gripper_hand_l_finger_vacuum_frame_contact_force")})
    goal_reached = DoneTerm(func=mdp.command_resample, params={"command_name": "ee_pose", "num_resamples": 2})# , time_out=True)

@configclass
class HSRBReachExperimentEnvCfg(MobileReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to HSRB
        self.scene.robot = HSRB_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"], scale=1.0, debug_vis=True
        )
        self.actions.base_action = mdp.MimicPureJointVelocityActionCfg(
            asset_name="robot", joint_names=["joint_x", "joint_y", "joint_rz"], use_default_offset=True, scale=1.0, debug_vis=True
        )

        self.rewards.end_effector_orientation_tracking = RewTerm(
            func=mdp.orientation_command_error_frame_shaped,
            weight=5.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "ee_pose", "frame_name": "ee_frame", "orientation_threshold": math.pi/6},
        )

        self.rewards.end_effector_position_tracking = RewTerm(
            func=mdp.position_command_error_frame_shaped,
            weight=5.0,
            params={"frame_name": "ee_frame", "command_name": "ee_pose", "position_threshold": 0.15, "goal_reward": 20.0},
        )

        self.commands.ee_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name="base_footprint",
            resampling_time_range=(100.0, 100.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(1.0, 1.0),
                pos_y=(0.0, 0.0),
                # pos_x=(-3.0, 3.0),
                # pos_y=(-3.0, 3.0),
                pos_z=(0.6, 0.6),
                roll=(math.pi, math.pi), # -math.pi, math.pi
                pitch=(math.pi/2, math.pi/2),  # depends on end-effector axis
                yaw=(-math.pi, -math.pi),
            ),
        )

        self.commands.risk_sensitivity = mdp.ScalarValueCommandCfg(
            value_range=(-1.0, 1.0),
            resampling_time_range=(100.0, 100.0),
        )

        # ground terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=HSRB_REACH_L_CORRIDOR_TERRAINS_CFG, # HSRB_REACH_L_CORRIDOR_TERRAINS_CFG, # HSRB_REACH_CORRIDOR_TERRAINS_CFG, 
            max_init_terrain_level=5,
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

        # override the curriculum
        self.curriculum = ExperimentCurriculumCfg()

        # override the terminations
        self.terminations = ExperimentTerminationsCfg()

        # Listen for the required transforms (end-effector)
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_footprint",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/hand_palm_link",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.065),
                    ),
                ),
            ],
        )

        # self.scene.depth_camera_frame = FrameTransformerCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/base_footprint",
        #     debug_vis=False,
        #     visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/DepthCameraFrameTransformer"),
        #     target_frames=[
        #         FrameTransformerCfg.FrameCfg(
        #             prim_path="{ENV_REGEX_NS}/Robot/head_rgbd_sensor_link",
        #             name="depth_camera",
        #             offset=OffsetCfg(
        #                 pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),
        #             ),
        #         ),
        #     ],
        # )

        # Tiled depth camera
        # depth_camera_tiled = ObsTerm(
        #     func=mdp.tiled_depth_camera,
        #     params={"sensor_cfg": SceneEntityCfg("depth_camera_tiled")}, # , "asset_cfg": SceneEntityCfg("robot")
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     # clip=(-1.0, 1.0),
        # )

        # priviledged scandots
        # self.scene.scandots = HSRB_SCANDOTS_CFG.copy()
        # self.observations.depth.scandots = ObsTerm(
        #     func=mdp.masked_scan_dot_points,
        #     params={"sensor_cfg": SceneEntityCfg("scandots"), "camera_frame_name": "depth_camera_frame", "camera_intrinsics": HSRB_DEFAULT_CAMERA_INTRINSICS},
        # )

        # priviledged height scan
        self.scene.height_scan = HSRB_SCANDOTS_CFG.copy()

        self.observations.depth = DepthCfg()

        # priviledged height scan observation
        self.observations.height_scan.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scan")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            # clip=(-1.0, 1.0),
        )

        # # masked priviledged height scan observation
        # self.observations.height_scan.height_scan = ObsTerm(
        #     func=mdp.height_scan_masked,
        #     params={"sensor_cfg": SceneEntityCfg("height_scan"), "camera_frame_name": "depth_camera_frame", "camera_intrinsics": HSRB_DEFAULT_CAMERA_INTRINSICS},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        self.scene.depth_camera_tiled = HSRB_TILED_DEPTH_CAMERA_CFG.copy()
        self.observations.depth.depth_camera_tiled = ObsTerm(
            func=mdp.tiled_depth_camera,
            params={"sensor_cfg": SceneEntityCfg("depth_camera_tiled")}, # , "asset_cfg": SceneEntityCfg("robot")
            noise=Unoise(n_min=-0.1, n_max=0.1),
            # clip=(-1.0, 1.0),
        )

        if self.scene.depth_camera_tiled is not None:
            self.scene.sky_light = AssetBaseCfg(
                prim_path="/World/skyLight",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                ),
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

        self.commands.ee_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name="base_footprint",
            resampling_time_range=(50.0, 100.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(3.0, 4.0),
                pos_y=(-1.0, 1.0),
                # pos_x=(-3.0, 3.0),
                # pos_y=(-3.0, 3.0),
                pos_z=(0.6, 0.8),
                roll=(0.0, 0.0), # -math.pi, math.pi
                pitch=(math.pi/2, math.pi/2),  # depends on end-effector axis
                yaw=(-math.pi, -math.pi),
            ),
        )

        self.commands.risk_sensitivity = mdp.ScalarValueCommandCfg(
            value_range=(-1.0, 1.0),
            resampling_time_range=(10.0, 20.0),
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


        contact_history_length = 1
        #### Base contact forces ####
        self.scene.base_b_bumper_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_b_bumper_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=["/World/ground/terrain/mesh"], track_air_time=True, # /World/ground/terrain/mesh
        )
        self.scene.base_f_bumper_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_f_bumper_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=["/World/ground/terrain/mesh"], track_air_time=True,
        )

        #### Arm contact forces ####
        # arm_lift_contact_force_prim_paths = robot_base_prim_paths + robot_gripper_prim_paths + robot_head_prim_paths + ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/torso_lift_link", "{ENV_REGEX_NS}/Robot/wrist_roll_link"]
        # self.scene.arm_lift_contact_force = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/arm_lift_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=arm_contact_force_prim_paths # ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/base.*/collisions/mesh.*", "{ENV_REGEX_NS}/Robot/wrist_roll_link", "{ENV_REGEX_NS}/Robot/hand.*", "{ENV_REGEX_NS}/Robot/torso_lift_link", "{ENV_REGEX_NS}/Robot/head.*"], # {ENV_REGEX_NS}/Robot/base_f_bumper_link
        # )
        arm_flex_contact_force_prim_paths = robot_base_prim_paths + robot_gripper_prim_paths + robot_head_prim_paths + ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/torso_lift_link", "{ENV_REGEX_NS}/Robot/wrist_roll_link"]
        self.scene.arm_flex_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/arm_flex_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=arm_flex_contact_force_prim_paths # ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/base.*/collisions/mesh.*", "{ENV_REGEX_NS}/Robot/wrist_roll_link", "{ENV_REGEX_NS}/Robot/hand.*", "{ENV_REGEX_NS}/Robot/torso_lift_link", "{ENV_REGEX_NS}/Robot/head.*"], # {ENV_REGEX_NS}/Robot/base_f_bumper_link
        )
        arm_roll_contact_force_prim_paths = robot_base_prim_paths + robot_gripper_prim_paths + robot_head_prim_paths + ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/torso_lift_link", "{ENV_REGEX_NS}/Robot/wrist_roll_link"]
        self.scene.arm_roll_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/arm_roll_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=arm_roll_contact_force_prim_paths # ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/base.*/collisions/mesh.*", "{ENV_REGEX_NS}/Robot/wrist_roll_link", "{ENV_REGEX_NS}/Robot/hand.*", "{ENV_REGEX_NS}/Robot/torso_lift_link", "{ENV_REGEX_NS}/Robot/head.*"], # {ENV_REGEX_NS}/Robot/base_f_bumper_link
        )

        #### Wrist contact forces ####
        wrist_roll_contact_force_prim_paths = robot_base_prim_paths + robot_head_prim_paths + ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/torso_lift_link"]
        self.scene.wrist_roll_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_roll_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=wrist_roll_contact_force_prim_paths
        )

        #### (Self) Gripper contact forces ####
        self_gripper_contact_force_prim_paths = robot_base_prim_paths + robot_arm_prim_paths + robot_head_prim_paths + ["/World/ground/terrain/mesh", "{ENV_REGEX_NS}/Robot/torso_lift_link"]
        self.scene.self_gripper_hand_palm_link_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/hand_palm_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=self_gripper_contact_force_prim_paths,
        )
        self.scene.self_gripper_hand_l_spring_proximal_link_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/hand_l_spring_proximal_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=self_gripper_contact_force_prim_paths,
        )
        self.scene.self_gripper_hand_r_spring_proximal_link_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/hand_r_spring_proximal_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=self_gripper_contact_force_prim_paths,
        )
        self.scene.self_gripper_hand_l_distal_link_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/hand_l_distal_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=self_gripper_contact_force_prim_paths,
        )
        self.scene.self_gripper_hand_r_distal_link_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/hand_r_distal_link", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=self_gripper_contact_force_prim_paths,
        )
        self.scene.self_gripper_hand_l_finger_vacuum_frame_contact_force = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/hand_l_finger_vacuum_frame", update_period=0.0, history_length=contact_history_length, debug_vis=True, filter_prim_paths_expr=self_gripper_contact_force_prim_paths,
        )


@configclass
class HSRBReachTeacher1aEnvCfg_PLAY(HSRBReachExperimentEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
