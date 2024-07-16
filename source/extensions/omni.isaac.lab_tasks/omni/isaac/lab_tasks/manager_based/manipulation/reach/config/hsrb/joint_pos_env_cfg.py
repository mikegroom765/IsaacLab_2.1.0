# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from omni.isaac.lab.utils import configclass

import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg, MobileReachEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.sensors import FrameTransformerCfg, ContactSensorCfg, patterns
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors import TiledCameraCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.hsrb import HSRB_CFG, HSRB_LIDAR_CFG, HSRB_DEPTH_CAMERA_CFG, HSRB_TILED_DEPTH_CAMERA_CFG, HSRB_DEFAULT_CAMERA_INTRINSICS  # isort:skip 
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip


FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

##
# Environment configuration
##

@configclass
class HSRBReachEnvCfg(MobileReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to HSRB
        self.scene.robot = HSRB_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # override actions
        self.actions.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"], scale=1.0, debug_vis=True
        )
        self.actions.base_action = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=["joint_x", "joint_y", "joint_rz"], use_default_offset=True, scale=1.0, debug_vis=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot", 
            joint_names=["hand_l_proximal_joint", "hand_r_proximal_joint"], 
            open_command_expr={"hand_l_proximal_joint": 1.2, "hand_r_proximal_joint": 1.2}, # 0.75 # revolute joint in articulation has a limit of 1.24 radians (71 degrees)
            close_command_expr={"hand_l_proximal_joint": 0.0, "hand_r_proximal_joint": 0.0}
        )
        self.actions.head_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["head_pan_joint", "head_tilt_joint"], scale=1.0, debug_vis=True
        )
        # override command generator body
        # end-effector is along z-direction
        # self.commands.ee_pose.body_name = "base_footprint"
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


        self.rewards.gripper_close_reward.params["open_joint_pos"] = 1.1
        self.rewards.gripper_close_reward.params["distance_threshold"] = 0.1
        self.rewards.gripper_close_reward.params["orientation_threshold"] = 0.2
        self.rewards.gripper_close_reward.params["asset_cfg"].joint_names = ["hand_l_proximal_joint", "hand_r_proximal_joint"]

        # # overwrite reward term parameters for is_goal_in_camera_view to use the when there is no TiledCamera in the scene
        # self.rewards.is_goal_in_camera_view.params["camera_intrinsics"] = HSRB_DEFAULT_CAMERA_INTRINSICS

        # Listen for the required transforms (end-effector)
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
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

        self.scene.depth_camera_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_footprint",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/DepthCameraFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/head_rgbd_sensor_link",
                    name="depth_camera",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )

        # add lidar
        self.scene.lidar = HSRB_LIDAR_CFG.copy()

        # add depth camera
        self.scene.depth_camera = HSRB_DEPTH_CAMERA_CFG.copy()
        self.scene.depth_camera_tiled = HSRB_TILED_DEPTH_CAMERA_CFG.copy()

        self.commands.ee_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name="base_footprint",
            resampling_time_range=(50.0, 100.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(-3.5,3.5),
                pos_y=(-3.5, 3.5),
                pos_z=(0.2, 1.4),
                roll=(-math.pi, math.pi),
                pitch=(math.pi/2, math.pi/2),  # depends on end-effector axis
                yaw=(-math.pi, -math.pi),
            ),
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
class HSRBReachEnvCfg_PLAY(HSRBReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
