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
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
import omni.isaac.lab.sim as sim_utils

##
# Pre-defined configs
##
from omni.isaac.lab_assets.hsrb import HSRB_CFG, HSRB_LIDAR_CFG  # isort:skip
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
        # override rewards
        # TODO: Update for the HSRB robot
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["hand_palm_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["hand_palm_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["hand_palm_link"]

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
            open_command_expr={"hand_l_proximal_joint": 0.75, "hand_r_proximal_joint": 0.75},
            close_command_expr={"hand_l_proximal_joint": 0.0, "hand_r_proximal_joint": 0.0}
        )
        self.actions.head_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["head_pan_joint", "head_tilt_joint"], scale=1.0, debug_vis=True
        )
        # override command generator body
        # end-effector is along z-direction
        # self.commands.ee_pose.body_name = "base_footprint"
        # self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)

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
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/wrist_roll_link",
                    name="wrist",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.0),
                    ),
                ),
            ],
        )

        # add lidar
        self.scene.lidar = HSRB_LIDAR_CFG.copy()

        self.commands.ee_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name="base_footprint",
            resampling_time_range=(50.0, 100.0),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(
                pos_x=(-3.5, 3.5),
                pos_y=(-3.5, 3.5),
                pos_z=(0.2, 0.5),
                roll=(0.0, 0.0),
                pitch=(-math.pi/2, -math.pi/2),  # depends on end-effector axis
                yaw=(-math.pi, -math.pi),
            ),
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
