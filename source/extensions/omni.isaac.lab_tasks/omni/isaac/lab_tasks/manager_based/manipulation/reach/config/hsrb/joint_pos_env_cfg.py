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
import omni.isaac.lab.sim as sim_utils

##
# Pre-defined configs
##
from omni.isaac.lab_assets.hsrb import HSRB_CFG  # isort:skip


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
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"], scale=0.5
        )
        self.actions.base_action = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=["joint_x", "joint_y", "joint_rz"], use_default_offset=True, scale=0.5
        )
        self.actions.gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot", joint_names=["hand_l_proximal_joint", "hand_r_proximal_joint"], scale=0.5
        )
        # self.actions.head_action = mdp.JointPositionActionCfg(
        #     asset_name="robot", joint_names=["head_pan_joint", "head_tilt_joint"], scale=0.5, use_default_offset=True
        # )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "base_footprint"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


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
