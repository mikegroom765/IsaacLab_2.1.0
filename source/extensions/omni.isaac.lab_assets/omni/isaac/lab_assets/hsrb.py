# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Toyota HSRB.

The following configurations are available:

* :obj:`HSRB_CFG`: hsrb4s robot 

Reference: https://git.hsr.io/tmc/hsr-omniverse

"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

HSRB_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/standalone/hsrb/hsrb4s.usd", 
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "wrist_roll_joint": 0.5,
            "wrist_flex_joint": -0.392699,
            "arm_roll_joint": 0.0,
            "arm_flex_joint": -1.570796,
            "arm_lift_joint": 0.1,
            "torso_lift_joint": 0.1,
            "hand_l_proximal_joint": 0.75,
            "hand_r_proximal_joint": 0.75,
            "joint_rz": 0.0,
            "head_pan_joint": 0.0,
            "joint_y": 0.0,
            "head_tilt_joint": 0.0,
            "joint_x": 0.0,            
        },
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["joint_x", "joint_y", "joint_rz"],
            velocity_limit={
                "joint_x": 0.2, # m/s?
                "joint_y": 0.2, # m/s?
                "joint_rz": 1.0, # rad/s? seen different values here: 1.5 and 1.0. moveit uses 1.0, paper uses 1.5
            },
            effort_limit=100000.0,
            stiffness=0.0,
            damping=100,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"],
            velocity_limit={
                "arm_lift_joint": 0.2,
                "arm_flex_joint": 1.2,
                "arm_roll_joint": 2.0,
                "wrist_flex_joint": 1.5,
                "wrist_roll_joint": 1.5,
            },
            effort_limit=100.0, # 10.0 
            stiffness={
                "arm_lift_joint": 900.0,
                "arm_flex_joint": 2000.0,
                "arm_roll_joint": 1000.0,
                "wrist_flex_joint": 900.0,
                "wrist_roll_joint": 900.0,
            },
            damping={
                "arm_lift_joint": 100.0,
                "arm_flex_joint": 20.0,
                "arm_roll_joint": 1.0,
                "wrist_flex_joint": 0.0,
                "wrist_roll_joint": 0.0,
            },
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_pan_joint", "head_tilt_joint"],
            effort_limit=50.0, # 5.0
            velocity_limit={
                "head_pan_joint": 1.0,
                "head_tilt_joint": 1.0,
            },
            stiffness={
                "head_pan_joint": 1200.0,
                "head_tilt_joint": 1200.0,
            },
            damping={
                "head_pan_joint": 10.0,
                "head_tilt_joint": 10.0, 
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["hand_l_proximal_joint", "hand_r_proximal_joint"],
            effort_limit=100.0,# 1.0
            # stiffness=1e5, below values are from moveit config, should they be higher?
            stiffness={
                "hand_l_proximal_joint": 5000.0,
                "hand_r_proximal_joint": 5000.0,
            },
            damping=1000, # 0.1
        ),
    },
)
"""Configuration of HSRB using implicit actuator models.

The following control configuration is used:

* Base: velocity control with damping
* Arm: position control with damping (contains default position offsets)
* Hand: currently position control - TODO: binary close/open control

"""