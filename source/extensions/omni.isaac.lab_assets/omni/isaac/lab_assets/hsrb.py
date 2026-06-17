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
from omni.isaac.lab.sensors import RayCasterCfg, patterns, TiledCameraCfg
from .velodyne import HOKUYO_UST_20LX_RAYCASTER_CFG
import torch

HSRB_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/standalone/hsrb/hsrb4s.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            # fix_root_link=False,
            # articulation_enabled=True,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        activate_contact_sensors=True,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True,contact_offset=0.001, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.01),
        joint_pos={
            "wrist_roll_joint": 0.0,
            "wrist_flex_joint": 0.0, # -0.392699,
            "arm_roll_joint": 0.0,
            "arm_flex_joint": -1.570796,
            "arm_lift_joint": 0.3,
            "torso_lift_joint": 0.15,
            "hand_l_proximal_joint": 0.75,
            "hand_r_proximal_joint": 0.75,
            # "head_pan_joint": 0.0,
            # "head_tilt_joint": -0.79,
            "base_l_drive_wheel_joint": 0.0,
            "base_r_drive_wheel_joint": 0.0,
            "base_roll_joint": 0.0,
        },
        joint_vel={
            "wrist_roll_joint": 0.0,
            "wrist_flex_joint": 0.0,
            "arm_roll_joint": 0.0,
            "arm_flex_joint": 0.0,
            "arm_lift_joint": 0.0,
            "torso_lift_joint": 0.0,
            "hand_l_proximal_joint": 0.0,
            "hand_r_proximal_joint": 0.0,
            "base_l_drive_wheel_joint": 0.0,
            "base_r_drive_wheel_joint": 0.0,
            "base_roll_joint": 0.0,
        },
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["base_l_drive_wheel_joint", "base_r_drive_wheel_joint", "base_roll_joint"],
            velocity_limit={
                "base_l_drive_wheel_joint": 8.0, # these values are from hsr-omniverse velocity limits
                "base_r_drive_wheel_joint": 8.0, 
                "base_roll_joint": 8.0, 
            },
            effort_limit={
                "base_l_drive_wheel_joint": 664.020019, # default values from offical hsr USD file - this is different than the URDF file
                "base_r_drive_wheel_joint": 664.020019,
                "base_roll_joint": 2067.599853,},
            stiffness=15000.0, # default values from offical hsr-omniverse hsr.py file
            damping=0.0, # default values from offical hsr-omniverse hsr.py file
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"],
            velocity_limit={
                "arm_lift_joint": 0.2, #  TODO: check these values - values in hsr-omniverse are different
                "arm_flex_joint": 1.2,
                "arm_roll_joint": 2.0,
                "wrist_flex_joint": 1.5,
                "wrist_roll_joint": 1.5,
            },
            effort_limit=10000.0, # 10.0 
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
                "wrist_flex_joint": 100.0,
                "wrist_roll_joint": 0.0,
            },
        ),
        # "head": ImplicitActuatorCfg(
        #     joint_names_expr=["head_pan_joint", "head_tilt_joint"],
        #     effort_limit=50.0, # 5.0
        #     velocity_limit={
        #         "head_pan_joint": 1.0,
        #         "head_tilt_joint": 1.0,
        #     },
        #     stiffness={
        #         "head_pan_joint": 1200.0,
        #         "head_tilt_joint": 1200.0,
        #     },
        #     damping={
        #         "head_pan_joint": 10.0,
        #         "head_tilt_joint": 10.0, 
        #     },
        # ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["hand_l_proximal_joint", "hand_r_proximal_joint", "hand_l_distal_joint", "hand_r_distal_joint"],
            effort_limit=200.0,# 1.0
            # stiffness=1e5, below values are from moveit config, should they be higher?
            stiffness={
                "hand_l_proximal_joint": 2.0,
                "hand_r_proximal_joint": 2.0,
                "hand_l_distal_joint": 2.0,
                "hand_r_distal_joint": 2.0,
            },
            damping=0.5, # 0.1
        ),
    },
)
"""Configuration of HSRB using implicit actuator models.

The following control configuration is used:

* Base: velocity control with damping
* Arm: position control with damping (contains default position offsets)
* Hand: binary close/open control

"""

HSRB_STUDENT_CFG = HSRB_CFG

# HSRB_STUDENT_CFG = HSRB_CFG.replace(
#     spawn=sim_utils.UsdFileCfg(
#         # usd_path="/workspace/isaaclab/source/standalone/hsrb/hsrb4s.usd",
#         usd_path="/workspace/isaaclab/source/standalone/hsrb/hsrb4s",
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=4,
#             sleep_threshold=0.005,
#             stabilization_threshold=0.001,
#         ),
#         activate_contact_sensors=True,
#     )
# )


HSRB_SCANDOTS_CFG = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",
    offset=RayCasterCfg.OffsetCfg(pos=(0.8, 0.0, 20.0), rot=(0.0, 0.0, 0.0, 1.0)),
    attach_yaw_only=True,
    pattern_cfg=patterns.GridPatternCfg(resolution=0.16, size=[1.6, 1.6]), # 121 points
    debug_vis=True,
    mesh_prim_paths=["/World/ground"],
)

### HSRB_SCANDOTS_CFG for active perception
# HSRB_SCANDOTS_CFG = RayCasterCfg(
#     prim_path="{ENV_REGEX_NS}/Robot/head_rgbd_sensor_link",
#     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 1.0, 20.0), rot=(1.0, 0.0, 0.0, 0.0)), # rot=(0.0, 0.0, 0.0, 1.0)
#     attach_yaw_only=True,
#     pattern_cfg=patterns.GridPatternCfg(resolution=0.16, size=[1.6, 1.6]), # 121 points
#     debug_vis=True,
#     mesh_prim_paths=["/World/ground"],
# )

HSRB_LIDAR_CFG = HOKUYO_UST_20LX_RAYCASTER_CFG.replace(
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
    prim_path="{ENV_REGEX_NS}/Robot/base_range_sensor_link",
    mesh_prim_paths=["/World/ground"],
    debug_vis=False,
)
"""Configuration of the HSRB's Hokuyo UST-20LX lidar sensor."""

HSRB_DEPTH_CAMERA_CFG = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_rgbd_sensor_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 1.0, 0.0)),
        attach_yaw_only=False,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
)
"""Configuration of the HSRB's depth camera sensor."""

HSRB_TILED_DEPTH_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/head_rgbd_sensor_link/tiled_camera",
    offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
    data_types=["depth"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=58, clipping_range=(0.1, 10.0)
    ),
    width=87,
    height=58,
)

"""Configuration of the HSRB's depth camera sensor, implemented as a tiled camera."""

HSRB_DEFAULT_CAMERA_INTRINSICS = torch.tensor([[264.8276, 0.0000, 320.0000],
                                                [0.0000, 264.8276, 240.0000],
                                                [0.0000, 0.0000, 1.0000]]).to("cuda")


CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/isaaclab/source/standalone/hsrb/cylinder.usd",
        activate_contact_sensors=True,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True,contact_offset=0.00, rest_offset=0.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"joint_x": 1.0, "joint_y": 1.0, "joint_z": 0.5},
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["joint_x", "joint_y", "joint_z"],
            velocity_limit={
                "joint_x": 0.5,
                "joint_y": 0.5,
                "joint_z": 0.5,
            },
            effort_limit=100000.0,
            stiffness=1000.0,
            damping=100,
        ),
    },
)