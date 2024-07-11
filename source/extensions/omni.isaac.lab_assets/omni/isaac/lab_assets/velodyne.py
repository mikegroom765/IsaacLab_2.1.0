# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Velodyne LiDAR sensors."""


from omni.isaac.lab.sensors import RayCasterCfg, patterns

##
# Configuration
##

VELODYNE_VLP_16_RAYCASTER_CFG = RayCasterCfg(
    attach_yaw_only=False,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=16, vertical_fov_range=(-15.0, 15.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=0.2
    ),
    debug_vis=True,
    max_distance=100,
)

"""Configuration for Velodyne Puck LiDAR (VLP-16) as a :class:`RayCasterCfg`.

Reference: https://velodynelidar.com/wp-content/uploads/2019/12/63-9229_Rev-K_Puck-_Datasheet_Web.pdf
"""

HOKUYO_UST_20LX_RAYCASTER_CFG = RayCasterCfg(
    attach_yaw_only=False,
    pattern_cfg=patterns.LidarPatternCfg(
        channels=1, vertical_fov_range=(0.0, 0.0), horizontal_fov_range=(-135.0, 135.0), horizontal_res=0.25
    ),
    debug_vis=True,
    max_distance=60,
)

"""Configuration for Hokuyo UST-20LX LiDAR as a :class:`RayCasterCfg`.

Reference: https://www.hokuyo-aut.jp/search/single.php?serial=167#spec
"""