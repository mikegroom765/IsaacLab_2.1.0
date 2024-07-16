# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen
import omni.isaac.lab.terrains.trimesh as mesh_gen


from ..terrain_generator_cfg import TerrainGeneratorCfg

min_num_objects = 5
max_num_objects = 30
min_height = 0.5
max_height = 1.0

HSRB_REACH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=2,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "repeated_box": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            object_type='box',
            object_params_start=mesh_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=min_num_objects, height=min_height, size=(0.5, 0.5), max_yx_angle=0.0, degrees=True
            ),
            object_params_end=mesh_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=max_num_objects, height=max_height, size=(0.5, 0.5), max_yx_angle=0.0, degrees=True
            ),
            platform_width=0.5,
        ),
        "repeated_cylinder": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            object_type='cylinder',
            object_params_start=mesh_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=min_num_objects, height=min_height, radius=0.25, max_yx_angle=0.0, degrees=True
            ),
            object_params_end=mesh_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=max_num_objects, height=max_height, radius=0.25, max_yx_angle=0.0, degrees=True
            ),
            platform_width=0.5,
        ),
            
    },
)
"""HSRB Reach terrains configuration."""
