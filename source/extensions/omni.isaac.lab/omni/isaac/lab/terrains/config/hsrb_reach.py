# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import random
import json

import omni.isaac.lab.terrains as terrain_gen
import omni.isaac.lab.terrains.trimesh as mesh_gen


from ..terrain_generator_cfg import TerrainGeneratorCfg, VariedGridTerrainGeneratorCfg

min_num_objects = 0
min_num_objects_student = 0
max_num_objects = 0
min_height = 0.6
max_height = 1.0
max_height_noise = 0.6

HSRB_REACH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(5.0, 5.0),
    border_width=20.0,
    num_rows=10,
    num_cols=25,
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
                num_objects=max_num_objects, height=max_height, size=(0.5, 0.5), max_yx_angle=20.0, degrees=True
            ),
            platform_width=0.5,
            max_height_noise=max_height_noise,
        ),
        "repeated_cylinder": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            object_type='cylinder',
            object_params_start=mesh_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=min_num_objects, height=min_height, radius=0.25, max_yx_angle=0.0, degrees=True
            ),
            object_params_end=mesh_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=max_num_objects, height=max_height, radius=0.25, max_yx_angle=20.0, degrees=True
            ),
            platform_width=0.5,
            max_height_noise=max_height_noise,
        ),
        "repeated_pyramid": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
            object_params_start=mesh_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=min_num_objects, height=min_height, radius=0.25, max_yx_angle=0.0, degrees=True
            ),
            object_params_end=mesh_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=max_num_objects, height=max_height, radius=0.25, max_yx_angle=20.0, degrees=True
            ),
            platform_width=0.5,
            max_height_noise=max_height_noise,
        ),
    },
)
"""HSRB Reach terrains configuration."""

HSRB_STUDENT_REACH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(5.0, 5.0),
    border_width=20.0,
    num_rows=25,
    num_cols=25,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "repeated_box": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            object_type='box',
            object_params_start=mesh_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=min_num_objects_student, height=min_height, size=(0.5, 0.5), max_yx_angle=0.0, degrees=True
            ),
            object_params_end=mesh_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=max_num_objects, height=max_height, size=(0.5, 0.5), max_yx_angle=20.0, degrees=True
            ),
            platform_width=0.5,
            max_height_noise=max_height_noise,
        ),
        "repeated_cylinder": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            object_type='cylinder',
            object_params_start=mesh_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=min_num_objects_student, height=min_height, radius=0.25, max_yx_angle=0.0, degrees=True
            ),
            object_params_end=mesh_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=max_num_objects, height=max_height, radius=0.25, max_yx_angle=20.0, degrees=True
            ),
            platform_width=0.5,
            max_height_noise=max_height_noise,
        ),
        "repeated_pyramid": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
            object_params_start=mesh_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=min_num_objects_student, height=min_height, radius=0.25, max_yx_angle=0.0, degrees=True
            ),
            object_params_end=mesh_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=max_num_objects, height=max_height, radius=0.25, max_yx_angle=20.0, degrees=True
            ),
            platform_width=0.5,
            max_height_noise=max_height_noise,
        ),
    },
)
"""HSRB Student Reach terrains configuration."""

HSRB_REACH_CORRIDOR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "corridor_terrain": terrain_gen.MeshCorridorTerrainCfg(
            corridor_height=2.0,
            corridor_width=3.0,
            corridor_length=6.0,
            num_obstacles=5,
            box_width=0.3,
            box_height=(0.2, 1.0),
            robot_width=0.4,
        ),
    },
)

HSRB_REACH_L_CORRIDOR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "corridor_terrain": terrain_gen.LShapedMeshCorridorTerrainCfg(
            corridor_height=2.0,
            corridor_width=3.0,
            corridor_length=6.0,
            num_obstacles=8,
            box_width=0.3,
            box_height=(0.2, 1.0),
            robot_width=0.4,
        ),
    },
)

def load_configurations(filename):
    """
    Load configurations (list of list of [i, j]) from JSON 
    and return them as a list of sets of tuples for direct iteration.
    """
    with open(filename, 'r') as f:
        configs_as_lists = json.load(f)
    # Convert each config to a set of (i, j) tuples
    configs = []
    for config in configs_as_lists:
        configs.append(set(tuple(cell) for cell in config))
    return configs

def generate_grid_reach_terrains_cfg(
    size: tuple[float, float],
    num_rows: int,
    num_cols: int,
    grid_cell_width: float,
    grid_cols: int,
    grid_rows: int,
    table_height_range: tuple[float, float],
    max_filled_cells: int,
    min_filled_cells: int,
    exclude_test_set: bool = False,
    test_set_config_filename: str | None = None,
    max_attempts: int = 1000,
):
    
    if exclude_test_set:
        if test_set_config_filename is None:
            raise ValueError("test_set_config_filename must be provided if exclude_test_set is True")
        test_set_configs = load_configurations(test_set_config_filename)
    else:
        test_set_configs = []
    
    # generate a grid of terrains (list of list of tuples(x, y)) where each sub-terrain is a grid of cells
    tables_locs = [[(i, j)] for i in range(num_cols) for j in range(num_rows)] # list of list of tuples(x, y)

    n_terrains = num_rows * num_cols # number of sub-terrains 

    for terrain in range(n_terrains): # for each sub-terrain
        num_filled_cells = random.randint(min_filled_cells, max_filled_cells)
        terrain_table_list = []
        attempt_count = 0

        while True: # loop to check against test set
            attempt_count += 1
            if attempt_count > max_attempts:
                # If we exceed max attempts, assign empty and move on 
                # (or raise an error, depending on your preference)
                print(f"Warning: Could not find a new configuration for terrain {terrain} after {max_attempts} attempts.")
                tables_locs[terrain] = []
                break
            # generate random locations for the filled cells - without duplicates
            if num_filled_cells == 0:
                candidate_set = set()
            for _ in range(num_filled_cells):
                while True: # loop to sample grid cells
                    i = random.randint(0, grid_rows-1)
                    j = random.randint(0, grid_cols-1)
                    # don't sample in the center of the grid plus a margin of 1 cell
                    if (i, j) not in tables_locs[terrain] and (i < (grid_rows//2)-1 or i > (grid_rows//2)+1) or (j < (grid_cols//2)-1 or j > (grid_cols//2)+1):
                        if ((i != 7) and (j != 4)):
                            terrain_table_list.append((i, j))
                            break
                # tables_locs[terrain] = terrain_table_list
                
            candidate_set = set(terrain_table_list)
            if num_filled_cells == 0:
                tables_locs[terrain] = []
            # Now check if this candidate set is in the test set
            if exclude_test_set:
                # Compare with each config in test_set_configs
                is_in_test_set = any(candidate_set == test_conf for test_conf in test_set_configs)
                if is_in_test_set:
                    # This config matches one in the test set; regenerate
                    continue
                else:
                    # It's a new configuration => accept and break
                    tables_locs[terrain] = terrain_table_list
                    break
            else:
                # If we're not excluding test set, just accept immediately
                tables_locs[terrain] = terrain_table_list
                break
            

    return HSRB_GRID_REACH_TERRAINS_CFG(size=size,
                                        num_rows=num_rows, 
                                        num_cols=num_cols, 
                                        grid_cell_width=grid_cell_width, 
                                        grid_cols=grid_cols, 
                                        grid_rows=grid_rows,
                                        tables_locs=tables_locs, 
                                        table_height_range=table_height_range)

def HSRB_GRID_REACH_TERRAINS_CFG(
        size: tuple[float, float],
        num_rows: int, 
        num_cols: int, 
        grid_cell_width: float, 
        grid_cols: int, 
        grid_rows: int,
        tables_locs: list[list[tuple[int, int]]],
        table_height_range: tuple[float, float],):
    return VariedGridTerrainGeneratorCfg(
        size=size,
        border_width=20.0,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        table_locs=tables_locs,
        sub_terrains={
            "grid_terrain": terrain_gen.MeshGridTerrainCfg(
                grid_cell_width=grid_cell_width,
                grid_cols=grid_cols,
                grid_rows=grid_rows,
                # tables_locs=tables_locs,
                table_height_range=table_height_range,
            ),
        },
    )
    
def HSRB_LIFT_CUBE_TERRAINS_CFG(
    size: tuple[float, float],
    table_height_range: tuple[float, float],
    table_length: float,
    table_width: float,
    num_rows: int = 1,
    num_cols: int = 1,
):
    num_envs = num_rows * num_cols
    table_heights = [random.uniform(*table_height_range) for _ in range(num_envs)]
    # table_heights = [0.2, 0.4, 0.6, 0.8]
    # create a dictionary of sub-terrain configurations each with a unique table height
    sub_terrains = {}
    for i in range(num_envs):
        sub_terrains[f"table_{i}"] = terrain_gen.LiftCubeEnvTerrainCfg(
            table_height=table_heights[int(i)],
            table_length=table_length,
            table_width=table_width,
        )
    
    return VariedGridTerrainGeneratorCfg(
        size=size,
        border_width=20.0,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains=sub_terrains,
        random_sample=False,
    )