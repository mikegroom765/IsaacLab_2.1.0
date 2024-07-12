# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporter

from omni.isaac.lab.utils.math import quat_error_magnitude
import math

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`omni.isaac.lab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("ee_pose")
    
    # compute the distance between the robot and the target ee_pose
    command = env.command_manager.get_command("ee_pose")

    command_position = command[env_ids, :3]
    command_orientation = command[env_ids, 3:7]

    data = env.scene["ee_frame"].data

    ee_pos = env.scene["ee_frame"].data.target_pos_source[..., 0, :]
    ee_quat = env.scene["ee_frame"].data.target_quat_source[..., 0, :]

    position_error = torch.norm(ee_pos[env_ids, :] - command_position, dim=1)

    orientation_error = []
    # calculate the orientation error for each environment
    for index, value in enumerate(env_ids):
        
        error = quat_error_magnitude(ee_quat[index, :], command_orientation[index, :])
        # append the error to the list
        orientation_error.append(error)

    orientation_error = torch.tensor(orientation_error).to("cuda")

    # robots that get ee close enough progress to harder terrains
    move_up = (position_error < 0.05) & (orientation_error < 0.1)
    # robots that don't get close enough go to simpler terrains
    move_down = (position_error > 1.0) | (orientation_error > math.pi / 2)
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
