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
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import CurriculumTermCfg

from .rewards import position_command_error_frame_shaped

from omni.isaac.lab.utils.math import quat_error_magnitude
import math

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def terrain_levels_pos_ori(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], position_thresholds: List[float], orientation_thresholds: List[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the error of positon and orientation of robot end effector when commanded to reach a target. 

    This term is used to increase the difficulty of the terrain when the robot gets close enough and decrease the
    difficulty when the robot is less than the thresholds required by the commanded pose.

    Args:
        position_thresholds: The thresholds for the position error of the robot end effector. The first value is the
            threshold for the robot to move to a harder terrain, and the second value is the threshold for the robot to
            move to an easier terrain. 
        orientation_thresholds: The thresholds for the orientation error of the robot end effector. The first value is the
            threshold for the robot to move to a harder terrain, and the second value is the threshold for the robot to
            move to an easier terrain.

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
    move_up = (position_error < position_thresholds[0]) & (orientation_error < orientation_thresholds[0])
    # robots that don't get close enough go to simpler terrains
    move_down = (position_thresholds[1] > 1.0) | (orientation_error > math.pi / orientation_thresholds[1])
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_successful_envs(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    terrain: TerrainImporter = env.scene.terrain

    if not hasattr(position_command_error_frame_shaped, 'successful_envs'):
        successful_envs = []
    else:
        successful_envs = position_command_error_frame_shaped.successful_envs
    
    # move up env ids that are successful - successful envs is list containing env ids that are successful
    move_up = torch.tensor([env_id in successful_envs for env_id in env_ids], dtype=torch.bool).to(env.unwrapped.device)
    # move down env ids that are not successful
    move_down = ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def goal_levels_range(
        env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str, min_x: torch.tensor, min_y: torch.tensor, max_x: torch.tensor, max_y: torch.tensor, level_intervals: float
):
    """Curriculum that modifies the goal position sampling range after successfully reaching a goal. 
    Samling range is increased when the goal is reached and decreased when the goal is not reached.
    
    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        min_y: The minimum y value for the goal position sampling range. Shape: [2], [-ve range min, +ve range min]
        min_x: The minimum x value for the goal position sampling range. Shape: [2], [-ve range min, +ve range min] 
        max_x: The maximum x value for the goal position sampling range. Shape: [2], [-ve range max, +ve range max]
        max_y: The maximum y value for the goal position sampling range. Shape: [2], [-ve range max, +ve range max]
        num_steps_list: The list of number of steps after which the change should be applied.
    """

    if not hasattr(position_command_error_frame_shaped, 'successful_envs'):
        successful_envs = []
    else:
        successful_envs = position_command_error_frame_shaped.successful_envs

    # get a mask of successful envs
    move_up = torch.tensor([env_id in successful_envs for env_id in env_ids], dtype=torch.bool).to(env.unwrapped.device)
    # get a mask of unsuccessful envs
    move_down = ~move_up

    # step size for each direction with level_intervals as the number of steps between min and max
    x_diff_upper = (max_x[1] - min_x[1]) / level_intervals
    x_diff_lower = (max_x[0] - min_x[0]) / level_intervals
    y_diff_upper = (max_y[1] - min_y[1]) / level_intervals
    y_diff_lower = (max_y[0] - min_y[0]) / level_intervals

    # obtain command settings
    command_term = env.command_manager.get_term(command_name)
    command_term.cfg.goal_level_range_sampling = True

    # update command settings
    # new min pos_x value and max pos_x value
    command_term.cfg.ranges[env_ids, 0, 0] = command_term.cfg.ranges[env_ids, 0, 0] + (x_diff_lower * move_up) - (x_diff_lower * move_down)
    command_term.cfg.ranges[env_ids, 0, 1] = command_term.cfg.ranges[env_ids, 0, 1] + (x_diff_upper * move_up) - (x_diff_upper * move_down)
    # new min pos_y value and max pos_y value
    command_term.cfg.ranges[env_ids, 1, 0] = command_term.cfg.ranges[env_ids, 1, 0] + (y_diff_lower * move_up) - (y_diff_lower * move_down)
    command_term.cfg.ranges[env_ids, 1, 1] = command_term.cfg.ranges[env_ids, 1, 1] + (y_diff_upper * move_up) - (y_diff_upper * move_down)

    # clip the values to the min and max values
    command_term.cfg.ranges[env_ids, 0, 0] = torch.clamp(command_term.cfg.ranges[env_ids, 0, 0], min=max_x[0], max=min_x[0])
    command_term.cfg.ranges[env_ids, 0, 1] = torch.clamp(command_term.cfg.ranges[env_ids, 0, 1], min=min_x[1], max=max_x[1])
    command_term.cfg.ranges[env_ids, 1, 0] = torch.clamp(command_term.cfg.ranges[env_ids, 1, 0], min=max_y[0], max=min_y[0])
    command_term.cfg.ranges[env_ids, 1, 1] = torch.clamp(command_term.cfg.ranges[env_ids, 1, 1], min=min_y[1], max=max_y[1])

    # set the updated command settings
    env.command_manager.set_term(name=command_name, term=command_term)

    # return the mean upper range
    return torch.mean(command_term.cfg.ranges[:, 0, 1] + command_term.cfg.ranges[:, 1, 1])


class goal_levels(ManagerTermBase):
    """
    Class based approach to the goals_level curriculum function.

    Curriculum that modifies the goal position sampling range after successfully reaching a goal.
    Both the upper and lower bounds are increased when the goal is reached and decreased when the goal is not reached.
    
    When highest level is solved, a random level is selected.

    This function uses a square shaped goal position sampling range.
    
    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        min_x: The new minimum x value for the goal position sampling range. Shape: [1]
        max_x: The new maximum x value for the goal position sampling range. Shape: [1]
        level_intervals: The number of levels to use to increment between mix_x and max_x. Levels are zero indexed.

    Note:
        In this function we are assuming that the goal position sampling range is square shaped. This means we do not
        need to update the y values of the goal position sampling range!
    """

    all_env_levels: torch.Tensor
    successful_envs: torch.Tensor

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        # initialise the base class
        super().__init__(cfg, env)

        self.all_env_levels = torch.zeros(env.num_envs).to(env.unwrapped.device)
        self.successful_envs = torch.zeros(env.num_envs).to(env.unwrapped.device)


    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str, min_x: torch.tensor, max_x: torch.tensor, level_intervals: int, successful_envs_term: str):

        self.successful_envs = env.reward_manager.get_term_cfg(successful_envs_term).func.successful_envs

        # 1. Build the move_up and move_down masks for successful and unsuccessful envs
        move_up = self.successful_envs[env_ids]
        move_down = ~move_up

        # 2. Compute level step size
        x_diff = (max_x - min_x) / level_intervals

        # 3. Obtain command settings
        command_term = env.command_manager.get_term(command_name)
        command_term.cfg.goal_level_sampling = True

        # 4. Update the levels for these envs
        self.all_env_levels[env_ids] += move_up.float() - move_down.float()
        
        # 4a. Clamp the levels here so they don't go below 0 or above level_intervals + 1
        # level_intervals + 1 is used to account for the random level selection when highest level is solved
        self.all_env_levels.clamp_(min=0, max=level_intervals + 1)

        # 5. Update the command ranges for these envs
        updated_levels = self.all_env_levels[env_ids]

        # For each env in env_ids, the range is [ x_diff*(lvl-1) + min_x, x_diff*(lvl) + min_x ]
        lower_bound = x_diff * (updated_levels - 1) + min_x
        upper_bound = x_diff * updated_levels + min_x

        # if level is 0, the range is [0, min_x] (assuming min_x > 0).
        is_level_zero = (updated_levels == 0)
        if is_level_zero.any():
            lower_bound[is_level_zero] = 0.0
            upper_bound[is_level_zero] = min_x

        is_above_max = (updated_levels > level_intervals)
        if is_above_max.any():
            random_levels = torch.randint(low=0, high=level_intervals, size=(is_above_max.sum(),), device=env.unwrapped.device)
            
            # index into lower/upper bounds 
            idx_above_max = torch.where(is_above_max)[0]
            lower_bound[idx_above_max] = x_diff * (random_levels - 1) + min_x
            upper_bound[idx_above_max] = x_diff * random_levels + min_x

            # if level is 0, the range is [0, min_x] (assuming min_x > 0).
            is_level_zero = (random_levels == 0)
            if is_level_zero.any():
                idx_l0 = idx_above_max[is_level_zero]
                lower_bound[idx_l0] = 0.0
                upper_bound[idx_l0] = min_x

            is_level_one = (random_levels == 1)
            if is_level_one.any():
                idx_l1 = idx_above_max[is_level_one]
                lower_bound[idx_l1] = min_x
                upper_bound[idx_l1] = min_x + x_diff

        self.all_env_levels.clamp_(min=0, max=level_intervals)

        # 6. Update the command settings
        command_term.cfg.ranges[env_ids, 0, 0] = lower_bound
        command_term.cfg.ranges[env_ids, 0, 1] = upper_bound

        # 7. Set the updated command settings
        env.command_manager.set_term(name=command_name, term=command_term)

        # 8. Return the average level
        return self.all_env_levels.mean()

    def set_succesful_envs(self, successful_envs: torch.Tensor):
        goal_levels.successful_envs = successful_envs
        
        
        
class lift_cube_curriculum(ManagerTermBase):
    """Class based approach to a curriculum for the lift_cube task.
    
    This curriculum alters three aspects of the environment:
        1. The intial distance of the robot to the cube.
        2. The initial position of the cube.
        3. The robot joint angles - specifically to avoid the tables.
        
    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        level_params (dict): A dictionary containing information on the curriculum levels. The dictionary should contain
                             keys corresponding to the event terms to be modified. The values should be
                             the names of the event terms to be updated with the new levels.:
                             {
                                 "initial_distance": NAME,
                                 "initial_position": NAME,
                                 "robot_joint": NAME"
                             }
        level_intervals (list): The number of levels to use to increment between min and max level values. Levels are 
                                zero indexed. The list should be ordered as follows:
                                [initial_distance_levels, initial_position_levels]
    """
    
    all_env_levels: torch.Tensor
    successful_envs: torch.Tensor

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        # initialise the base class
        super().__init__(cfg, env)

        self.all_env_levels = torch.zeros(env.num_envs).to(env.unwrapped.device)
        self.successful_envs = torch.zeros(env.num_envs).to(env.unwrapped.device)
        self.initialised = False
        
    def __call__(self, env: ManagerBasedRLEnv, env_ids: Sequence[int], event_terms: list[str], level_intervals: int, successful_envs_term: str):
        """This function is called when the curriculum term is executed. All we do here is keep track and update the levels
        for the environments in env_ids. Event settings are updated based on the levels, this is handled in the respective
        event terms."""        
        
        self.successful_envs = env.reward_manager.get_term_cfg(successful_envs_term).func.successful_envs
        
        # 1. Build the move_up and move_down masks for successful and unsuccessful envs
        move_up = self.successful_envs[env_ids]
        move_down = ~move_up

        # 2. Compute level step size
        # initial_distance_diff = (level_params[event_terms[0]][1] - level_params[event_terms[0]][0]) / level_intervals
        # initial_position_diff = (level_params[event_terms[1]][1] - level_params[event_terms[1]][0]) / level_intervals
        
        # 3. Obtain event settings
        initial_distance_event = env.event_manager.get_term_cfg(event_terms[0])
        initial_position_event = env.event_manager.get_term_cfg(event_terms[1])
        robot_joint_event = env.event_manager.get_term_cfg(event_terms[2])
        
        if not self.initialised:
            initial_distance_event.func.level_intervals = level_intervals
            initial_position_event.func.level_intervals = level_intervals
            robot_joint_event.func.level_intervals = level_intervals
            self.initialised = True
        
        # 4. Update the levels for these envs
        self.all_env_levels[env_ids] += move_up.float() - move_down.float()
        self.all_env_levels.clamp_(min=0, max=level_intervals)
        
        # 5. Update the event settings for these envs
        updated_levels = self.all_env_levels[env_ids]
        
        initial_distance_event.func.current_levels[env_ids] = updated_levels
        initial_position_event.func.current_levels[env_ids] = updated_levels
        robot_joint_event.func.current_levels[env_ids] = updated_levels
        
        return self.all_env_levels.mean()
        