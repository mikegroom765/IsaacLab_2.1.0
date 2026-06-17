# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`omni.isaac.lab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight after given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

def modify_reward_parameters(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, parameters: dict, num_steps: int):
    """Curriculum that modifies reward parameters after given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        parameters: The updated parameters of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """    
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # check if term_cfg.params dict has same keys as parameters
        term_cfg.params=parameters
        env.reward_manager.set_term_cfg(term_name, term_cfg)


def modify_command_parameters(env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str, parameters: dict, num_steps: int):
    """Curriculum that modifies command parameters after given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        command_name: The name of the command.
    """
    if env.common_step_counter > num_steps:
        # obtain command settings
        command_term = env.command_manager.get_term(command_name)
        # update command settings
        command_term.cfg.ranges.pos_x = parameters['pos_x']
        command_term.cfg.ranges.pos_y = parameters['pos_y']
        command_term.cfg.ranges.pos_z = parameters['pos_z']

        env.command_manager.set_term(name=command_name, term=command_term)
        # print(f"Updated command term: {env.command_manager.get_term(command_name)}")