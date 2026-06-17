# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv
    from omni.isaac.lab.managers.command_manager import CommandTerm

from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import TerminationTermCfg

from copy import deepcopy

"""
MDP terminations.
"""


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length


def command_resample(env: ManagerBasedRLEnv, command_name: str, num_resamples: int = 1) -> torch.Tensor:
    """Terminate the episode based on the total number of times commands have been re-sampled.

    This makes the maximum episode length fluid in nature as it depends on how the commands are
    sampled. It is useful in situations where delayed rewards are used :cite:`rudin2022advanced`.
    """
    command: CommandTerm = env.command_manager.get_term(command_name)
    # return torch.logical_and((command.time_left <= env.step_dt), (command.command_counter == num_resamples))
    return command.command_counter == num_resamples

def goal_reached(env: ManagerBasedRLEnv, frame_name: str, command_name: str, reward_term_name: str) -> torch.Tensor:
    """Terminate the episode when the agent successfully reaches the goal.

    Args:
        frame_name (str): Name of the FrameTransformer sensor used to compute the current position of the agent.
        command_name (str): Name of the command term used to compute the target position of the agent.
    """
    asset: Articulation = env.scene["robot"]
    frame_pos = asset.data.body_pos_w[:, asset.find_bodies(frame_name)[0], :].squeeze(1)    
    command = env.command_manager.get_command(command_name) 
    position_threshold = env.reward_manager.get_term_cfg(reward_term_name).params["position_threshold"]
    
    distance = torch.sqrt(((frame_pos - command[:, :3]) ** 2).sum(dim=-1))
    terminations = (distance < position_threshold)
    
    return terminations

class give_up_action(ManagerTermBase):
    
    distance_success_envs: torch.Tensor
    time_success_envs: torch.Tensor
    
    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        
        
        self._initialized = False
        self.env = env
        self.command_name = cfg.params["command_name"]
        self.frame_name = cfg.params["frame_name"]
        self.initial_ee_distance = torch.ones(env.num_envs, device=env.unwrapped.device)
        self.command_storage = deepcopy(env.command_manager.get_command(self.command_name))
        
    def _initialize_buffers(self, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        
        
        self.asset = self.env.scene[asset_cfg.name]
        frame_pos = self.asset.data.body_pos_w[:, self.asset.find_bodies(self.frame_name)[0], :].squeeze(1)
        
        # frame_pos = self.env.scene[self.frame_name].data.target_pos_source[..., 0, :] 
        command = self.env.command_manager.get_command(self.command_name)
        delta = frame_pos - command[:, :3]
        self.initial_ee_distance = torch.sqrt((delta ** 2).sum(dim=1))
        self.command_storage = torch.zeros_like(command, device=self.env.unwrapped.device)
        self.distance_success_envs = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.unwrapped.device)
        self.time_success_envs = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.unwrapped.device)
        self.initial_base_position = self.asset.data.body_pos_w[:, self.asset.find_bodies("base_link")[0], :2].squeeze(1)
        self._initialized = True
    
    def __call__(self, env: ManagerBasedRLEnv, 
                 command_name: str, 
                 frame_name: str, 
                 home_threshold: float, 
                 time_threshold: float | None, 
                 distance_threshold: float | None,
                 asset_cfg: SceneEntityCfg,
                 action_name: str):
        """Terminate the episode when the agent successfully gives up on the task."""
        
        # Throw an error when both time and distance thresholds are None
        if time_threshold is None and distance_threshold is None:
            raise ValueError("Both time and distance thresholds for the give_up_reward_term cannot be None.")

        if not self._initialized:
            self._initialize_buffers()

        # extract the asset (to enable type hinting)
        # frame_pos = env.scene[frame_name].data.target_pos_source[..., 0, :]
        frame_pos = self.asset.data.body_pos_w[:, self.asset.find_bodies(self.frame_name)[0], :].squeeze(1)
        
        command = env.command_manager.get_command(command_name)
        asset = env.scene[asset_cfg.name]
        
        # get the action term
        give_up_action = env.action_manager.get_term(action_name).processed_actions.squeeze(1) # [num_envs]
        
        # 1. Detect command changes and reset positions and _success_envs if needed
        changed_mask = (command != self.command_storage).any(dim=1)
        if changed_mask.any():
            self.command_storage[changed_mask] = command[changed_mask]
            self.distance_success_envs[changed_mask] = False
            self.time_success_envs[changed_mask] = False
            self.initial_ee_distance[changed_mask] = torch.sqrt(((frame_pos - command[:, :3]) ** 2).sum(dim=1))[changed_mask]
            self.initial_base_position[changed_mask] = asset.data.body_pos_w[changed_mask, asset.find_bodies("base_link")[0], :2].squeeze(1)
            
        # 2. Compute the distance between the current position and the target position
        distance = torch.sqrt(((frame_pos - command[:, :3]) ** 2).sum(dim=-1))
        # 3. Compute the time elapsed
        time_elapsed = env.episode_length_buf * env.step_dt
        
        # 4. Compute the success masks
        if distance_threshold is not None: # If distance threshold is set
            distance_success_mask = ((self.initial_ee_distance - distance) / self.initial_ee_distance) > distance_threshold
        else:
            distance_success_mask = torch.ones_like(distance, dtype=torch.bool)
            
        if time_threshold is not None: # If time threshold is set
            time_success_mask = (time_elapsed > time_threshold)
        else:
            time_success_mask = torch.ones_like(time_elapsed, dtype=torch.bool)
            
        home_delta = self.initial_base_position - self.asset.data.body_pos_w[:, self.asset.find_bodies("base_link")[0], :2].squeeze(1) # [num_envs, 2]
        home_distance = torch.sqrt((home_delta ** 2).sum(dim=1)) # [num_envs]
        home_success_mask = home_distance < home_threshold
        
        # 5. Termination logic
        self.distance_success_envs |= distance_success_mask # remaining True if has been already True, [num_envs]
        self.time_success_envs |= time_success_mask # remaining True if has been already True, [num_envs]

        # terminate only on successful give up action
        terminations = give_up_action & self.distance_success_envs & self.time_success_envs & home_success_mask
        
        return terminations

"""
Root terminations.
"""


def bad_orientation(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle


def root_height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height

"""
Joint terminations.
"""


def joint_pos_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos > asset.data.soft_joint_pos_limits[..., 1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos < asset.data.soft_joint_pos_limits[..., 0], dim=1)
    return torch.logical_or(out_of_upper_limits[:, asset_cfg.joint_ids], out_of_lower_limits[:, asset_cfg.joint_ids])


def joint_pos_out_of_manual_limit(
    env: ManagerBasedRLEnv, bounds: tuple[float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint positions are outside of the configured bounds.

    Note:
        This function is similar to :func:`joint_pos_out_of_limit` but allows the user to specify the bounds manually.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None:
        asset_cfg.joint_ids = slice(None)
    # compute any violations
    out_of_upper_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] > bounds[1], dim=1)
    out_of_lower_limits = torch.any(asset.data.joint_pos[:, asset_cfg.joint_ids] < bounds[0], dim=1)
    return torch.logical_or(out_of_upper_limits, out_of_lower_limits)


def joint_vel_out_of_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside of the soft joint limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    limits = asset.data.soft_joint_vel_limits
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > limits[:, asset_cfg.joint_ids], dim=1)


def joint_vel_out_of_manual_limit(
    env: ManagerBasedRLEnv, max_velocity: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's joint velocities are outside the provided limits."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute any violations
    return torch.any(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) > max_velocity, dim=1)


def joint_effort_out_of_limit(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when effort applied on the asset's joints are outside of the soft joint limits.

    In the actuators, the applied torque are the efforts applied on the joints. These are computed by clipping
    the computed torques to the joint limits. Hence, we check if the computed torques are equal to the applied
    torques.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # check if any joint effort is out of limit
    out_of_limits = torch.isclose(
        asset.data.computed_torque[:, asset_cfg.joint_ids], asset.data.applied_torque[:, asset_cfg.joint_ids]
    )
    return torch.any(out_of_limits, dim=1)


"""
Contact sensor.
"""


def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
    
def filtered_illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix_w = contact_sensor.data.force_matrix_w # [num_envs, num_bodies, num_filtered_bodies, 3]
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(force_matrix_w, dim=-1), dim=1)[0] > threshold, dim=1
    )
