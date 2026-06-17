# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence
from copy import deepcopy

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_error_magnitude_unique, quat_mul, transform_points, project_points, quat_conjugate, matrix_from_quat
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


"""Utility functions"""

def exp_scale_with_steps(env: ManagerBasedRLEnv, reward: float, scale: float) -> torch.Tensor:
    """This function scales a reward term with the number of steps taken in the episode."""
    return reward * torch.exp(torch.Tensor([(env.episode_length_buf % env.max_episode_length) * scale])).to(env.device)


"""Reward functions"""

def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

class position_command_error_frame_shaped(ManagerTermBase):
    """
    Class based approach for the position command error shaped reward function.

    For now this class does not implement the logic to check if we are doing a reward for goal ee orientation also.
    If we want to implement that, it's probably best to do it in a single class that handles both position and orientation.
    """

    previous_distance: torch.Tensor
    command_storage: torch.Tensor
    successful_envs: torch.Tensor
    env: ManagerBasedRLEnv
    # orientation_shaper: bool

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)

        self.command_name = cfg.params["command_name"] # this is now in the world frame!
        self.frame_name = cfg.params["frame_name"]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

        command = env.command_manager.get_command(self.command_name)

        self.frame_pos = self.asset.data.body_pos_w[:, self.asset.find_bodies(self.frame_name)[0], :].squeeze(1)
        self.previous_distance = torch.zeros_like(self.asset.data.body_pos_w[..., 0, 0]).to(env.unwrapped.device)
        self.command_storage = deepcopy(command)
        self.successful_envs = torch.zeros(env.num_envs, dtype=torch.bool).to(env.unwrapped.device)
        # self.orientation_shaper = cfg.params[]

        self.env = env

        self._initialized = False

    def _initialize_buffers(self):
        """
        Allocate and initialize any buffers that depend on env.num_envs or shapes that
        aren't known until env is fully set up.
        """
        # We assume the shape is known from your env
        frame_pos = self.asset.data.body_pos_w[:, self.asset.find_bodies(self.frame_name)[0], :].squeeze(1)
        command = self.env.command_manager.get_command(self.command_name)             # shape [num_envs, ...]

        num_envs = frame_pos.shape[0]
        device = frame_pos.device

        # Initialize state buffers
        self.previous_distance = torch.zeros(num_envs, device=device)
        self.command_storage = torch.zeros_like(command, device=device)
        self.successful_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Store the initial command
        self.command_storage[:] = command
        self._initialized = True


    def __call__(self, env: ManagerBasedRLEnv, 
                 command_name: str, 
                 frame_name: str, 
                 position_threshold: float, 
                 goal_reward: float, 
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 only_shaped: bool = False,
                 only_goal: bool = False) -> torch.Tensor:

        # throw error if both only_shaped and only_goal are True
        assert not (only_shaped and only_goal), "only_shaped and only_goal cannot be True at the same time."

        if not self._initialized:
            self._initialize_buffers()

        # extract the asset (to enable type hinting)
        frame_pos = self.asset.data.body_pos_w[:, self.asset.find_bodies(self.frame_name)[0], :].squeeze(1)
        command = env.command_manager.get_command(command_name)

        # ----------------------------------------------------
        # 1. Compute current distance
        # ----------------------------------------------------
        delta = frame_pos - command[:, :3]
        distance = torch.sqrt((delta ** 2).sum(dim=1))  # or torch.norm(delta, dim=1)
        
        # ----------------------------------------------------
        # 2. Detect command changes and reset distances if needed
        # ----------------------------------------------------
        changed_mask = (command != self.command_storage).any(dim=1)
        if changed_mask.any():
            self.previous_distance[changed_mask] = distance[changed_mask]
            self.command_storage[changed_mask] = command[changed_mask]
            
        # Shaped reward = (previous_distance - distance).
        #  -> positive if distance decreased, negative if it increased.
        shaped_reward = self.previous_distance - distance

        # ----------------------------------------------------
        # 3. Check success conditions
        # ----------------------------------------------------
        position_success_mask = (distance < position_threshold)

        # # If there's an orientation shaper with an orientation threshold, check that too
        # if self.orientation_shaper is not None and hasattr(self.orientation_shaper, "successful_envs"):
        #     orientation_success_mask = self.orientation_shaper.successful_envs
        #     success_mask = position_success_mask & orientation_success_mask
        # else:
        success_mask = position_success_mask

        # Reward bonus for envs that meet the success mask
        # shaped_reward[success_mask] += goal_reward

        # Optionally, mark them as successful for future reference
        self.successful_envs |= success_mask

        # ----------------------------------------------------
        # 4. Handle timeouts or "done" envs
        # ----------------------------------------------------
        timeout_mask = (env.episode_length_buf >= env.max_episode_length)
        self.successful_envs[timeout_mask] = False

        # ----------------------------------------------------
        # 5. Update persistent state
        # ----------------------------------------------------
        self.previous_distance[:] = distance
        
        goal_reward_tensor = torch.zeros_like(distance)
        goal_reward_tensor[success_mask] += goal_reward
        
        if only_shaped:
            return shaped_reward
        if only_goal:
            return goal_reward_tensor
        
        return shaped_reward + goal_reward_tensor
    
    def reset(self, env_ids=None):
        """
        Optional: Reset specified environments in the internal buffers.
        If env_ids is None, reset all environments.
        """
        if not self._initialized:
            self._initialize_buffers()

        if env_ids is None:
            self._initialize_buffers()  # resets everything
        else:
            self.successful_envs[env_ids] = False


class give_up_reward_term(ManagerTermBase):
    """
    Class based approach for the give up reward term.

    This class is used to model the action of giving up on a task. This action term is useful in scenarios
    where the agent is unable to complete the task and decides to give up. The action term is a binary action
    which is either 1 (give up) or 0 (continue).
    
    For a give up action to be valid, some threshold conditions should be met. These can be distance based relative to the goal,
    or time based. Distance based conditions are met by reaching a certain percentage of the initial goal distance, AND then returning
    to a "home" positon. Time based conditions are met by staying alive for a certain amount of time, AND then returning to a "home" position.
    
    "Home" position is defined as the initial position of the agent base_link at the start of the episode. This is stored in the initial_position
    attribute of the give_up_reward_term class.
    
    For a give up action term, there is a reward give_up_reward for giving up (this should be smaller than the 
    reward for completing the task), no reward/penalty for continuing, and a penalty give_up_penalty for giving up when the
    give up conditions are not met.
    """

    initial_base_position: torch.Tensor # initial position of the agent (base_link, [x, y]) at the start of the episode, shape [num_envs, 2]
    initial_ee_distance: torch.Tensor # initial distance to the goal, shape [num_envs]
    current_base_position: torch.Tensor # current position of the agent (base_link, [x, y]), shape [num_envs, 2]
    command_storage: torch.Tensor # storage for the current ee_pose command, shape [num_envs, ...]
    distance_success_envs: torch.Tensor # envs that have reached past the distance threshold
    time_success_envs: torch.Tensor # envs that have reached past the time threshold
    give_up_success_envs: torch.Tensor # envs that have successfully given up
    give_up_reward: float # reward for giving up
    give_up_penalty: float # penalty for giving up when the give up conditions are not met
    time_threshold: float # time threshold for the give up conditions
    ee_distance_threshold: float # ee distance threshold for the give up conditions
    asset: Articulation # the agent's articulation asset
    env: ManagerBasedRLEnv

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)

        self.env = env
        self._initialized = False
        
        self.command_name = cfg.params["command_name"]
        self.frame_name = cfg.params["frame_name"]
        
        self.command_storage = deepcopy(env.command_manager.get_command(self.command_name))
        self.give_up_success_envs = torch.zeros(env.num_envs, dtype=torch.bool).to(env.unwrapped.device)
        

    def _initialize_buffers(self, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
        """
        Allocate and initialize any buffers that depend on env.num_envs or shapes that
        aren't known until env is fully set up.
        """

        num_envs = self.env.num_envs
        device = self.env.unwrapped.device
        
        self.asset = self.env.scene[asset_cfg.name]

        # Initialize state buffers
        frame_pos = self.asset.data.body_pos_w[:, self.asset.find_bodies(self.frame_name)[0], :].squeeze(1)
        command = self.env.command_manager.get_command(self.command_name)
        delta = frame_pos - command[:, :3]
        self.initial_ee_distance = torch.sqrt((delta ** 2).sum(dim=1))
        self.command_storage = torch.zeros_like(command, device=device)
        self.distance_success_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.time_success_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.give_up_success_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.command_storage[:] = self.env.command_manager.get_command(self.command_name)
        self.initial_base_position = self.asset.data.body_pos_w[:, self.asset.find_bodies("base_link")[0], :2].squeeze(1)
        
        self._initialized = True


    def __call__(self, env: ManagerBasedRLEnv, 
                 command_name: str, 
                 frame_name: str, 
                 home_threshold: float, 
                 time_threshold: float | None, 
                 distance_threshold: float | None,
                 give_up_reward: float,
                 give_up_penalty: float, 
                 asset_cfg: SceneEntityCfg,
                 action_name: str) -> torch.Tensor:
        
        """
        This is the function that gets called to calculate the reward for the give up action term. 
        
        Args:
            command_name (str): The name of the goal command position that the agent is trying to achieve.
            frame_name (str): The name of the FrameTransformer describing the end effector pose of the robot.
            home_threshold (float): The distance threshold used to describe the "home" position of the agent. Distances 
                                    are calculated using the robot base_link position.
            time_threshold (float): The time threshold used to describe the time elapsed before the agent can give up in seconds.
                                    If set to None, the time threshold is not used.
            distance_threshold (float): The distance threshold used to describe the percentage of the initial distance to the goal
                                        that the agent must achieve before it can give up. If set to None, the distance threshold is not used.
            give_up_reward (float): The reward given to the agent for giving up.
            give_up_penalty (float): The penalty given to the agent for giving up when the give up conditions are not met.
            asset_cfg (SceneEntityCfg): The SceneEntityCfg object describing the agent's articulation asset.
            
        Returns:
            torch.Tensor: The reward tensor for the give up action term.
        """
        
        # Throw an error when both time and distance thresholds are None
        if time_threshold is None and distance_threshold is None:
            raise ValueError("Both time and distance thresholds for the give_up_reward_term cannot be None.")

        if not self._initialized:
            self._initialize_buffers(asset_cfg)
            
        # extract the asset (to enable type hinting)
        frame_pos = self.asset.data.body_pos_w[:, self.asset.find_bodies(frame_name)[0], :].squeeze(1)
        command = env.command_manager.get_command(command_name)
        
        # get the action term
        give_up_action = env.action_manager.get_term(action_name).processed_actions.squeeze(1) # [num_envs]
        
        # 1. Detect command changes and reset positions and _success_envs if needed
        changed_mask = (command != self.command_storage).any(dim=1)
        if changed_mask.any():
            self.command_storage[changed_mask] = command[changed_mask]
            self.distance_success_envs[changed_mask] = False
            self.time_success_envs[changed_mask] = False
            self.initial_ee_distance[changed_mask] = torch.sqrt(((frame_pos - command[:, :3]) ** 2).sum(dim=1))[changed_mask]
            self.initial_base_position[changed_mask] = self.asset.data.body_pos_w[changed_mask, self.asset.find_bodies("base_link")[0], :2].squeeze(1)

        # 2. Compute current distance
        goal_delta = frame_pos - command[:, :3]
        distance = torch.sqrt((goal_delta ** 2).sum(dim=1))  # or torch.norm(delta, dim=1)
        
        # 3. Compute time elapsed
        time_elapsed = env.episode_length_buf * env.step_dt
        
        # 4. Check success conditions
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
        # print(f"reward home_distance: {home_distance}")
        home_success_mask = home_distance < home_threshold # [num_envs]
        
        # 5. Reward logic
        self.distance_success_envs |= distance_success_mask # remaining True if has been already True, [num_envs]
        self.time_success_envs |= time_success_mask # remaining True if has been already True, [num_envs]
        
        # Reward bonus for envs that meet the success mask
        reward = torch.zeros_like(distance)
        
        reward_mask = give_up_action & self.distance_success_envs & self.time_success_envs & home_success_mask
        penalty_mask = give_up_action & ~(self.distance_success_envs & self.time_success_envs & home_success_mask)
        
        # create reward and penalty tensors that are the same size as reward and penalty masks
        reward[reward_mask] = give_up_reward
        reward[penalty_mask] = give_up_penalty
        
        return reward

    def reset_success_envs_buffer(self, env_ids=None):
        """
        Optional: Reset specified environments in the internal buffers.
        If env_ids is None, reset all environments.
        """
        if not self._initialized:
            self._initialize_buffers()

        if env_ids is None:
            self.distance_success_envs[:] = False
            self.time_success_envs[:] = False
            self.give_up_success_envs[:] = False
        else:
            self.distance_success_envs[env_ids] = False
            self.time_success_envs[env_ids] = False
            self.give_up_success_envs[env_ids] = False


def position_command_error_frame(env: ManagerBasedRLEnv, command_name: str, frame_name: str) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    specified transform listener (FrameTransformer). The position error is computed as the L2-norm
    of the difference between the desired and current positions.

    This function also includes a shaped reward for the position error, which is computed as the difference
    between the previous distance and the current distance. If the current distance to the goal is less than 0.1, 
    the shaped reward is increased by 20.0.
    """
    # extract the asset (to enable type hinting)
    frame_pos = env.scene[frame_name].data.target_pos_source[..., 0, :] 
    command = env.command_manager.get_command(command_name)

    # obtain the distance between the desired and current positions
    distance = torch.norm(frame_pos - command[:, :3], dim=1, p=2)
    return distance

def height_command_error_frame(env: ManagerBasedRLEnv, command_name: str, frame_name: str) -> torch.Tensor:
    """Penalize tracking of the task-space height error.

    The function computes the height error between the desired position (from the command) and the
    specified transform listener (FrameTransformer). The height error is computed as 
    difference between the desired and current z-heights.
    """
    # extract the asset (to enable type hinting)
    frame_height = env.scene[frame_name].data.target_pos_source[..., 0, 2] 
    command = env.command_manager.get_command(command_name)
    # obtain the distance between the desired and current z-positions, taking the absolute value
    height_error = torch.abs(frame_height - command[:, 2])
    return height_error

def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def position_command_error_tanh_frame(env: ManagerBasedRLEnv, command_name: str, frame_name: str, std: float) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    specified transform listener (FrameTransformer). The position error is computed as the L2-norm
    of the difference between the desired and current positions, and is mapped using a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    frame_pos = env.scene[frame_name].data.target_pos_source[..., 0, :] 
    command = env.command_manager.get_command(command_name)

    # obtain the distance between the desired and current positions
    distance = torch.norm(frame_pos - command[:, :3], dim=1, p=2)
    return 1 - torch.tanh(distance / std)

def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def orientation_command_error_frame(env: ManagerBasedRLEnv, command_name: str, frame_name: str, asset_cfg: SceneEntityCfg, orientation_threshold: float = None) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of transform listener (FrameTransformer) (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.

    orientation_threshold is an optional parameter that can be used to set a threshold for the orientation error. If the orientation error
    is less than the threshold, this is considered a task success and the command is reset, only if the distance to the goal is also less than
    the distance_threshold given to position_command_error_frame_shaped(). If no orientation_threshold is given, only the position error is considered.
    """

    if not hasattr(orientation_command_error_frame, 'orientation_threshold'):
        orientation_command_error_frame.orientation_threshold = orientation_threshold
        
    
    if not hasattr(orientation_command_error_frame, 'successful_envs'):
        orientation_command_error_frame.successful_envs = []

    frame_ori_w = env.scene[frame_name].data.target_quat_w[..., 0, :] 
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)

    quat_error = quat_error_magnitude_unique(frame_ori_w, des_quat_w)

    if orientation_threshold is not None:
        if torch.any(quat_error < orientation_threshold):
            idx = torch.where(quat_error < orientation_threshold)

            # reset the command terms of the successful env ids
            orientation_command_error_frame.successful_envs = idx[0].tolist()

    return quat_error

def orientation_command_error_frame_shaped(env: ManagerBasedRLEnv, command_name: str, frame_name: str, asset_cfg: SceneEntityCfg, orientation_threshold: float = None) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path. This reward is shaped, and is computed as the difference
    between the previous orientation error and the current orientation error. If the current orientation error is less than
    the orientation_threshold, the shaped reward is increased by 20.0. This reward is positive if the orientation error is decreasing,
    and negative if the orientation error is increasing, so it's weighting should be postive to encourage the agent to decrease the orientation error.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of transform listener (FrameTransformer) (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.

    orientation_threshold is an optional parameter that can be used to set a threshold for the orientation error. If the orientation error
    is less than the threshold, this is considered a task success and the command is reset, only if the distance to the goal is also less than
    the distance_threshold given to position_command_error_frame_shaped(). If no orientation_threshold is given, only the position error is considered.
    """

    if not hasattr(orientation_command_error_frame_shaped, 'orientation_threshold'):
        orientation_command_error_frame_shaped.orientation_threshold = orientation_threshold
        
    if not hasattr(orientation_command_error_frame_shaped, 'successful_envs'):
        orientation_command_error_frame_shaped.successful_envs = []

    frame_ori_w = env.scene[frame_name].data.target_quat_w[..., 0, :] 
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)

    if not hasattr(orientation_command_error_frame_shaped, 'previous_orientation_error'):
        # initialise the previous orientation error to zeros the same size as the frame_ori_w
        orientation_command_error_frame_shaped.previous_orientation_error = torch.zeros_like(frame_ori_w[:, 0])
        orientation_command_error_frame_shaped.command_storage = deepcopy(command)

    # if any of the individual commands have changed this indicates a episode ending or command resampling, so reset the
    # relevant previous orientation error, element-wise
    mask = torch.any(command != orientation_command_error_frame_shaped.command_storage, dim=1)
    if torch.any(mask):
        orientation_command_error_frame_shaped.previous_orientation_error[mask] = torch.zeros_like(frame_ori_w[:, 0][mask])
        orientation_command_error_frame_shaped.command_storage[mask] = command[mask]

    quat_error = quat_error_magnitude_unique(frame_ori_w, des_quat_w)

    # compute the shaped reward
    shaped_reward = orientation_command_error_frame_shaped.previous_orientation_error - quat_error
    # print(f"Shaped reward: {shaped_reward}")

    if orientation_threshold is not None:
        if torch.any(quat_error < orientation_threshold):
            idx = torch.where(quat_error < orientation_threshold)

            # reset the command terms of the successful env ids
            orientation_command_error_frame_shaped.successful_envs = idx[0].tolist()

    orientation_command_error_frame_shaped.previous_orientation_error = quat_error
    return shaped_reward


def contact_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalise when the contact force on the sensor exceeds the force threshold.
    
    The function computes the net contact forces on the sensor and penalizes when the maximum contact force exceeds the
    threshold. The penalty is -1.0 if the contact force exceeds the threshold and 0.0 otherwise.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    bool_tensor = torch.any(torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1)
    rewards = bool_tensor.float() * -1.0
    return rewards

def contact_penalty_velocity_scaled(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, link_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalise when the contact force on the sensor exceeds the force threshold. This penalty is scaled by the linear velocity of the body
    experiencing the contact force.
    
    The function computes the net contact forces on the sensor and penalizes when the maximum contact force exceeds the
    threshold. The penalty is -1.0 if the contact force exceeds the threshold and 0.0 otherwise.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    #get index of matching string from link_name 
    link_id = asset.body_names.index(link_name)
    velocities = asset.data.body_vel_w[:, link_id, :3]
    velocities_scaled = torch.norm(velocities, dim=-1)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    bool_tensor = torch.any(torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1)
    rewards = bool_tensor.float() * -1.0 * velocities_scaled
    return rewards

def grasp_close(env: RLTaskEnv, distance_threshold: float, orientation_threshold: float, command_name: str, frame_name:str, open_joint_pos: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward for closing the fingers when being close to the handle.

    The :attr:`distance_threshold` is the distance from the handle at which the gripper should be closed.
    The :attr:`orientation_threshold` is the orientation error threshold between the handle and the end effector at which the gripper should be closed.
    The :attr:`open_joint_pos` is the joint position when the fingers are open.

    Note:
        It is assumed that zero joint position corresponds to the fingers being closed.
    """
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    distance = position_command_error_frame(env=env, command_name=command_name, frame_name=frame_name)
    orientation_error = orientation_command_error_frame(env=env, command_name="ee_pose", frame_name="ee_frame", asset_cfg=asset_cfg)

    is_close = (distance <= distance_threshold) & (orientation_error <= orientation_threshold)

    return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)

def is_goal_in_camera_view(env: ManagerBasedRLEnv, camera_name: str, goal_name: str) -> torch.Tensor:
    """Reward if the goal object is in the camera's view.

    The function computes the visibility of the goal object in the camera's view. The reward is 1 if the goal object
    is visible in the camera's view and -0.2 otherwise.
    """
    camera_image_shape: torch.Tensor = env.scene.sensors[camera_name].data.image_shape
    camera_intrinsics: torch.Tensor = env.scene.sensors[camera_name].data.intrinsic_matrices

    goal_pos: torch.Tensor = env.command_manager.get_command(goal_name)[:, :3]
    goal_quat: torch.Tensor = env.command_manager.get_command(goal_name)[:, 3:7]

    camera_pos = env.scene["depth_camera_frame"].data.target_pos_source[..., 0, :] # depth_camera_frame
    camera_quat = env.scene["depth_camera_frame"].data.target_quat_source[..., 0, :]

    # compute the goal position in the camera's frame
    p_WG_W = goal_pos # position of goal relative to world in world frame
    p_WC_W = camera_pos # position of camera relative to world in world frame
    q_WC = camera_quat # orientation of camera relative to world frame

    p_CG_W = -p_WC_W + p_WG_W # position of the goal relative to the camera in world frame
    q_CW = quat_conjugate(q_WC) # orientation from world to the camera
    R_CW = matrix_from_quat(q_CW) # rotation matrix from world to camera

    p_CG_W_reshaped = p_CG_W.unsqueeze(-1)

    p_CG_C = torch.bmm(R_CW, p_CG_W_reshaped).squeeze(-1) # position of the goal relative to the camera in camera frame

    # project the goal position in the camera's frame to the image plane
    goal_pixel = project_points(p_CG_C, camera_intrinsics)
    goal_pixel = goal_pixel[0]

    # check if the goal pixel is within the image shape, and if the goal is in front of the camera
    infront_of_camera = goal_pixel[:, 2] > 0
    goal_visible = (goal_pixel[:, 0] >= 0.0) & (goal_pixel[:, 0] <= camera_image_shape[1]) & (goal_pixel[:, 1] >= 0.0) & (goal_pixel[:, 1] <= camera_image_shape[0])
    
    # false if the goal is behind the camera
    goal_visible = goal_visible & infront_of_camera

    # if the goal is visible, return 1.0, else return -0.2
    rewards = torch.where(goal_visible, torch.tensor(1.0), torch.tensor(-0.2))

    return rewards


def is_goal_in_camera_view_frame(env: ManagerBasedRLEnv, camera_frame_name: str, goal_name: str, camera_intrinsics: torch.Tensor) -> torch.Tensor:
    """Reward if the goal object is a fake camera's view. This function is used when the camera position is a 
    described by a FrameTransformer, and we don't include a TiledCamera. Additionally, this function needs to be
    passed a camera matrix, since we don't have access to the camera's intrinsic matrix from the simulation.

    The function computes the visibility of the goal object in the camera's view. The reward is 1 if the goal object
    is visible in the camera's view and -0.2 otherwise.
    """
    camera_intrinsics: torch.Tensor = camera_intrinsics

    goal_pos: torch.Tensor = env.command_manager.get_command(goal_name)[:, :3]
    goal_quat: torch.Tensor = env.command_manager.get_command(goal_name)[:, 3:7]

    # get image shape from camera_intrinsics
    height = camera_intrinsics[1, 2] * 2
    width = camera_intrinsics[0, 2] * 2
    camera_image_shape = torch.tensor([height, width]).to(goal_pos.device)

    camera_pos = env.scene[camera_frame_name].data.target_pos_source[..., 0, :] # depth_camera_frame
    camera_quat = env.scene[camera_frame_name].data.target_quat_source[..., 0, :]

    # compute the goal position in the camera's frame
    p_WG_W = goal_pos # position of goal relative to world in world frame
    p_WC_W = camera_pos # position of camera relative to world in world frame
    q_WC = camera_quat # orientation of camera relative to world frame

    p_CG_W = -p_WC_W + p_WG_W # position of the goal relative to the camera in world frame
    q_CW = quat_conjugate(q_WC) # orientation from camera to the world
    R_CW = matrix_from_quat(q_CW) # rotation matrix from camera to world

    p_CG_W_reshaped = p_CG_W.unsqueeze(-1)

    p_CG_C = torch.bmm(R_CW, p_CG_W_reshaped).squeeze(-1) # position of the goal relative to the camera in camera frame

    # project the goal position in the camera's frame to the image plane
    goal_pixel = project_points(p_CG_C, camera_intrinsics)
    goal_pixel = goal_pixel[0]

    # check if the goal pixel is within the image shape, and if the goal is in front of the camera
    infront_of_camera = goal_pixel[:, 2] > 0
    goal_visible = (goal_pixel[:, 0] >= 0.0) & (goal_pixel[:, 0] <= camera_image_shape[1]) & (goal_pixel[:, 1] >= 0.0) & (goal_pixel[:, 1] <= camera_image_shape[0])
    
    # false if the goal is behind the camera
    goal_visible = goal_visible & infront_of_camera

    # if the goal is visible, return 1.0, else return -0.2
    rewards = torch.where(goal_visible, torch.tensor(1.0), torch.tensor(-0.2))

    return rewards

def body_lin_acc_l2_filtered(env: ManagerBasedRLEnv, body_names: Sequence[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel. Bodies are filtered by name_key."""
    asset: Articulation = env.scene[asset_cfg.name]
    # print(f"Asset joint names: {asset.joint_names}")
    body_ids = asset.find_bodies(body_names)
    return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, body_ids[0], :], dim=-1), dim=1)