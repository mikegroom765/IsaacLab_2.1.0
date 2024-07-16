# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, transform_points, project_points, quat_conjugate, matrix_from_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


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

def position_command_error_frame(env: ManagerBasedRLEnv, command_name: str, frame_name: str) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    specified transform listener (FrameTransformer). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    frame_pos = env.scene[frame_name].data.target_pos_source[..., 0, :] 
    command = env.command_manager.get_command(command_name)

    # obtain the distance between the desired and current positions
    distance = torch.norm(frame_pos - command[:, :3], dim=1, p=2)
    return distance

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

def orientation_command_error_frame(env: ManagerBasedRLEnv, command_name: str, frame_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of transform listener (FrameTransformer) (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """

    frame_ori_w = env.scene[frame_name].data.target_quat_w[..., 0, :] 
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    return quat_error_magnitude(frame_ori_w, des_quat_w)

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



def is_goal_in_camera_view(env: ManagerBasedRLEnv, camera_name: str, goal_name: str) -> torch.Tensor:
    """Reward if the goal object is in the camera's view.

    The function computes the visibility of the goal object in the camera's view. The reward is 1 if the goal object
    is visible in the camera's view and -0.2 otherwise.
    """

    camera_pos_w: torch.Tensor = env.scene.sensors[camera_name].data.pos_w
    camera_quat_w: torch.Tensor = env.scene.sensors[camera_name].data.quat_w_world
    camera_image_shape: torch.Tensor = env.scene.sensors[camera_name].data.image_shape
    camera_intrinsics: torch.TensorW = env.scene.sensors[camera_name].data.intrinsic_matrices

    goal_pos: torch.Tensor = env.command_manager.get_command(goal_name)[:, :3]
    goal_quat: torch.Tensor = env.command_manager.get_command(goal_name)[:, 3:7]

    depth_pos = env.scene["depth_camera_frame"].data.target_pos_source[..., 0, :] # depth_camera_frame
    depth_quat = env.scene["depth_camera_frame"].data.target_quat_source[..., 0, :]

    # compute the goal position in the camera's frame
    p_WG_W = goal_pos # position of goal relative to world in world frame
    p_WC_W = depth_pos # position of camera relative to world in world frame
    q_WC = depth_quat # orientation of camera relative to world frame

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