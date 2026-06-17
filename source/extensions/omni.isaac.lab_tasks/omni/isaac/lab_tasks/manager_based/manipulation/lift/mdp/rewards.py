# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject, Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer, ContactSensor
from omni.isaac.lab.utils.math import combine_frame_transforms
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import RewardTermCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] - object.data.default_root_state[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)

def object_ee_distance_bodies(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_body_name: str = "ee_tcp",
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    asset: Articulation = env.scene[robot_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_pos_w = asset.data.body_pos_w[..., asset.find_bodies(ee_body_name)[0], :].squeeze(1) # (num_envs, 3)
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_pos_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


class is_grasping_object(ManagerTermBase):
    
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.env = env
        self.successful_envs = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    def __call__(self,
        env: ManagerBasedRLEnv,
        min_force: float = 0.5,
        min_angle: float = 85,
        l_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_gripper_object_contact_force"),
        r_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_gripper_object_contact_force"),
    ) -> torch.Tensor:
        """Reward for grasping the object."""
        # extract the used quantities (to enable type-hinting)
        l_contact_sensor: ContactSensor = env.scene[l_contact_sensor_cfg.name]
        r_contact_sensor: ContactSensor = env.scene[r_contact_sensor_cfg.name]
        
        l_contact_sensor_quat_w = l_contact_sensor.data.quat_w
        r_contact_sensor_quat_w = r_contact_sensor.data.quat_w
        
        if (l_contact_sensor_quat_w == None and r_contact_sensor_quat_w == None):
            return torch.zeros(env.num_envs, device=l_contact_sensor.device)
        
        l_contact_forces = l_contact_sensor.data.force_matrix_w[:, 0, 0, :] # (N, B, M, 3) = (num_envs, 1, 1, 3) -> (num_envs, 3)
        r_contact_forces = r_contact_sensor.data.force_matrix_w[:, 0, 0, :] # (N, B, M, 3) = (num_envs, 1, 1, 3) -> (num_envs, 3)
        
        l_force = torch.linalg.norm(l_contact_forces, dim=1)
        r_force = torch.linalg.norm(r_contact_forces, dim=1)
        
        l_contact_sensor_rot_w = math_utils.matrix_from_quat(l_contact_sensor_quat_w).squeeze(1)
        r_contact_sensor_rot_w = math_utils.matrix_from_quat(r_contact_sensor_quat_w).squeeze(1)
        
        l_open_direction = l_contact_sensor_rot_w @ torch.tensor([0.0, -1.0, 0.0], device=l_contact_sensor.device)
        r_open_direction = r_contact_sensor_rot_w @ torch.tensor([0.0, 1.0, 0.0], device=r_contact_sensor.device)
        
        l_angle = compute_angle_between_vectors(l_open_direction, l_contact_forces)
        r_angle = compute_angle_between_vectors(r_open_direction, r_contact_forces)
        
        l_flag = torch.logical_and(l_force > min_force, torch.rad2deg(l_angle) <= min_angle)
        r_flag = torch.logical_and(r_force > min_force, torch.rad2deg(r_angle) <= min_angle)
        
        # update the successful environments
        self.successful_envs = torch.logical_or(torch.logical_and(l_flag, r_flag), self.successful_envs)
        
        return torch.logical_and(l_flag, r_flag).float()
    
    def reset(self, env_ids=None):
        if env_ids is None:
            self.successful_envs = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        else:
            self.successful_envs[env_ids] = False
            

@torch.jit.script
def normalize_vector(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalise a tensor of vectors."""
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

@torch.jit.script
def compute_angle_between_vectors(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1, x2 = normalize_vector(x1), normalize_vector(x2)
    dot_product = torch.clip(torch.einsum("ij,ij->i", x1, x2), -1.0, 1.0)
    return torch.acos(dot_product)

def is_object_placed(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_goal_distance_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return a binary reward that is 1 if the object is within the threshold distance of the goal position.
    Zero otherwise."""
    
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_w = command[:, :3]
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return torch.where(distance < object_goal_distance_threshold, 1.0, 0.0)

def static_vel_reward_when_placed(
    env: ManagerBasedRLEnv, 
    command_name: str,
    arm_joint_names: list[str],
    object_goal_distance_threshold: float,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Penalise joint velocity when the object is placed, i.e., it is within a threshold distance to pick target position.
    reward = (1 - torch.tanh(torch.linalg.norm(qvel_without_gripper, axis=1)/std)) * is_placed"""
    
    robot: Articulation = env.scene[robot_cfg.name]
    filtered_joint_ids = robot.find_joints(arm_joint_names)[0]
    
    linear_robot_base_vel = robot.data.root_vel_w[:, :2]
    angular_robot_base_vel = robot.data.root_ang_vel_w[..., 2].unsqueeze(1)
    joint_vels = robot.data.joint_vel[:, filtered_joint_ids]
    
    robot_vel = torch.linalg.norm(torch.cat((linear_robot_base_vel, angular_robot_base_vel, joint_vels), dim=1), axis=1)
    
    is_placed = is_object_placed(env, command_name, object_goal_distance_threshold, object_cfg)
    return (1 - torch.tanh(robot_vel/std)) * is_placed
    

def success_reward(
    env: ManagerBasedRLEnv,
    arm_joint_names: list[str],
    command_name: str,
    object_goal_distance_threshold: float,
    robot_joint_vel_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    return_dtype_bool: bool = False,
) -> torch.Tensor:
    """Reward the agent for successfully completing the task. This is a binary reward.
    Task completion is defined as the object being close to the goal region (< object_goal_distance_threshold)
    and the robot joint velocities being below a threshold (< robot_joint_vel_threshold)."""
    
    robot: Articulation = env.scene[robot_cfg.name]
    filtered_joint_ids = robot.find_joints(arm_joint_names)[0]
    
    linear_robot_base_vel = robot.data.root_vel_w[:, :2]
    angular_robot_base_vel = robot.data.root_ang_vel_w[..., 2].unsqueeze(1)
    joint_vels = robot.data.joint_vel[:, filtered_joint_ids]
    
    robot_vel = torch.linalg.norm(torch.cat((linear_robot_base_vel, angular_robot_base_vel, joint_vels), dim=1), axis=1)
    
    object_placed = is_object_placed(env, command_name, object_goal_distance_threshold, object_cfg)
    is_robot_static = torch.where(robot_vel < robot_joint_vel_threshold, 1.0, 0.0)
    
    if return_dtype_bool:
        return object_placed.bool() * is_robot_static.bool()
    return object_placed * is_robot_static

def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

def object_goal_distance_when_grasped(
    env: ManagerBasedRLEnv,
    command_name: str,
    grasped_reward_term_name: str,
    std: float,
    min_force: float = 0.5,
    min_angle: float = 85,
    l_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_gripper_object_contact_force"),
    r_contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_gripper_object_contact_force"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using binary reward."""
    # extract the used quantities (to enable type-hinting)
    # is_grasped = is_grasping_object(env,
    #                                 min_force=min_force,
    #                                 min_angle=min_angle,
    #                                 l_contact_sensor_cfg=l_contact_sensor_cfg,
    #                                 r_contact_sensor_cfg=r_contact_sensor_cfg)
    
    is_grasped = env.reward_manager.get_term_cfg(grasped_reward_term_name).func(env,
                                                                                min_force=min_force,
                                                                                min_angle=min_angle,
                                                                                l_contact_sensor_cfg=l_contact_sensor_cfg,
                                                                                r_contact_sensor_cfg=r_contact_sensor_cfg)
    
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    distance = torch.norm(command[:, :3] - object.data.root_pos_w[:, :3], dim=1)
    
    object_ee_distance_reward = 1 - torch.tanh(distance / std)
    
    return is_grasped * object_ee_distance_reward
    