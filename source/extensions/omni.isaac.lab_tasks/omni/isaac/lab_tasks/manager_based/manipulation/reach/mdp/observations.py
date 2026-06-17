# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import ArticulationData
from omni.isaac.lab.sensors import FrameTransformerData
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.assets import Articulation

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

""" Taken from source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/cabinet/mdp/observations.py"""

def rel_ee_object_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    object_data: ArticulationData = env.scene["object"].data

    return object_data.root_pos_w - ee_tf_data.target_pos_w[..., 0, :]


# def rel_ee_drawer_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """The distance between the end-effector and the object."""
#     ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
#     cabinet_tf_data: FrameTransformerData = env.scene["cabinet_frame"].data

#     return cabinet_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]


# def fingertips_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """The position of the fingertips relative to the environment origins."""
#     ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
#     fingertips_pos = ee_tf_data.target_pos_w[..., 1:, :] - env.scene.env_origins.unsqueeze(1)

#     return fingertips_pos.view(env.num_envs, -1)


def ee_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector relative to the environment origins."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_w[..., 0, :] - env.scene.env_origins

    return ee_pos

def ee_pos_source(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The position of the end-effector described by FrameTransfromer named ee_frame."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_pos = ee_tf_data.target_pos_source[..., 0, :]
    return ee_pos

def ee_quat_source(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector described by FrameTransfromer named ee_frame."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_source[..., 0, :]
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat

def ee_pose_robot_base_frame(env: ManagerBasedRLEnv, make_quat_unique: bool = True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The pose of the end-effector relative to the robot base_link, in the robot base_link frame."""
    asset: Articulation = env.scene[asset_cfg.name]

    # W = world, R = robot, EE = end-effector
    p_WR_W = asset.data.body_pos_w[:, asset.find_bodies("base_link")[0], :] # [num_envs, 1, 3]
    r_WR = asset.data.body_quat_w[:, asset.find_bodies("base_link")[0], :] # [num_envs, 1, 4]

    p_WEE_W = asset.data.body_pos_w[:, asset.find_bodies("ee_tcp")[0], :] # [num_envs, 1, 3]
    r_WEE = asset.data.body_quat_w[:, asset.find_bodies("ee_tcp")[0], :] # [num_envs, 1, 4]

    p_REE_W = p_WEE_W - p_WR_W # [num_envs, 1, 3]
    r_RW = math_utils.quat_inv(r_WR) # [num_envs, 1, 4]

    p_REE_R = math_utils.transform_points(p_REE_W, None, r_RW.squeeze(1))

    r_REE = math_utils.quat_mul(r_RW.squeeze(1), r_WEE.squeeze(1))

    if make_quat_unique:
        r_REE = math_utils.quat_unique(r_REE)

    return torch.cat([p_REE_R.squeeze(1), r_REE], dim=-1)

def ee_quat(env: ManagerBasedRLEnv, make_quat_unique: bool = True) -> torch.Tensor:
    """The orientation of the end-effector in the environment frame.

    If :attr:`make_quat_unique` is True, the quaternion is made unique by ensuring the real part is positive.
    """
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    ee_quat = ee_tf_data.target_quat_w[..., 0, :]
    # make first element of quaternion positive
    return math_utils.quat_unique(ee_quat) if make_quat_unique else ee_quat
