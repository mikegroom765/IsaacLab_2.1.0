# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from collections.abc import Sequence

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, combine_frame_transforms
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.envs.mdp.observations import filter_points_from_camera_view
from omni.isaac.lab.managers.manager_base import ManagerTermBase
from omni.isaac.lab.managers.manager_term_cfg import ObservationTermCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

class masked_object_position_robot_root_frame(ManagerTermBase):
    """Class based approach to provide the position of the object in the robot's root frame. 
    
    Position is set to the initial object position if not visible
    in the robot camera frame.
    
    The mask is also returned to indicate if the object is visible in the camera frame.
"""
    
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        self.last_seen_object_pos_b = torch.zeros(env.num_envs, 3, device=env.device)
        self.last_seen_object_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
        
    def __call__(self,
        env: ManagerBasedRLEnv,
        camera_frame_name: str, 
        camera_intrinsics: torch.Tensor, 
        mask: bool = True,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        noise: Unoise | None = Unoise(n_min=-0.1, n_max=0.1),
    ) -> torch.Tensor:
        """The position of the object in the robot's root frame. 
        Set to last seen position if the object is not visible in the robot camera frame.
        Also, returns a mask to indicate if the object is visible in the camera frame.
        """
        
        robot: RigidObject = env.scene[robot_cfg.name]
        object: RigidObject = env.scene[object_cfg.name]
        object_pos_w = object.data.root_pos_w[:, :3].unsqueeze(1)
        # convert object_pos_w from [num_envs, 3] to [num_envs, 1, 3]

        last_seen_object_pos_b = self.last_seen_object_pos_b

        if mask:
            object_pos_w = filter_points_from_camera_view(env, object_pos_w, camera_intrinsics=camera_intrinsics, camera_frame_name=camera_frame_name)
            # NOTE: There is a edge case here - when points are exactly at zero, they are considered not visible, even when they are visible.
            # visible = object_pos_w != 0
            visible = torch.all(object_pos_w != 0, dim=-1).squeeze(-1)
        else:
            visible = torch.ones_like(object_pos_w, dtype=torch.bool)
        
        if torch.any(visible):
            # set objects that are visible in the camera view to the current object position in robot frame
            last_seen_object_pos_b[visible, :], _ = subtract_frame_transforms(
                robot.data.root_state_w[visible, :3], robot.data.root_state_w[visible, 3:7], object_pos_w[visible].squeeze(1)
            )
            # update last seen object position in world frame
            self.last_seen_object_pos_w[visible, :] = object_pos_w[visible].squeeze(1)
            
        if torch.any(~visible):
            # set objects that are not visible in the camera view to the last seen object position in robot frame
            # convert last seen object position in the world frame to the robot frame
            last_seen_object_pos_b[~visible, :], _ = subtract_frame_transforms(
                robot.data.root_state_w[~visible, :3], robot.data.root_state_w[~visible, 3:7], self.last_seen_object_pos_w[~visible]
            )
            
        if noise is not None:
            # only apply noise to the visible objects
            last_seen_object_pos_b[visible] = noise.func(last_seen_object_pos_b[visible], noise)
            # update last seen object position in world frame to the noisy object position we just calculated
            self.last_seen_object_pos_w[visible, :], _ = combine_frame_transforms(
                robot.data.root_state_w[visible, :3], robot.data.root_state_w[visible, 3:7], last_seen_object_pos_b[visible]
            )
            
        
        return torch.cat([last_seen_object_pos_b, visible.unsqueeze(1)], dim=1)

    def reset(self, env_ids: Sequence[int] | None = None, positions: torch.tensor | None = None) -> None:
        """Reset the initial object position."""
        if positions is None:
            return
        self.last_seen_object_pos_b[env_ids] = positions