# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import GOAL_REGION_MARKER_FAR_CFG, GOAL_REGION_MARKER_CLOSE_CFG, FRAME_MARKER_CFG
from omni.isaac.lab.sensors import ContactSensor
import omni.isaac.lab.utils.math as math_utils

from omni.isaac.lab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique, quat_from_matrix, matrix_from_euler

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from .commands_cfg import UniformPoseCommandCfg, GridUniformPoseCommandCfg, GoalRegionCommandCfg


class UniformPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        if self.cfg.return_quat:
            return self.pose_command_b
        return self.pose_command_b[:, :3]

    # @property
    # def command_w(self) -> torch.Tensor:
    #     """The desired pose command in the simulation world frame. Shape is (num_envs, 7).

    #     The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
    #     """
    #     return self.pose_command_w
    
    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    # def _resample_command(self, env_ids: Sequence[int]):
    #     # sample new pose targets
    #     # -- position
    #     r = torch.empty(len(env_ids), device=self.device)
    #     self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
    #     self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
    #     self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
    #     # -- orientation
    #     euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
    #     euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
    #     euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
    #     euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
    #     quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
    #     # make sure the quaternion has real part as positive
    #     self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _resample_command(self, env_ids: Sequence[int]):
        """_resample_command implemented by applying an orientation given by the average
        of the specified ranges in rpy, then sampling about that point using the range
        
        This is as apposed to the original implementation which samples uniformly from the range
        directly in rpy without applying any average orientation first
        """

        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])

        # average of the ranges
        avg_roll = sum(self.cfg.ranges.roll) / 2
        avg_pitch = sum(self.cfg.ranges.pitch) / 2
        avg_yaw = sum(self.cfg.ranges.yaw) / 2

        # perturb about the average
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)

        # subtract the average
        euler_angles[:, 0] -= avg_roll
        euler_angles[:, 1] -= avg_pitch
        euler_angles[:, 2] -= avg_yaw

        avg_rot_mat = matrix_from_euler(torch.Tensor([avg_roll, avg_pitch, avg_yaw]).to("cuda"), "XYZ")

        for idx, value in enumerate(env_ids):
            perturb_tensor = torch.Tensor([euler_angles[idx, 0], euler_angles[idx, 1], euler_angles[idx, 2]]).to("cuda")
            perturb_rot_mat = matrix_from_euler(perturb_tensor, "XYZ")

            sampled_rot_mat = torch.matmul(avg_rot_mat, perturb_rot_mat)

            quat = quat_from_matrix(sampled_rot_mat)
            # make sure the quaternion has real part as positive
            self.pose_command_b[value, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                marker_cfg = FRAME_MARKER_CFG.copy()
                marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_pose"
                self.goal_pose_visualizer = VisualizationMarkers(marker_cfg)
                # -- current body pose
                marker_cfg.prim_path = "/Visuals/Command/body_pose"
                self.body_pose_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.body_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.body_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.body_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])


class GridUniformPoseCommand(UniformPoseCommand):
    """Command generator for generating pose commands near grid cells for grid environment."""

    cfg: GridUniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: GridUniformPoseCommandCfg, env: ManagerBasedEnv):
        super(UniformPoseCommand, self).__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        # initial ranges as a tensor
        ranges = torch.stack([cfg.pos_x, cfg.pos_y, cfg.pos_z, cfg.roll, cfg.pitch, cfg.yaw]).to(self.device)
        # store seperate ranges for each environment
        self.cfg.ranges = torch.stack([ranges for _ in range(self.num_envs)]).to(self.device)

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

    def __str__(self) -> str:
        msg = "GridUniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command in the simulation world frame. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        if self.cfg.return_quat:
            return self.pose_command_w
        return self.pose_command_w[:, :3]
    
    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current body pose
        # body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        # tensor shaped [num_envs, 4] with default quaternion values [1, 0, 0, 0]
        env_quat = torch.zeros(self.num_envs, 4, device=self.device)
        env_quat[:, 0].fill_(1.0)
        self.body_pose_visualizer.visualize(self._env.scene.env_origins, env_quat)

    def _update_metrics(self):
        pass

    def uniform_sample_point_in_level(self, upper_range_size, lower_range_size):
        """Function to uniformly sample a point inside a 'level' defined by the upper and lower range sizes.
        
        Two squares are defined by the upper and lower range sizes. The function uniformly samples a point
        inside the larger square (upper_range_size) but is outside the smaller square (lower_range_size).
        
        This doesn't sample z height, only x and y positions."""

        # upper_range_half_width = upper_range_size / 2
        # lower_range_half_width = lower_range_size / 2

        while True:
            # Generate a batch of points inside the larger square
            points = torch.rand(100, 2).to(upper_range_size.device) * 2 * upper_range_size - upper_range_size

            # Filter points outside the smaller square
            mask = (torch.abs(points[:, 0]) > lower_range_size) & (torch.abs(points[:, 1]) > lower_range_size)
            if not mask.any():
                continue
            return points[mask][0]
            
    def _resample_command(self, env_ids: Sequence[int]):
        """_resample_command implemented by applying an orientation given by the average
        of the specified ranges in rpy, then sampling about that point using the range
        
        This is as apposed to the original implementation which samples uniformly from the range
        directly in rpy without applying any average orientation first
        """

        table_locs = self.cfg.table_locations # type list[list[tuple[int, int]]]
        grid_rows = self.cfg.grid_rows
        grid_cols = self.cfg.grid_cols
        grid_cell_width = self.cfg.grid_cell_width

        if self.cfg.goal_level_range_sampling == False and self.cfg.goal_level_sampling == False:
            raise ValueError("At least one of the goal_level_range_sampling or goal_level_sampling must be enabled when using this command (GridUniformPoseCommand)!")
        elif self.cfg.goal_level_range_sampling == True and self.cfg.goal_level_sampling == True:
            raise ValueError("Only one of the goal_level_range_sampling or goal_level_sampling can be enabled when using this command (GridUniformPoseCommand)!")

        # using the env_ids, randomly select a table location for each env_id from the table_locs[env_id]
        for idx, value in enumerate(env_ids):
            env_table_locs = table_locs[value]
            perm = torch.randperm(len(env_table_locs))
            env_table_locs = [env_table_locs[j] for j in perm]
            # if there are no table locations, sample a random pose
            if len(env_table_locs) == 0:
                if self.cfg.goal_level_range_sampling == True:
                    r = torch.empty(len(env_ids), device=self.device)
                    # TODO: For some reason we are resampling the position for all envs in env_ids here. This means
                    # we resample them more than once if env_ids has more than one element! Dumb. 
                    self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges[value, 0])
                    self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges[value, 1])
                    self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges[value, 2])
                    continue
                elif self.cfg.goal_level_sampling == True:
                    r = torch.empty(1, device=self.device)
                    sampled_point = self.uniform_sample_point_in_level(self.cfg.ranges[value, 0, 1], self.cfg.ranges[value, 0, 0])
                    self.pose_command_b[value, 0] = sampled_point[0]
                    self.pose_command_b[value, 1] = sampled_point[1]
                    self.pose_command_b[value, 2] = r.uniform_(*self.cfg.ranges[value, 2])
                    continue 
            
            tables_within_bounds = False

            # if there are table locations, 50/50 sample a pose near the table or random pose
            for i in range(len(env_table_locs)):

                table_loc = env_table_locs[i]
                # from the table_loc, sample a position within that grid cell
                # -- position
                table_x = (table_loc[0] * self.cfg.grid_cell_width) - (grid_rows*grid_cell_width/2) + self.cfg.grid_cell_width/2 # + torch.rand(1) * self.cfg.grid_cell_width
                table_y = (table_loc[1] * self.cfg.grid_cell_width) - (grid_cols*grid_cell_width/2) + self.cfg.grid_cell_width/2 # + torch.rand(1) * self.cfg.grid_cell_width
                sampled_height = torch.rand(1).to(self.cfg.ranges.device) * (self.cfg.ranges[value, 2, 1] - self.cfg.ranges[value, 2, 0]) + self.cfg.ranges[value, 2, 0] 
                # TODO: Change how sample is taken above so min is max_table_height
                # check if the sampled position is not within the range
                if table_x < self.cfg.ranges[value, 0, 0] or table_x > self.cfg.ranges[value, 0, 1] or table_y < self.cfg.ranges[value, 1, 0] or table_y > self.cfg.ranges[value, 1, 1]:
                    # if not within the range, continue to the next table location and check
                    continue
                else:
                    # is within the range
                    # 50% chance of sampling a random position within the grid cell, 50% chance of sampling random pose
                    tables_within_bounds = True
                    if torch.rand(1) < 0.5:
                        self.pose_command_b[value, 0] = table_x  + (torch.rand(1) * self.cfg.grid_cell_width/2)
                        self.pose_command_b[value, 1] = table_y  + (torch.rand(1) * self.cfg.grid_cell_width/2)
                        self.pose_command_b[value, 2] = sampled_height
                        # lower bound of the height is the max_table_height
                        if self.pose_command_b[value, 2] < self.cfg.max_table_height:
                            self.pose_command_b[value, 2] = self.cfg.max_table_height + 0.1
                    else:
                        if self.cfg.goal_level_range_sampling == True:
                            # if goal_level_range_sampling is enabled, sample a random pose from a uniform range
                            r = torch.empty(len(env_ids), device=self.device)
                            self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges[value, 0])
                            self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges[value, 1])
                            self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges[value, 2])
                        elif self.cfg.goal_level_sampling == True:
                            r = torch.empty(1, device=self.device)
                            sampled_point = self.uniform_sample_point_in_level(self.cfg.ranges[value, 0, 1], self.cfg.ranges[value, 0, 0])
                            self.pose_command_b[value, 0] = sampled_point[0]
                            self.pose_command_b[value, 1] = sampled_point[1]
                            self.pose_command_b[value, 2] = r.uniform_(*self.cfg.ranges[value, 2])
                            continue
                    break

            if not tables_within_bounds:
                if self.cfg.goal_level_range_sampling == True:
                    r = torch.empty(len(env_ids), device=self.device)
                    self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges[value, 0])
                    self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges[value, 1])
                    self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges[value, 2])
                elif self.cfg.goal_level_sampling == True:
                    r = torch.empty(1, device=self.device)
                    sampled_point = self.uniform_sample_point_in_level(self.cfg.ranges[value, 0, 1], self.cfg.ranges[value, 0, 0])
                    self.pose_command_b[value, 0] = sampled_point[0]
                    self.pose_command_b[value, 1] = sampled_point[1]
                    self.pose_command_b[value, 2] = r.uniform_(*self.cfg.ranges[value, 2])
                    continue

        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])

        # average of the ranges
        avg_roll = sum(self.cfg.ranges[value, 3]) / 2
        avg_pitch = sum(self.cfg.ranges[value, 4]) / 2
        avg_yaw = sum(self.cfg.ranges[value, 5]) / 2

        # perturb about the average
        euler_angles[:, 0].uniform_(*self.cfg.ranges[value, 3])
        euler_angles[:, 1].uniform_(*self.cfg.ranges[value, 4])
        euler_angles[:, 2].uniform_(*self.cfg.ranges[value, 5])

        # subtract the average
        euler_angles[:, 0] -= avg_roll
        euler_angles[:, 1] -= avg_pitch
        euler_angles[:, 2] -= avg_yaw

        avg_rot_mat = matrix_from_euler(torch.Tensor([avg_roll, avg_pitch, avg_yaw]).to("cuda"), "XYZ")

        for idx, value in enumerate(env_ids):
            perturb_tensor = torch.Tensor([euler_angles[idx, 0], euler_angles[idx, 1], euler_angles[idx, 2]]).to("cuda")
            perturb_rot_mat = matrix_from_euler(perturb_tensor, "XYZ")

            sampled_rot_mat = torch.matmul(avg_rot_mat, perturb_rot_mat)

            quat = quat_from_matrix(sampled_rot_mat)
            # make sure the quaternion has real part as positive
            self.pose_command_b[value, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat
            
        # put the command into the world frame by adding env origins
        self.pose_command_w[env_ids, :3] = self.pose_command_b[env_ids, :3] + self._env.scene.env_origins[env_ids]
        self.pose_command_w[env_ids, 3:] = self.pose_command_b[env_ids, 3:]
        

class GoalRegionCommand(CommandTerm):
    """Command generator for generating goal region commands for lift tasks."""
    
    cfg: GoalRegionCommandCfg
    goal_region_size: float
    
    def __init__(self, cfg: GoalRegionCommandCfg, env: ManagerBasedEnv):
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.frame_name = cfg.frame_name
        self.frame_idx = self.robot.find_bodies(self.frame_name)[0]
        self.table_heights = torch.tensor(cfg.table_heights).to(env.device)
        self.table_centre = cfg.table_centre
        self.goal_region_size = cfg.goal_region_size
        super().__init__(cfg, env)
        
        # create buffers
        # -- command goal region centres (x, y, z) in root frame
        self.goal_region_centres_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_region_centres_w = torch.zeros_like(self.goal_region_centres_b)
        self.debug_last_vis = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.debug_initialised = False
    
    def __str__(self) -> str:
        msg = "GoalRegionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg
    
    @property
    def command(self) -> torch.Tensor:
        """The desired goal region command position in the simulation world frame. Shape is (num_envs, 3)."""
        if self.cfg.return_quat:
            return torch.cat([self.goal_region_centres_w, torch.tensor([1, 0, 0, 0]).expand(self.goal_region_centres_w.shape[0], -1).to(self.device)], dim=1)
        return self.goal_region_centres_w

    # @property
    # def command_w(self) -> torch.Tensor:
    #     """The desired goal region command position in the simulation world frame. Shape is (num_envs, 3)."""
    #     return self.goal_region_centres_w
    
    def _resample_command(self, env_ids: Sequence[int]):
        """The _resample_command function for the GoalRegionCommand.
        
        Resamples the goal region centres for the specified environments.
        """
        # sample new goal region centres in the table frame
        r = torch.empty(len(env_ids), device=self.device)
        self.goal_region_centres_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x) + self.table_centre[0]
        self.goal_region_centres_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y) + self.table_centre[1]
        self.goal_region_centres_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z) + self.table_heights[env_ids]
        
        # put the command into the world frame by adding env origins
        self.goal_region_centres_w[env_ids] = self.goal_region_centres_b[env_ids] + self._env.scene.env_origins[env_ids]
    
    def _update_metrics(self):
        pass
    
    def _update_command(self):
        pass
    
    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        
        frame_pos = self.object.data.body_pos_w[:, 0, :]
        distance = torch.norm(frame_pos - self.goal_region_centres_w, dim=-1)
        distance_mask = distance <= self.goal_region_size
        offset_goal_region_centres_w = self.goal_region_centres_w + torch.tensor([0, 0, 0.0]).to(self.goal_region_centres_w.device)
        self.goal_region_visualizer_far.visualize(marker_indices=~distance_mask, translations=offset_goal_region_centres_w)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_region_visualizer_far"):
                marker_cfg = GOAL_REGION_MARKER_FAR_CFG.copy()
                marker_cfg.markers["goal_region_inside"].scale = (self.goal_region_size, self.goal_region_size, self.goal_region_size)
                marker_cfg.markers["goal_region_outside"].scale = (self.goal_region_size, self.goal_region_size, self.goal_region_size)
                marker_cfg.prim_path = "/Visuals/Command/goal_region/far"
                self.goal_region_visualizer_far = VisualizationMarkers(marker_cfg)
                
            self.goal_region_visualizer_far.set_visibility(True)
        else:
            if hasattr(self, "goal_region_visualizer_far"):
                self.goal_region_visualizer_far.set_visibility(False)