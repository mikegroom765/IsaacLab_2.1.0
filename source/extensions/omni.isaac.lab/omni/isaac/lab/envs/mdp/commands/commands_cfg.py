# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
import torch

from omni.isaac.lab.managers import CommandTermCfg
from omni.isaac.lab.utils import configclass

from .null_command import NullCommand
from .pose_2d_command import TerrainBasedPose2dCommand, UniformPose2dCommand
from .pose_command import UniformPoseCommand, GridUniformPoseCommand, GoalRegionCommand
from .velocity_command import NormalVelocityCommand, UniformVelocityCommand
from .scalar_value import ScalarValueCommand


@configclass
class NullCommandCfg(CommandTermCfg):
    """Configuration for the null command generator."""

    class_type: type = NullCommand

    def __post_init__(self):
        """Post initialization."""
        # set the resampling time range to infinity to avoid resampling
        self.resampling_time_range = (math.inf, math.inf)


@configclass
class ScalarValueCommandCfg(CommandTermCfg):
    """Configuration for the scalar value command generator."""

    class_type: type = ScalarValueCommand

    value_range: tuple[float, float] = MISSING # min max
    """Range for the scalar value to be uniformly sampled from."""

@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    heading_command: bool = MISSING
    """Whether to use heading command or angular velocity command.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """
    heading_control_stiffness: float = MISSING
    """Scale factor to convert the heading error to angular velocity command."""
    rel_standing_envs: float = MISSING
    """Probability threshold for environments where the robots that are standing still."""
    rel_heading_envs: float = MISSING
    """Probability threshold for environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command)."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING  # min max [m/s]
        lin_vel_y: tuple[float, float] = MISSING  # min max [m/s]
        ang_vel_z: tuple[float, float] = MISSING  # min max [rad/s]
        heading: tuple[float, float] = MISSING  # min max [rad]

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


@configclass
class NormalVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = NormalVelocityCommand
    heading_command: bool = False  # --> we don't use heading command for normal velocity command.

    @configclass
    class Ranges:
        """Normal distribution ranges for the velocity commands."""

        mean_vel: tuple[float, float, float] = MISSING
        """Mean velocity for the normal distribution.

        The tuple contains the mean linear-x, linear-y, and angular-z velocity.
        """
        std_vel: tuple[float, float, float] = MISSING
        """Standard deviation for the normal distribution.

        The tuple contains the standard deviation linear-x, linear-y, and angular-z velocity.
        """
        zero_prob: tuple[float, float, float] = MISSING
        """Probability of zero velocity for the normal distribution.

        The tuple contains the probability of zero linear-x, linear-y, and angular-z velocity.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


@configclass
class UniformPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """
    return_quat: bool = True
    """Whether to return the quaternion or not. Defaults to True."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING  # min max [m]
        pos_y: tuple[float, float] = MISSING  # min max [m]
        pos_z: tuple[float, float] = MISSING  # min max [m]
        roll: tuple[float, float] = MISSING  # min max [rad]
        pitch: tuple[float, float] = MISSING  # min max [rad]
        yaw: tuple[float, float] = MISSING  # min max [rad]

    ranges: Ranges = MISSING
    """Ranges for the commands."""
    
@configclass
class GoalRegionCommandCfg(CommandTermCfg):
    """Configuration for the goal region command generator.
    
    This command generator create a goal region around a randomly sampled goal position.
    """

    class_type: type = GoalRegionCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    
    frame_name: str = MISSING
    """Name of the frame in the asset for which the commands are generated, i.e., ee_tcp."""
    
    object_name: str = MISSING
    """Name of the object which the goal command is for."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the goal region commands.
        This is used to sample the center of the goal region."""

        pos_x: tuple[float, float] = MISSING  # min max [m]
        pos_y: tuple[float, float] = MISSING  # min max [m]
        pos_z: tuple[float, float] = MISSING  # min max [m]
        
    ranges: Ranges = MISSING
    """Ranges for the center of the goal region."""
    
    goal_region_size: float = MISSING
    """Radius of the spherical goal region in meters."""

    table_heights: list[float] = MISSING
    """List of heights of the tables in the environment. Used to sample the z position of the goal region."""
    
    table_centre: tuple[float, float] = MISSING
    """Centre of the table in the environment. Used to sample the x and y position of the goal region."""
    
    return_quat: bool = True
    """Whether to return the quaternion or not. Defaults to True."""
    
    def __post_init__(self):
        """Post initialization."""
        # set the resampling time range to 1000 to avoid resampling
        self.resampling_time_range = (1000.0, 1000.0)

@configclass
class GridUniformPoseCommandCfg(CommandTermCfg):
    """Configuration for the grid uniform pose command generator.
    
    This command generator samples the pose commands from nearby the filled grid cells in the grid env.
    """

    class_type: type = GridUniformPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    # body_name: str = MISSING
    # """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """
    return_quat: bool = True
    """Whether to return the quaternion or not. Defaults to True."""

    pos_x: torch.tensor = MISSING  # min max [m]
    pos_y: torch.tensor = MISSING  # min max [m]
    pos_z: torch.tensor = MISSING  # min max [m]
    roll: torch.tensor = MISSING  # min max [rad]
    pitch: torch.tensor = MISSING  # min max [rad]
    yaw: torch.tensor = MISSING  # min max [rad]

    # initial_range: list[torch.tensor] = MISSING
    # """Ranges for the initial commands (before level progression)."""

    ranges: torch.tensor = torch.tensor([])
    """Ranges for the commands for each env. shape: [num_envs, 6, 2].
    Final dimensions are [min, max] for each of the 6 pose values."""

    goal_level_range_sampling: bool = False
    """Whether to sample from a uniform range during level progression."""

    goal_level_sampling: bool = False
    """Whether to sample from a incremental increasing/decreasing range during level progression*."""

    table_locations: list[list[tuple[int, int]]] = MISSING
    """List of list of tuples representing the grid cells that are filled in the grid env."""

    max_table_height: float = MISSING
    """Maximum height of the table in meters. Used to sample ensure the sampled z position is above this height when near a table."""

    grid_rows: int = MISSING
    """Number of rows in each grid env."""

    grid_cols: int = MISSING
    """Number of columns in each grid env."""

    grid_cell_width: float = MISSING
    """Width of a grid cell in meters."""

    terrain_size: tuple[float, float] = MISSING
    """Size of the sub-terrain in meters."""

@configclass
class UniformPose2dCommandCfg(CommandTermCfg):
    """Configuration for the uniform 2D-pose command generator."""

    class_type: type = UniformPose2dCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""
        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""
        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""


@configclass
class TerrainBasedPose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration for the terrain-based position command generator."""

    class_type = TerrainBasedPose2dCommand

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the sampled commands."""
