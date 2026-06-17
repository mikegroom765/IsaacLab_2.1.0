# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`omni.isaac.lab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import RayCaster, TiledCamera, ContactSensor
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise


from omni.isaac.lab.utils.math import transform_points, project_points, quat_conjugate, matrix_from_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedRLEnv

"""
Root state.
"""


def base_pos_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def root_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def root_quat_w(
    env: ManagerBasedEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    # make the quaternion real-part positive if configured
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def root_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


def root_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w


"""
Joint state.
"""


def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]

def filtered_joint_pos(env: ManagerBasedEnv, joint_names: list[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset, filtered by joint_names.

    Note: Only the joints configured in :attr:`joint_names` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = [asset.find_joints(joint_name)[0] for joint_name in joint_names]
    return asset.data.joint_pos[:, joint_ids].squeeze(-1)


def hsr_base_odom_pos(env: ManagerBasedEnv, base_velocity_action_name: str) -> torch.Tensor:
    """The base odom 2D pose (x, y, rz) of the HSR robot in the environment frame."""
    if base_velocity_action_name not in env.action_manager.active_terms:
        raise ValueError(f"Action term {base_velocity_action_name} is not an active action term.")
    return env.action_manager.get_term(base_velocity_action_name).wheel_odometry

def hsr_base_vel(env: ManagerBasedEnv, base_velocity_action_name: str) -> torch.Tensor:
    """The base velocity (linear x, linear y, angular z) of the HSR robot in the world frame
       calculated using forward dynamics from current joint velocities and steering angle."""
    if base_velocity_action_name not in env.action_manager.active_terms:
        raise ValueError(f"Action term {base_velocity_action_name} is not an active action term.")
    return env.action_manager.get_term(base_velocity_action_name).base_velocity

def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def joint_pos_limit_normalized(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.scale_transform(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1],
    )


def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]

def filtered_joint_vel(env: ManagerBasedEnv, joint_names: list[str], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset, filtered by joint_names.

    Note: Only the joints configured in :attr:`joint_names` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = [asset.find_joints(joint_name)[0] for joint_name in joint_names]
    return asset.data.joint_vel[:, joint_ids].squeeze(-1)


def joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]


"""
Sensors.
"""


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

def height_scan_masked(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, camera_frame_name: str, camera_intrinsics: torch.Tensor) -> torch.Tensor:
    """Height scan from the given sensor w.r.t the sensor's frame. The height scan is filtered using the 
    filter_points_from_camera_view function.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    points = sensor.data.ray_hits_w
    filtered_points = filter_points_from_camera_view(env, points, camera_intrinsics, camera_frame_name=camera_frame_name)
    filtered_height_scan = sensor.data.pos_w[:, 2].unsqueeze(1) - filtered_points[..., 2]
    return filtered_height_scan

def lidar_2d_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """2D Lidar scan from the given sensor w.r.t. the sensor's frame.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # calculate the distance from the sensor to the hit points
    distance = torch.norm(sensor.data.ray_hits_w[..., :2] - sensor.data.pos_w[:, None, :2], dim=-1)
    # if the distance is greater than the max distance, set it to the max distance
    max_distance = sensor.cfg.max_distance
    distance = torch.where(distance > max_distance, torch.tensor(max_distance), distance)
    return distance

def tiled_depth_camera(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, max_distance: float | None = None) -> torch.Tensor:
    """
    Retrieve and optionally clamp the depth images from a tiled depth camera sensor.

    Args:
        env (ManagerBasedEnv): The environment containing the scene.
        sensor_cfg (SceneEntityCfg): The configuration identifying the TiledCamera in the scene.
        max_distance (float, optional): If set, all depth values exceeding this value 
            will be clamped to `max_distance`. Defaults to None (no clamping).

    Returns:
        torch.Tensor:
            Depth tensor of shape (num_cameras, channels, height, width). Typically, channels=1.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera = env.scene.sensors[sensor_cfg.name]
    # check if the sensor has depth data type
    if "depth" not in sensor.cfg.data_types:
        raise ValueError("The sensor does not have depth data type.")
    # get the depth data
    depth = sensor.data.output["depth"]
    # reshape from (num_cameras, height, width, channels) to (num_cameras, channels, height, width)
    depth = depth.permute(0, 3, 1, 2)
    # if max_distance is provided, clamp the depth values
    if max_distance is not None:
        # Ensure depth is a float tensor if not already
        if not torch.is_floating_point(depth):
            depth = depth.float()
        depth.clamp_(max=max_distance)
    return depth

def depth_camera_points(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """3D point cloud from the given depth camera sensor w.r.t. the robot base frame.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    points_w = sensor.data.ray_hits_w
    # convert the points to the robot base frame
    p_WB_W = env.scene[asset_cfg.name].data.root_pos_w # position of the robot base frame in world frame
    R_WB = env.scene[asset_cfg.name].data.root_quat_w # quaternion of the robot base frame in world frame
    points_b = math_utils.transform_points(points_w, p_WB_W, R_WB) # transform the points to the robot base frame

    # if points are outside the max distance, set them to the zero vector
    max_distance = sensor.cfg.max_distance
    # create a tensor of zeros with the same shape as points_b
    zeros = torch.zeros(points_b.shape).to("cuda")
    points_b = torch.where(torch.norm(points_w - sensor.data.pos_w[:, None, :3], dim=-1, keepdim=True).expand_as(points_b) > max_distance, zeros, points_b)
    return points_b

def depth_camera(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Depth image from the given depth camera sensor
    NOT TESTED YET...
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    points_w = sensor.data.ray_hits_w
    # calculate the depth from the sensor to the hit points
    p_WR_W = points_w # position of the hit points in world frame
    p_WC_W = sensor.data.pos_w # position of the camera in world frame
    r_WC = sensor.data.quat_w # quaternion of the camera in world frame

    # print(f"p_WR_W shape: {p_WR_W.shape}")
    # print(f"p_WC_W shape: {p_WC_W.shape}")
    # print(f"p_WC_W.expand_as(p_WR_W) shape: {p_WC_W.unsqueeze(1).expand_as(p_WR_W).shape}")

    # subtract the position of the camera from the position of the hit points
    # p_WR_W.shape = (batch_size, num_points, 3)
    # p_WC_W.shape = (batch_size, 3)

    # print(f"p_WC_W: {p_WC_W}")
    # print(f"p_WC_W[:, None, :]: {p_WC_W[:, None, :]}")
    # print(f"p_WC_W[:, None, :].shape: {p_WC_W[:, None, :].shape}")
    
    p_CR_W = -p_WC_W[:, None, :] + p_WR_W # position of the hit points relative to the camera frame in world frame

    # print(f"p_CR_W: {p_CR_W}")

    # p_CR_W = -p_WC_W + p_WR_W
    r_CW = math_utils.quat_inv(r_WC) # quaternion of the world frame in camera frame

    # p_CR_C = math_utils.quat_rotate(R_CW, p_CR_W)

    p_CR_C = math_utils.transform_points(p_CR_W, None, r_CW) # points in camera frame
    # p_CR_C_test = math_utils.transform_points(p_WC_W, , r_CW) # points in camera frame

    depth = p_CR_C[..., 2]

    # if the depth is greater than the max distance, set it to the max distance
    max_distance = sensor.cfg.max_distance
    # print(f"STILL NEED TO TEST IF DEPTH IS DONE CORRECTLY")
    grid_size = sensor.cfg.pattern_cfg.size
    resolution = sensor.cfg.pattern_cfg.resolution
    width = round(((grid_size[0] + resolution) / resolution))
    height = round(((grid_size[1] + resolution) / resolution))
    depth = torch.where(depth > sensor.cfg.max_distance, max_distance, depth)

    # reshape to height x width
    depth = depth.view(-1, height, width)

    # print(f"depth: {depth}")
    # print(f"depth shape: {depth.shape}")

    return depth

def body_incoming_wrench(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # obtain the link incoming forces in world frame
    link_incoming_forces = asset.root_physx_view.get_link_incoming_joint_force()[:, asset_cfg.body_ids]
    return link_incoming_forces.view(env.num_envs, -1)


"""
Actions.
"""


def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions


"""
Commands.
"""


def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)

def pose_command_base_frame(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Retrieves the 'command' pose in the coordinate frame of the robot's base link.
    
    Args:
        env (ManagerBasedRLEnv): The environment managing scenes and commands.
        command_name (str): Name of the command to retrieve from the CommandManager.
        asset_cfg (SceneEntityCfg): Scene entity cfg to identify the robot asset in the scene.

    Returns:
        torch.Tensor of shape [num_envs, 7], where the first 3 elements are position in base frame,
        and the last 4 elements are orientation in base frame (quaternion).

    Note:
        p_AB_C means the position of B relative to A in the coordinate frame of C. Translation from A to B in frame C.
        r_AB means the orientation of frame B relative to frame A. Rotation from A to B.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # get the indices of the base footprint and base link bodies to avoid repeated calls    
    base_footprint_idx = asset.find_bodies("base_footprint")[0]

    # C is the command frame, W is the world frame, R is the robot base frame (base_footprint)

    p_WC_W = command[:, :3] # [num_envs, 3]
    r_WC = command[:, 3:] # [num_envs, 4]

    p_WR_W = asset.data.body_pos_w[:, base_footprint_idx, :] # [num_envs, 1, 3]
    r_WR = asset.data.body_quat_w[:, base_footprint_idx, :] # [num_envs, 1, 4]
    
    p_RC_W = p_WC_W - p_WR_W.squeeze(1) # [num_envs, 3]
    r_RW = math_utils.quat_inv(r_WR) # [num_envs, 1, 4]

    p_RC_R = math_utils.transform_points(p_RC_W.unsqueeze(1), None, r_RW.squeeze(1)) # [num_envs, 1, 3]


    r_RC = math_utils.quat_mul(r_RW.squeeze(1), r_WC) # [num_envs, 4]

    # return the command in the robot base frame, [x, y, z, qx, qy, qz, qw] [num_envs, 7].
    return torch.cat([p_RC_R.squeeze(1), r_RC], dim=-1)


def normalized_pose_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name.
    
    This command takes pose commands and limits translation component to be a unit vector.
    """
    command = env.command_manager.get_command(command_name)
    translation = command[:, :3]
    translation = translation / torch.norm(translation, dim=-1, keepdim=True)
    return torch.cat([translation, command[:, 3:]], dim=-1)

def distance_to_goal(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Calculates the distance between the robot end-effector and the goal in the robot base frame.
    
    Args:
        env (ManagerBasedRLEnv): The environment managing scenes and commands.
        command_name (str): Name of the command to retrieve.
        asset_cfg (SceneEntityCfg): Scene entity cfg to identify the robot asset in the scene.
    Returns:
        torch.Tensor of shape [num_envs, 3], where the elements are the distance to the goal.
    """
    
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # get the indices of the base footprint and base link bodies to avoid repeated calls    
    base_footprint_idx = asset.find_bodies("base_footprint")[0]

    # C is the command frame, W is the world frame, R is the robot base frame (base_footprint), EE is the end-effector frame
    p_WC_W = command[:, :3] # [num_envs, 3]
    
    p_WEE_W = asset.data.body_pos_w[:, asset.find_bodies("ee_tcp")[0], :] # [num_envs, 1, 3]

    # position of the command frame relative to the EE frame in the world frame    
    p_EEC_W = p_WC_W - p_WEE_W.squeeze(1) # [num_envs, 3]

    r_WR = asset.data.body_quat_w[:, base_footprint_idx, :] # [num_envs, 1, 4]
    r_RW = math_utils.quat_inv(r_WR) # [num_envs, 1, 4]

    # position of the command frame relative to the EE frame in the robot base frame
    p_EEC_R = math_utils.transform_points(p_EEC_W.unsqueeze(1), None, r_RW.squeeze(1)) # [num_envs, 1, 3]

    return p_EEC_R.squeeze(1)

def ee_distance_to_object(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """
    Calculates the distance between the robot end-effector and the object in the robot base frame.
    
    Args:
        env (ManagerBasedRLEnv): The environment managing scenes and commands.
        asset_cfg (SceneEntityCfg): Scene entity cfg to identify the robot asset in the scene.
        object_cfg (SceneEntityCfg): Scene entity cfg to identify the object asset in the scene.
    Returns:
        torch.Tensor of shape [num_envs, 3], where the elements are the distance to the object.
    """
    
    asset: Articulation = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # get the indices of the base footprint and base link bodies to avoid repeated calls    
    base_footprint_idx = asset.find_bodies("base_footprint")[0]
    
    p_WEE_W = asset.data.body_pos_w[:, asset.find_bodies("ee_tcp")[0], :] # [num_envs, 1, 3]
    p_WO_W = object.data.root_pos_w

    p_EEO_W = p_WO_W - p_WEE_W.squeeze(1) # [num_envs, 3]
    
    r_WR = asset.data.body_quat_w[:, base_footprint_idx, :] # [num_envs, 1, 4]
    r_RW = math_utils.quat_inv(r_WR) # [num_envs, 1, 4]
    
    p_EEO_R = math_utils.transform_points(p_EEO_W.unsqueeze(1), None, r_RW.squeeze(1)) # [num_envs, 1, 3]
    
    return p_EEO_R.squeeze(1)


def object_distance_to_goal(env: ManagerBasedRLEnv, command_name: str,  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Calculates the distance between the object and the goal region in the robot base frame.
    
    Args:
        env (ManagerBasedRLEnv): The environment managing scenes and commands.
        command_name (str): Name of the command to retrieve.
        asset_cfg (SceneEntityCfg): Scene entity cfg to identify the robot asset in the scene.
        object_cfg (SceneEntityCfg): Scene entity cfg to identify the object asset in the scene.
    Returns:
        torch.Tensor of shape [num_envs, 3], where the elements are the distance to the goal.
    """
    
    asset: Articulation = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # get the indices of the base footprint and base link bodies to avoid repeated calls
    base_footprint_idx = asset.find_bodies("base_footprint")[0]
    
    p_WO_W = object.data.root_pos_w
    p_WC_W = command[:, :3] # [num_envs, 3]
    
    p_OC_W = p_WC_W - p_WO_W # [num_envs, 3]
    
    r_WR = asset.data.body_quat_w[:, base_footprint_idx, :] # [num_envs, 1, 4]
    r_RW = math_utils.quat_inv(r_WR) # [num_envs, 1, 4]
    
    p_OC_R = math_utils.transform_points(p_OC_W.unsqueeze(1), None, r_RW.squeeze(1)) # [num_envs, 1, 3]
    
    return p_OC_R.squeeze(1)

"""
Privileged information.
"""

def rel_cylinder_pos(env: ManagerBasedRLEnv, camera_frame_name: str, camera_intrinsics: torch.Tensor, mask: bool = True, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), cylinder_cfg: SceneEntityCfg = SceneEntityCfg("cylinder"), noise: Unoise | None = Unoise(n_min=-0.1, n_max=0.1)) -> torch.Tensor:
    """Relative position of the cylinder to the robot base frame. X, Y distance to the cylinder from the robot base frame. 
    The position masked to zero if the cylinder is not visible in the camera view.

    Args:
        env (ManagerBasedRLEnv): The environment managing scenes and commands.
        camera_frame_name (str): Name of the camera frame to identify the camera in the scene.
        camera_intrinsics (torch.Tensor): 3x3 camera intrinsic matrix.
        mask (bool): If True, the output is masked to zero if the cylinder is not visible in the camera view.
        asset_cfg (SceneEntityCfg): Scene entity cfg to identify the robot asset in the scene.
        cylinder_cfg (SceneEntityCfg): Scene entity cfg to identify the cylinder asset in the scene.
        noise (UNoise): Noise configuration to add noise to the cylinder location if visible. If None, no noise is added.

    Returns:
        torch.Tensor of shape [num_envs, 2], where the elements are [X, Y] distances to the cylinder.
        Zero if the cylinder is not visible in the camera.

    Note:
        p_AB_C means the position of B relative to A in the coordinate frame of C. Translation from A to B in frame C.
        r_AB means the orientation of frame B relative to frame A. Rotation from A to B.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cylinder: Articulation = env.scene[cylinder_cfg.name]

    robot_base_link_idx = asset.find_bodies("base_link")[0]
    p_WR_W = asset.data.body_pos_w[:, robot_base_link_idx, :] # [num_envs, 1, 3]
    p_WC_W = cylinder.data.body_pos_w[:, cylinder.find_bodies("base_link")[0], :] # [num_envs, 1, 3]
    
    # filter out the cylinder position that are not within the camera view
    if mask:
        p_WC_W = filter_points_from_camera_view(env, p_WC_W, camera_intrinsics, camera_frame_name=camera_frame_name)
        # get as mask of which points are visible
        # NOTE: There is a edge case here - when points are exactly at zero, they are considered not visible, even when they are visible.
        visible = p_WC_W != 0
    else:
        visible = torch.ones_like(p_WC_W, dtype=torch.bool)

    # get the quaternion of the robot base frame relative to the world frame
    r_WR = asset.data.body_quat_w[:, robot_base_link_idx, :]
    # calculate the quaternion of the world frame relative to the robot base frame
    r_RW = math_utils.quat_inv(r_WR)

    # create a tensor of zeros with the same shape as p_WC_W
    p_RC_W = torch.zeros_like(p_WC_W)
    # only calculate the relative position if the points are visible
    p_RC_W[visible] = p_WC_W[visible] - p_WR_W[visible]

    # transform the relative position to the robot base frame
    p_RC_R = math_utils.transform_points(p_RC_W, None, r_RW.squeeze(1))

    # add noise to the visible points using the mask 
    if noise is not None:
        p_RC_R[visible] = noise.func(p_RC_R[visible], noise)
    
    return p_RC_R.squeeze(1)[:,:2]

def is_grasping_object_observation(
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
    
    return torch.logical_and(l_flag, r_flag).float().unsqueeze(-1)


@torch.jit.script
def normalize_vector(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalise a tensor of vectors."""
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)

@torch.jit.script
def compute_angle_between_vectors(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    x1, x2 = normalize_vector(x1), normalize_vector(x2)
    dot_product = torch.clip(torch.einsum("ij,ij->i", x1, x2), -1.0, 1.0)
    return torch.acos(dot_product)

def scan_dot_points(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, world_frame: bool = True) -> torch.Tensor:
    """Scan dots from the given RayCaster sensor.

    Points are 3D.

    Arg world_frame is used to toggle whether the points are returned in the world frame (True) or the sensor frame (False).
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    if world_frame:
        return sensor.data.ray_hits_w
    points_WP_W = sensor.data.ray_hits_w # points relative to the world in the world frame
    p_WS_W = sensor.data.pos_w # position of the sensor relative to the world in the world frame
    sensor_offset = sensor.cfg.offset
    q_WS = sensor.data.quat_w # quaternion of the sensor relative to the world frame
    q_SW = quat_conjugate(q_WS) # quaternion of the world frame relative to the sensor frame
    R_SW = matrix_from_quat(q_SW) # rotation matrix from the sensor to the world frame
    # pad the position of the sensor to the same shape as the points
    p_WS_W = p_WS_W.unsqueeze(1).expand_as(points_WP_W)
    points_SP_W = -p_WS_W + points_WP_W # points relative to the sensor in the world frame
    points_SP_S = torch.einsum('bij,bmj->bmi', R_SW, points_SP_W) # points relative to the sensor in the sensor frame
    return points_SP_S


def masked_scan_dot_points(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, camera_frame_name: str, camera_intrinsics: torch.Tensor) -> torch.Tensor:
    """Scan dots from the given RayCaster sensor w.r.t. the sensor's frame.

    Scan dots are filtered using the filter_points_from_camera_view function. This returns the points that are within the camera view, with other points zero'd.

    Points are 3D.
    """
    # Get the points from the sensor in the world frame
    points = scan_dot_points(env, sensor_cfg, world_frame=True)
    # Filter out the points that are not within the camera view
    return filter_points_from_camera_view(env, points, camera_intrinsics, camera_frame_name=camera_frame_name)


"""
Utility functions
"""

def filter_points_from_camera_view(env: ManagerBasedRLEnv, 
                                   points_WP_W: torch.Tensor, 
                                   camera_intrinsics: torch.Tensor, 
                                   max_distance: float | None = None, 
                                   camera_frame_name: str = "head_rgbd_sensor_link",
                                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Filter out points that are outside the camera view and (optionally) beyond `max_distance`. 
    
    Assumptions:
      - `points_WP_W` is shaped [num_envs, num_points, 3], in the world frame.
      - The camera location is described by a link of the robot, the name of which is passed as the parameter camera_frame_name.
        For the HSR, this is typically the "head_rgbd_sensor_link" link. 
      - `camera_intrinsics` is shaped [3, 3], with [fx, 0, cx; 0, fy, cy; 0, 0, 1].
      - If a point is not visible or out of (optional) distance range, we zero it out in the output.

    Args:
        env (ManagerBasedRLEnv): The environment with scene data.
        camera_frame_name (str): Frame name for the camera in the scene. Default is "head_rgbd_sensor_link".
        asset_cfg (SceneEntityCfg): Scene entity cfg to identify the robot asset in the scene.
        points_WP_W (torch.Tensor): World-frame points of shape [num_envs, num_points, 3].
        camera_intrinsics (torch.Tensor): 3x3 camera intrinsic matrix.
        max_distance (float, optional): If provided, points farther than `max_distance` from the 
                                        camera are considered invisible.
    
    Returns:
        (torch.Tensor): The same shape [num_envs, num_points, 3], where invisible or out-of-range 
                        points are replaced with (0, 0, 0).

    Note:
        https://arxiv.org/abs/2405.07991, here they get improved performance by limiting agents view to a max_distance of 2.0 m.

        p_AB_C means the position of B relative to A in the coordinate frame of C. Translation from A to B in frame C.
        r_AB means the orientation of frame B relative to frame A. Rotation from A to B.
    """
    
    # extract image shape from camera intrinsics
    height = camera_intrinsics[1, 2] * 2
    width = camera_intrinsics[0, 2] * 2
    camera_image_shape = torch.tensor([height, width], device=points_WP_W.device)
    asset: Articulation = env.scene[asset_cfg.name]
    
    p_WC_W = asset.data.body_pos_w[:, asset.find_bodies(camera_frame_name)[0], :] # position of the camera in the world frame
    q_WC = asset.data.body_quat_w[:, asset.find_bodies(camera_frame_name)[0], :] # quaternion of the camera in the world frame

    # Squeeze out any singleton dimension if needed
    if p_WC_W.dim() == 3 and p_WC_W.shape[1] == 1:
        p_WC_W = p_WC_W.squeeze(1)
    if q_WC.dim() == 3 and q_WC.shape[1] == 1:
        q_WC = q_WC.squeeze(1)

    return filter_points_from_camera_view_jit(points_WP_W, p_WC_W, q_WC, camera_intrinsics, camera_image_shape, max_distance)


@torch.jit.script
def filter_points_from_camera_view_jit(
    points_WP_W: torch.Tensor,             # [num_envs, num_points, 3]
    p_WC_W: torch.Tensor,                  # [num_envs, 3]
    q_WC: torch.Tensor,                    # [num_envs, 4]
    camera_intrinsics: torch.Tensor,       # [3, 3]
    camera_image_shape: torch.Tensor,      # [2] -> [height, width]
    max_distance: float | None = None,
) -> torch.Tensor:
    """
    Pure-tensor version of point filtering, suitable for TorchScript.
    It assumes all inputs are PyTorch tensors of compatible shapes.
    
    Args:
        points_WP_W (torch.Tensor): World-frame points, [num_envs, num_points, 3].
        p_WC_W (torch.Tensor): Camera position in world frame, [num_envs, 3].
        r_CW (torch.Tensor): Rotation matrix for camera->world, shape [num_envs, 3, 3].
        camera_intrinsics (torch.Tensor): 3x3 camera intrinsic matrix.
        camera_image_shape (torch.Tensor): [2] = [height, width].
        max_distance (float, optional): If set, points beyond this range from camera are invisible.

    Returns:
        (torch.Tensor): Filtered points of same shape as `points_WP_W`. 
                        Out-of-view points are zeroed out.
    """
    q_CW = quat_conjugate(q_WC) # quaternion of the world frame in the camera frame
    r_CW = matrix_from_quat(q_CW).squeeze(1) # rotation matrix from camera to world frame

    # # pad the position of the camera to the same shape as the points
    # p_WC_W = p_WC_W.expand_as(points_WP_W)
    # points relative to the camera in the world frame
    points_CP_W = points_WP_W - p_WC_W.unsqueeze(1)
    
    points_CP_C = torch.einsum('bij,bmj->bmi', r_CW, points_CP_W) # points relative to the camera in the camera frame

    points_image = project_points(points_CP_C, camera_intrinsics)

    # check if the points are within the imag plane, and if the depth is positive
    infront_of_camera = points_image[..., 2] > 0
    # only do the check if there are points are in front of the camera
    if torch.any(infront_of_camera):
        points_visible = (points_image[..., 0] >= 0.0) & (points_image[..., 0] <= camera_image_shape[1]) & (points_image[..., 1] >= 0.0) & (points_image[..., 1] <= camera_image_shape[0])
    else:
        points_visible = torch.zeros_like(infront_of_camera)

    # if max_distance is provided, check if the points are within the max_distance
    if max_distance is not None:
        points_visible = points_visible & (torch.norm(points_CP_C, dim=-1) <= max_distance)

    points_visible = points_visible & infront_of_camera

    # set points that are not visible to zero
    points_WP_W = points_WP_W * points_visible.unsqueeze(-1).expand_as(points_WP_W)
    return points_WP_W