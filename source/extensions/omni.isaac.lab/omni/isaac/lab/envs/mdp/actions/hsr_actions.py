from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import math
from omni.isaac.lab.utils.math import euler_xyz_from_quat

import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.utils.string as string_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm

from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import RED_ARROW_NEGATIVE_Y_MARKER_CFG, RED_ARROW_Y_MARKER_CFG
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from . import actions_cfg

## All of this code is based off https://git.hsr.io/tmc/hsr-omniverse 

class BaseOdometry:
    def __init__(self, num_envs: int, device: str) -> None:
        self.values = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        
    @property
    def x(self) -> torch.Tensor:
        return self.values[:, 0]
    
    @x.setter
    def x(self, value: torch.Tensor) -> None:
        self.values[:, 0] = value
        
    @property
    def y(self) -> torch.Tensor:
        return self.values[:, 1]
    
    @y.setter
    def y(self, value: torch.Tensor) -> None:
        self.values[:, 1] = value
        
    @property
    def ang(self) -> torch.Tensor:
        return self.values[:, 2]
    
    @ang.setter
    def ang(self, value: torch.Tensor) -> None:
        self.values[:, 2] = value
        
    def __call__(self) -> torch.Tensor:
        return self.values

class JointSpace:
    def __init__(self, num_envs: int, device: str) -> None:
        self.values = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        
    @property
    def vel_wheel_l(self) -> torch.Tensor:
        return self.values[:, 0]
    
    @vel_wheel_l.setter
    def vel_wheel_l(self, value: torch.Tensor) -> None:
        self.values[:, 0] = value
        
    @property
    def vel_wheel_r(self) -> torch.Tensor:
        return self.values[:, 1]
    
    @vel_wheel_r.setter
    def vel_wheel_r(self, value: torch.Tensor) -> None:
        self.values[:, 1] = value
        
    @property
    def vel_steer(self) -> torch.Tensor:
        return self.values[:, 2]
    
    @vel_steer.setter
    def vel_steer(self, value: torch.Tensor) -> None:
        self.values[:, 2] = value
        
    def __call__(self) -> torch.Tensor:
        return self.values

class CartSpace:
    def __init__(self, num_envs: int, device: str) -> None:
        self.values = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
    
    @property
    def dot_x(self) -> torch.Tensor:
        return self.values[:, 0]
    
    @dot_x.setter
    def dot_x(self, value: torch.Tensor) -> None:
        self.values[:, 0] = value
        
    @property
    def dot_y(self) -> torch.Tensor:
        return self.values[:, 1]
    
    @dot_y.setter
    def dot_y(self, value: torch.Tensor) -> None:
        self.values[:, 1] = value
        
    @property
    def dot_r(self) -> torch.Tensor:
        return self.values[:, 2]
    
    @dot_r.setter
    def dot_r(self, value: torch.Tensor) -> None:
        self.values[:, 2] = value
        
    def __call__(self) -> torch.Tensor:
        return self.values

class VehicleStateSteer:
    def __init__(self, num_envs: int, device: str) -> None:
        self.steer_angle = torch.zeros(num_envs, dtype=torch.float32, device=device)
        
    def __call__(self) -> torch.Tensor:
        return self.steer_angle
    
class VehicleStateVel:
    def __init__(self, num_envs: int, device: str) -> None:
        self.values = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
        
    @property
    def x_vel(self) -> torch.Tensor:
        return self.values[:, 0]
    
    @x_vel.setter
    def x_vel(self, value: torch.Tensor) -> None:
        self.values[:, 0] = value
        
    @property
    def y_vel(self) -> torch.Tensor:
        return self.values[:, 1]
    
    @y_vel.setter
    def y_vel(self, value: torch.Tensor) -> None:
        self.values[:, 1] = value
        
    @property
    def ang_vel(self) -> torch.Tensor:
        return self.values[:, 2]
    
    @ang_vel.setter
    def ang_vel(self, value: torch.Tensor) -> None:
        self.values[:, 2] = value
        
    def __call__(self) -> torch.Tensor:
        return self.values
        
# Dynamics of offset diff drive vehicle
#  Equations are from the paper written by Masayoshi Wada etal.
#  https://www.jstage.jst.go.jp/article/jrsj1983/18/8/18_8_1166/_pdf

class HSRBaseVelocityControl(ActionTerm):
    """Base class for HSR velocity control action terms.
    
    Minimal class that computes wheel velocities from an offset diff-drive inverse dynamics model.
    
    wheel_separation : distance between left and right wheels
    wheel_radius     : radius of each drive wheel
    wheel_offset     : offset for the steering axis"""
    
    cfg: actions_cfg.HSRBaseVelocityControlCfg
    _asset: Articulation
    
    def __init__(self, cfg: actions_cfg.HSRBaseVelocityControlCfg, 
                 env: ManagerBasedEnv,
                 wheel_separation: float = 0.266,
                 wheel_radius: float = 0.04,
                 wheel_offset: float = 0.11) -> None:
        # initialize the action term
        # perform ManagerTermBase initialization first 
        super().__init__(cfg, env)
        
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)
        
        self.wheel_separation = wheel_separation
        self.wheel_radius = wheel_radius
        self.wheel_offset = wheel_offset

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
        
        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._processed_actions = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
                
        self.left_wheel_idx = self._asset.find_joints("base_l_drive_wheel_joint")[0] # 6
        self.right_wheel_idx = self._asset.find_joints("base_r_drive_wheel_joint")[0] # 8
        self.roll_idx = self._asset.find_joints("base_roll_joint")[0] # 3
        
        self.odometry_ = BaseOdometry(num_envs=self.num_envs, device=self.device)
        self.state_ = VehicleStateSteer(num_envs=self.num_envs, device=self.device)
        self.state_vel_ = VehicleStateVel(num_envs=self.num_envs, device=self.device)
        self.joint_param_ = JointSpace(num_envs=self.num_envs, device=self.device)
        self.cmd = CartSpace(num_envs=self.num_envs, device=self.device)
        self.relcmd = CartSpace(num_envs=self.num_envs, device=self.device)
        
        self.dt = env.step_dt

        self.vel_limit_steer_ = 8.0
        self.vel_limit_wheel_ = 8.0
        self.cmd_vel_limit_x = 0.5
        self.cmd_vel_limit_y = 0.5
        self.cmd_vel_limit_rz = 0.5
        
        # precompute repeated constrains for efficiency in forward and inverse dynamics
        self.wheel_radius = torch.tensor([self.wheel_radius], device=self.device)
        # forward dynamics
        self.half_wheel_radius = torch.tensor([self.wheel_radius / 2.0], device=self.device)
        self.forward_wheel_radius_factor = torch.tensor([self.wheel_radius * self.wheel_offset / self.wheel_separation], device=self.device)
        self.wheel_radius_over_separation = torch.tensor([self.wheel_radius / self.wheel_separation], device=self.device)
        # inverse dynamics
        self.inverse_wheel_radius_factor = torch.tensor([self.wheel_separation / 2.0 / self.wheel_radius / self.wheel_offset], device=self.device)
        
        self.compiled_forward_dynamics = torch.compile(self.forward_dynamics)
        self.compiled_inverse_dynamics = torch.compile(self.inverse_dynamics)
        
    def forward_dynamics(self, joint_params: torch.Tensor, steer_angle: torch.Tensor) -> torch.Tensor:
        """Given the joint velocities and steering angles, compute the cartesian space velocities.
        
        Args:
            joint_params:  (N x 3)  = [vel_wheel_l, vel_wheel_r, vel_steer]
            steer_angle:   (N,)     = per-environment steering angle
        Returns:
            cartesian_params:  (N x 3) = [dot_x, dot_y, dot_r] in local base frame"""
        
        cos_s = torch.cos(steer_angle)
        sin_s = torch.sin(steer_angle)
              
        dot_x = (self.half_wheel_radius * cos_s - self.forward_wheel_radius_factor * sin_s) * joint_params[:, 1] + (self.half_wheel_radius * cos_s + self.forward_wheel_radius_factor * sin_s) * joint_params[:, 0]
        dot_y = (self.half_wheel_radius * sin_s + self.forward_wheel_radius_factor * cos_s) * joint_params[:, 1] + (self.half_wheel_radius * sin_s - self.forward_wheel_radius_factor * cos_s) * joint_params[:, 0]
        dot_r = self.wheel_radius_over_separation * joint_params[:, 1] - self.wheel_radius_over_separation * joint_params[:, 0] - joint_params[:, 2]
        
        return torch.stack([dot_x, dot_y, dot_r], dim=1)
        
    def inverse_dynamics(self, cmd_vel: torch.Tensor, steer_angle: torch.Tensor) -> torch.Tensor:
        """Given the desired cartesian space velocities and steering angles, compute the joint velocities.
        
        Args:
            cmd_vel:      (N x 3)  = desired [dot_x, dot_y, dot_r] in local base frame
            steer_angle:  (N,)     = per-environment steering angle
        Returns:
            joint_params:  (N x 3) = [vel_wheel_l, vel_wheel_r, vel_steer]"""
        
        cos_s = torch.cos(steer_angle)
        sin_s = torch.sin(steer_angle)
        
        vel_wheel_r = (cos_s / self.wheel_radius - sin_s / self.inverse_wheel_radius_factor) * cmd_vel[:, 0] + (sin_s / self.wheel_radius + cos_s / self.inverse_wheel_radius_factor) * cmd_vel[:, 1]
        vel_wheel_l = (cos_s / self.wheel_radius + sin_s / self.inverse_wheel_radius_factor) * cmd_vel[:, 0] + (sin_s / self.wheel_radius - cos_s / self.inverse_wheel_radius_factor) * cmd_vel[:, 1]
        vel_steer = -sin_s / self.wheel_offset * cmd_vel[:, 0] + cos_s / self.wheel_offset * cmd_vel[:, 1] - cmd_vel[:, 2]
        
        return torch.stack([vel_wheel_l, vel_wheel_r, vel_steer], dim=1)

    def process_actions(self, raw_actions: torch.Tensor):
        """Given the desired (dot_x, dot_y, dot_r) in local base frame and current steering angles,
        compute the corresponding wheel velocities (with per-env velocity clamping).
        
        This function in one control step does the following:
            1. Calculate the cartesian space velocities by using forward dynamics equations (joint_params -> cartesian_params)
            2. Integrate velocities to update wheel odometry
            3. Calculate relative cartesian space velocities w.r.t. the current odom estimate
            4. Calculate relative cmd joint velocities w.r.t. the current joint velocities
            5. Calculate joint velocities by using inverse dynamics equations (relcmd -> joint_params)
            6. Apply scaled velocity limits
            7. Return the processed actions
        
        Args:
            raw_actions:  (N x 3)  = desired [dot_x, dot_y, dot_r]
            steer_angle:  (N,)     = per-environment steering angle
        Returns:
            processed_actions:  (N x 3) = [vel_wheel_l, vel_wheel_r, vel_steer] after clamping
        """
        # process the raw actions
        # here we convert from cmd_vel to joint space velocities
        self._raw_actions[:] = raw_actions.clamp(-1.0, 1.0) # .clamp_(-1.0, 1.0) # [num_envs, 3]
        
        # assuming raw actions are from [-1, 1] range - scale them to the HSR limits
        self._raw_actions[:, 0] = self._raw_actions[:, 0] * self.cmd_vel_limit_x
        self._raw_actions[:, 1] = self._raw_actions[:, 1] * self.cmd_vel_limit_y
        self._raw_actions[:, 2] = self._raw_actions[:, 2] * self.cmd_vel_limit_rz
        
        # get current joint velocities and steering angles
        cur_left_wheel_vel = self._asset.data.joint_vel[:, self.left_wheel_idx] # [num_envs, 1]
        cur_right_wheel_vel = self._asset.data.joint_vel[:, self.right_wheel_idx] # [num_envs, 1]
        cur_roll_vel = self._asset.data.joint_vel[:, self.roll_idx] # [num_envs, 1]
        self.state_.steer_angle = self._asset.data.joint_pos[:, self.roll_idx] # [num_envs, 1]
        
        joint_param = torch.cat([cur_left_wheel_vel, cur_right_wheel_vel, cur_roll_vel], dim=-1) # [num_envs, 3]

        # Calculate cartesian space velocities by using forward dynamics equations
        cartesian_param_ = self.compiled_forward_dynamics(joint_param, self.state_.steer_angle[:,0]) # [num_envs, 3]
        
        self.state_vel_.x_vel = cartesian_param_[:, 0] # [num_envs]
        self.state_vel_.y_vel = cartesian_param_[:, 1] # [num_envs]
        self.state_vel_.ang_vel = cartesian_param_[:, 2] # [num_envs]

        # Integrate velocities to update wheel odometry
        diff_r = cartesian_param_[:, 2] * self.dt
        ang = self.odometry_.ang + 0.5 * diff_r
        cosr = torch.cos(ang)  # use Runge-Kutta 2nd
        sinr = torch.sin(ang)
        abs_dot_x = cartesian_param_[:, 0] * cosr - cartesian_param_[:, 1] * sinr
        abs_dot_y = cartesian_param_[:, 0] * sinr + cartesian_param_[:, 1] * cosr
        self.odometry_.x += abs_dot_x[:] * self.dt
        self.odometry_.y += abs_dot_y[:] * self.dt
        self.odometry_.ang += diff_r
    
        # Convert local frame velocity commands to world frame
        ang = self.odometry_.ang + 0.5 * self._raw_actions[:, 2] * self.dt
        cosr = torch.cos(ang)
        sinr = torch.sin(ang)
        self.cmd.dot_x = self._raw_actions[:, 0] * cosr - self._raw_actions[:, 1] * sinr
        self.cmd.dot_y = self._raw_actions[:, 0] * sinr + self._raw_actions[:, 1] * cosr
        self.cmd.dot_r = self._raw_actions[:, 2]
        
        # Calculate updated cartesian space velocities w.r.t. the current odom estimate
        diff_r = self.cmd.dot_r[:] * self.dt
        self.relcmd.dot_r = self.cmd.dot_r[:]
        ang = self.odometry_.ang + 0.5 * diff_r  # use Runge-Kutta 2nd
        cosr = torch.cos(-ang)
        sinr = torch.sin(-ang)
        self.relcmd.dot_x = self.cmd.dot_x * cosr - self.cmd.dot_y * sinr
        self.relcmd.dot_y = self.cmd.dot_x * sinr + self.cmd.dot_y * cosr

        # Calculate updated joint velocities given the current steering angle
        joint_param_ = self.compiled_inverse_dynamics(self.relcmd(), self.state_.steer_angle[:,0]) # [num_envs, 3]
        self.joint_param_.vel_wheel_l = joint_param_[:, 0]
        self.joint_param_.vel_wheel_r = joint_param_[:, 1]
        self.joint_param_.vel_steer = joint_param_[:, 2]

        # apply velocity limits
        ratio = torch.max(
                    torch.max(
                        torch.abs(self.joint_param_.vel_steer) / self.vel_limit_steer_,
                        torch.abs(self.joint_param_.vel_wheel_l) / self.vel_limit_wheel_
                    ),
                    torch.abs(self.joint_param_.vel_wheel_r) / self.vel_limit_wheel_
                )
        
        scale = torch.clamp(ratio, min=1.0)
        # if any of the joint velocities exceed the limits, scale them down
        self.joint_param_.vel_wheel_l /= scale
        self.joint_param_.vel_wheel_r /= scale
        self.joint_param_.vel_steer /= scale
            
        self._processed_actions[:, 1] = self.joint_param_.vel_wheel_l
        self._processed_actions[:, 2] = self.joint_param_.vel_wheel_r
        self._processed_actions[:, 0] = self.joint_param_.vel_steer


    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # reset the processed actions
        if env_ids is None:
            self._raw_actions[:] = 0.0
            root_states = self._asset.data.root_state_w[env_ids].clone() 
            self.odometry_.x[:] = root_states[:, 0] - self._env.scene.env_origins[:, 0]
            self.odometry_.y[:] = root_states[:, 1] - self._env.scene.env_origins[:, 1]
            root_state_quat = root_states[:, 3:7]
            self.odometry_.ang[:] = euler_xyz_from_quat(root_state_quat)[2]
            self.state_.steer_angle = self.odometry_.ang.clone()
            self.state_vel_.x_vel[env_ids], self.state_vel_.y_vel[env_ids], self.state_vel_.ang_vel[env_ids] = 0.0, 0.0, 0.0
            self.joint_param_.vel_wheel_l[:], self.joint_param_.vel_wheel_r[:], self.joint_param_.vel_steer[:] = 0.0, 0.0, 0.0
            self.cmd.dot_x[:], self.cmd.dot_y[:], self.cmd.dot_r[:] = 0.0, 0.0, 0.0
            self.relcmd.dot_x[:], self.relcmd.dot_y[:], self.relcmd.dot_r[:] = 0.0, 0.0, 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            root_states = self._asset.data.root_state_w[env_ids].clone() 
            self.odometry_.x[env_ids] = root_states[:, 0] - self._env.scene.env_origins[env_ids, 0]
            self.odometry_.y[env_ids] = root_states[:, 1] - self._env.scene.env_origins[env_ids, 1]
            root_state_quat = root_states[:, 3:7]
            self.odometry_.ang[env_ids] = euler_xyz_from_quat(root_state_quat)[2]
            self.state_.steer_angle = self.odometry_.ang.clone()
            self.state_vel_.x_vel[env_ids], self.state_vel_.y_vel[env_ids], self.state_vel_.ang_vel[env_ids] = 0.0, 0.0, 0.0
            self.joint_param_.vel_wheel_l[env_ids], self.joint_param_.vel_wheel_r[env_ids], self.joint_param_.vel_steer[env_ids] = 0.0, 0.0, 0.0
            self.cmd.dot_x[env_ids], self.cmd.dot_y[env_ids], self.cmd.dot_r[env_ids] = 0.0, 0.0, 0.0
            self.relcmd.dot_x[env_ids], self.relcmd.dot_y[env_ids], self.relcmd.dot_r[env_ids] = 0.0, 0.0, 0.0
        
        
    @property
    def action_dim(self) -> int:
        return self._num_joints
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    @property
    def wheel_odometry(self) -> torch.Tensor:
        """Return the current wheel odometry position."""
        return torch.stack([self.odometry_.x, self.odometry_.y, self.odometry_.ang], dim=1)

    @property
    def base_velocity(self) -> torch.Tensor:
        """Return the current base velocity."""
        return torch.stack([self.state_vel_.x_vel, self.state_vel_.y_vel, self.state_vel_.ang_vel], dim=1)
    
    def apply_actions(self):
        # apply the processed actions
        self._asset.set_joint_velocity_target(self._processed_actions, joint_ids=self._joint_ids)
        
        
class HSRGripperAction(ActionTerm):
    """Continuous gripper action term for the HSR robot."""
    
    cfg: actions_cfg.HSRGripperActionCfg
    _asset: Articulation
    
    def __init__(self, cfg: actions_cfg.HSRGripperActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        # perform ManagerTermBase initialization first 
        super().__init__(cfg, env)
        
        gripper_joint_names = ["hand_l_proximal_joint", "hand_l_distal_joint", "hand_r_proximal_joint", "hand_r_distal_joint"]
        self._joint_ids, self._joint_names = self._asset.find_joints(gripper_joint_names)
        self._hand_motor_id = self._asset.find_joints("hand_l_proximal_joint")[0]
        self._num_joints = len(self._joint_ids)
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)
        
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
        
    """
    Properties.
    """
    
    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """
    
        
    def process_actions(self, raw_actions: torch.Tensor):
        """Given the desired gripper action, compute the corresponding joint position targets."""
        # process the raw actions
        self._raw_actions[:] = raw_actions
        # self._raw_actions[:] = torch.ones_like(self._raw_actions[:]) * -1.0
        actions = self._raw_actions.clamp(-1.0, 1.0)
        actions = math_utils.unscale_transform(
                actions,
                self._asset.data.soft_joint_pos_limits[:, self._hand_motor_id, 0],
                self._asset.data.soft_joint_pos_limits[:, self._hand_motor_id, 1],
            )
        self._processed_actions[:, 0] = actions[:, 0]
        self._processed_actions[:, 1] = actions[:, 0]
        self._processed_actions[:, 2] = -actions[:, 0]
        self._processed_actions[:, 3] = -actions[:, 0]
        
    def apply_actions(self):
        # set position targets
        jos_pos = self._asset.data.joint_pos[:, self._joint_ids] 
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)
        
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # reset the processed actions
         self._raw_actions[env_ids] = 0.0
        
    def _debug_vis_callback(self, event):
        if not self._asset.is_initialized:
            return
        
        l_contact_sensor: ContactSensor = self._env.scene["left_gripper_object_contact_force"]
        r_contact_sensor: ContactSensor = self._env.scene["right_gripper_object_contact_force"]
        
        if not l_contact_sensor.is_initialized or not r_contact_sensor.is_initialized:
            return
        l_contact_sensor_quat_w = l_contact_sensor.data.quat_w
        r_contact_sensor_quat_w = r_contact_sensor.data.quat_w
           
        self.l_grasping_visualizer.visualize(translations=l_contact_sensor.data.pos_w.squeeze(1), orientations=l_contact_sensor_quat_w.squeeze(1))
        self.r_grasping_visualizer.visualize(translations=r_contact_sensor.data.pos_w.squeeze(1), orientations=r_contact_sensor_quat_w.squeeze(1))
        
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "l_grasping_visualizer"):
                marker_cfg = RED_ARROW_NEGATIVE_Y_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/debug/l_grasping"
                self.l_grasping_visualizer = VisualizationMarkers(marker_cfg)
                
            self.l_grasping_visualizer.set_visibility(True)
            if not hasattr(self, "r_grasping_visualizer"):
                marker_cfg = RED_ARROW_Y_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/debug/r_grasping"
                self.r_grasping_visualizer = VisualizationMarkers(marker_cfg)
            
        else:
            if hasattr(self, "l_grasping_visualizer"):
                self.l_grasping_visualizer.set_visibility(False)
            if hasattr(self, "r_grasping_visualizer"):
                self.r_grasping_visualizer.set_visibility(False)
                

class HSRBBinaryGripperAction(HSRGripperAction):
    """Binary gripper action term for the HSR robot."""
    
    cfg: actions_cfg.HSRBBinaryGripperActionCfg
    _asset: Articulation
    
    def process_actions(self, raw_actions: torch.Tensor):
        """Given the desired gripper action, compute the corresponding joint position targets."""
        # process the raw actions
        self._raw_actions[:] = raw_actions
        
        actions = torch.where(
            self._raw_actions > 0.0,
            self._asset.data.soft_joint_pos_limits[:, self._hand_motor_id, 1],
            self._asset.data.soft_joint_pos_limits[:, self._hand_motor_id, 0],
        )
        
        self._processed_actions[:, 0] = actions[:, 0]
        self._processed_actions[:, 1] = actions[:, 0]
        self._processed_actions[:, 2] = -actions[:, 0]
        self._processed_actions[:, 3] = -actions[:, 0]