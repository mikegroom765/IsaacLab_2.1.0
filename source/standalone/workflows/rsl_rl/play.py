# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--risk_sensitivity", type=float, default=0.0, help="Risk sensitivity for the agent.")
parser.add_argument("--visualise_critic", type=bool, default=False, help="Visualise critic for the teacher agent.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.algorithms import *
from rsl_rl.runners import Runner # OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

from omni.isaac.lab_tasks.manager_based.manipulation.reach import mdp
import math
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import RewardTermCfg as RewTerm

import threading, asyncio
from tornado.platform.asyncio import AnyThreadEventLoopPolicy
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.animation import FFMpegWriter

import numpy as np
from scipy.stats import gaussian_kde, norm
from torch.distributions import Normal
from typing import Tuple

from scipy.interpolate import InterpolatedUnivariateSpline, PchipInterpolator, UnivariateSpline


from omni.isaac.lab.terrains.config.hsrb_reach import generate_grid_reach_terrains_cfg, HSRB_LIFT_CUBE_TERRAINS_CFG

def wang_transform(u, lam):
    """Wang transform w(u) = Φ(Φ⁻¹(u) + λ)."""
    return norm.cdf(norm.ppf(u) + lam)

def plot_wang_pdf(Q, lam):
    """
    Q   : array‐like, shape (N,)  
          Sorted quantile values from your QR‐DQN (i.e. Q[i] ≈ F⁻¹((i+0.5)/N))
    lam : float  
          Wang shift parameter (positive→risk‐seeking, negative→risk‐averse)
    """
    Q = np.asarray(Q).flatten()
    N = len(Q)
    # 1) original CDF levels for each quantile
    tau = (np.arange(N) + 0.5) / N

    # 2) distorted CDF levels
    w_tau = wang_transform(tau, lam)

    # 3) compute piecewise masses & mid‐points
    delta_p = w_tau[1:] - w_tau[:-1]             # mass in each interval
    mids    = 0.5 * (Q[1:] + Q[:-1])            # midpoint of each bin

    # 4) approximate density = mass / width
    widths  = Q[1:] - Q[:-1]
    pdf     = delta_p / widths
    
    return mids, pdf

def wang_distorted_quantile_cloud(
    quantiles: torch.Tensor,
    taus_orig: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given:
      quantiles    Tensor of shape [B, N] at original taus_orig [N],
      taus_orig    1D Tensor of N increasing values in (0,1),
      beta         Wang distortion parameter,
    Returns:
      u               uniform midpoints [N],
      q_prime         interpolated quantiles [B, N] = Q(g^{-1}(u)),
      distorted_mean  their mean per batch [B].
    """
    B, N = quantiles.shape
    device = quantiles.device
    dtype  = quantiles.dtype

    # 1) uniform midpoints u_i = (i+0.5)/N
    u = (torch.arange(N, device=device, dtype=dtype) + 0.5) / N  # [N]

    # 2) inverse Wang distortion: tau_prime = Phi(Phi^{-1}(u) - beta)
    std = Normal(0., 1.)
    tau_prime = std.cdf(std.icdf(u) - beta)                    # [N]

    # 3) ensure quantiles are sorted along last dim
    q_sorted, _ = quantiles.sort(dim=1)                        # [B, N]

    # 4) find positions in taus_orig for each tau_prime
    #    searchsorted gives indices in [0..N], we clamp to [1..N-1]
    idx = torch.searchsorted(taus_orig, tau_prime, right=True)  # [N]
    idx = idx.clamp(1, N-1)

    # 5) build low/high index tensors for gather
    idx_lo = (idx - 1).unsqueeze(0).expand(B, N)  # [B, N]
    idx_hi = idx.unsqueeze(0).expand(B, N)       # [B, N]

    # 6) gather y0 = Q(taus_orig[idx_lo]), y1 = Q(taus_orig[idx_hi])
    y0 = q_sorted.gather(1, idx_lo)  # [B, N]
    y1 = q_sorted.gather(1, idx_hi)  # [B, N]

    # 7) corresponding taus at those indices
    t0 = taus_orig[idx_lo[0]]  # [N]
    t1 = taus_orig[idx_hi[0]]  # [N]

    # 8) blend weights and interpolate
    w = ((tau_prime - t0) / (t1 - t0)).unsqueeze(0).expand(B, N)  # [B, N]
    q_prime = y0 + (y1 - y0) * w                                  # [B, N]

    # 9) ordinary mean of q_prime is exactly the Wang-distorted expectation
    distorted_mean = q_prime.mean(dim=1)                         # [B]

    return u, q_prime, distorted_mean
def update_plot(line, ax, fig, xs, ys):
    line.set_xdata(xs)
    line.set_ydata(ys)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
def update_expected_value(mean_line, fig, mean):
    mean_line.set_xdata([mean, mean])
    fig.canvas.draw()
    fig.canvas.flush_events()

def quantile_hist_pdf(q: np.ndarray, eps: float = 1e-6):
    """
    Build a piecewise-constant PDF from sorted quantiles q:
      x_i = midpoint between q[i] and q[i+1]
      f_i = density = 1 / ((N-1) * Δq)
    Bins with Δq < eps are dropped to avoid infinities.
    """
    q_sorted = np.sort(q)
    # differences and midpoints
    dq = q_sorted[1:] - q_sorted[:-1]
    x_mid = 0.5 * (q_sorted[:-1] + q_sorted[1:])
    # mask out zero or tiny widths
    valid = dq > eps
    x = x_mid[valid]
    f = 1.0 / ((q_sorted.size - 1) * dq[valid])
    return x, f



def main():
    """Play with RSL-RL agent."""

    algorithms = {
        'PPO': PPO,
        'PPOMulti': PPOMulti,
        'DPPO': DPPO,
        'DPPOMulti': DPPOMulti,
    }

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # env_cfg.rewards.end_effector_orientation_tracking = RewTerm(
    #     func=mdp.orientation_command_error_frame_shaped,
    #     weight=5.0,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "ee_pose", "frame_name": "ee_frame", "orientation_threshold": math.pi/6},
    # )

    # env_cfg.rewards.end_effector_position_tracking = RewTerm(
    #     func=mdp.position_command_error_frame_shaped,
    #     weight=5.0,
    #     params={"frame_name": "ee_frame", "command_name": "ee_pose", "position_threshold": 0.20, "goal_reward": 20.0},
    # )

    if args_cli.task == "Isaac-Reach-HSRB-DPPOMulti-Student-v0" or args_cli.task == "Isaac-Reach-HSRB-PPOMulti-Student-v0":
        env_cfg.scene.terrain.terrain_generator.num_cols = 4
        env_cfg.scene.terrain.terrain_generator.num_rows = 4

    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    a = int(args_cli.num_envs**0.5)  # Start at the square root of n
    while a > 0:
        if args_cli.num_envs % a == 0:  # Check if a divides num_envs
            b = args_cli.num_envs // a
            num_rows, num_cols = a, b
            break
        a -= 1
    print(f"[INFO] Using {num_rows} x {num_cols} grid for training.")

    if args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-Teacher-v0"  or args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-Depth-v0" \
        or args_cli.task == "Isaac-Grid-HSRB-PPOMulti-Teacher-v0":

        size=(6.0,6.0) # size of the sub-terrain in meters
        # num_rows = 2 # number of rows of sub-terrains
        # num_cols = 2 # number of columns of sub-terrains
        grid_rows = 9 # number of rows of grid cells
        grid_cols = 9 # number of columns of grid cells
        grid_cell_width = 0.5 # width of a grid cell in meters
        max_filled_cells = 12 # maximum number of filled cells in a grid cell
        min_filled_cells = 6 # minimum number of filled cells in a grid cell
        table_height_range = (0.25, 0.6) # height range of the tables

        hsrb_grid_reach_terrain = generate_grid_reach_terrains_cfg(size=size,
                                                                num_rows=num_rows,
                                                                num_cols=num_cols,
                                                                grid_cell_width=grid_cell_width,
                                                                grid_cols=grid_cols,
                                                                grid_rows=grid_rows,
                                                                table_height_range=table_height_range,
                                                                max_filled_cells=max_filled_cells,
                                                                min_filled_cells=min_filled_cells)
        env_cfg.scene.terrain.terrain_generator = hsrb_grid_reach_terrain

        # env_cfg.commands.ee_pose = mdp.GridUniformPoseCommandCfg(
        #     asset_name="robot",
        #     resampling_time_range=(50.0, 100.0),
        #     debug_vis=True,
        #     pos_x=torch.tensor([-1.5, 1.5]),
        #     pos_y=torch.tensor([-1.5, 1.5]),
        #     pos_z=torch.tensor([0.2, 1.4]),
        #     roll=torch.tensor([-math.pi, math.pi]),
        #     pitch=torch.tensor([math.pi/2, math.pi/2]),  # depends on end-effector axis
        #     yaw=torch.tensor([-math.pi, -math.pi]),
        #     goal_level_range_sampling=True,
        #     goal_level_sampling=False,
        #     table_locations=hsrb_grid_reach_terrain.table_locs,
        #     max_table_height=table_height_range[1],
        #     grid_rows=grid_rows,
        #     grid_cols=grid_cols,
        #     grid_cell_width=grid_cell_width,
        #     terrain_size=size,
        #     return_quat=True,
        # )

    if args_cli.task == 'Isaac-Lift-Cube-HSRB-DPPOMulti-Teacher-Play-v0' or args_cli.task == "Isaac-Lift-Cube-HSRB-PPOMulti-Teacher-Play-v0" \
        or args_cli.task == 'Isaac-Lift-Cube-HSRB-DPPOMulti-Student-Play-v0':

        size=(5.0, 5.0) # size of the sub-terrain in meters
        table_height_range = (0.55, 0.55) # height range of the tables

        hsrb_lift_cube_reach_terrain = HSRB_LIFT_CUBE_TERRAINS_CFG(size=size,
                                                                num_rows=num_rows,
                                                                num_cols=num_cols,
                                                                table_length=1.0,
                                                                table_width=3.0,
                                                                table_height_range=table_height_range)
        env_cfg.scene.terrain.terrain_generator = hsrb_lift_cube_reach_terrain
        table_heights_list = [hsrb_lift_cube_reach_terrain.sub_terrains[f"table_{i}"].table_height for i in range(len(hsrb_lift_cube_reach_terrain.sub_terrains))]
        env_cfg.events.reset_object_position.params["set_heights"] = table_heights_list
        env_cfg.commands.object_goal_region.table_heights = table_heights_list
        
        # set initial_level_zero to False on all curriculum reset events
        env_cfg.events.reset_robot_joints.params["initial_level_zero"] = False
        env_cfg.events.reset_object_position.params["initial_level_zero"] = False
        env_cfg.events.reset_base.params["initial_level_zero"] = False
        env_cfg.events.reset_base.params["x_pose_range"] = (0.0, 0.0)
        
    

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    algorithm = algorithms[agent_cfg.to_dict()["algorithm"]['class_name']]
    # remove the class_name key from the dictionary
    algorithm_dict = agent_cfg.to_dict()["algorithm"]
    policy_dict = agent_cfg.to_dict()["policy"]
    del policy_dict['class_name']
    del algorithm_dict['class_name']

    print(f"[INFO] algorithm_dict: {algorithm_dict}")
    print(f"[INFO] policy_dict: {policy_dict}")

    kwargs = agent_cfg.to_dict()
    kwargs.pop("policy")
    kwargs.pop("device")

    agent: Agent = algorithm(env, device=agent_cfg.device, **algorithm_dict, **policy_dict)
    runner = Runner(env, agent, log_dir=None, device=agent_cfg.device, **kwargs)
    # runner = Runner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    if "Student" in args_cli.task:
        runner.load_student(resume_path)
        policy = runner.get_student_policy(device=env.unwrapped.device)
    else:
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    if args_cli.visualise_critic:
        if "DPPOMulti-Teacher-Play" not in args_cli.task:
            raise ValueError("Visualisation of critic is only available for DPPOMulti-Teacher-Play tasks.")
        critic = runner.get_critic_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     runner.alg.actor_critic, runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     runner.alg.actor_critic, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    # risk_sensitivity = 1

    # plt.ion()  # turn on interactive mode
    dpi_target = 200
    text_width = 397.48499 / (72.27 * 3)
    fig_w, fig_h = 960 / dpi_target, 1080 / dpi_target
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi_target, constrained_layout=True)
    epsilon = 1e-6 # Small value for numerical stability

    # Define configurations for each PDF plot
    plot_configs = {
        'neutral': {'beta': 0.0, 'color': '#0173b2', 'linestyle': '-', 'label': 'Risk Neutral (β=0.0)', 'linewidth': 3},
        'seeking': {'beta': -1.0, 'color': '#de8f05', 'linestyle': '--', 'label': 'Risk Seeking (β=-1.0)', 'linewidth': 3},
        'averse':  {'beta': 1.0, 'color': '#cc78bc', 'linestyle': '--', 'label': 'Risk Averse (β=+1.0)', 'linewidth': 3}, # Changed averse color
    }

    # Create dictionaries to hold plot line and mean line objects
    plot_lines = {}
    mean_lines = {}
    
    
    # Initialize plot lines and mean lines based on configurations
    for name, config in plot_configs.items():
        # PDF Curve Line uses linewidth directly from config
        plot_lines[name], = ax.plot([], [],
                                marker=None,
                                color=config['color'],
                                linestyle=config['linestyle'],
                                label=f"{config['label']} PDF",
                                linewidth=config['linewidth']) # Uses the value set above (e.g., 1.5)

        # Vertical Mean Line uses half the PDF linewidth
        mean_lines[name] = ax.axvline(0,
                                    color=config['color'],
                                    linestyle=':', # Dotted linestyle for means
                                    label=f"{config['label']} Mean",
                                    linewidth=config['linewidth'] * 0.5) # Automatically becomes 0.75 if PDF lw is 1.5

    # Configure plot appearance
    ax.legend(loc="upper left")
    ax.set_xlabel(r"Value $Z_{\phi}(s)$")
    ax.set_ylabel("Estimated Probability Density")
    ax.set_ylim(bottom=0) # Ensure y-axis starts at 0
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # anim = animation.FuncAnimation(fig, update_plot, fargs=(line, ax), interval=30*10)
    # anim = animation.FuncAnimation(
    #     fig,
    #     update_plot,
    #     frames=180,
    #     fargs=(line, ax, fig, xs, ys_all),
    #     blit=True
    # )
    step = 0
    rollout_counter = 0
    # writer = animation.FFMpegWriter(fps=10, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    # fname = f"/workspace/isaaclab/source/models/quantile_figures/distribution_{rollout_counter}.mp4"
    # writer.setup(fig, fname, dpi=300)
    canvas  = FigureCanvasAgg(fig)
    
    writer = None 
    current_video_fname = "" # To keep track of the current filename

    
    # fig.canvas.draw()
    # plt.show() 
    # ax.set_title("Time‐step 0")


    # plt.figure(figsize=(6,3))
    
    dones = torch.zeros((env.num_envs,), dtype=torch.bool).to(env.unwrapped.device)

    # reset environment
    obs, env_info = env.get_observations()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            
            # risk sensitivity 
            if algorithm == DPPOMulti:
                actor_obs, critic_obs, lidar_obs, depth_obs, scandots_obs, height_scan_obs = agent._process_observations(obs, env_info)
                risk_sensitivity_obs = torch.full((actor_obs.shape[0], 1), args_cli.risk_sensitivity).to(env.unwrapped.device)
                # risk_sensitivity_obs = torch.full((actor_obs.shape[0], risk_sensitivity), 1.0).to(env.unwrapped.device)
                # replace final observation with risk sensitivity
                # actor_obs = torch.cat((actor_obs[:, :-1], risk_sensitivity_obs), dim=-1).to(env.unwrapped.device)
                actor_obs = torch.cat((actor_obs[:, :18], risk_sensitivity_obs, actor_obs[:, 19:]), dim=-1).to(env.unwrapped.device)
                
            
                # agent stepping
                actions = policy(actor_obs, lidar_obs, depth_obs, scandots_obs, height_scan_obs)
                
                if args_cli.visualise_critic:
                    
                        # Ensure the base directory exists
                        if writer is None: # Check if a writer was active
                            video_base_path = "/workspace/isaaclab/source/models/quantile_figures"
                            os.makedirs(video_base_path, exist_ok=True)

                            current_video_fname = f"{video_base_path}/distribution_{rollout_counter}.mp4"
                            print(f"[INFO] Setting up initial video writer for: {current_video_fname}")
                            writer = animation.FFMpegWriter(fps=10, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
                            try:
                                writer.setup(fig, current_video_fname, dpi=300)
                            except Exception as e_setup:
                                print(f"[ERROR] Initial writer setup failed: {e_setup}. Disabling video output.")
                                writer = None # Ensure writer is None if setup fails
                    # with writer.saving(fig, fname, dpi=100):
                        critic_obs = torch.cat((critic_obs[:, :18], risk_sensitivity_obs, critic_obs[:, 19:]), dim=-1).to(env.unwrapped.device)
                        values, last_quantiles_tensor = critic(critic_obs, lidar_obs, depth_obs, scandots_obs, height_scan_obs)
                        last_quantiles_numpy = last_quantiles_tensor.detach().cpu().numpy()
                        # Select quantiles for the first environment and sort
                        q        = last_quantiles_numpy[0].ravel()
                        q_sorted = np.sort(q)
                        N = q_sorted.shape[0]
                        if N < 4: # UnivariateSpline k=3 needs at least k+1=4 points
                            print(f"Warning: Less than 4 quantiles ({N}), skipping plot update for UnivariateSpline.")
                            continue

                        # --- Calculate Quantile Levels (tau) ---
                        taus = (np.arange(N) + 0.5) / N

                        # --- Calculate Base Smooth PDF (Neutral Case, beta=0) using UnivariateSpline ---
                        z_fine = None
                        pdf_fine_all = {}
                        means_all = {}

                        try:
                            # 1. Fit smoothing spline Q(tau)
                            unique_taus, unique_indices = np.unique(taus, return_index=True)
                            unique_q_sorted = q_sorted[unique_indices]
                            if len(unique_taus) < 4: # Need k+1 points
                                raise ValueError(f"Not enough unique taus ({len(unique_taus)}) for UnivariateSpline(k=3).")

                            # Use UnivariateSpline. k=3 (cubic) is default.
                            # s=0 means it tries to interpolate through all points (can wiggle).
                            # Small s > 0 provides smoothing. Start with s=0 or s=1e-6.
                            Q_spline = UnivariateSpline(unique_taus, unique_q_sorted, k=3, s=0.5, ext='raise') # ext='raise' avoids extrapolation issues

                            # 2. Create fine grid for evaluation (within original tau range)
                            taus_fine = np.linspace(unique_taus[0] + epsilon, unique_taus[-1] - epsilon, 1000) # Increased points for smoother visual

                            # 3. Evaluate quantiles (z) and derivative (Q') on fine grid
                            z_fine = Q_spline(taus_fine)
                            Q_prime_fine = Q_spline.derivative(n=1)(taus_fine) # Get the first derivative
                            Q_prime_fine = np.maximum(Q_prime_fine, epsilon) # Ensure positive

                            # --- Calculate Distorted PDFs and Means (Logic mostly unchanged) ---
                            # 4. Calculate *unnormalized* base smooth PDF f(z) = 1 / Q'(tau)
                            pdf_fine_base = 1.0 / Q_prime_fine

                            inv_cdf_fine = norm.ppf(taus_fine)
                            pdf_at_inv_cdf = norm.pdf(inv_cdf_fine)

                            for name, config in plot_configs.items():
                                beta = config['beta']
                                wang_deriv_fine = norm.pdf(inv_cdf_fine + beta) / np.maximum(pdf_at_inv_cdf, epsilon)
                                pdf_distorted = pdf_fine_base * wang_deriv_fine
                                area = np.trapz(pdf_distorted, z_fine)
                                pdf_normalized = pdf_distorted / area if area > epsilon else np.zeros_like(pdf_distorted)
                                pdf_fine_all[name] = pdf_normalized
                                means_all[name] = np.trapz(z_fine * pdf_normalized, z_fine)

                        except Exception as e: # Catch broader exceptions during spline fitting/evaluation
                            print(f"Warning: Spline/PDF calculation failed: {e}. Skipping plot update.")
                            continue

                        # --- Update Plot Lines and Mean Lines ---
                        if z_fine is not None:
                            max_y_value = 0.0
                            for name, pdf_data in pdf_fine_all.items():
                                plot_lines[name].set_data(z_fine, pdf_data)
                                mean_lines[name].set_xdata([means_all[name], means_all[name]])
                                max_y_value = max(max_y_value, np.max(pdf_data))

                        # --- Dynamic Axis Scaling ---
                        # ax.set_xlim(z_fine.min(), z_fine.max())
                        ax.set_xlim(-20, 140)
                        ax.set_ylim(0, max_y_value * 1.15)

                        # --- Optional: Add tight_layout if constrained_layout isn't enough ---
                        # If the y-label still gets cut off sometimes, try uncommenting this:
                        fig.tight_layout()

                        # --- Redraw Canvas and Save Frame ---
                        fig.canvas.draw()
                        fig.canvas.flush_events()

                        writer.grab_frame()
                        # create directory if it does not exist
                        os.makedirs(f"/workspace/isaaclab/source/models/quantile_figures/distribution_{rollout_counter}", exist_ok=True)
                        # fname = f"/workspace/isaaclab/source/models/quantile_figures/distribution_{rollout_counter}/step_{step}.pdf"
                        # plt.savefig(fname)
                        step += 1
                        print(f"Step: {step}")


                       # --- Handle Rollout End (with new writer instance) ---
                        if torch.any(dones): # 'dones' here is from the *previous* step
                            if writer is not None: # Check if a writer was active
                                print(f"[INFO] Rollout {rollout_counter} ended. Finalizing video: {current_video_fname}")
                                try:
                                    writer.finish()
                                    print(f"Saved → {current_video_fname}")
                                except Exception as e_finish:
                                    print(f"[ERROR] Error finishing video {current_video_fname}: {e_finish}")
                            
                            rollout_counter += 1
                            step = 0 # Reset step counter for the new video's frames if you use 'step' for PDF naming

                            # CHANGE 2: Create a NEW writer for the next segment if still visualising
                            if args_cli.visualise_critic: # Check if we should continue making videos
                                video_base_path = "/workspace/isaaclab/source/models/quantile_figures" # ensure defined
                                current_video_fname = f"{video_base_path}/distribution_{rollout_counter}.mp4"
                                print(f"[INFO] Setting up new video writer for: {current_video_fname}")
                                
                                # Create a new FFMpegWriter instance
                                writer = animation.FFMpegWriter(fps=10, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
                                try:
                                    writer.setup(fig, current_video_fname, dpi=300)
                                except Exception as e_write:
                                    print(f"[ERROR] Error setting up writer for {current_video_fname}: {e_write}. Disabling further video output.")
                                    writer = None # Ensure writer is None if setup fails
                                    args_cli.visualise_critic = False # Optionally disable
                            else:
                                writer = None # Ensure writer is None if visualization was turned off
                        
                        # x_wang, y_wang = plot_wang_pdf(q, args_cli.risk_sensitivity)
                        # plt.scatter(x_wang, y_wang, label='Wang PDF', color='red')

                        
                        # describe the controls:
                        # print(f"[Paused] Press Enter to continue without saving, or type 's' + Enter to save…")

                        # # read one line of input  
                        # key = input().strip().lower()

                        # fname = f"/workspace/isaaclab/source/models/quantile_figures/quantiles_{step}.png"
                        # plt.savefig(fname)
                        # print(f"Saved → {fname}")
                        # step += 1

                        # # now close (or clear) the figure for the next iteration
                        # plt.close()

            if algorithm == DPPO:
                risk_sensitivity_obs = torch.full((actor_obs.shape[0], risk_sensitivity), -1.0).to(env.unwrapped.device)
                # replace final observation with risk sensitivity
                actor_obs = torch.cat((actor_obs[:, :-1], risk_sensitivity_obs), dim=-1).to(env.unwrapped.device)
                actions = policy(obs)

            if algorithm == PPO:
                actions = policy(obs)

            if algorithm == PPOMulti:
                actor_obs, critic_obs, lidar_obs, depth_obs, scandots_obs, height_scan_obs = agent._process_observations(obs, env_info)
                actions = policy(actor_obs, lidar_obs, depth_obs, scandots_obs, height_scan_obs)
            # env stepping
            obs, rewards, dones, env_info = env.step(actions)

    # close the simulator
    env.close()
    # plt.ioff()
    if writer is not None:
        print(f"[INFO] Simulation ended. Finalizing last video: {current_video_fname}")
        try:
            writer.finish()
            print(f"Saved → {current_video_fname}")
        except Exception as e_final_finish:
            print(f"[ERROR] Error finishing final video {current_video_fname}: {e_final_finish}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
