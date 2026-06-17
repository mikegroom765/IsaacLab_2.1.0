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
from datetime import datetime

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
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import CurriculumCfg, TerminationsCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp.rewards import position_command_error_frame_shaped
from omni.isaac.lab.terrains.config.hsrb_reach import generate_grid_reach_terrains_cfg
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

def convert_tensors_to_lists(obj):
    """
    Recursively traverse a nested structure (dict, list, etc.) 
    and convert all PyTorch tensors to Python lists.
    """
    # Case 1: If `obj` is a dictionary, process each key-value pair
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = convert_tensors_to_lists(v)
        return obj

    # Case 2: If `obj` is a list (or tuple), process each element
    elif isinstance(obj, (list, tuple)):
        converted_list = []
        for item in obj:
            converted_list.append(convert_tensors_to_lists(item))
        # If original was a tuple, convert back to tuple. Otherwise return list.
        return tuple(converted_list) if isinstance(obj, tuple) else converted_list

    # Case 3: If `obj` is a PyTorch tensor, convert to Python list
    elif torch.is_tensor(obj):
        return obj.tolist()

def main():
    """Play with RSL-RL agent."""

    algorithms = {
        'PPO': PPO,
        'PPOMulti': PPOMulti,
        'DPPO': DPPO,
        'DPPOMulti': DPPOMulti,
    }

    if not ((args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-Student-v0") or (args_cli.task == "Isaac-Grid-HSRB-PPOMulti-Student-v0") \
        or (args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-CVaR-Student-v0") or (args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-Neutral-Student-v0")):
        raise ValueError(f"[ERROR] Unsupported task: {args_cli.task}. Please use Isaac-Grid-HSRB-DPPOMulti-Student-v0, Isaac-Grid-HSRB-PPOMulti-Student-v0, Isaac-Grid-HSRB-DPPOMulti-Neutral-Student-v0 or Isaac-Grid-HSRB-DPPOMulti-CVaR-Student-v0")

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # env_cfg.episode_length_s = 10.0

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    import random
    import warp as wp
    import numpy as np

    random.seed(agent_cfg.seed)
    np.random.seed(agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(agent_cfg.seed)
    torch.cuda.manual_seed(agent_cfg.seed)
    torch.cuda.manual_seed_all(agent_cfg.seed)
    wp.rand_init(agent_cfg.seed)

    print(f"[INFO] Number of environments: {args_cli.num_envs}")
    a = int(args_cli.num_envs**0.5)  # Start at the square root of n
    while a > 0:
        if args_cli.num_envs % a == 0:  # Check if a divides num_envs
            b = args_cli.num_envs // a
            num_rows, num_cols = a, b
            break
        a -= 1
    print(f"[INFO] Using {num_rows} x {num_cols} grid for training.")

    size=(6.0,6.0) # size of the sub-terrain in meters
    # num_rows = 2 # number of rows of sub-terrains
    # num_cols = 2 # number of columns of sub-terrains
    grid_rows = 9 # number of rows of grid cells
    grid_cols = 9 # number of columns of grid cells
    grid_cell_width = 0.5 # width of a grid cell in meters
    max_filled_cells = 16 # maximum number of filled cells in a grid cell
    min_filled_cells = 10 # minimum number of filled cells in a grid cell
    filename = "hard"
    table_height_range = (0.20, 0.6) # height range of the tables

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

    env_cfg.commands.ee_pose = mdp.GridUniformPoseCommandCfg(
            asset_name="robot",
            # body_name="base_footprint",
            resampling_time_range=(50.0, 100.0),
            debug_vis=True,
            pos_x=torch.tensor([-2.0, 2.0]),
            pos_y=torch.tensor([-2.0, 2.0]),
            pos_z=torch.tensor([0.2, 1.4]),
            roll=torch.tensor([-math.pi, math.pi]),
            pitch=torch.tensor([math.pi/2, math.pi/2]),  # depends on end-effector axis
            yaw=torch.tensor([-math.pi, -math.pi]),
            table_locations=hsrb_grid_reach_terrain.table_locs,
            max_table_height=table_height_range[1],
            goal_level_range_sampling=True,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            grid_cell_width=grid_cell_width,
            terrain_size=size,
            return_quat=True,
        )

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
    runner.load_student(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    # if teacher task - if task name contains "Teacher"
    if "Teacher" in args_cli.task:
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    elif "Student" in args_cli.task:
        policy = runner.get_student_policy(device=env.unwrapped.device)

    print(f"[INFO] Policy loaded successfully!")

    # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     runner.alg.actor_critic, runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     runner.alg.actor_critic, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    # risk_sensitivity = 1 # risk sensitivity (Wangs metric, >0 is risk seeking, <0 is risk averse)
    time_limit = 100
    current_time = 0
    number_of_rollouts = 25

    successful_envs = env.unwrapped.reward_manager.get_term_cfg("end_effector_position_tracking").func.successful_envs
    all_logs = {}

    while simulation_app.is_running():
        # simulate environment
        if "DPPO" in args_cli.task:
            risk_sensitivities = [-1.0, -0.5, 0.0, 0.5, 1.0]
            if "Neutral" in args_cli.task:
                risk_sensitivities = [args_cli.risk_sensitivity]
            if "CVaR" in args_cli.task:
                risk_sensitivities = [0.01, 0.05, 0.15, 0.5, 1.0]
                # risk_sensitivities = [1.0]
        else: 
            risk_sensitivities = [args_cli.risk_sensitivity]
        for risk_sensitivity in risk_sensitivities:
            print(f"[INFO] Running with risk sensitivity: {risk_sensitivity}")
            for rollout in range(number_of_rollouts):
                # binary tensor to keep track of environments that are done
                dones = torch.zeros(args_cli.num_envs, dtype=torch.bool).to(env.unwrapped.device)
                dones_list = torch.zeros(args_cli.num_envs, dtype=torch.bool).to(env.unwrapped.device)
                success_list = torch.zeros(args_cli.num_envs, dtype=torch.bool).to(env.unwrapped.device)
                give_up_list = torch.zeros(args_cli.num_envs, dtype=torch.bool).to(env.unwrapped.device)
                failed_list = torch.zeros(args_cli.num_envs, dtype=torch.bool).to(env.unwrapped.device)
                env_contact_list_per_env = torch.zeros(args_cli.num_envs, dtype=torch.bool).to(env.unwrapped.device)
                cylinder_contact_list_per_env = torch.zeros(args_cli.num_envs, dtype=torch.bool).to(env.unwrapped.device)
                time_out_list_per_env = torch.zeros(args_cli.num_envs, dtype=torch.bool).to(env.unwrapped.device)
                success_list_per_env = torch.zeros(args_cli.num_envs, dtype=torch.bool).to(env.unwrapped.device)
                give_up_list_per_env = torch.zeros(args_cli.num_envs, dtype=torch.bool).to(env.unwrapped.device)
                success_times = []
                give_up_times = []
                failed_times = []
                all_times = [] 
                per_step_rewards = torch.tensor([]).to(env.unwrapped.device)
                cumulative_rewards = torch.zeros(env.num_envs, dtype=torch.float).to(env.unwrapped.device)
                current_time = 0
                logs = {}
                initialised_logs = False
                while not torch.all(dones_list):
            
                    # run everything in inference mode
                    with torch.inference_mode():

                        # reset environment
                        obs, env_info = env.get_observations()
                    
                        # risk sensitivity 
                        if algorithm == DPPOMulti and "Neutral" not in args_cli.task:
                            # actor_obs, critic_obs, lidar_obs, depth_obs, scandots_obs, height_scan_obs = agent._process_observations(obs, env_info)
                            # risk_sensitivity_obs = torch.full((actor_obs.shape[0], 1), args_cli.risk_sensitivity).to(env.unwrapped.device)
                            # actor_obs = torch.cat((risk_sensitivity_obs, actor_obs[:, 1:]), dim=-1).to(env.unwrapped.device)
                        
                            actor_obs, critic_obs, lidar_obs, depth_obs, scandots_obs, height_scan_obs = agent._process_observations(obs, env_info)
                            risk_sensitivity_obs = torch.full((actor_obs.shape[0], 1), risk_sensitivity).to(env.unwrapped.device)
                            actor_obs = torch.cat((actor_obs[:, :18], risk_sensitivity_obs, actor_obs[:, 19:]), dim=-1).to(env.unwrapped.device)
                            
                            # agent stepping
                            actions = policy(actor_obs, lidar_obs, depth_obs, scandots_obs, height_scan_obs)

                        if algorithm == DPPO:
                            risk_sensitivity_obs = torch.full((actor_obs.shape[0], 1), -1.0).to(env.unwrapped.device)
                            # replace final observation with risk sensitivity
                            actor_obs = torch.cat((actor_obs[:, :-1], risk_sensitivity_obs), dim=-1).to(env.unwrapped.device)
                            actions = policy(obs)

                        if algorithm == PPO:
                            actions = policy(obs)

                        if algorithm == PPOMulti or "Neutral" in args_cli.task:
                            actor_obs, critic_obs, lidar_obs, depth_obs, scandots_obs, height_scan_obs = agent._process_observations(obs, env_info)
                            actions = policy(actor_obs, lidar_obs, depth_obs, scandots_obs, height_scan_obs)
                        
                        # env stepping
                        # zero out the actions for the environments that are done

                        # print(f"[INFO] actions.shape: {actions.shape}")
                        # print(f"[INFO] dones_list.shape: {dones_list.shape}")
                        # print(f"[INFO] actions: {actions}")
                        number_of_dones = torch.sum(dones_list)
                        dones_mask = torch.ones_like(dones, dtype=torch.bool)
                        for idx, value in enumerate(dones_list):
                            if value:
                                dones_mask[idx] = True
                            else:
                                dones_mask[idx] = False
                        actions[dones_mask] = torch.zeros_like(actions[number_of_dones, :])
                        # print(f"[INFO] actions after zeroing out: {actions}")

                        # print(f"env.unwrapped.scene.articulations = {env.unwrapped.scene.articulations}")
                        
                        # set cylinder joint velocity to zero for the environments that are done
                        if torch.any(dones_list):
                            env.unwrapped.scene.reset(env_ids=torch.where(dones_list)[0])
                        
                        # env.unwrapped.termination_manager.get_termination("Episode Termination/goal_reached").successful_envs = successful_envs

                        obs, rewards, dones, env_info = env.step(actions)
                        # append this timesteps rewards (shape [num_envs]) to per_step_rewards
                        per_step_rewards = torch.cat((per_step_rewards, rewards.unsqueeze(0)), dim=0)
                        current_time += env_cfg.sim.dt * env_cfg.decimation

                        if not initialised_logs:
                            if "log" in env_info.keys():
                                # print(f"env_info['log']: {env_info['log']}")
                                for key, value in env_info["log"].items():
                                    if "Episode Termination" in key:
                                        logs.update({key: 0.0})
                                initialised_logs = True

                        if torch.any(dones):
                            # Add all the environments that are done to the list
                            dones_list = dones_list | dones

                            # add the successful environments to the list if they are not already there
                            for env_id in range(dones.shape[0]):
                                if dones[env_id]:
                                    if (not failed_list[env_id]) and (not success_list[env_id]) and (not give_up_list[env_id]) and (env.unwrapped.termination_manager.get_term('goal_reached')[env_id]): 
                                        # successful for the first time
                                        success_list[env_id] = True
                                        print(f"env_id {env_id} was added to successful_envs")
                                        success_times.append(current_time)
                                        all_times.append(current_time)
                                        if initialised_logs:
                                            if "log" in env_info.keys():
                                                for key, value in env_info["log"].items():
                                                    if "Episode Termination/goal_reached" in key:
                                                        # Log the number of successful tasks
                                                        # remove "Episode Termination/" from the key
                                                        split_key = key.split("/")[1]
                                                        logs[key] += env.unwrapped.termination_manager.get_term(split_key)[env_id].float()
                                                        #  Log task success per environment
                                                        success_list_per_env[env_id] = env.unwrapped.termination_manager.get_term(split_key)[env_id].float()
                                    elif (not failed_list[env_id]) and (not success_list[env_id]) and (not give_up_list[env_id]) and (env.unwrapped.termination_manager.get_term('give_up_action')[env_id]):
                                        # successful give up action for the first time
                                        give_up_list[env_id] = True
                                        print(f"env_id {env_id} was added to give_up_list")
                                        give_up_times.append(current_time)
                                        all_times.append(current_time)
                                        if initialised_logs:
                                            if "log" in env_info.keys():
                                                for key, value in env_info["log"].items():
                                                    if "Episode Termination/give_up_action" in key:
                                                        # Log the number of give up actions
                                                        # remove "Episode Termination/" from the key
                                                        split_key = key.split("/")[1]
                                                        logs[key] += env.unwrapped.termination_manager.get_term(split_key)[env_id].float()
                                                        #  Log task success per environment
                                                        give_up_list_per_env[env_id] = env.unwrapped.termination_manager.get_term(split_key)[env_id].float()
                                        
                                    elif (not failed_list[env_id]) and (success_list[env_id]) and (not give_up_list[env_id]) and (env.unwrapped.termination_manager.get_term('goal_reached')[env_id]): 
                                        # successfuly, but already in the success_list
                                        pass
                                    elif (not failed_list[env_id]) and (not success_list[env_id]) and (give_up_list[env_id]) and (env.unwrapped.termination_manager.get_term('give_up_action')[env_id]):
                                        # give up action, but already in the give_up_list
                                        pass
                                    elif (not success_list[env_id]) and (not failed_list[env_id]) and (not env.unwrapped.termination_manager.get_term('goal_reached')[env_id]) \
                                        and (not give_up_list[env_id]) and (not env.unwrapped.termination_manager.get_term('give_up_action')[env_id]): 
                                        # not in successful_envs or give_up_list, and failed 
                                        failed_list[env_id] = True
                                        print(f"env_id {env_id} was added to failed_list")
                                        failed_times.append(current_time)
                                        all_times.append(current_time)
                                        if initialised_logs:
                                            if "log" in env_info.keys():
                                                # sometimes more than one env contact is recorded in the logs
                                                # use a flag to ensure that only one contact is logged
                                                env_contact_bool = False
                                                for key, value in env_info["log"].items():
                                                    if "contact" in key:
                                                        if "cylinder" in key:
                                                            cylinder_split_key = key.split("/")[1]
                                                            logs[key] += env.unwrapped.termination_manager.get_term(cylinder_split_key)[env_id].float()
                                                            if env_contact_bool and (env.unwrapped.termination_manager.get_term(cylinder_split_key)[env_id].float() != 0.0):
                                                                env_contact_bool = False
                                                            # log cylinder contact per environment
                                                            cylinder_contact_list_per_env[env_id] = env.unwrapped.termination_manager.get_term(cylinder_split_key)[env_id].float()
                                                            
                                                        elif "contact_" in key or "_contact" in key:
                                                            if not env_contact_bool:
                                                                # remove "Episode Termination/" from the key
                                                                split_key = key.split("/")[1]
                                                                logs[key] += env.unwrapped.termination_manager.get_term(split_key)[env_id].float()
                                                                
                                                                if env.unwrapped.termination_manager.get_term(split_key)[env_id].float() != 0.0:
                                                                    env_contact_bool = True
                                                                # log env contact per environment
                                                                env_contact_list_per_env[env_id] = env.unwrapped.termination_manager.get_term(split_key)[env_id].float()

                                                    if "time_out" in key:
                                                        split_key = key.split("/")[1]
                                                        logs[key] += env.unwrapped.termination_manager.get_term(split_key)[env_id].float()
                                                        if env.unwrapped.termination_manager.get_term(split_key)[env_id].float() != 0.0:
                                                            print(f"Env {env_id} timed out!")
                                                            time_out_list_per_env[env_id] = env.unwrapped.termination_manager.get_term(split_key)[env_id].float()
                                                    
                                    elif (not success_list[env_id]) and (failed_list[env_id]) and (not env.unwrapped.termination_manager.get_term('goal_reached')[env_id]) \
                                        and (not give_up_list[env_id]) and (not env.unwrapped.termination_manager.get_term('give_up_action')[env_id]): 
                                        # already in the failed list, and failed
                                        pass
                                # calculate the number of completed tasks
                        n_completed_tasks = torch.sum(success_list) + torch.sum(give_up_list) + torch.sum(failed_list)
                        print(f"[INFO] {n_completed_tasks} tasks completed! {torch.sum(success_list)} successful tasks, {torch.sum(give_up_list)} given up and {torch.sum(failed_list)} failed tasks.")
                        print(f"[INFO] Rollout {rollout} percent completed: {n_completed_tasks/args_cli.num_envs*100}%")


                        if current_time >= env_cfg.episode_length_s:
                            print(f"[INFO] Time limit reached!")
                            # print number of environments that are done
                            print(f"[INFO] {torch.sum(dones_list)} tasks completed!")
                            break

                    if torch.all(dones_list):
                        print(f"[INFO] All tasks completed!")
                        for key, value in logs.items():
                            print(f"[INFO] {key}: {value}")
                        # For now instead get task success numbers from log prints... sometimes there is a "successful" task that has
                        # a goal_reached value of 0.0. For now report conservative numbers.
                        print(f"[INFO] Number of successful tasks: {torch.sum(success_list)}")
                        print(f"[INFO] Number of given up tasks: {torch.sum(give_up_list)}")
                        print(f"[INFO] Number of failed tasks: {torch.sum(failed_list)}")
                        print(f"[INFO] Average time for successful tasks: {torch.mean(torch.tensor(success_times))}")
                        print(f"[INFO] Average time for failed tasks: {torch.mean(torch.tensor(failed_times))}")
                        print(f"[INFO] Average time for all tasks: {torch.mean(torch.tensor(all_times))}")
                        break
        
                # calculate cumulative rewards for each environment across num_steps (shape [num_envs] from shape [num_envs, num_steps]) 
                cumulative_rewards = torch.sum(per_step_rewards, dim=0)

                # add per_step_rewards and cumulative_rewards to logs
                logs.update({"per_step_rewards": per_step_rewards})
                logs.update({"cumulative_rewards": cumulative_rewards})

                # add task success/failure information to logs
                logs.update({"success_list": success_list})
                logs.update({"give_up_list": give_up_list})
                logs.update({"failed_list": failed_list})
                logs.update({"success_times": torch.tensor(success_times)})
                logs.update({"give_up_times": torch.tensor(give_up_times)})
                logs.update({"failed_times": torch.tensor(failed_times)})
                logs.update({"all_times": torch.tensor(all_times)})

                # add per env information to logs
                logs.update({"env_contact_list_per_env": env_contact_list_per_env})
                logs.update({"cylinder_contact_list_per_env": cylinder_contact_list_per_env})
                logs.update({"time_out_list_per_env": time_out_list_per_env})
                logs.update({"give_up_list_per_env": give_up_list_per_env})
                logs.update({"success_list_per_env": success_list_per_env})
                logs.update({"failed_list_per_env": failed_list})

                with torch.inference_mode():
                    # reset the environment
                    env.reset()

                print(f"[INFO] Rollout {rollout} completed!")

                if (args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-Student-v0") or (args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-CVaR-Student-v0"):
                    # create copy of risk_sensitivity
                    # risk_sensitivity = args_cli.risk_sensitivity
                    
                    # add risk_sensitivity to logs
                    print(f"[INFO] Adding risk_sensitivity to logs. Risk sensitivity: {risk_sensitivity}")
                    
                    logs.update({"risk_sensitivity": torch.tensor(risk_sensitivity)})
                    


                # append logs to all_logs
                all_logs.update({rollout: logs})


                if rollout == number_of_rollouts - 1:
                    break


            print(f"[INFO] All rollouts completed!")
            print(f"[INFO] Convert tensors to lists.")
            all_logs = convert_tensors_to_lists(all_logs)
            print(f"[INFO] Saving logs to {log_dir}")
            dump_yaml(os.path.join(log_dir, "eval", filename + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".yaml"), all_logs)
            
            if (args_cli.task == "Isaac-Lift-Cube-HSRB-PPOMulti-Teacher-v0"):
                    # break after one "risk_sensitivity" run
                    break
        # dump_yaml(os.path.join(log_dir, "eval", datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".yaml"), all_logs)

        # close the simulator
        env.close()
        break


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
