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
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_tasks.manager_based.manipulation.reach.reach_env_cfg import CurriculumCfg, TerminationsCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp.rewards import position_command_error_frame_shaped



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

    env_cfg.episode_length_s = 100.0

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
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    print(f"[INFO] Policy loaded successfully!")

    # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     runner.alg.actor_critic, runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     runner.alg.actor_critic, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    risk_sensitivity = 1
    time_limit = 1000
    current_time = 0
    # binary tensor to keep track of environments that are done
    dones = torch.zeros(args_cli.num_envs, dtype=torch.bool).to(env.unwrapped.device)
    dones_list = torch.zeros(args_cli.num_envs, dtype=torch.bool).to(env.unwrapped.device)
    success_list = torch.zeros(args_cli.num_envs, dtype=torch.bool).to(env.unwrapped.device)
    failed_list = torch.zeros(args_cli.num_envs, dtype=torch.bool).to(env.unwrapped.device)
    success_times = []
    failed_times = []
    all_times = [] 

    if not hasattr(position_command_error_frame_shaped, 'successful_envs'):
        successful_envs = []
    else:
        successful_envs = successful_envs = env.unwrapped.reward_manager.get_term_cfg("end_effector_position_tracking").func.successful_envs

    # reset environment
    obs, env_info = env.get_observations()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            
            # risk sensitivity 
            if algorithm == DPPOMulti:
                actor_obs, critic_obs, lidar_obs, depth_obs, scandots_obs, height_scan_obs = agent._process_observations(obs, env_info)
                risk_sensitivity_obs = torch.full((actor_obs.shape[0], risk_sensitivity), 1.0).to(env.unwrapped.device)
                # replace final observation with risk sensitivity
                actor_obs = torch.cat((actor_obs[:, :-1], risk_sensitivity_obs), dim=-1).to(env.unwrapped.device)
            
                # agent stepping
                actions = policy(actor_obs, lidar_obs, depth_obs, scandots_obs, height_scan_obs)

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
            # zero out the actions for the environments that are done

            # print(f"[INFO] actions.shape: {actions.shape}")
            # print(f"[INFO] dones_list.shape: {dones_list.shape}")
            # print(f"[INFO] actions: {actions}")
            number_of_dones = torch.sum(dones_list)
            # print(f"dones_list: {dones_list}")
            dones_mask = torch.ones_like(dones, dtype=torch.bool)
            for idx, value in enumerate(dones_list):
                if value:
                    dones_mask[idx] = True
                else:
                    dones_mask[idx] = False
            # print(f"dones_mask: {dones_mask}")
            actions[dones_mask] = torch.zeros_like(actions[number_of_dones, :])
            # print(f"[INFO] actions after zeroing out: {actions}")

            obs, rewards, dones, env_info = env.step(actions)
            current_time += env_cfg.sim.dt

            if torch.any(dones):
                # Add all the environments that are done to the list
                print(f"[INFO] A task was completed!")
                dones_list = dones_list | dones
                # print(f"dones_list: {dones_list}")
                successful_envs = successful_envs = env.unwrapped.reward_manager.get_term_cfg("end_effector_position_tracking").func.successful_envs
            
                # add the successful environments to the list if they are not already there
                for env_id in range(dones.shape[0]):
                    if dones[env_id]:
                        # print(f"success_list: {success_list}")
                        # print(f"failed_list: {failed_list}")
                        # print(f"successful_envs: {successful_envs}")
                        if (not success_list[env_id]) and (env_id in successful_envs):
                            success_list[env_id] = True
                            print(f"env_id {env_id} was added to successful_envs")
                            success_times.append(current_time)
                            all_times.append(current_time)
                        elif (success_list[env_id]) and (env_id in successful_envs):
                            print(f"env_id {env_id} was already in successful_envs")
                        elif (not failed_list[env_id]) and (env_id not in successful_envs):
                            failed_list[env_id] = True
                            print(f"env_id {env_id} was added to failed_list")
                            failed_times.append(current_time)
                            all_times.append(current_time)
                        elif (failed_list[env_id]) and (env_id not in successful_envs):
                            print(f"env_id {env_id} was already in failed_list")


            if current_time >= time_limit:
                print(f"[INFO] Time limit reached!")
                # print number of environments that are done
                print(f"[INFO] {torch.sum(dones_list)} tasks completed!")
                break

            if torch.all(dones_list):
                print(f"[INFO] All tasks completed!")
                print(f"[INFO] Number of successful tasks: {torch.sum(success_list)}")
                print(f"[INFO] Number of failed tasks: {torch.sum(failed_list)}")
                print(f"[INFO] Average time for successful tasks: {torch.mean(torch.tensor(success_times))}")
                print(f"[INFO] Average time for failed tasks: {torch.mean(torch.tensor(failed_times))}")
                print(f"[INFO] Average time for all tasks: {torch.mean(torch.tensor(all_times))}")
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
