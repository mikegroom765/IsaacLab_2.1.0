# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# import atexit
import cProfile

# Create the profiler
profiler = cProfile.Profile()

# Start profiling immediately
profiler.enable()


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--dagger", action="store_true", default=False, help="Use Dagger for training.")
parser.add_argument("--teacher_run", type=str, default=None, help="Run name suffix to the log directory of the teacher.")
parser.add_argument("--teacher_checkpoint", type=str, default=None, help="Checkpoint file to use for the teacher.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli, profiler)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime
import math

from rsl_rl.runners import Runner # OnPolicyRunner
from rsl_rl.algorithms import *

from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlRunnerCfg, RslRlVecEnvWrapper
# I know this is terrible, but I am lazy
import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab.terrains.config.hsrb_reach import generate_grid_reach_terrains_cfg, HSRB_LIFT_CUBE_TERRAINS_CFG

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False # should this be True?
torch.backends.cudnn.benchmark = False


def main():
    
    agent_cfg: RslRlRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

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

    algorithms = {
        'PPO': PPO,
        'PPOMulti': PPOMulti,
        'DPPO': DPPO,
        'DPPOMulti': DPPOMulti,
    }

    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    print(f"[INFO] Agent configuration: {agent_cfg}")
    # print(f"[INFO] Environment configuration: {env_cfg}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    # print(type(env_cfg))
    # print(f"env_cfg: {env_cfg['scene']}")
    
    size=(6.0, 6.0) # size of the sub-terrain in meters
    grid_rows = 9 # number of rows of grid cells
    grid_cols = 9 # number of columns of grid cells
    grid_cell_width = 0.5 # width of a grid cell in meters
    max_filled_cells = 16 # maximum number of filled cells in a grid cell
    min_filled_cells = 10 # minimum number of filled cells in a grid cell
    table_height_range = (0.2, 0.6) # height range of the tables

    if args_cli.dagger:

        # calculate number of rows and columns for terrains based on number of environments
        print(f"[INFO] Number of environments: {args_cli.num_envs}")
        a = int(args_cli.num_envs**0.5)  # Start at the square root of n
        while a > 0:
            if args_cli.num_envs % a == 0:  # Check if a divides num_envs
                b = args_cli.num_envs // a
                num_rows, num_cols = a, b
                break
            a -= 1
        print(f"[INFO] Using {num_rows} x {num_cols} grid for training.")

        if args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-Student-v0" or args_cli.task == "Isaac-Grid-HSRB-PPOMulti-Student-v0" \
            or args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-CVaR-Student-v0" or args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-Neutral-Student-v0":

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
            env_cfg.commands.ee_pose.table_locations = hsrb_grid_reach_terrain.table_locs
            env_cfg.commands.ee_pose.grid_rows = grid_rows
            env_cfg.commands.ee_pose.grid_cols = grid_cols
            env_cfg.commands.ee_pose.grid_cell_width = grid_cell_width
            
        if args_cli.task == "Isaac-Lift-Cube-HSRB-DPPOMulti-Student-v0" or args_cli.task == "Isaac-Lift-Cube-HSRB-PPOMulti-Student-v0" \
            or args_cli.task == "Isaac-Lift-Cube-HSRB-DPPOMulti-CVaR-Student-v0" or args_cli.task =="Isaac-Lift-Cube-HSRB-DPPOMulti-Neutral-Student-v0":

            size=(5.0, 5.0) # size of the sub-terrain in meters
            table_height_range = (0.4, 0.8) # height range of the tables

            hsrb_lift_cube_reach_terrain = HSRB_LIFT_CUBE_TERRAINS_CFG(size=size,
                                                                    num_rows=num_rows,
                                                                    num_cols=num_cols,
                                                                    table_length=1.0,
                                                                    table_width=3.0,
                                                                    table_height_range=table_height_range)
            env_cfg.scene.terrain.terrain_generator = hsrb_lift_cube_reach_terrain
            table_heights_list = [hsrb_lift_cube_reach_terrain.sub_terrains[f"table_{i}"].table_height for i in range(len(hsrb_lift_cube_reach_terrain.sub_terrains))]
            env_cfg.events.reset_object_position.params["set_heights"] = table_heights_list
            env_cfg.events.reset_robot_joints.params["table_heights"] = table_heights_list
            env_cfg.commands.object_goal_region.table_heights = table_heights_list
            # env_cfg.terminations.object_dropping.params["minimum_heights"] = table_heights_list
            
            if args_cli.task == "Isaac-Lift-Cube-HSRB-DPPOMulti-CVaR-Student-v0":
                env_cfg.commands.risk_sensitivity.value_range = (0.01, 1.0)

    else:
        # calculate number of rows and columns for terrains based on number of environments
        print(f"[INFO] Number of environments: {args_cli.num_envs}")
        a = int(args_cli.num_envs**0.5)  # Start at the square root of n
        while a > 0:
            if args_cli.num_envs % a == 0:  # Check if a divides num_envs
                b = args_cli.num_envs // a
                num_rows, num_cols = a, b
                break
            a -= 1
        print(f"[INFO] Using {num_rows} x {num_cols} grid for training.")

        if args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-Teacher-v0" or args_cli.task == "Isaac-Grid-HSRB-PPOMulti-Teacher-v0" \
            or args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-CVaR-Teacher-v0" or args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-Depth-v0" \
                or args_cli.task == "Isaac-Grid-HSRB-PPOMulti-Depth-v0" or args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-Neutral-Teacher-v0":

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
            env_cfg.commands.ee_pose.table_locations = hsrb_grid_reach_terrain.table_locs
            env_cfg.commands.ee_pose.grid_rows = grid_rows
            env_cfg.commands.ee_pose.grid_cols = grid_cols
            env_cfg.commands.ee_pose.grid_cell_width = grid_cell_width
            
            if args_cli.task == "Isaac-Grid-HSRB-DPPOMulti-CVaR-Teacher-v0":
                env_cfg.commands.risk_sensitivity.value_range = (0.01, 1.0)
            
        if args_cli.task == "Isaac-Lift-Cube-HSRB-DPPOMulti-Teacher-v0" or args_cli.task == "Isaac-Lift-Cube-HSRB-PPOMulti-Teacher-v0" \
            or args_cli.task == "Isaac-Lift-Cube-HSRB-DPPOMulti-CVaR-Teacher-v0" or args_cli.task =="Isaac-Lift-Cube-HSRB-DPPOMulti-Neutral-Teacher-v0":

            size=(5.0, 5.0) # size of the sub-terrain in meters
            table_height_range = (0.4, 0.8) # height range of the tables

            hsrb_lift_cube_reach_terrain = HSRB_LIFT_CUBE_TERRAINS_CFG(size=size,
                                                                    num_rows=num_rows,
                                                                    num_cols=num_cols,
                                                                    table_length=1.0,
                                                                    table_width=3.0,
                                                                    table_height_range=table_height_range)
            env_cfg.scene.terrain.terrain_generator = hsrb_lift_cube_reach_terrain
            table_heights_list = [hsrb_lift_cube_reach_terrain.sub_terrains[f"table_{i}"].table_height for i in range(len(hsrb_lift_cube_reach_terrain.sub_terrains))]
            env_cfg.events.reset_object_position.params["set_heights"] = table_heights_list
            env_cfg.events.reset_robot_joints.params["table_heights"] = table_heights_list
            env_cfg.commands.object_goal_region.table_heights = table_heights_list
            # env_cfg.terminations.object_dropping.params["minimum_heights"] = table_heights_list
            
            if args_cli.task == "Isaac-Lift-Cube-HSRB-DPPOMulti-CVaR-Teacher-v0":
                env_cfg.commands.risk_sensitivity.value_range = (0.01, 1.0)
           
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create agent and runner from rsl-rl
    # runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # print(f"**agent_cfg.to_dict()['algorithm']: {agent_cfg.to_dict()['algorithm']}")
    # print(f"**agent_cfg.to_dict()['policy']: {agent_cfg.to_dict()['policy']}")

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
    runner = Runner(env, agent, log_dir=log_dir, device=agent_cfg.device, **kwargs)

    experiment_name = agent_cfg.experiment_name

    # TODO: Figure out what this does...
    if not args_cli.dagger:
        runner._learn_cb = [lambda *args, **kwargs: Runner._log(*args, prefix=f"{algorithm.__name__}_{experiment_name}", **kwargs)]
    else:
        runner._learn_cb = [lambda *args, **kwargs: Runner._log_dagger(*args, prefix=f"Dagger_{experiment_name}", **kwargs)]
    # print(f"[INFO] train kwargs: {**kwargs}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir

    # run training
    if args_cli.dagger:
        # load teacher policy
        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging teacher student experiment in directory: {log_root_path} / {args_cli.teacher_run}")
        # specify directory for logging runs: {time-stamp}_{run_name}
        log_dir = args_cli.teacher_run
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

        teacher_path = get_checkpoint_path(log_root_path, args_cli.teacher_run, args_cli.teacher_checkpoint)
        print(f"[INFO]: Loading teacher model checkpoint from: {teacher_path}")
        runner.load(teacher_path)
        
        # if not resuming, initialize the student policy with the teacher policy weights
        if not agent_cfg.resume:
            runner.initalise_student_policy()
        else:
            # get path to previous checkpoint
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            # load previously trained model
            runner.load_student(resume_path)

        if "Lift-Cube" in args_cli.task:
            # pretrain for 1201 iterations - twice the number of iterations for Grid task (but same number of env steps!!!)
            runner.learn_dagger(iterations=agent_cfg.max_iterations, num_pretrain_iterations=1201)
        else:
            runner.learn_dagger(iterations=agent_cfg.max_iterations)
    else:
        if agent_cfg.resume:
            # get path to previous checkpoint
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            # load previously trained model
            runner.load(resume_path)
        # dump the configuration into log-directory
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
        runner.learn(iterations=agent_cfg.max_iterations)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
