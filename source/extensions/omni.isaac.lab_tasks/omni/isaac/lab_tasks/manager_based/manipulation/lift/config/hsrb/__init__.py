# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import agents, joint_pos_env_cfg, joint_pos_student_policy

##
# Register Gym environments.
##


##
# Using the new RSL-RL runner
## 

##
# Joint Position Control
##


## HSRB Lift Cube - DPPO-Wang
gym.register(
    id="Isaac-Lift-Cube-HSRB-DPPOMulti-Teacher-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.HSRBCubeLiftTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HSRBLiftDPPOMultiTeacherRunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

## HSRB Lift Cube - DPPO-Wang PLAY
gym.register(
    id="Isaac-Lift-Cube-HSRB-DPPOMulti-Teacher-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.HSRBCubeLiftTeacherEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HSRBLiftDPPOMultiTeacherRunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

## HSRB Lift Cube - DPPO-Neutral
gym.register(
    id="Isaac-Lift-Cube-HSRB-DPPOMulti-Neutral-Teacher-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.HSRBCubeLiftTeacherEnvCfg_Neutral,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HSRBLiftDPPOMultiNeutralTeacherRunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

## HSRB Lift Cube - DPPO-Neutral PLAY
gym.register(
    id="Isaac-Lift-Cube-HSRB-DPPOMulti-Neutral-Teacher-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.HSRBCubeLiftTeacherEnvCfg_Neutral_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HSRBLiftDPPOMultiNeutralTeacherRunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

## HSRB Lift Cube - PPO
gym.register(
    id="Isaac-Lift-Cube-HSRB-PPOMulti-Teacher-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.HSRBCubeLiftTeacherEnvCfg_PPO,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HSRBLiftPPOMultiTeacherRunnerCfg,
    },
)


## HSRB Lift Cube - PPO PLAY
gym.register(
    id="Isaac-Lift-Cube-HSRB-PPOMulti-Teacher-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.HSRBCubeLiftTeacherEnvCfg_PPO_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HSRBLiftPPOMultiTeacherRunnerCfg,
    },
)


## HSRB Lift Cube - DPPO-CVaR
gym.register(
    id="Isaac-Lift-Cube-HSRB-DPPOMulti-CVaR-Teacher-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_env_cfg.HSRBCubeLiftTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.HSRBLiftDPPOMultiCVaRTeacherRunnerCfg,
    },
)


## HSRB Lift Cube - DPPO-Wang student
# Grid task for student DPPO training
gym.register(
    id="Isaac-Lift-Cube-HSRB-DPPOMulti-Student-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_student_policy.HSRBLiftStudentEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HSRBLiftDPPOMultiStudentRunnerCfg",
    },
)

## HSRB Lift Cube - PPO student
gym.register(
    id="Isaac-Lift-Cube-HSRB-PPOMulti-Student-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_student_policy.HSRBLiftStudentEnvCfg_PPO,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HSRBLiftPPOMultiStudentRunnerCfg",
    },
)

## HSRB Lift Cube - DPPO-CVaR student
gym.register(
    id="Isaac-Lift-Cube-HSRB-DPPOMulti-CVaR-Student-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_student_policy.HSRBLiftStudentEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HSRBLiftDPPOMultiCVaRStudentRunnerCfg",
    },
)

# HSRB Lift Cube - DPPO-Neutral student
gym.register(
    id="Isaac-Lift-Cube-HSRB-DPPOMulti-Neutral-Student-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": joint_pos_student_policy.HSRBLiftStudentEnvCfg_PPO,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:HSRBLiftDPPOMultiNeutralStudentRunnerCfg",
    },
)