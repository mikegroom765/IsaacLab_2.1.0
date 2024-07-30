# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class FrankaCabinetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_cabinet_direct"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        actor_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        actor_activations=["tanh", "tanh", "tanh", "tanh"],
        critic_activations=["tanh", "tanh", "tanh"],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_coeff=1.0,
        # use_clipped_value_loss=True,
        clip_ratio=0.2,
        entropy_coeff=0.0,
        # num_learning_epochs=8,
        batch_count=8,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=0.008,
        gradient_clip=1.0,
    )
