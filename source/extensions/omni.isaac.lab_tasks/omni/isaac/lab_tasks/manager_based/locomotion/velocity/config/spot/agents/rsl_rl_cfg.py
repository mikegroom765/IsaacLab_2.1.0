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
class SpotFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 50
    experiment_name = "spot_flat"
    empirical_normalization = False
    store_code_state = False
    policy = RslRlPpoActorCriticCfg(
        actor_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_activations=["tanh", "tanh", "tanh", "tanh"],
        critic_activations=["tanh", "tanh", "tanh"],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_coeff=0.5,
        # use_clipped_value_loss=True,
        clip_ratio=0.2,
        entropy_coeff=0.0025,
        # num_learning_epochs=5,
        batch_count=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=0.01,
        gradient_clip=1.0,
    )
