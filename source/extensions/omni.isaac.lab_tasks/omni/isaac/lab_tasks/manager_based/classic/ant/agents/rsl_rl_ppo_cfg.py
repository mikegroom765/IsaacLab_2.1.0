# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlDppoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlDppoAlgorithmCfg
)

from rsl_rl.modules import QuantileNetwork


@configclass
class AntPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "ant"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        actor_noise_std=1.0,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        actor_activations=["tanh", "tanh", "tanh", "tanh"],
        critic_activations=["tanh", "tanh", "tanh", "tanh"],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_coeff=1.0,
        # use_clipped_value_loss=True,
        clip_ratio=0.2,
        entropy_coeff=0.0,
        # num_learning_epochs=5,
        batch_count=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=0.01,
        gradient_clip=1.0,
    )

@configclass
class AntDPPORunnerCfg(RslRlRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "ant_DPPO"
    empirical_normalization = True
    policy = RslRlDppoActorCriticCfg(
        actor_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        actor_activations=["tanh", "tanh", "tanh", "tanh"],
        critic_activations=["tanh", "tanh", "tanh"],
    )
    algorithm = RslRlDppoAlgorithmCfg(
        value_coeff=0.9,
        clip_ratio=0.2,
        entropy_coeff=0.006,
        batch_count=4, #4096
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        gae_lambda=0.95,
        target_kl=0.01,
        gradient_clip=1.0,
        iqn_action_samples=32,
        iqn_embedding_size=64,
        iqn_feature_layers=1,
        iqn_value_samples=8,
        qrdqn_quantile_count=200,
        value_lambda=0.95,
        value_loss="quantile_l1",
        value_loss_kwargs={},
        value_measure=QuantileNetwork.measure_wang,
        value_measure_adaptation=(60,),
        value_measure_kwargs={},
    )
