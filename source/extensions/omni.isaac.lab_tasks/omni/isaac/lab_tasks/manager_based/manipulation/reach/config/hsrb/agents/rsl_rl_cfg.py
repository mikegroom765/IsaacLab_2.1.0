# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticCustomCfg,
    RslRlPpoAlgorithmCfg,
    RslRlDppoAlgorithmCfg,
    RslRlDppoActorCriticCfg,
    RslRlRunnerCfg,
    RslRlDppoMultiInputAlgorithmCfg,
)


@configclass
class HSRBReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "hsrb_reach"
    run_name = ""
    resume = False
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        actor_noise_std=1.0,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        actor_activations=["tanh", "tanh", "tanh"],
        critic_activations=["tanh", "tanh", "tanh"],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_coeff=1.0,
        clip_ratio=0.2,
        entropy_coeff=0.001,
        batch_count=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=0.01,
        gradient_clip=1.0,
    )


@configclass
class HSRBReachDPPORunnerCfg(RslRlRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "hsrb_reach_DPPO"
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
        batch_count=16, #4096
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
        value_measure=None,
        value_measure_adaptation=None,
        value_measure_kwargs={},
    )

@configclass
class HSRBReachDPPOMultiInputRunnerCfg(RslRlRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "hsrb_reach_DPPO_multi_input"
    empirical_normalization = True
    policy = RslRlDppoMultiInputAlgorithmCfg(
        actor_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        actor_activations=["tanh", "tanh", "tanh", "tanh"],
        critic_activations=["tanh", "tanh", "tanh"],
        lidar_encode_conv_activations=["relu", "relu", "relu", "relu"],
        lidar_encode_conv_channels=[32, 64, 64],
        lidar_encode_conv_kernels=[3, 3, 3],
        lidar_encode_conv_strides=[4, 2, 1],
        lidar_encode_conv_paddings=[0, 0, 0],
        lidar_encode_output_dim=64,
        depth_encode_conv_activations=["relu", "relu", "relu"],
        depth_encode_conv_channels=[32, 64],
        depth_encode_conv_kernels=[5, 3],
        depth_encode_conv_strides=[1, 1],
        depth_encode_conv_paddings=[0, 0],
        depth_encode_max_pool_kernels=[2, 2],
        depth_encode_max_pool_strides=[2, 2],
        depth_encode_output_dim=64,
    )
    algorithm = RslRlDppoAlgorithmCfg(
        class_name="DPPOMulti",
        critic_network="multi_qrdqn",
        value_coeff=0.9,
        clip_ratio=0.2,
        entropy_coeff=0.006,
        batch_count=16, #4096
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
        value_measure=None,
        value_measure_adaptation=None,
        value_measure_kwargs={},
    )