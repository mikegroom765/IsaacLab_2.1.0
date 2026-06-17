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
    RslRlRunnerCfg, #  This is actually for DPPO, these configs need tidying....
    RslRlPPORunnerCfg,
    RslRlPpoMultiInputAlgorithmCfg,
    RslRlDppoMultiInputAlgorithmCfg,
)

from rsl_rl.modules import QuantileNetwork


@configclass
class HSRBReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96
    max_iterations = 15000
    save_interval = 100
    experiment_name = "hsrb_reach_PPO"
    run_name = ""
    resume = False
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        actor_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_activations=["tanh", "tanh", "tanh", "linear"],
        critic_activations=["tanh", "tanh", "tanh", "linear"],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_coeff=1.0,
        num_learning_epochs=5,
        clip_ratio=0.2,
        entropy_coeff=0.01,
        batch_count=64,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=0.01,
        gradient_clip=1.0,
    )


@configclass
class HSRBReachDPPORunnerCfg(RslRlRunnerCfg):
    num_steps_per_env = 96
    max_iterations = 15000
    save_interval = 500
    experiment_name = "hsrb_reach_DPPO"
    empirical_normalization = True
    policy = RslRlDppoActorCriticCfg(
        actor_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        actor_activations=["tanh", "tanh", "tanh", "linear"],
        critic_activations=["tanh", "tanh", "tanh"],
    )
    algorithm = RslRlDppoAlgorithmCfg(
        value_coeff=0.9,
        num_learning_epochs=5,
        clip_ratio=0.2,
        entropy_coeff=0.01,
        batch_count=64, #4096
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
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
        value_measure_adaptation=(48,), # (48,) is the default value - (not using a cylinder in the env!)
        value_measure_kwargs={},
    )

@configclass
class HSRBReachDPPOMultiInputRunnerCfg(RslRlRunnerCfg):
    num_steps_per_env = 96
    max_iterations = 10000
    save_interval = 500
    experiment_name = "hsrb_reach_DPPO_multi_input"
    empirical_normalization = True
    policy = RslRlDppoMultiInputAlgorithmCfg(
        actor_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        actor_activations=["tanh", "tanh", "tanh", "linear"],
        critic_activations=["tanh", "tanh", "tanh"],
        lidar_encode=False,
        lidar_encode_conv_activations=["relu", "relu", "relu", "relu"],
        lidar_encode_conv_channels=[32, 64, 64],
        lidar_encode_conv_kernels=[3, 3, 3],
        lidar_encode_conv_strides=[4, 2, 1],
        lidar_encode_conv_paddings=[0, 0, 0],
        lidar_encode_output_dim=64,
        depth_encode=False,
        depth_encode_conv_activations=["relu", "relu", "relu"],
        depth_encode_conv_channels=[32, 64],
        depth_encode_conv_kernels=[5, 3],
        depth_encode_conv_strides=[1, 1],
        depth_encode_conv_paddings=[0, 0],
        depth_encode_max_pool_kernels=[2, 2],
        depth_encode_max_pool_strides=[2, 2],
        depth_encode_output_dim=16,
        scandot_encode=True,
        num_scandots=121,
        num_global_features=16,
        height_scan_encode=False,
        num_height_scan_points=121,
        height_scan_encode_output_dim=16,
        height_scan_encode_layer_dims=[256, 128],
        height_scan_encode_activations=["relu", "relu", "relu"],
        dropout_LSTM=0.1,
        dropout_MLP=0.1,
        dropout_height_scan_encoder=0.1,
        dropout_depth_encoder=0.1,
    )
    algorithm = RslRlDppoAlgorithmCfg(
        class_name="DPPOMulti",
        critic_network="multi_qrdqn",
        value_coeff=0.9,
        num_learning_epochs=5,
        clip_ratio=0.2,
        entropy_coeff=0.01,
        batch_count=64, #4096
        learning_rate=1.0e-3, # 3e-4
        schedule="adaptive",
        gamma=0.99, # 0.998
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
        value_measure_adaptation=(48,),
        value_measure_kwargs={},
    )

@configclass
class HSRBReachDPPOMultiTeacher1aRunnerCfg(RslRlRunnerCfg):
    num_steps_per_env = 96
    max_iterations = 10000
    save_interval = 100
    experiment_name = "hsrb_reach_DPPOMulti_teacher_1a"
    empirical_normalization = True
    policy = RslRlDppoMultiInputAlgorithmCfg(
        actor_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        actor_activations=["tanh", "tanh", "tanh", "tanh"],
        critic_activations=["tanh", "tanh", "tanh"],
        lidar_encode=False,
        lidar_encode_conv_activations=["relu", "relu", "relu", "relu"],
        lidar_encode_conv_channels=[32, 64, 64],
        lidar_encode_conv_kernels=[3, 3, 3],
        lidar_encode_conv_strides=[4, 2, 1],
        lidar_encode_conv_paddings=[0, 0, 0],
        lidar_encode_output_dim=64,
        depth_encode=False,
        depth_encode_conv_activations=["relu", "relu", "relu"],
        depth_encode_conv_channels=[32, 64],
        depth_encode_conv_kernels=[5, 3],
        depth_encode_conv_strides=[1, 1],
        depth_encode_conv_paddings=[0, 0],
        depth_encode_max_pool_kernels=[2, 2],
        depth_encode_max_pool_strides=[2, 2],
        depth_encode_output_dim=16,
        scandot_encode=False,
        num_scandots=121,
        num_global_features=16,
        height_scan_encode=True,
        num_height_scan_points=121,
        height_scan_encode_output_dim=16,
        height_scan_encode_layer_dims=[256, 128],
        height_scan_encode_activations=["relu", "relu", "relu"],
    )
    algorithm = RslRlDppoAlgorithmCfg(
        class_name="DPPOMulti",
        critic_network="multi_qrdqn",
        value_coeff=0.9,
        num_learning_epochs=5,
        clip_ratio=0.2,
        entropy_coeff=0.001,
        batch_count=64, #4096
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
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
        value_measure_adaptation=(18,),
        value_measure_kwargs={},
    )

@configclass
class HSRBGridDPPOMultiTeacherRunnerCfg(HSRBReachDPPOMultiTeacher1aRunnerCfg):
    experiment_name = "hsrb_grid_DPPOMulti_teacher"
    
@configclass
class HSRBGridDPPOMultiDepthRunnerCfg(HSRBReachDPPOMultiTeacher1aRunnerCfg):
    experiment_name = "hsrb_grid_DPPOMulti_depth"
    policy = RslRlDppoMultiInputAlgorithmCfg(
        actor_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        actor_activations=["tanh", "tanh", "tanh", "tanh"],
        critic_activations=["tanh", "tanh", "tanh"],
        lidar_encode=False,
        lidar_encode_conv_activations=["relu", "relu", "relu", "relu"],
        lidar_encode_conv_channels=[32, 64, 64],
        lidar_encode_conv_kernels=[3, 3, 3],
        lidar_encode_conv_strides=[4, 2, 1],
        lidar_encode_conv_paddings=[0, 0, 0],
        lidar_encode_output_dim=64,
        depth_encode=True,
        depth_encode_conv_activations=["relu", "relu", "relu"],
        depth_encode_conv_channels=[32, 64],
        depth_encode_conv_kernels=[5, 3],
        depth_encode_conv_strides=[1, 1],
        depth_encode_conv_paddings=[0, 0],
        depth_encode_max_pool_kernels=[2, 2],
        depth_encode_max_pool_strides=[2, 2],
        depth_encode_output_dim=16,
        scandot_encode=False,
        num_scandots=121,
        num_global_features=16,
        height_scan_encode=False,
        num_height_scan_points=121,
        height_scan_encode_output_dim=16,
        height_scan_encode_layer_dims=[256, 128],
        height_scan_encode_activations=["relu", "relu", "relu"],
    )
    algorithm = RslRlDppoAlgorithmCfg(
        class_name="DPPOMulti",
        critic_network="multi_qrdqn",
        value_coeff=0.9,
        num_learning_epochs=5,
        clip_ratio=0.2,
        entropy_coeff=0.001,
        batch_count=8, #4096
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
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
        value_measure_adaptation=(18,),
        value_measure_kwargs={},
    )

@configclass
class HSRBReachPPOMultiTeacher1aRunnerCfg(RslRlPPORunnerCfg):
    num_steps_per_env = 96
    max_iterations = 10000
    save_interval = 100
    experiment_name = "hsrb_reach_PPOMulti_teacher_1a"
    empirical_normalization = True
    policy = RslRlPpoMultiInputAlgorithmCfg(
        actor_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        actor_activations=["tanh", "tanh", "tanh", "tanh"],
        critic_activations=["tanh", "tanh", "tanh", "tanh"],
        lidar_encode=False,
        lidar_encode_conv_activations=["relu", "relu", "relu", "relu"],
        lidar_encode_conv_channels=[32, 64, 64],
        lidar_encode_conv_kernels=[3, 3, 3],
        lidar_encode_conv_strides=[4, 2, 1],
        lidar_encode_conv_paddings=[0, 0, 0],
        lidar_encode_output_dim=64,
        depth_encode=False,
        depth_encode_conv_activations=["relu", "relu", "relu"],
        depth_encode_conv_channels=[32, 64],
        depth_encode_conv_kernels=[5, 3],
        depth_encode_conv_strides=[1, 1],
        depth_encode_conv_paddings=[0, 0],
        depth_encode_max_pool_kernels=[2, 2],
        depth_encode_max_pool_strides=[2, 2],
        depth_encode_output_dim=16,
        scandot_encode=False,
        num_scandots=121,
        num_global_features=16,
        height_scan_encode=True,
        num_height_scan_points=121,
        height_scan_encode_output_dim=16,
        height_scan_encode_layer_dims=[256, 128],
        height_scan_encode_activations=["relu", "relu", "relu"],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOMulti",
        value_coeff=0.9,
        num_learning_epochs=5,
        clip_ratio=0.2,
        entropy_coeff=0.001,
        batch_count=64, #4096
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=0.01,
        gradient_clip=1.0,
    )

@configclass
class HSRBGridPPOMultiTeacherRunnerCfg(HSRBReachPPOMultiTeacher1aRunnerCfg):
    experiment_name = "hsrb_grid_PPOMulti_teacher"
    
@configclass
class HSRBGridPPOMultiDepthRunnerCfg(HSRBReachPPOMultiTeacher1aRunnerCfg):
    experiment_name = "hsrb_grid_PPOMulti_depth"
    policy = RslRlPpoMultiInputAlgorithmCfg(
        actor_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        actor_activations=["tanh", "tanh", "tanh", "tanh"],
        critic_activations=["tanh", "tanh", "tanh", "tanh"],
        lidar_encode=False,
        lidar_encode_conv_activations=["relu", "relu", "relu", "relu"],
        lidar_encode_conv_channels=[32, 64, 64],
        lidar_encode_conv_kernels=[3, 3, 3],
        lidar_encode_conv_strides=[4, 2, 1],
        lidar_encode_conv_paddings=[0, 0, 0],
        lidar_encode_output_dim=64,
        depth_encode=True,
        depth_encode_conv_activations=["relu", "relu", "relu"],
        depth_encode_conv_channels=[32, 64],
        depth_encode_conv_kernels=[5, 3],
        depth_encode_conv_strides=[1, 1],
        depth_encode_conv_paddings=[0, 0],
        depth_encode_max_pool_kernels=[2, 2],
        depth_encode_max_pool_strides=[2, 2],
        depth_encode_output_dim=16,
        scandot_encode=False,
        num_scandots=121,
        num_global_features=16,
        height_scan_encode=False,
        num_height_scan_points=121,
        height_scan_encode_output_dim=16,
        height_scan_encode_layer_dims=[256, 128],
        height_scan_encode_activations=["relu", "relu", "relu"],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOMulti",
        value_coeff=0.9,
        num_learning_epochs=5,
        clip_ratio=0.2,
        entropy_coeff=0.001,
        batch_count=8, #4096
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=0.01,
        gradient_clip=1.0,
    )

@configclass
class HSRBReachDPPOMultiStudentRunnerCfg(HSRBReachDPPOMultiTeacher1aRunnerCfg):
    experiment_name = "hsrb_reach_DPPOMulti_student"
    max_iterations = 100000

@configclass
class HSRBGridDPPOMultiStudentRunnerCfg(HSRBGridDPPOMultiTeacherRunnerCfg):
    experiment_name = "hsrb_grid_DPPOMulti_student"
    max_iterations = 100000

@configclass
class HSRBReachPPOMultiStudentRunnerCfg(HSRBReachPPOMultiTeacher1aRunnerCfg):
    experiment_name = "hsrb_reach_PPOMulti_student"
    max_iterations = 100000

@configclass
class HSRBGridPPOMultiStudentRunnerCfg(HSRBGridPPOMultiTeacherRunnerCfg):
    experiment_name = "hsrb_grid_PPOMulti_student"
    max_iterations = 100000
    
#### CVaR Runner Configs ####
@configclass
class HSRBGridDPPOMultiCVaRTeacherRunnerCfg(HSRBReachDPPOMultiTeacher1aRunnerCfg):
    experiment_name = "hsrb_grid_DPPOMulti_CVaR_teacher"
    algorithm = RslRlDppoAlgorithmCfg(
        class_name="DPPOMulti",
        critic_network="multi_qrdqn",
        value_coeff=0.9,
        num_learning_epochs=5,
        clip_ratio=0.2,
        entropy_coeff=0.001,
        batch_count=64, #4096
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
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
        value_measure=QuantileNetwork.measure_cvar,
        value_measure_adaptation=(18,),
        value_measure_kwargs={},
    )
    
@configclass
class HSRBGridDPPOMultiCVaRStudentRunnerCfg(HSRBGridDPPOMultiCVaRTeacherRunnerCfg):
    experiment_name = "hsrb_grid_DPPOMulti_CVaR_student"
    max_iterations = 100000
    
    
#### Neutral Runner Configs ####
@configclass
class HSRBGridDPPOMultiNeutralTeacherRunnerCfg(HSRBReachDPPOMultiTeacher1aRunnerCfg):
    experiment_name = "hsrb_grid_DPPOMulti_Neutral_teacher"
    algorithm = RslRlDppoAlgorithmCfg(
        class_name="DPPOMulti",
        critic_network="multi_qrdqn",
        value_coeff=0.9,
        num_learning_epochs=5,
        clip_ratio=0.2,
        entropy_coeff=0.001,
        batch_count=64, #4096
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
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
        value_measure=QuantileNetwork.measure_neutral,
        value_measure_adaptation=None,
        value_measure_kwargs={},
    )
    
@configclass
class HSRBGridDPPOMultiNeutralStudentRunnerCfg(HSRBGridDPPOMultiNeutralTeacherRunnerCfg):
    experiment_name = "hsrb_grid_DPPOMulti_Neutral_student"
    max_iterations = 100000