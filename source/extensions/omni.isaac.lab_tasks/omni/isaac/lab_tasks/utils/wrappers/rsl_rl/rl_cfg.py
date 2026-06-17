# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal, Dict, Union, Tuple

from omni.isaac.lab.utils import configclass


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    actor_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    actor_activations: list[str] = MISSING
    """The activation functions for the actor network. Add "linear" for the last layer."""

    critic_activations: list[str] = MISSING
    """The activation functions for the critic network. Add "linear" for the last layer."""

    recurrent: bool = True
    """Whether to use a recurrent network."""
    
    input_normalization: bool = False
    """Whether to use input normalization."""

    recurrent_layers: int = 1
    """The number of recurrent layers."""

    recurrent_module: str = "LSTM"
    """The recurrent module to use. Must be one of Network.recurrent_modules."""

    recurrent_tf_context_length: int = 64
    """The context length of the Transformer (if a transformer is used)."""

    recurrent_tf_head_count: int = 8
    """The head count of the Transformer (if a transformer is used)."""



@configclass
class RslRlPpoActorCriticCustomCfg(RslRlPpoActorCriticCfg):
    """Configuration for PPO actor-critic networks with custom architecture."""

    class_name: str = "ActorCriticCustom"
    """The policy class name. Default is ActorCriticCustom."""

    actor_noise_std: float = MISSING

    lidar_input_hidden_dims: list[int] = MISSING

    depth_input_hidden_dims: list[int] = MISSING

    actor_hidden_dims: list[int] = MISSING

    critic_hidden_dims: list[int] = MISSING

    actor_activations: list[str] = MISSING
    """The activation functions for the actor network. Add "linear" for the last layer."""

    critic_activations: list[str] = MISSING
    """The activation functions for the critic network. Add "linear" for the last layer."""


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is PPO."""

    value_coeff: float = MISSING
    """The coefficient for the value loss."""

    # use_clipped_value_loss: bool = MISSING
    # """Whether to use clipped value loss."""

    clip_ratio: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coeff: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    batch_count: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    gae_lambda: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    target_kl: float = MISSING
    """The desired KL divergence."""

    gradient_clip: float = MISSING
    """The maximum gradient norm."""


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

@configclass
class RslRlDppoActorCriticCfg:
    """Configuration for the DPPO actor-critic networks."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    actor_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    actor_activations: list[str] = MISSING
    """The activation functions for the actor network. Add "linear" for the last layer."""

    critic_activations: list[str] = MISSING
    """The activation functions for the critic network. Add "linear" for the last layer."""

    recurrent: bool = True
    """Whether to use a recurrent network."""

    input_normalization: bool = False
    """Whether to use input normalization."""

    recurrent_layers: int = 1
    """The number of recurrent layers."""

    recurrent_module: str = "LSTM"
    """The recurrent module to use. Must be one of Network.recurrent_modules."""

    recurrent_tf_context_length: int = 64
    """The context length of the Transformer (if a transformer is used)."""

    recurrent_tf_head_count: int = 8
    """The head count of the Transformer (if a transformer is used)."""

    # TODO: Add more parameters relevent to the recurrent module. (There are seperate parameters for actor and critic networks)

@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = "PPO"
    """The algorithm class name. Default is DPPO."""

    value_coeff: float = MISSING
    """The coefficient for the value loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    clip_ratio: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coeff: float = MISSING
    """The coefficient for the entropy loss."""

    batch_count: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    gae_lambda: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    target_kl: float = MISSING
    """The target KL divergence."""

    gradient_clip: float = MISSING
    """The maximum gradient norm."""

@configclass
class RslRlDppoAlgorithmCfg:
    """Configuration for the DPPO algorithm."""

    class_name: str = "DPPO"
    """The algorithm class name. Default is DPPO."""

    critic_network: str = "qrdqn"
    """The critic network to use. Default is qrdqn."""

    iqn_action_samples: int = MISSING
    """The number of samples to use for the critic IQN network when acting."""

    iqn_embedding_size: int = MISSING
    """The embedding size to use for the critic IQN network."""

    iqn_feature_layers: int = MISSING
    """The number of feature layers to use for the critic IQN network."""

    iqn_value_samples: int = MISSING
    """The number of samples to use for the critic IQN network when computing the value."""

    qrdqn_quantile_count: int = MISSING
    """The number of quantiles to use for the critic QR network."""

    value_lambda: float = MISSING
    """The lambda parameter for the SR(lambda) value target computation."""

    value_loss: str = MISSING
    """The loss function to use for the critic network."""

    value_loss_kwargs: Dict = MISSING
    """Keyword arguments for computing the value loss."""

    value_measure: str = MISSING
    """The probability measure to apply to the critic network output distribution when
                updating the policy."""
    
    value_measure_adaptation: Union[Tuple, None] = MISSING
    """Controls adaptation of the value measure. If None, no
                adaptation is performed. If a tuple, the tuple specifies the observations that are passed to the value
                measure."""

    value_measure_kwargs: Dict = MISSING
    """The keyword arguments to pass to the value measure."""

    value_coeff: float = MISSING
    """The coefficient for the value loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    clip_ratio: float = MISSING
    """The clipping parameter for the policy."""

    entropy_coeff: float = MISSING
    """The coefficient for the entropy loss."""

    batch_count: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    gae_lambda: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    target_kl: float = MISSING
    """The target KL divergence."""

    gradient_clip: float = MISSING
    """The maximum gradient norm."""

@configclass
class RslRlDppoMultiInputAlgorithmCfg(RslRlDppoActorCriticCfg):
    """Configuration for the DPPO multi-input networks."""

    lidar_encode: bool = True
    """Whether to encode the lidar input."""

    lidar_encode_output_dim: int = 64
    """The output dimensions of the lidar encoder."""

    lidar_encode_conv_channels: list[int] = [32, 64, 64]
    """The number of output channels for each convolutional layer of the lidar encoder."""

    lidar_encode_conv_kernels: list[int] = [3, 3, 3]
    """The kernel size for each convolutional layer of the lidar encoder."""

    lidar_encode_conv_strides: list[int] = [4, 2, 1]
    """The stride for each convolutional layer of the lidar encoder."""

    lidar_encode_conv_paddings: list[int] = [0, 0, 0]
    """The padding for each convolutional layer of the lidar encoder."""

    lidar_encode_conv_activations: list[str] = ["relu", "relu", "relu"]
    """The activation function for each convolutional layer of the lidar encoder."""

    depth_encode: bool = True
    """Whether to encode the depth input."""

    depth_encode_output_dim: int = 64
    """The output dimensions of the depth encoder."""

    depth_encode_conv_channels: list[int] = [32, 64]
    """The number of output channels for each convolutional layer of the depth encoder."""

    depth_encode_conv_kernels: list[int] = [5, 3]
    """The kernel size for each convolutional layer of the depth encoder."""

    depth_encode_conv_strides: list[int] = [1, 1]
    """The stride for each convolutional layer of the depth encoder."""

    depth_encode_conv_paddings: list[int] = [0, 0]
    """The padding for each convolutional layer of the depth encoder."""

    depth_encode_conv_activations: list[str] = ["relu", "relu", "relu"]
    """The activation function for each convolutional layer of the depth encoder."""

    depth_encode_max_pool_kernels: list[int] = [2, 2]
    """The kernel size for the max pooling layer of the depth encoder."""

    depth_encode_max_pool_strides: list[int] = [2, 2]
    """The stride for the max pooling layer of the depth encoder."""

    scandot_encode: bool = False
    """Whether to encode the scandot input."""

    num_scandots: int = 121
    """The number of scandots."""

    num_global_features: int = 16
    """The number of global features output by the pointnet encoder."""

    height_scan_encode: bool = True
    """Whether to encode the height scan input."""

    num_height_scan_points: int = 121
    """The number of height scan points."""

    height_scan_encode_output_dim: int = 16
    """The output dimensions of the height scan encoder."""

    height_scan_encode_layer_dims: list[int] = [256, 128]
    """The hidden dimensions of the height scan encoder."""

    height_scan_encode_activations: list[str] = ["relu", "relu", "relu"]
    """The activation functions for the height scan encoder."""

    """Dropouts values!"""
    dropout_LSTM: float = 0.0
    """The dropout value for the LSTM layer (only if more than one LSTM layer)."""
    dropout_MLP: float = 0.0
    """The dropout value for the final MLP layer."""
    dropout_depth_encoder: float = 0.0
    """The dropout value for the depth encoder."""
    dropout_height_scan_encoder: float = 0.0
    """The dropout value for the height scan encoder."""
    dropout_critic: bool = False
    """Whether to use dropout in the critic network."""
    

@configclass
class RslRlPpoMultiInputAlgorithmCfg(RslRlPpoActorCriticCfg):
    """Configuration for the DPPO multi-input networks."""

    lidar_encode: bool = True
    """Whether to encode the lidar input."""

    lidar_encode_output_dim: int = 64
    """The output dimensions of the lidar encoder."""

    lidar_encode_conv_channels: list[int] = [32, 64, 64]
    """The number of output channels for each convolutional layer of the lidar encoder."""

    lidar_encode_conv_kernels: list[int] = [3, 3, 3]
    """The kernel size for each convolutional layer of the lidar encoder."""

    lidar_encode_conv_strides: list[int] = [4, 2, 1]
    """The stride for each convolutional layer of the lidar encoder."""

    lidar_encode_conv_paddings: list[int] = [0, 0, 0]
    """The padding for each convolutional layer of the lidar encoder."""

    lidar_encode_conv_activations: list[str] = ["relu", "relu", "relu"]
    """The activation function for each convolutional layer of the lidar encoder."""

    depth_encode: bool = True
    """Whether to encode the depth input."""

    depth_encode_output_dim: int = 64
    """The output dimensions of the depth encoder."""

    depth_encode_conv_channels: list[int] = [32, 64]
    """The number of output channels for each convolutional layer of the depth encoder."""

    depth_encode_conv_kernels: list[int] = [5, 3]
    """The kernel size for each convolutional layer of the depth encoder."""

    depth_encode_conv_strides: list[int] = [1, 1]
    """The stride for each convolutional layer of the depth encoder."""

    depth_encode_conv_paddings: list[int] = [0, 0]
    """The padding for each convolutional layer of the depth encoder."""

    depth_encode_conv_activations: list[str] = ["relu", "relu", "relu"]
    """The activation function for each convolutional layer of the depth encoder."""

    depth_encode_max_pool_kernels: list[int] = [2, 2]
    """The kernel size for the max pooling layer of the depth encoder."""

    depth_encode_max_pool_strides: list[int] = [2, 2]
    """The stride for the max pooling layer of the depth encoder."""

    scandot_encode: bool = False
    """Whether to encode the scandot input."""

    num_scandots: int = 121
    """The number of scandots."""

    num_global_features: int = 16
    """The number of global features output by the pointnet encoder."""

    height_scan_encode: bool = True
    """Whether to encode the height scan input."""

    num_height_scan_points: int = 121
    """The number of height scan points."""

    height_scan_encode_output_dim: int = 16
    """The output dimensions of the height scan encoder."""

    height_scan_encode_layer_dims: list[int] = [256, 128]
    """The hidden dimensions of the height scan encoder."""

    height_scan_encode_activations: list[str] = ["relu", "relu", "relu"]
    """The activation functions for the height scan encoder."""

    """Dropouts values!"""
    dropout_LSTM: float = 0.0
    """The dropout value for the LSTM layer (only if more than one LSTM layer)."""
    dropout_MLP: float = 0.0
    """The dropout value for the final MLP layer."""
    dropout_depth_encoder: float = 0.0
    """The dropout value for the depth encoder."""
    dropout_height_scan_encoder: float = 0.0
    """The dropout value for the height scan encoder."""
    dropout_critic: bool = False
    """Whether to use dropout in the critic network."""


@configclass
class RslRlRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda"
    """The device for the rl-agent. Default is cuda."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlDppoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlDppoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "orbit"
    """The neptune project name. Default is "orbit"."""

    wandb_project: str = "orbit"
    """The wandb project name. Default is "orbit"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """

@configclass
class RslRlPPORunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: int = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda"
    """The device for the rl-agent. Default is cuda."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string.

    The name of the run directory is typically the time-stamp at execution. If the run name is not empty,
    then it is appended to the run directory's name, i.e. the logging directory's name will become
    ``{time-stamp}_{run_name}``.
    """

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "orbit"
    """The neptune project name. Default is "orbit"."""

    wandb_project: str = "orbit"
    """The wandb project name. Default is "orbit"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all).

    If regex expression, the latest (alphabetical order) matching run will be loaded.
    """

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is ``"model_.*.pt"`` (all).

    If regex expression, the latest (alphabetical order) matching file will be loaded.
    """