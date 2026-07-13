# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import Memory
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class _ScaledL2Normalize(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = math.sqrt(float(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=2.0, dim=-1) * self.scale


class ActorCriticDepthCnnGruTeacher(nn.Module):
    """Teacher actor-critic network with an explicit distillation latent.

    Observation layout is assumed to be:
        [low_dim_proprio_or_task_history, flattened_external_perception_image]

    The external perception image can be a depth image or a height-scan map. It is encoded with
    a CNN, then passed through a GRU. The resulting temporal perception feature is concatenated
    with an MLP feature from the low-dimensional observation path.
    The teacher actor predicts a compact latent from the fused feature and produces actions from
    that latent. A future student can imitate both the latent and the action.
    """

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        depth_shape=(1, 36, 64),
        actor_cnn_channels=(16, 32, 64),
        critic_cnn_channels=(16, 32, 64),
        actor_depth_feature_dim=256,
        critic_depth_feature_dim=256,
        actor_gru_hidden_dim=256,
        critic_gru_hidden_dim=256,
        actor_gru_num_layers=1,
        critic_gru_num_layers=1,
        actor_proprio_hidden_dims=(256, 256),
        critic_proprio_hidden_dims=(256, 256),
        teacher_latent_dim=64,
        teacher_latent_unit_norm: bool = True,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticDepthCnnGruTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.num_actions = num_actions
        self.depth_shape = tuple(depth_shape)
        self.depth_dim = math.prod(self.depth_shape)
        self.num_actor_proprio_obs = num_actor_obs - self.depth_dim
        self.num_critic_proprio_obs = num_critic_obs - self.depth_dim
        if self.num_actor_proprio_obs <= 0:
            raise ValueError(
                f"Actor observation dim {num_actor_obs} is not larger than depth dim {self.depth_dim}."
            )
        if self.num_critic_proprio_obs <= 0:
            raise ValueError(
                f"Critic observation dim {num_critic_obs} is not larger than depth dim {self.depth_dim}."
            )

        self.actor_depth_encoder = self._build_depth_encoder(
            self.depth_shape, actor_cnn_channels, actor_depth_feature_dim, activation
        )
        self.critic_depth_encoder = self._build_depth_encoder(
            self.depth_shape, critic_cnn_channels, critic_depth_feature_dim, activation
        )
        self.actor_depth_memory = Memory(
            actor_depth_feature_dim,
            type="gru",
            num_layers=actor_gru_num_layers,
            hidden_size=actor_gru_hidden_dim,
        )
        self.critic_depth_memory = Memory(
            critic_depth_feature_dim,
            type="gru",
            num_layers=critic_gru_num_layers,
            hidden_size=critic_gru_hidden_dim,
        )

        self.actor_proprio_encoder = self._build_mlp(
            self.num_actor_proprio_obs, actor_proprio_hidden_dims, None, activation
        )
        self.critic_proprio_encoder = self._build_mlp(
            self.num_critic_proprio_obs, critic_proprio_hidden_dims, None, activation
        )

        actor_proprio_feature_dim = self._mlp_output_dim(self.num_actor_proprio_obs, actor_proprio_hidden_dims)
        critic_proprio_feature_dim = self._mlp_output_dim(self.num_critic_proprio_obs, critic_proprio_hidden_dims)

        actor_fusion_dim = actor_proprio_feature_dim + actor_gru_hidden_dim
        critic_fusion_dim = critic_proprio_feature_dim + critic_gru_hidden_dim
        self.teacher_latent_dim = int(teacher_latent_dim)
        if self.teacher_latent_dim <= 0:
            self.teacher_latent = nn.Identity()
            actor_input_dim = actor_fusion_dim
        else:
            latent_layers = [
                nn.Linear(actor_fusion_dim, self.teacher_latent_dim),
                nn.LayerNorm(self.teacher_latent_dim),
                nn.Tanh(),
            ]
            if teacher_latent_unit_norm:
                latent_layers.append(_ScaledL2Normalize(self.teacher_latent_dim))
            self.teacher_latent = nn.Sequential(*latent_layers)
            actor_input_dim = self.teacher_latent_dim
        self.last_teacher_latent = None

        self.actor = self._build_mlp(
            actor_input_dim,
            actor_hidden_dims,
            num_actions,
            activation,
        )
        self.critic = self._build_mlp(
            critic_fusion_dim,
            critic_hidden_dims,
            1,
            activation,
        )

        print(f"Actor proprio encoder: {self.actor_proprio_encoder}")
        print(f"Actor depth encoder: {self.actor_depth_encoder}")
        print(f"Actor depth GRU: {self.actor_depth_memory}")
        print(f"Teacher latent head: {self.teacher_latent}")
        print(f"Actor fusion MLP: {self.actor}")
        print(f"Critic proprio encoder: {self.critic_proprio_encoder}")
        print(f"Critic depth encoder: {self.critic_depth_encoder}")
        print(f"Critic depth GRU: {self.critic_depth_memory}")
        print(f"Critic fusion MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _mlp_output_dim(input_dim, hidden_dims):
        return input_dim if len(hidden_dims) == 0 else hidden_dims[-1]

    @staticmethod
    def _build_mlp(input_dim, hidden_dims, output_dim, activation_name):
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(resolve_nn_activation(activation_name))
            last_dim = hidden_dim
        if output_dim is not None:
            layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    @classmethod
    def _build_depth_encoder(cls, depth_shape, channels, feature_dim, activation_name):
        in_channels, height, width = depth_shape
        conv_layers = []
        last_channels = in_channels
        for index, out_channels in enumerate(channels):
            kernel_size = 5 if index == 0 else 3
            conv_layers.append(
                nn.Conv2d(last_channels, out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
            )
            conv_layers.append(resolve_nn_activation(activation_name))
            last_channels = out_channels
        conv_body = nn.Sequential(*conv_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, height, width)
            flattened_dim = conv_body(dummy).reshape(1, -1).shape[-1]

        return nn.Sequential(
            conv_body,
            nn.Flatten(),
            nn.Linear(flattened_dim, feature_dim),
            resolve_nn_activation(activation_name),
        )

    def reset(self, dones=None):
        self.actor_depth_memory.reset(dones)
        self.critic_depth_memory.reset(dones)

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _split_observations(self, observations):
        proprio_obs = observations[..., : -self.depth_dim]
        depth_obs = observations[..., -self.depth_dim :]
        return proprio_obs, depth_obs

    def _encode_depth(self, depth_obs, encoder):
        leading_shape = depth_obs.shape[:-1]
        depth_images = depth_obs.reshape(-1, *self.depth_shape)
        depth_features = encoder(depth_images)
        return depth_features.reshape(*leading_shape, -1)

    def _encode_actor(self, observations, masks=None, hidden_states=None):
        proprio_obs, depth_obs = self._split_observations(observations)
        proprio_features = self.actor_proprio_encoder(proprio_obs)
        depth_features = self._encode_depth(depth_obs, self.actor_depth_encoder)
        depth_temporal_features = self.actor_depth_memory(depth_features, masks, hidden_states)
        if masks is not None:
            proprio_features = unpad_trajectories(proprio_features, masks)
        else:
            depth_temporal_features = depth_temporal_features.squeeze(0)
        return torch.cat([proprio_features, depth_temporal_features], dim=-1)

    def _encode_critic(self, observations, masks=None, hidden_states=None):
        proprio_obs, depth_obs = self._split_observations(observations)
        proprio_features = self.critic_proprio_encoder(proprio_obs)
        depth_features = self._encode_depth(depth_obs, self.critic_depth_encoder)
        depth_temporal_features = self.critic_depth_memory(depth_features, masks, hidden_states)
        if masks is not None:
            proprio_features = unpad_trajectories(proprio_features, masks)
        else:
            depth_temporal_features = depth_temporal_features.squeeze(0)
        return torch.cat([proprio_features, depth_temporal_features], dim=-1)

    def update_distribution(self, observations, masks=None, hidden_states=None):
        actor_features = self._encode_actor(observations, masks, hidden_states)
        teacher_latent = self.teacher_latent(actor_features)
        self.last_teacher_latent = teacher_latent
        mean = self.actor(teacher_latent)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, observations, masks=None, hidden_states=None):
        self.update_distribution(observations, masks, hidden_states)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actor_features = self._encode_actor(observations)
        teacher_latent = self.teacher_latent(actor_features)
        self.last_teacher_latent = teacher_latent
        return self.actor(teacher_latent)

    def get_teacher_latent(self, observations):
        actor_features = self._encode_actor(observations)
        return self.teacher_latent(actor_features)

    def act_inference_with_latent(self, observations):
        teacher_latent = self.get_teacher_latent(observations)
        return self.actor(teacher_latent), teacher_latent

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        critic_features = self._encode_critic(critic_observations, masks, hidden_states)
        return self.critic(critic_features)

    def get_hidden_states(self):
        return self.actor_depth_memory.hidden_states, self.critic_depth_memory.hidden_states

    def load_state_dict(self, state_dict, strict=True):
        if "std" in state_dict and hasattr(self, "log_std") and "log_std" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["log_std"] = torch.log(torch.clamp(state_dict.pop("std"), min=1.0e-6))
        elif "log_std" in state_dict and hasattr(self, "std") and "std" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["std"] = torch.exp(state_dict.pop("log_std"))
        super().load_state_dict(state_dict, strict=strict)
        return True
