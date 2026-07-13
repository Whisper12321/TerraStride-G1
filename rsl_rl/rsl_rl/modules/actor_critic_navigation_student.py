from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCriticNavigationStudent(nn.Module):
    """Compass-only student actor-critic with a distillation-ready latent.

    The actor observes only the deployable student observation and predicts a compact latent before
    producing actions. The latent is intentionally the same default size as the teacher latent so a
    later distillation loss can align them without changing the policy interface.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_encoder_hidden_dims=(256, 256),
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
        init_noise_std=0.5,
        noise_std_type: str = "log",
        student_latent_dim: int = 64,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticNavigationStudent.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.num_actions = num_actions
        self.student_latent_dim = int(student_latent_dim)
        if self.student_latent_dim <= 0:
            raise ValueError("student_latent_dim must be positive.")

        self.actor_encoder = self._build_mlp(num_actor_obs, actor_encoder_hidden_dims, None, activation)
        encoder_out_dim = self._mlp_output_dim(num_actor_obs, actor_encoder_hidden_dims)
        self.student_latent = nn.Sequential(
            nn.Linear(encoder_out_dim, self.student_latent_dim),
            nn.LayerNorm(self.student_latent_dim),
            nn.Tanh(),
        )
        self.actor = self._build_mlp(self.student_latent_dim, actor_hidden_dims, num_actions, activation)
        self.critic = self._build_mlp(num_critic_obs, critic_hidden_dims, 1, activation)
        self.last_student_latent = None

        print(f"Student actor encoder: {self.actor_encoder}")
        print(f"Student latent head: {self.student_latent}")
        print(f"Student actor MLP: {self.actor}")
        print(f"Student critic MLP: {self.critic}")

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
    def _mlp_output_dim(input_dim: int, hidden_dims) -> int:
        return input_dim if len(hidden_dims) == 0 else int(hidden_dims[-1])

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims, output_dim: int | None, activation_name: str) -> nn.Sequential:
        layers = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(resolve_nn_activation(activation_name))
            last_dim = int(hidden_dim)
        if output_dim is not None:
            layers.append(nn.Linear(last_dim, int(output_dim)))
        return nn.Sequential(*layers)

    def reset(self, dones=None):
        pass

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

    def get_student_latent(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.actor_encoder(observations)
        latent = self.student_latent(features)
        self.last_student_latent = latent
        return latent

    def update_distribution(self, observations: torch.Tensor) -> None:
        latent = self.get_student_latent(observations)
        mean = self.actor(latent)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        latent = self.get_student_latent(observations)
        return self.actor(latent)

    def act_inference_with_latent(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.get_student_latent(observations)
        return self.actor(latent), latent

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    def get_hidden_states(self):
        return None

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
