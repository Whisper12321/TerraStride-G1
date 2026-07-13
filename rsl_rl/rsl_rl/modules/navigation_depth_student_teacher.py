from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic_depth_cnn_gru_teacher import ActorCriticDepthCnnGruTeacher
from rsl_rl.modules.actor_critic_navigation_depth_student import ActorCriticNavigationDepthStudent


class NavigationDepthStudentTeacher(nn.Module):
    """Student-teacher wrapper for depth navigation distillation.

    The student consumes deployable goal/proprio/depth observations. The teacher consumes
    privileged teacher observations and is frozen after loading an RL teacher checkpoint.
    """

    is_recurrent = True

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        activation="elu",
        init_noise_std=0.5,
        noise_std_type: str = "log",
        depth_shape=(1, 18, 32),
        teacher_depth_shape=None,
        student_actor_hidden_dims=(256, 128),
        student_critic_hidden_dims=(256, 128),
        student_actor_cnn_channels=(16, 32, 64),
        student_critic_cnn_channels=(16, 32, 64),
        student_actor_depth_feature_dim=128,
        student_critic_depth_feature_dim=128,
        student_actor_gru_hidden_dim=128,
        student_critic_gru_hidden_dim=128,
        student_actor_gru_num_layers=1,
        student_critic_gru_num_layers=1,
        student_actor_proprio_hidden_dims=(128, 128),
        student_critic_proprio_hidden_dims=(128, 128),
        student_latent_dim=64,
        student_latent_unit_norm=True,
        teacher_actor_hidden_dims=(512, 256, 128),
        teacher_critic_hidden_dims=(512, 256, 128),
        teacher_actor_cnn_channels=(16, 32, 64),
        teacher_critic_cnn_channels=(16, 32, 64),
        teacher_actor_depth_feature_dim=256,
        teacher_critic_depth_feature_dim=256,
        teacher_actor_gru_hidden_dim=256,
        teacher_critic_gru_hidden_dim=256,
        teacher_actor_gru_num_layers=1,
        teacher_critic_gru_num_layers=1,
        teacher_actor_proprio_hidden_dims=(256, 256),
        teacher_critic_proprio_hidden_dims=(256, 256),
        teacher_latent_dim=64,
        teacher_latent_unit_norm=True,
        teacher_init_noise_std=0.7,
        teacher_noise_std_type: str = "log",
        **kwargs,
    ):
        if kwargs:
            print(
                "NavigationDepthStudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.loaded_teacher = False
        self.student_latent_dim = int(student_latent_dim)
        self.teacher_latent_dim = int(teacher_latent_dim)
        if teacher_depth_shape is None:
            teacher_depth_shape = depth_shape

        self.student = ActorCriticNavigationDepthStudent(
            num_student_obs,
            num_student_obs,
            num_actions,
            actor_hidden_dims=student_actor_hidden_dims,
            critic_hidden_dims=student_critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            depth_shape=depth_shape,
            actor_cnn_channels=student_actor_cnn_channels,
            critic_cnn_channels=student_critic_cnn_channels,
            actor_depth_feature_dim=student_actor_depth_feature_dim,
            critic_depth_feature_dim=student_critic_depth_feature_dim,
            actor_gru_hidden_dim=student_actor_gru_hidden_dim,
            critic_gru_hidden_dim=student_critic_gru_hidden_dim,
            actor_gru_num_layers=student_actor_gru_num_layers,
            critic_gru_num_layers=student_critic_gru_num_layers,
            actor_proprio_hidden_dims=student_actor_proprio_hidden_dims,
            critic_proprio_hidden_dims=student_critic_proprio_hidden_dims,
            student_latent_dim=student_latent_dim,
            student_latent_unit_norm=student_latent_unit_norm,
        )
        self.teacher = ActorCriticDepthCnnGruTeacher(
            num_teacher_obs,
            num_teacher_obs,
            num_actions,
            actor_hidden_dims=teacher_actor_hidden_dims,
            critic_hidden_dims=teacher_critic_hidden_dims,
            activation=activation,
            init_noise_std=teacher_init_noise_std,
            noise_std_type=teacher_noise_std_type,
            depth_shape=teacher_depth_shape,
            actor_cnn_channels=teacher_actor_cnn_channels,
            critic_cnn_channels=teacher_critic_cnn_channels,
            actor_depth_feature_dim=teacher_actor_depth_feature_dim,
            critic_depth_feature_dim=teacher_critic_depth_feature_dim,
            actor_gru_hidden_dim=teacher_actor_gru_hidden_dim,
            critic_gru_hidden_dim=teacher_critic_gru_hidden_dim,
            actor_gru_num_layers=teacher_actor_gru_num_layers,
            critic_gru_num_layers=teacher_critic_gru_num_layers,
            actor_proprio_hidden_dims=teacher_actor_proprio_hidden_dims,
            critic_proprio_hidden_dims=teacher_critic_proprio_hidden_dims,
            teacher_latent_dim=teacher_latent_dim,
            teacher_latent_unit_norm=teacher_latent_unit_norm,
        )
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    @property
    def action_mean(self):
        return self.student.action_mean

    @property
    def action_std(self):
        return self.student.action_std

    @property
    def entropy(self):
        return self.student.entropy

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is None:
            hidden_states = (None, None)
        self.student.reset(dones, hidden_states[0])
        self.teacher.reset(dones)

    def detach_hidden_states(self, dones=None):
        self.student.detach_hidden_states(dones)

    def forward(self):
        raise NotImplementedError

    def act(self, observations):
        return self.student.act(observations)

    def act_inference(self, observations):
        return self.student.act_inference(observations)

    def act_inference_with_latent(self, observations):
        return self.student.act_inference_with_latent(observations)

    def evaluate(self, teacher_observations):
        with torch.no_grad():
            return self.teacher.act_inference(teacher_observations)

    def evaluate_with_latent(self, teacher_observations):
        with torch.no_grad():
            return self.teacher.act_inference_with_latent(teacher_observations)

    def get_hidden_states(self):
        return self.student.get_hidden_states(), None

    def load_state_dict(self, state_dict, strict=True):
        """Load either a full distillation checkpoint or an RL teacher checkpoint."""
        if any(key.startswith("student.") or key.startswith("teacher.") for key in state_dict.keys()):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            return True

        # RL teacher checkpoints contain actor and critic weights. Distillation only queries the
        # frozen teacher actor, while this wrapper is constructed from the teacher actor observation
        # stream only. The original PPO critic can therefore have a larger privileged input shape.
        teacher_actor_state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("critic_") and not key.startswith("critic.")
        }
        self.teacher.load_state_dict(teacher_actor_state_dict, strict=False)
        self.loaded_teacher = True
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)
        return False
