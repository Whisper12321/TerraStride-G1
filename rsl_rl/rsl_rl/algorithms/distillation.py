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
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.storage import RolloutStorage


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

    policy: StudentTeacher | StudentTeacherRecurrent
    """The student teacher model."""

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        loss_type="mse",
        latent_loss_coef=0.0,
        rollout_strategy="dagger",
        teacher_action_prob_start=1.0,
        teacher_action_prob_end=0.0,
        teacher_action_prob_warmup_iters=500,
        teacher_action_prob_decay_iters=5000,
        student_rollout_deterministic=True,
        device="cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None  # TODO: remove when runner has a proper base class

        # distillation components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None
        self.rollout_initial_hidden_states = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.latent_loss_coef = float(latent_loss_coef)
        self.rollout_strategy = str(rollout_strategy).lower()
        if self.rollout_strategy not in ("student", "teacher", "dagger"):
            raise ValueError(
                f"Unknown rollout_strategy: {rollout_strategy}. Supported values are: student, teacher, dagger"
            )
        self.teacher_action_prob_start = float(teacher_action_prob_start)
        self.teacher_action_prob_end = float(teacher_action_prob_end)
        self.teacher_action_prob_warmup_iters = max(int(teacher_action_prob_warmup_iters), 0)
        self.teacher_action_prob_decay_iters = max(int(teacher_action_prob_decay_iters), 1)
        self.student_rollout_deterministic = bool(student_rollout_deterministic)
        self._rollout_teacher_action_sum = 0.0
        self._rollout_action_count = 0
        self._last_rollout_teacher_fraction = 0.0
        self._last_rollout_teacher_probability = 0.0

        # initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0

    @staticmethod
    def _clone_hidden_states(hidden_states):
        if hidden_states is None:
            return None
        if isinstance(hidden_states, tuple):
            return tuple(Distillation._clone_hidden_states(hidden_state) for hidden_state in hidden_states)
        return hidden_states.detach().clone()

    def _teacher_action_probability(self) -> float:
        if self.rollout_strategy == "teacher":
            return 1.0
        if self.rollout_strategy == "student":
            return 0.0

        if self.num_updates < self.teacher_action_prob_warmup_iters:
            return max(0.0, min(self.teacher_action_prob_start, 1.0))

        progress = float(self.num_updates - self.teacher_action_prob_warmup_iters) / float(
            self.teacher_action_prob_decay_iters
        )
        progress = max(0.0, min(progress, 1.0))
        teacher_prob = self.teacher_action_prob_start + progress * (
            self.teacher_action_prob_end - self.teacher_action_prob_start
        )
        return max(0.0, min(teacher_prob, 1.0))

    def _student_rollout_action(self, obs):
        if self.student_rollout_deterministic and hasattr(self.policy, "act_inference"):
            return self.policy.act_inference(obs).detach()
        return self.policy.act(obs).detach()

    def _select_rollout_actions(self, student_actions, teacher_actions):
        teacher_prob = self._teacher_action_probability()
        self._last_rollout_teacher_probability = teacher_prob
        num_envs = student_actions.shape[0]
        if teacher_prob >= 1.0:
            self._rollout_teacher_action_sum += float(num_envs)
            self._rollout_action_count += int(num_envs)
            return teacher_actions.detach()
        if teacher_prob <= 0.0:
            self._rollout_action_count += int(num_envs)
            return student_actions.detach()

        teacher_mask = torch.rand(num_envs, 1, device=student_actions.device) < teacher_prob
        self._rollout_teacher_action_sum += float(teacher_mask.float().sum().item())
        self._rollout_action_count += int(teacher_mask.numel())
        return torch.where(teacher_mask, teacher_actions, student_actions).detach()

    def init_storage(
        self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape
    ):
        teacher_latent_dim = int(getattr(self.policy, "teacher_latent_dim", 0))
        privileged_latent_shape = [teacher_latent_dim] if self.latent_loss_coef > 0.0 and teacher_latent_dim > 0 else None
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            None,
            self.device,
            privileged_latent_shape=privileged_latent_shape,
        )

    def act(self, obs, teacher_obs):
        if self.storage is not None and self.storage.step == 0:
            self.rollout_initial_hidden_states = self._clone_hidden_states(self.policy.get_hidden_states())

        # Compute the expert label first. The expert action is always stored as the supervised target.
        if self.latent_loss_coef > 0.0 and hasattr(self.policy, "evaluate_with_latent"):
            privileged_actions, privileged_latents = self.policy.evaluate_with_latent(teacher_obs)
            self.transition.privileged_actions = privileged_actions.detach()
            self.transition.privileged_latents = privileged_latents.detach()
        else:
            privileged_actions = self.policy.evaluate(teacher_obs)
            self.transition.privileged_actions = privileged_actions.detach()

        # Mature distillation uses an explicit rollout policy. DAgger starts from expert-controlled
        # trajectories and anneals toward the student-controlled state distribution.
        student_actions = self._student_rollout_action(obs)
        self.transition.actions = self._select_rollout_actions(student_actions, self.transition.privileged_actions)

        # record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_latent_loss = 0
        loss = 0
        cnt = 0
        rollout_initial_hidden_states = (
            self.rollout_initial_hidden_states
            if self.rollout_initial_hidden_states is not None
            else self.last_hidden_states
        )

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self._clone_hidden_states(rollout_initial_hidden_states))
            self.policy.detach_hidden_states()
            for obs, _, _, privileged_actions, privileged_latents, dones in self.storage.generator():

                # inference the student for gradient computation
                if privileged_latents is not None and hasattr(self.policy, "act_inference_with_latent"):
                    actions, student_latents = self.policy.act_inference_with_latent(obs)
                else:
                    actions = self.policy.act_inference(obs)
                    student_latents = None

                # behavior cloning loss
                behavior_loss = self.loss_fn(actions, privileged_actions)
                if privileged_latents is not None and student_latents is not None:
                    latent_loss = self.loss_fn(student_latents, privileged_latents)
                else:
                    latent_loss = behavior_loss.new_zeros(())

                # total loss
                loss = loss + behavior_loss + self.latent_loss_coef * latent_loss
                mean_behavior_loss += behavior_loss.item()
                mean_latent_loss += latent_loss.item()
                cnt += 1

                # gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        if cnt > 0 and not isinstance(loss, int):
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            self.optimizer.step()
            self.policy.detach_hidden_states()

        mean_behavior_loss /= cnt
        mean_latent_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self._clone_hidden_states(self.policy.get_hidden_states())
        self.rollout_initial_hidden_states = None
        self.policy.detach_hidden_states()

        # construct the loss dictionary
        loss_dict = {"behavior": mean_behavior_loss}
        if self.latent_loss_coef > 0.0:
            loss_dict["latent"] = mean_latent_loss
        if self._rollout_action_count > 0:
            self._last_rollout_teacher_fraction = self._rollout_teacher_action_sum / float(self._rollout_action_count)
        loss_dict["rollout_teacher_fraction"] = self._last_rollout_teacher_fraction
        loss_dict["rollout_teacher_probability"] = self._last_rollout_teacher_probability
        self._rollout_teacher_action_sum = 0.0
        self._rollout_action_count = 0

        return loss_dict

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
