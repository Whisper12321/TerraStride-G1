# High-level navigation environment that reuses the source G1LocomotionEnv.
#
# Key separation:
#   - Low-level locomotion policy uses the original G1LocomotionEnv observation path,
#     including locomotion noise, locomotion history buffers, gait state, and height scan concatenation.
#   - High-level navigation policy has independent observation history buffers and independent noise settings.

from __future__ import annotations

import glob
import math
import os

import isaaclab.sim as sim_utils
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.managers import RewardManager
from isaaclab.utils.buffers import CircularBuffer

from legged_lab.envs.g1.g1_locomotion_env import G1LocomotionEnv
from legged_lab.envs.g1.g1_navigation_cfg import G1NavigationEnvCfg
from legged_lab.terrains import maze_terrains
from legged_lab.utils.path_planning import path_reference_points, sparsify_reference_points


class G1NavigationEnv(G1LocomotionEnv):
    """Navigation environment with a frozen low-level locomotion policy.

    The environment inherits source locomotion initialization and utilities.
    Navigation training samples one high-level action [vx, vy, wz]. Inside each high-level step,
    the frozen low-level policy is evaluated several times at the locomotion frequency.
    """

    def __init__(self, cfg: G1NavigationEnvCfg, headless):
        # During parent __init__, G1LocomotionEnv calls self.reset().
        # We guard navigation reset until navigation tensors are allocated.
        self._navigation_initialized = False
        super().__init__(cfg, headless)

        # Replace/confirm reward manager uses navigation reward cfg.
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self._init_navigation_buffers()
        self.low_level_policy = self._load_low_level_policy(self.cfg.nav.locomotion_policy_path)
        self._navigation_initialized = True

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset(env_ids)

    # --------------------------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------------------------
    def _init_navigation_buffers(self) -> None:
        self.low_level_steps_per_nav_step = int(self.cfg.nav.navigation_decimation)
        if self.low_level_steps_per_nav_step <= 0:
            raise ValueError("navigation_decimation must be positive.")

        # Source locomotion step_dt is one low-level control period.
        self.low_level_step_dt = self.step_dt
        self.nav_step_dt = self.low_level_step_dt * self.low_level_steps_per_nav_step

        self.num_actions = 3  # high-level nav action: [vx, vy, wz]

        self.nav_cmd = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.prev_nav_cmd = torch.zeros_like(self.nav_cmd)
        self.prev_prev_nav_cmd = torch.zeros_like(self.nav_cmd)
        self.interp_cmd = torch.zeros_like(self.nav_cmd)

        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.final_goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.full_route_goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.start_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_marker_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.start_marker_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.current_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.prev_final_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.current_final_goal_dist = torch.zeros(self.num_envs, device=self.device)
        self.nav_attempt_dist = torch.zeros(self.num_envs, device=self.device)
        self.goal_reached_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.nav_success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.waypoint_reached_step_buf = torch.zeros(self.num_envs, device=self.device)
        self.final_goal_reached_step_buf = torch.zeros(self.num_envs, device=self.device)
        self.path_difficulty_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.path_guidance_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.waypoint_sparsity_levels = self.path_guidance_levels
        self.path_failure_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.path_guidance_failure_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.waypoint_failure_streak = self.path_guidance_failure_streak
        self.terrain_failure_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.path_guidance_success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.path_success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.terrain_success_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.curriculum_promotion_cooldown = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.nav_episode_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.max_navigation_waypoints = int(self.cfg.nav.max_navigation_waypoints)
        self.nav_waypoints_w = torch.zeros(
            self.num_envs, self.max_navigation_waypoints, 3, device=self.device
        )
        self.nav_waypoint_counts = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.nav_waypoint_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.nav_path_metric_score = torch.zeros(self.num_envs, device=self.device)
        self.nav_path_metric_length = torch.zeros(self.num_envs, device=self.device)
        self.nav_path_metric_detour = torch.zeros(self.num_envs, device=self.device)
        self.nav_path_metric_turns = torch.zeros(self.num_envs, device=self.device)
        self.nav_route_full_length = torch.zeros(self.num_envs, device=self.device)
        self.nav_route_prefix_fraction = torch.ones(self.num_envs, device=self.device)
        self.prev_nav_path_progress = torch.zeros(self.num_envs, device=self.device)
        self.nav_path_progress = torch.zeros(self.num_envs, device=self.device)
        self.nav_path_total_length = torch.ones(self.num_envs, device=self.device)
        self.nav_path_remaining = torch.zeros(self.num_envs, device=self.device)
        self.nav_path_lateral_error = torch.zeros(self.num_envs, device=self.device)
        self.nav_path_lookahead_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.nav_path_guidance_weight = torch.ones(self.num_envs, device=self.device)
        self.nav_route_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.nav_route_refresh_required = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._active_navigation_routes: list[dict | None] = [None for _ in range(self.num_envs)]

        self.nav_actor_obs_buffer: CircularBuffer | None = None
        self.nav_critic_obs_buffer: CircularBuffer | None = None
        self.nav_noise_scale_vec: torch.Tensor | None = None
        self._skip_parent_navigation_curriculum_update = False
        self._pending_navigation_curriculum_log: dict[str, torch.Tensor] = {}
        self._nav_actor_obs_dim = -1
        self._nav_critic_obs_dim = -1
        self._navigation_point_markers: VisualizationMarkers | None = None
        self._learning_iteration = 0
        self._learning_total_iterations = 0

    def _load_low_level_policy(self, path: str):
        path = self._resolve_low_level_policy_path(path)
        try:
            policy = torch.jit.load(path, map_location=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load locomotion policy from {path}. "
                "Default loader expects TorchScript. If this is an RSL-RL checkpoint, "
                "replace _load_low_level_policy() with ActorCritic + load_state_dict."
            ) from exc
        policy.eval()
        return policy

    def _resolve_low_level_policy_path(self, path: str) -> str:
        if path and path != "auto":
            return os.path.abspath(path)

        candidates = glob.glob(os.path.abspath("logs/g1_locomotion/*/exported/policy.pt"))
        candidates = [candidate for candidate in candidates if os.path.isfile(candidate)]
        if not candidates:
            raise FileNotFoundError(
                "No exported locomotion policy was found. Set cfg.nav.locomotion_policy_path to a TorchScript "
                "policy.pt path, or export a locomotion policy under logs/g1_locomotion/*/exported/policy.pt."
            )
        return max(candidates, key=os.path.getmtime)

    # --------------------------------------------------------------------------------------------
    # High-level navigation observations
    # --------------------------------------------------------------------------------------------
    def compute_current_navigation_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute current high-level navigation obs before history stacking.

        Actor observation uses navigation-specific noise.
        Critic observation adds privileged path-reference state for value estimation only.
        """
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        root_lin_vel_b = self.robot.data.root_lin_vel_b
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        base_vel = torch.cat([root_lin_vel_b[:, :2], root_ang_vel_b[:, 2:3]], dim=-1)
        projected_gravity = self.robot.data.projected_gravity_b

        final_goal_vec_w = self.final_goal_pos_w[:, :2] - root_pos_w[:, :2]
        final_goal_dist = torch.norm(final_goal_vec_w, dim=-1, keepdim=True)
        waypoint_vec_w = self.goal_pos_w[:, :2] - root_pos_w[:, :2]
        waypoint_dist = torch.norm(waypoint_vec_w, dim=-1, keepdim=True)
        self.current_goal_dist = waypoint_dist.squeeze(-1)
        self.current_final_goal_dist = final_goal_dist.squeeze(-1)
        self._update_path_reference_metrics()

        yaw = self._quat_to_yaw(root_quat_w)
        c = torch.cos(-yaw)
        s = torch.sin(-yaw)
        goal_x_b = c * final_goal_vec_w[:, 0] - s * final_goal_vec_w[:, 1]
        goal_y_b = s * final_goal_vec_w[:, 0] + c * final_goal_vec_w[:, 1]
        goal_vec_b = torch.stack([goal_x_b, goal_y_b], dim=-1)

        desired_heading = torch.atan2(final_goal_vec_w[:, 1], final_goal_vec_w[:, 0])
        heading_error = self._wrap_to_pi(desired_heading - yaw).unsqueeze(-1)

        current_actor_obs = torch.cat(
            [goal_vec_b * self.cfg.nav_obs_scales.goal_xy,  # 2
             final_goal_dist * self.cfg.nav_obs_scales.goal_distance, # 1
             heading_error * self.cfg.nav_obs_scales.heading_error, # 1
             self.prev_nav_cmd * self.cfg.nav_obs_scales.previous_command, # 3
             base_vel * self.cfg.nav_obs_scales.base_velocity, # 3
             projected_gravity * self.cfg.nav_obs_scales.projected_gravity, # 3
            ],
            dim=-1,
        ) 

        current_critic_obs = torch.cat(
            [current_actor_obs, self._compute_privileged_path_observations(root_pos_w, yaw)],
            dim=-1,
        )

        return current_actor_obs, current_critic_obs

    def _compute_privileged_path_observations(self, root_pos_w: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Privileged critic-only features from the training-time global path reference."""
        lookahead_vec_w = self.nav_path_lookahead_pos_w[:, :2] - root_pos_w[:, :2]
        c = torch.cos(-yaw)
        s = torch.sin(-yaw)
        lookahead_x_b = c * lookahead_vec_w[:, 0] - s * lookahead_vec_w[:, 1]
        lookahead_y_b = s * lookahead_vec_w[:, 0] + c * lookahead_vec_w[:, 1]
        lookahead_vec_b = torch.stack([lookahead_x_b, lookahead_y_b], dim=-1)
        lookahead_dist = torch.norm(lookahead_vec_w, dim=-1, keepdim=True)
        lookahead_heading = self._wrap_to_pi(torch.atan2(lookahead_vec_w[:, 1], lookahead_vec_w[:, 0]) - yaw).unsqueeze(-1)

        total_length = torch.clamp(self.nav_path_total_length, min=1.0e-6)
        progress_fraction = (self.nav_path_progress / total_length).unsqueeze(-1)
        guidance_max_level = max(int(self.cfg.nav.num_path_guidance_levels) - 1, 1)
        guidance_level = (self.path_guidance_levels.float() / float(guidance_max_level)).unsqueeze(-1)

        return torch.cat(
            [
                lookahead_vec_b * self.cfg.nav_obs_scales.path_xy,  # 2
                lookahead_dist * self.cfg.nav_obs_scales.path_distance,  # 1
                lookahead_heading,  # 1
                self.nav_path_remaining.unsqueeze(-1) * self.cfg.nav_obs_scales.path_distance,  # 1
                self.nav_path_lateral_error.unsqueeze(-1) * self.cfg.nav_obs_scales.path_lateral_error,  # 1
                progress_fraction,  # 1
                self.nav_path_guidance_weight.unsqueeze(-1),  # 1
                guidance_level,  # 1
                self.nav_path_metric_score.unsqueeze(-1),  # 1
                self.nav_path_metric_length.unsqueeze(-1) * self.cfg.nav_obs_scales.path_metric_length,  # 1
                self.nav_path_metric_detour.unsqueeze(-1),  # 1
                self.nav_path_metric_turns.unsqueeze(-1) * self.cfg.nav_obs_scales.path_metric_turns,  # 1
            ],
            dim=-1,
        )

    def compute_observations(self):
        """Return high-level navigation observations for PPO training.

        This overrides G1LocomotionEnv.compute_observations() for the trainer only.
        Low-level frozen policy uses G1LocomotionEnv.compute_observations(self) explicitly.
        """
        current_actor_obs, current_critic_obs = self.compute_current_navigation_observations()
        self._ensure_nav_obs_buffers(current_actor_obs.shape[-1], current_critic_obs.shape[-1])

        if self.cfg.nav_noise.add_noise:
            current_actor_obs = current_actor_obs + (
                2 * torch.rand_like(current_actor_obs) - 1
            ) * self._get_nav_noise_scale_vec(current_actor_obs)

        assert self.nav_actor_obs_buffer is not None
        assert self.nav_critic_obs_buffer is not None
        self.nav_actor_obs_buffer.append(current_actor_obs)
        self.nav_critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.nav_actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.nav_critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        # Navigation-side sensor observations are appended after the high-level history stack.

        if self.cfg.nav_obs.use_depth_camera and self.cfg.scene.noisy_camera.enable_camera:
            depth_obs = self._get_depth_obs_for_navigation()
            actor_obs = torch.cat([actor_obs, depth_obs], dim=-1)
            critic_obs = torch.cat([critic_obs, depth_obs], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)
        return actor_obs, critic_obs

    def _ensure_nav_obs_buffers(self, actor_dim: int, critic_dim: int) -> None:
        if (
            self.nav_actor_obs_buffer is None
            or self.nav_critic_obs_buffer is None
            or self._nav_actor_obs_dim != actor_dim
            or self._nav_critic_obs_dim != critic_dim
        ):
            self.nav_actor_obs_buffer = CircularBuffer(
                max_len=self.cfg.nav_obs.actor_obs_history_length,
                batch_size=self.num_envs,
                device=self.device,
            )
            self.nav_critic_obs_buffer = CircularBuffer(
                max_len=self.cfg.nav_obs.critic_obs_history_length,
                batch_size=self.num_envs,
                device=self.device,
            )
            self._nav_actor_obs_dim = actor_dim
            self._nav_critic_obs_dim = critic_dim
            self.nav_noise_scale_vec = None

    def _get_nav_noise_scale_vec(self, current_actor_obs: torch.Tensor) -> torch.Tensor:
        if self.nav_noise_scale_vec is None or self.nav_noise_scale_vec.shape != current_actor_obs[0].shape:
            noise_vec = torch.zeros_like(current_actor_obs[0])
            noise_vec[:2] = self.cfg.nav_noise.goal_xy * self.cfg.nav_obs_scales.goal_xy
            noise_vec[2:3] = self.cfg.nav_noise.goal_distance * self.cfg.nav_obs_scales.goal_distance
            noise_vec[3:4] = self.cfg.nav_noise.heading_error * self.cfg.nav_obs_scales.heading_error
            noise_vec[4:7] = self.cfg.nav_noise.previous_command * self.cfg.nav_obs_scales.previous_command
            noise_vec[7:10] = self.cfg.nav_noise.base_velocity * self.cfg.nav_obs_scales.base_velocity
            noise_vec[10:13] = self.cfg.nav_noise.projected_gravity * self.cfg.nav_obs_scales.projected_gravity
            self.nav_noise_scale_vec = noise_vec
        return self.nav_noise_scale_vec

    def _reset_nav_obs_buffers(self, env_ids: torch.Tensor) -> None:
        if self.nav_actor_obs_buffer is not None:
            self.nav_actor_obs_buffer.reset(env_ids)
        if self.nav_critic_obs_buffer is not None:
            self.nav_critic_obs_buffer.reset(env_ids)

    def _compute_locomotion_observations(self) -> torch.Tensor:
        """Use the original locomotion observation path for frozen low-level policy."""
        actor_obs, _ = G1LocomotionEnv.compute_observations(self)
        return actor_obs


    def _get_depth_obs_for_navigation(self) -> torch.Tensor:
        sensor = self.noisy_camera if hasattr(self, "noisy_camera") else self.depth_camera
        data_type = self.cfg.nav_obs.depth_data_type
        output = sensor.data.output
        if data_type not in output:
            fallback = (
                "distance_to_image_plane_noised"
                if "distance_to_image_plane_noised" in output
                else "distance_to_image_plane"
            )
            data_type = fallback
        depth = output[data_type]
        depth = torch.nan_to_num(
            depth,
            nan=self.cfg.nav_obs.depth_clip_max,
            posinf=self.cfg.nav_obs.depth_clip_max,
            neginf=self.cfg.nav_obs.depth_clip_min,
        )
        depth = torch.clamp(depth, self.cfg.nav_obs.depth_clip_min, self.cfg.nav_obs.depth_clip_max)
        if self.cfg.nav_noise.add_noise and self.cfg.nav_noise.depth > 0.0:
            depth += (2.0 * torch.rand_like(depth) - 1.0) * self.cfg.nav_noise.depth
            depth = torch.clamp(depth, self.cfg.nav_obs.depth_clip_min, self.cfg.nav_obs.depth_clip_max)
        return depth.reshape(self.num_envs, -1)

    # --------------------------------------------------------------------------------------------
    # Step
    # --------------------------------------------------------------------------------------------
    def step(self, actions: torch.Tensor):
        """One high-level navigation step.

        A navigation action is converted to a target velocity command. Then the frozen locomotion policy
        is evaluated low_level_steps_per_nav_step times at the original locomotion frequency.
        """
        self.extras.pop("log", None)

        self.prev_prev_nav_cmd.copy_(self.prev_nav_cmd)
        self.prev_nav_cmd.copy_(self.nav_cmd)
        raw_nav_cmd = self._scale_and_clip_nav_action(actions)
        alpha = max(0.0, min(float(self.cfg.nav.command_smoothing_alpha), 1.0))
        self.nav_cmd = alpha * raw_nav_cmd + (1.0 - alpha) * self.prev_nav_cmd
        self.waypoint_reached_step_buf.zero_()
        self.final_goal_reached_step_buf.zero_()

        done_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for substep in range(self.low_level_steps_per_nav_step):
            active_ids = (~done_pending).nonzero(as_tuple=False).flatten()
            if len(active_ids) == 0:
                break
            active_mask = None
            if len(active_ids) != self.num_envs:
                active_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                active_mask[active_ids] = True

            ratio = float(substep + 1) / float(self.low_level_steps_per_nav_step)
            if self.cfg.nav.command_interpolation == "linear":
                next_interp_cmd = (1.0 - ratio) * self.prev_nav_cmd + ratio * self.nav_cmd
            else:
                next_interp_cmd = self.nav_cmd

            # Feed interpolated high-level command into the low-level locomotion observation path.
            self.interp_cmd[active_ids] = next_interp_cmd[active_ids]
            self.command_generator.command[active_ids, :3] = self.interp_cmd[active_ids]

            low_level_obs = self._compute_locomotion_observations()
            with torch.no_grad():
                low_level_action = self.low_level_policy(low_level_obs)
            self._apply_low_level_action_one_step(low_level_action, active_mask=active_mask)

            self.episode_length_buf[active_ids] += 1
            self._update_gait_cycle_from_action(active_ids)
            self._calculate_gait_para(active_ids)
            self.update_foothold_planner(active_ids)
            self.finalize_foothold_planner_step(active_ids)

            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.low_level_step_dt)

            reset_now, self.time_out_buf = self.check_reset()
            if active_mask is not None:
                reset_now &= active_mask
                time_out_now = self.time_out_buf & active_mask
            else:
                time_out_now = self.time_out_buf
            goal_reached_now = self._check_goal_reached(active_ids)
            if self.cfg.nav.terminate_on_goal:
                reset_now |= goal_reached_now

            newly_done = reset_now & ~done_pending
            done_pending |= reset_now
            time_out_pending |= time_out_now
            if newly_done.any():
                newly_done_ids = newly_done.nonzero(as_tuple=False).flatten()
                self.interp_cmd[newly_done_ids] = 0.0
                self.command_generator.command[newly_done_ids, :3] = 0.0

        self.reset_buf = done_pending
        self.time_out_buf = time_out_pending
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        dones = self.reset_buf.clone()
        time_outs = time_out_pending.clone()
        self._update_path_reference_metrics()

        # Reward is computed by RewardManager only, at the high-level step interval.
        reward_buf = self.reward_manager.compute(self.nav_step_dt)

        active_ids = (~self.reset_buf).nonzero(as_tuple=False).flatten()
        if len(active_ids) > 0:
            self.prev_goal_dist[active_ids] = self.current_goal_dist[active_ids]
            self.prev_final_goal_dist[active_ids] = self.current_final_goal_dist[active_ids]
            self.prev_nav_path_progress[active_ids] = self.nav_path_progress[active_ids]

        if len(self.reset_env_ids) > 0:
            self.reset(self.reset_env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        self.extras["time_outs"] = time_outs

        return actor_obs, reward_buf, dones, self.extras

    def _apply_low_level_action_one_step(
        self,
        low_level_action: torch.Tensor,
        active_mask: torch.Tensor | None = None,
    ) -> None:
        """Apply one locomotion control step using the same mechanics as the source locomotion env."""
        if active_mask is not None:
            low_level_action = low_level_action.clone()
            low_level_action[~active_mask] = self.action[~active_mask]

        self.prev_action.copy_(self.action)
        delayed_actions = self.action_buffer.compute(low_level_action)
        next_action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        if active_mask is not None:
            self.action = torch.where(active_mask.unsqueeze(-1), next_action, self.action)
        else:
            self.action = next_action

        default_joint_pos = self.robot.data.default_joint_pos
        joint_actions = self.action[:, : self.num_joint_actions]
        processed_actions = joint_actions * self.action_scale + default_joint_pos

        self.avg_feet_force_per_step.zero_()
        self.avg_feet_speed_per_step.zero_()

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(
                self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1
            )

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation
        if active_mask is not None:
            active_scale = active_mask.to(dtype=self.avg_feet_force_per_step.dtype).unsqueeze(-1)
            self.avg_feet_force_per_step *= active_scale
            self.avg_feet_speed_per_step *= active_scale
        if not self.headless:
            self.sim.render()

    # --------------------------------------------------------------------------------------------
    # Reset and curricula
    # --------------------------------------------------------------------------------------------
    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        if not getattr(self, "_navigation_initialized", False):
            # During parent construction, only reset locomotion state.
            return G1LocomotionEnv.reset(self, env_ids)

        terminal_navigation_log = self._navigation_terminal_log_dict(env_ids)
        curriculum_log = self._update_navigation_curricula(
            env_ids,
            update_terrain=self._terrain_curriculum_enabled(),
        )
        self._pending_navigation_curriculum_log = curriculum_log
        self._skip_parent_navigation_curriculum_update = True

        # Source reset handles robot, scene, events, locomotion obs/action buffers, gait buffers, reward reset, etc.
        try:
            G1LocomotionEnv.reset(self, env_ids)
        finally:
            self._skip_parent_navigation_curriculum_update = False

        self._reset_navigation_task(env_ids)
        self._reset_nav_obs_buffers(env_ids)
        self.nav_cmd[env_ids] = 0.0
        self.prev_nav_cmd[env_ids] = 0.0
        self.prev_prev_nav_cmd[env_ids] = 0.0
        self.interp_cmd[env_ids] = 0.0
        if hasattr(self, "command_generator"):
            self.command_generator.command[env_ids, :3] = 0.0

        self.scene.write_data_to_sim()
        self.sim.forward()

        self.extras.setdefault("log", {})
        self.extras["log"].update(self._pending_navigation_curriculum_log)
        self.extras["log"].update(terminal_navigation_log)
        self.extras["log"].update(self._navigation_curriculum_stage_log_dict(env_ids))
        self._pending_navigation_curriculum_log = {}

    def _reset_navigation_task(self, env_ids: torch.Tensor) -> None:
        start_pos, final_goal_pos, waypoints, waypoint_counts = self._sample_navigation_task(env_ids)
        self.start_pos_w[env_ids] = start_pos
        self.final_goal_pos_w[env_ids] = final_goal_pos
        self.nav_waypoints_w[env_ids] = waypoints
        self.nav_waypoint_counts[env_ids] = waypoint_counts
        self.nav_waypoint_indices[env_ids] = 0
        self.goal_pos_w[env_ids] = waypoints[:, 0]
        self.start_marker_pos_w[env_ids] = start_pos
        self.start_marker_pos_w[env_ids, 2] -= self.robot.data.default_root_state[env_ids, 2]
        self.goal_marker_pos_w[env_ids] = final_goal_pos
        self.nav_success_buf[env_ids] = False
        self.goal_reached_buf[env_ids] = False
        self.nav_attempt_dist[env_ids] = torch.norm(final_goal_pos[:, :2] - start_pos[:, :2], dim=-1)
        self.current_goal_dist[env_ids] = torch.norm(self.goal_pos_w[env_ids, :2] - start_pos[:, :2], dim=-1)
        self.current_final_goal_dist[env_ids] = self.nav_attempt_dist[env_ids]
        self.prev_goal_dist[env_ids] = self.current_goal_dist[env_ids]
        self.prev_final_goal_dist[env_ids] = self.current_final_goal_dist[env_ids]
        self._update_path_reference_metrics(env_ids)
        self.prev_nav_path_progress[env_ids] = self.nav_path_progress[env_ids]
        self.waypoint_reached_step_buf[env_ids] = 0.0
        self.final_goal_reached_step_buf[env_ids] = 0.0
        self.nav_episode_valid[env_ids] = True

        if self.cfg.nav.randomize_start:
            root_state = self.robot.data.root_state_w[env_ids].clone()
            root_state[:, :3] = start_pos
            yaw_to_goal = torch.atan2(self.goal_pos_w[env_ids, 1] - start_pos[:, 1], self.goal_pos_w[env_ids, 0] - start_pos[:, 0])
            root_state[:, 3:7] = self._yaw_to_quat(yaw_to_goal)
            self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
            self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

    def _sample_navigation_task(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.nav_path_metric_score[env_ids] = 0.0
        self.nav_path_metric_length[env_ids] = 0.0
        self.nav_path_metric_detour[env_ids] = 0.0
        self.nav_path_metric_turns[env_ids] = 0.0
        self.nav_route_full_length[env_ids] = 0.0
        self.nav_route_prefix_fraction[env_ids] = 1.0
        path_sample = self._sample_start_goal_and_waypoints_from_maze_paths(env_ids)
        if path_sample is not None:
            return path_sample

        start_pos, goal_pos = self._sample_start_and_goal(env_ids)
        waypoints = torch.zeros(len(env_ids), self.max_navigation_waypoints, 3, device=self.device)
        waypoints[:, 0] = goal_pos
        waypoint_counts = torch.ones(len(env_ids), dtype=torch.long, device=self.device)
        self.full_route_goal_pos_w[env_ids] = goal_pos
        self.nav_route_full_length[env_ids] = torch.norm(goal_pos[:, :2] - start_pos[:, :2], dim=-1)
        return start_pos, goal_pos, waypoints, waypoint_counts

    def _sample_start_goal_and_waypoints_from_maze_paths(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        start_pos = torch.zeros(len(env_ids), 3, device=self.device)
        final_goal_pos = torch.zeros_like(start_pos)
        waypoints = torch.zeros(len(env_ids), self.max_navigation_waypoints, 3, device=self.device)
        waypoint_counts = torch.ones(len(env_ids), dtype=torch.long, device=self.device)
        default_root_z = self.robot.data.default_root_state[env_ids, 2]
        fallback_rows: list[int] = []

        for out_i, env_id_tensor in enumerate(env_ids):
            env_id = int(env_id_tensor.item())
            metadata = self._get_maze_metadata_for_env(env_id)
            if metadata is None:
                return None

            graph = metadata.get("graph")
            if graph is None:
                fallback_rows.append(out_i)
                continue

            route = self._get_or_sample_navigation_route(env_id, metadata)
            if route is None:
                fallback_rows.append(out_i)
                continue

            candidate = route["candidate"]
            start_local_xy = route["start_local_xy"]
            goal_local_xy = route["goal_local_xy"]
            env_origin = self.scene.env_origins[env_id]
            start_world_xy = self._local_xy_to_world_xy(metadata, start_local_xy, env_origin)
            full_goal_world_xy = self._local_xy_to_world_xy(metadata, goal_local_xy, env_origin)
            ground_z = env_origin[2]

            reference_points = path_reference_points(
                graph,
                candidate.cell_path,
                start_xy=start_local_xy,
                goal_xy=goal_local_xy,
                include_cell_centers=True,
            )
            if bool(getattr(self.cfg.nav, "fixed_route_prefix_curriculum", True)):
                dense_targets = reference_points[1:] if len(reference_points) > 1 else [goal_local_xy]
                if not dense_targets:
                    dense_targets = [goal_local_xy]
                dense_count = len(dense_targets)
                prefix_count = self._route_prefix_count_for_level(
                    dense_count,
                    int(self.path_difficulty_levels[env_id].item()),
                )
                prefix_targets = dense_targets[:prefix_count]
                if not prefix_targets:
                    prefix_targets = [goal_local_xy]
                prefix_points = [start_local_xy] + prefix_targets
                stride = self._path_reference_stride_for_env(env_id)
                sparse_points = sparsify_reference_points(
                    prefix_points,
                    stride=stride,
                    max_points=self.max_navigation_waypoints,
                )
                if not sparse_points:
                    sparse_points = [prefix_targets[-1]]
                task_goal_local_xy = prefix_targets[-1]
                prefix_fraction = float(prefix_count) / float(max(dense_count, 1))
            else:
                stride = self._path_reference_stride_for_env(env_id)
                sparse_points = sparsify_reference_points(
                    reference_points,
                    stride=stride,
                    max_points=self.max_navigation_waypoints,
                )
                if not sparse_points:
                    sparse_points = [goal_local_xy]
                task_goal_local_xy = goal_local_xy
                prefix_points = [start_local_xy] + sparse_points
                prefix_fraction = 1.0

            task_goal_world_xy = self._local_xy_to_world_xy(metadata, task_goal_local_xy, env_origin)

            start_pos[out_i, 0] = start_world_xy[0]
            start_pos[out_i, 1] = start_world_xy[1]
            start_pos[out_i, 2] = ground_z + default_root_z[out_i]
            final_goal_pos[out_i, 0] = task_goal_world_xy[0]
            final_goal_pos[out_i, 1] = task_goal_world_xy[1]
            final_goal_pos[out_i, 2] = ground_z
            self.full_route_goal_pos_w[env_id, 0] = full_goal_world_xy[0]
            self.full_route_goal_pos_w[env_id, 1] = full_goal_world_xy[1]
            self.full_route_goal_pos_w[env_id, 2] = ground_z

            count = min(len(sparse_points), self.max_navigation_waypoints)
            waypoint_counts[out_i] = count
            for waypoint_i in range(count):
                waypoint_world_xy = self._local_xy_to_world_xy(metadata, sparse_points[waypoint_i], env_origin)
                waypoints[out_i, waypoint_i, 0] = waypoint_world_xy[0]
                waypoints[out_i, waypoint_i, 1] = waypoint_world_xy[1]
                waypoints[out_i, waypoint_i, 2] = ground_z
            waypoints[out_i, count - 1] = final_goal_pos[out_i]

            self._set_path_metrics_from_local_polyline(
                env_id=env_id,
                points=prefix_points,
                full_route_score=float(candidate.metrics.score),
                prefix_fraction=prefix_fraction,
            )
            self.nav_route_full_length[env_id] = self._local_polyline_length(reference_points)
            self.nav_route_prefix_fraction[env_id] = prefix_fraction

        if fallback_rows:
            fallback_rows_tensor = torch.tensor(fallback_rows, dtype=torch.long, device=self.device)
            fallback_env_ids = env_ids[fallback_rows_tensor]
            fallback_start, fallback_goal = self._sample_start_and_goal(fallback_env_ids)
            start_pos[fallback_rows_tensor] = fallback_start
            final_goal_pos[fallback_rows_tensor] = fallback_goal
            waypoints[fallback_rows_tensor, 0] = fallback_goal
            waypoint_counts[fallback_rows_tensor] = 1
            self.full_route_goal_pos_w[fallback_env_ids] = fallback_goal
            self.nav_route_full_length[fallback_env_ids] = torch.norm(
                fallback_goal[:, :2] - fallback_start[:, :2], dim=-1
            )
            self.nav_route_prefix_fraction[fallback_env_ids] = 1.0

        return start_pos, final_goal_pos, waypoints, waypoint_counts

    def _get_or_sample_navigation_route(self, env_id: int, metadata: dict) -> dict | None:
        route = self._active_navigation_routes[env_id]
        needs_new_route = bool(self.nav_route_refresh_required[env_id].item()) or not self._route_matches_metadata(
            route, metadata
        )
        if not needs_new_route:
            return route

        route = self._sample_navigation_route_from_metadata(env_id, metadata)
        if route is None:
            self._active_navigation_routes[env_id] = None
            self.nav_route_active[env_id] = False
            self.nav_route_refresh_required[env_id] = True
            return None

        self._active_navigation_routes[env_id] = route
        self.nav_route_active[env_id] = True
        self.nav_route_refresh_required[env_id] = False
        return route

    def _sample_navigation_route_from_metadata(self, env_id: int, metadata: dict) -> dict | None:
        graph = metadata.get("graph")
        if graph is None:
            return None

        if bool(getattr(self.cfg.nav, "fixed_route_prefix_curriculum", True)):
            candidates = self._fixed_route_candidates_for_metadata(metadata)
        else:
            level = int(self.path_difficulty_levels[env_id].item())
            candidates = self._path_candidates_for_level(metadata, level)
        if not candidates:
            return None

        max_tries = int(self.cfg.nav.path_start_goal_sample_tries)
        if bool(getattr(self.cfg.nav, "fixed_route_prefix_curriculum", True)):
            primary_count = self._fixed_route_primary_candidate_count(len(candidates))
            first_index = int(env_id) % primary_count
            ordered_candidates = (
                candidates[first_index:primary_count]
                + candidates[:first_index]
                + candidates[primary_count:]
            )
        else:
            ordered_candidates = []
            for _ in range(max_tries):
                pick = int(torch.randint(len(candidates), (1,), device=self.device).item())
                ordered_candidates.append(candidates[pick])

        for candidate in ordered_candidates:
            for _ in range(max_tries):
                start_local_xy = self._sample_local_xy_in_cell(metadata, candidate.start_cell)
                goal_local_xy = self._sample_local_xy_in_cell(metadata, candidate.goal_cell)
                if start_local_xy is None or goal_local_xy is None:
                    continue
                return {
                    "metadata_call_index": metadata.get("call_index"),
                    "graph": graph,
                    "candidate": candidate,
                    "start_local_xy": start_local_xy,
                    "goal_local_xy": goal_local_xy,
                }
        return None

    def _fixed_route_candidates_for_metadata(self, metadata: dict):
        candidates_by_level = metadata.get("path_candidates_by_level", {})
        if not candidates_by_level:
            return []
        candidates_by_level = {
            int(level): candidates
            for level, candidates in candidates_by_level.items()
            if candidates
        }
        if not candidates_by_level:
            return []

        requested_level = int(getattr(self.cfg.nav, "fixed_route_candidate_level", -1))
        if requested_level < 0:
            requested_level = max(int(level) for level in candidates_by_level.keys())
        elif requested_level not in candidates_by_level:
            available_levels = sorted(int(level) for level in candidates_by_level.keys())
            requested_level = min(available_levels, key=lambda level: abs(level - requested_level))

        candidates = candidates_by_level.get(requested_level, [])
        if not candidates:
            return []
        return sorted(candidates, key=lambda candidate: candidate.metrics.score, reverse=True)

    def _fixed_route_primary_candidate_count(self, total_count: int) -> int:
        total_count = max(int(total_count), 1)
        candidate_count = int(getattr(self.cfg.nav, "fixed_route_candidate_count", 0))
        if candidate_count <= 0:
            return total_count
        return max(1, min(candidate_count, total_count))

    def _route_prefix_count_for_level(self, full_count: int, level: int) -> int:
        full_count = max(int(full_count), 1)
        num_levels = max(int(self.cfg.nav.num_path_difficulty_levels), 1)
        if num_levels <= 1 or full_count <= 1:
            return full_count

        level = max(0, min(int(level), num_levels - 1))
        prefix_count = 1 + int(math.ceil(float(full_count - 1) * float(level) / float(num_levels - 1)))
        return max(1, min(prefix_count, full_count))

    @staticmethod
    def _local_polyline_length(points: list[tuple[float, float]]) -> float:
        if len(points) < 2:
            return 0.0
        length = 0.0
        for start, end in zip(points[:-1], points[1:]):
            dx = float(end[0] - start[0])
            dy = float(end[1] - start[1])
            length += math.sqrt(dx * dx + dy * dy)
        return float(length)

    def _set_path_metrics_from_local_polyline(
        self,
        *,
        env_id: int,
        points: list[tuple[float, float]],
        full_route_score: float,
        prefix_fraction: float,
    ) -> None:
        if len(points) < 2:
            self.nav_path_metric_score[env_id] = 0.0
            self.nav_path_metric_length[env_id] = 0.0
            self.nav_path_metric_detour[env_id] = 1.0
            self.nav_path_metric_turns[env_id] = 0.0
            return

        length = 0.0
        turns = 0
        prev_dir: tuple[float, float] | None = None
        for start, end in zip(points[:-1], points[1:]):
            dx = float(end[0] - start[0])
            dy = float(end[1] - start[1])
            segment_length = math.sqrt(dx * dx + dy * dy)
            length += segment_length
            if segment_length <= 1.0e-6:
                continue
            direction = (dx / segment_length, dy / segment_length)
            if prev_dir is not None:
                dot = prev_dir[0] * direction[0] + prev_dir[1] * direction[1]
                if dot < 0.985:
                    turns += 1
            prev_dir = direction

        dx = float(points[-1][0] - points[0][0])
        dy = float(points[-1][1] - points[0][1])
        euclidean = max(math.sqrt(dx * dx + dy * dy), 1.0e-6)
        prefix_fraction = max(0.0, min(float(prefix_fraction), 1.0))
        self.nav_path_metric_score[env_id] = float(full_route_score) * prefix_fraction
        self.nav_path_metric_length[env_id] = float(length)
        self.nav_path_metric_detour[env_id] = float(length / euclidean)
        self.nav_path_metric_turns[env_id] = float(turns)

    @staticmethod
    def _route_matches_metadata(route: dict | None, metadata: dict) -> bool:
        if route is None:
            return False
        route_call_index = route.get("metadata_call_index")
        metadata_call_index = metadata.get("call_index")
        if route_call_index is not None and metadata_call_index is not None:
            return route_call_index == metadata_call_index
        return route.get("graph") is metadata.get("graph")

    def _invalidate_navigation_routes(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.nav_route_refresh_required[env_ids] = True
        self.nav_route_active[env_ids] = False
        for env_id_tensor in env_ids:
            self._active_navigation_routes[int(env_id_tensor.item())] = None

    def _path_reference_stride_for_env(self, env_id: int) -> int:
        levels = tuple(int(level) for level in self.cfg.nav.path_reference_stride_levels)
        if not levels:
            return 1
        guidance_level = int(self.path_guidance_levels[env_id].item())
        guidance_level = max(0, min(guidance_level, len(levels) - 1))
        return levels[guidance_level]

    def _waypoint_stride_for_env(self, env_id: int) -> int:
        """Backward-compatible name for the current path-reference stride."""
        return self._path_reference_stride_for_env(env_id)

    def _path_guidance_weights_for_levels(self, levels: torch.Tensor) -> torch.Tensor:
        scales = tuple(float(scale) for scale in self.cfg.nav.path_guidance_reward_scales)
        if not scales:
            return torch.ones_like(levels, dtype=torch.float32)
        scale_tensor = torch.tensor(scales, device=self.device, dtype=torch.float32)
        indices = torch.clamp(levels.to(dtype=torch.long), min=0, max=len(scales) - 1)
        return scale_tensor[indices]

    def _update_path_reference_metrics(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        counts = torch.clamp(self.nav_waypoint_counts[env_ids], min=1, max=self.max_navigation_waypoints)
        root_xy = self.robot.data.root_pos_w[env_ids, :2]
        start_xy = self.start_pos_w[env_ids, :2]
        reference_xy = torch.cat(
            [start_xy.unsqueeze(1), self.nav_waypoints_w[env_ids, :, :2]],
            dim=1,
        )
        seg_start = reference_xy[:, :-1]
        seg_end = reference_xy[:, 1:]
        seg_vec = seg_end - seg_start
        seg_len = torch.norm(seg_vec, dim=-1)
        seg_len_sq = torch.clamp(torch.square(seg_len), min=1.0e-8)

        segment_ids = torch.arange(self.max_navigation_waypoints, device=self.device).unsqueeze(0)
        valid = segment_ids < counts.unsqueeze(1)
        rel = root_xy.unsqueeze(1) - seg_start
        projection_t = torch.clamp(torch.sum(rel * seg_vec, dim=-1) / seg_len_sq, 0.0, 1.0)
        projection = seg_start + projection_t.unsqueeze(-1) * seg_vec
        dist_sq = torch.sum(torch.square(root_xy.unsqueeze(1) - projection), dim=-1)
        dist_sq = torch.where(valid, dist_sq, torch.full_like(dist_sq, float("inf")))

        nearest_index = torch.argmin(dist_sq, dim=1)
        gather_index = nearest_index.unsqueeze(-1)
        nearest_t = torch.gather(projection_t, 1, gather_index).squeeze(1)
        nearest_len = torch.gather(seg_len, 1, gather_index).squeeze(1)

        prefix = torch.cumsum(seg_len * valid.float(), dim=1) - seg_len * valid.float()
        nearest_prefix = torch.gather(prefix, 1, gather_index).squeeze(1)
        progress = nearest_prefix + nearest_t * nearest_len
        total_length = torch.clamp(torch.sum(seg_len * valid.float(), dim=1), min=1.0e-6)
        progress = torch.clamp(progress, min=0.0)
        progress = torch.minimum(progress, total_length)

        lookahead_distance = float(self.cfg.nav.path_lookahead_distance)
        lookahead_progress = torch.minimum(progress + lookahead_distance, total_length)
        segment_end_progress = prefix + seg_len * valid.float()
        lookahead_candidate = torch.where(
            valid & (segment_end_progress >= lookahead_progress.unsqueeze(1)),
            segment_end_progress,
            torch.full_like(segment_end_progress, float("inf")),
        )
        lookahead_index = torch.argmin(lookahead_candidate, dim=1)
        lookahead_gather = lookahead_index.unsqueeze(-1)
        lookahead_prefix = torch.gather(prefix, 1, lookahead_gather).squeeze(1)
        lookahead_len = torch.gather(seg_len, 1, lookahead_gather).squeeze(1)
        lookahead_t = torch.clamp(
            (lookahead_progress - lookahead_prefix) / torch.clamp(lookahead_len, min=1.0e-6),
            0.0,
            1.0,
        )
        lookahead_start = torch.gather(
            seg_start,
            1,
            lookahead_gather.unsqueeze(-1).expand(-1, -1, 2),
        ).squeeze(1)
        lookahead_vec = torch.gather(
            seg_vec,
            1,
            lookahead_gather.unsqueeze(-1).expand(-1, -1, 2),
        ).squeeze(1)
        lookahead_xy = lookahead_start + lookahead_t.unsqueeze(-1) * lookahead_vec

        self.nav_path_progress[env_ids] = progress
        self.nav_path_total_length[env_ids] = total_length
        self.nav_path_remaining[env_ids] = torch.clamp(total_length - progress, min=0.0)
        self.nav_path_lateral_error[env_ids] = torch.sqrt(torch.clamp(torch.min(dist_sq, dim=1).values, min=0.0))
        self.nav_path_lookahead_pos_w[env_ids, :2] = lookahead_xy
        self.nav_path_lookahead_pos_w[env_ids, 2] = self.final_goal_pos_w[env_ids, 2]
        self.nav_path_guidance_weight[env_ids] = self._path_guidance_weights_for_levels(
            self.path_guidance_levels[env_ids]
        )

    def _get_maze_metadata_for_env(self, env_id: int) -> dict | None:
        maze_cfg = self._get_maze_terrain_cfg()
        if maze_cfg is None:
            return None
        metadata_entries = getattr(maze_cfg, "generated_maze_metadata", None)
        if not metadata_entries:
            metadata_entries = maze_terrains.get_generated_maze_metadata()
        if not metadata_entries:
            return None

        terrain = getattr(self.scene, "terrain", None)
        if terrain is None or getattr(terrain, "terrain_levels", None) is None:
            return None

        num_cols = int(getattr(self.cfg.scene.terrain_generator, "num_cols", 1))
        expected = max(int(getattr(self.cfg.scene.terrain_generator, "num_rows", 1)) * num_cols, 1)
        entries = metadata_entries[-expected:] if len(metadata_entries) >= expected else metadata_entries

        terrain_level = int(terrain.terrain_levels[env_id].item())
        terrain_type = int(terrain.terrain_types[env_id].item()) if hasattr(terrain, "terrain_types") else 0
        candidate_index = terrain_level * num_cols + terrain_type
        if 0 <= candidate_index < len(entries):
            return entries[candidate_index]

        row_start = terrain_level * num_cols
        if 0 <= row_start < len(entries):
            return entries[row_start]
        return entries[min(max(terrain_level, 0), len(entries) - 1)]

    def _path_candidates_for_level(self, metadata: dict, level: int):
        candidates_by_level = metadata.get("path_candidates_by_level", {})
        if not candidates_by_level:
            return []
        max_level = max(int(self.cfg.nav.num_path_difficulty_levels) - 1, 0)
        level = max(0, min(int(level), max_level))
        if candidates_by_level.get(level):
            return candidates_by_level[level]
        for delta in range(1, max_level + 1):
            lower = level - delta
            upper = level + delta
            if lower >= 0 and candidates_by_level.get(lower):
                return candidates_by_level[lower]
            if upper <= max_level and candidates_by_level.get(upper):
                return candidates_by_level[upper]
        return []

    def _sample_local_xy_in_cell(self, metadata: dict, cell: tuple[int, int]) -> tuple[float, float] | None:
        graph = metadata.get("graph")
        if graph is None:
            return None
        center_x, center_y = graph.cell_center(cell)
        cell_x, cell_y = graph.cell_size
        base_clearance = max(float(self.cfg.nav.path_start_goal_clearance), 0.0)
        clearance_levels = (
            base_clearance,
            0.5 * base_clearance,
            0.25 * base_clearance,
            0.0,
        )
        blocking_rects = metadata.get("navigation_blocking_rects", metadata.get("occupied_rects", ()))
        for clearance in clearance_levels:
            max_dx = max(0.5 * cell_x - clearance, 0.0)
            max_dy = max(0.5 * cell_y - clearance, 0.0)
            for _ in range(int(self.cfg.nav.path_start_goal_sample_tries)):
                dx = (2.0 * torch.rand((), device=self.device).item() - 1.0) * max_dx
                dy = (2.0 * torch.rand((), device=self.device).item() - 1.0) * max_dy
                x = float(center_x + dx)
                y = float(center_y + dy)
                rect = (x - clearance, x + clearance, y - clearance, y + clearance)
                if self._rect_is_free(rect, blocking_rects):
                    return x, y
        return None

    def _local_xy_to_world_xy(
        self, metadata: dict, local_xy: tuple[float, float], env_origin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        origin_xy = metadata.get("origin_xy", (0.0, 0.0))
        world_x = env_origin[0] + float(local_xy[0] - origin_xy[0])
        world_y = env_origin[1] + float(local_xy[1] - origin_xy[1])
        return world_x, world_y

    @staticmethod
    def _rect_is_free(rect: tuple[float, float, float, float], occupied_rects) -> bool:
        for occupied in occupied_rects:
            if not (
                rect[1] <= occupied[0]
                or rect[0] >= occupied[1]
                or rect[3] <= occupied[2]
                or rect[2] >= occupied[3]
            ):
                return False
        return True

    def _sample_start_and_goal(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat_patches = self._get_navigation_flat_patches()
        if flat_patches is not None:
            return self._sample_start_and_goal_from_flat_patches(env_ids, flat_patches)
        if self._has_maze_reserved_clear_zones():
            return self._sample_start_and_goal_from_reserved_clear_zones(env_ids)
        return self._sample_start_and_goal_in_terrain_box(env_ids)

    def _sample_start_and_goal_from_flat_patches(
        self, env_ids: torch.Tensor, flat_patches: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        levels = self.scene.terrain.terrain_levels[env_ids]
        terrain_types = self.scene.terrain.terrain_types[env_ids]

        start_pos = torch.zeros(len(env_ids), 3, device=self.device)
        goal_pos = torch.zeros_like(start_pos)
        default_root_z = self.robot.data.default_root_state[env_ids, 2]

        for out_i in range(len(env_ids)):
            patches = flat_patches[levels[out_i], terrain_types[out_i]]
            valid_patch_mask = torch.isfinite(patches).all(dim=-1)
            patches = patches[valid_patch_mask]
            if patches.shape[0] < 2:
                if self._has_maze_reserved_clear_zones():
                    return self._sample_start_and_goal_from_reserved_clear_zones(env_ids)
                return self._sample_start_and_goal_in_terrain_box(env_ids)

            best_start_idx = torch.randint(patches.shape[0], (1,), device=self.device).item()
            best_goal_idx = (best_start_idx + 1) % patches.shape[0]

            for _ in range(int(self.cfg.nav.path_start_goal_sample_tries)):
                dists = torch.norm(patches[:, :2] - patches[best_start_idx, :2], dim=-1)
                candidate_mask = dists >= self.cfg.nav.min_start_goal_separation
                candidate_ids = torch.nonzero(candidate_mask, as_tuple=False).flatten()
                if len(candidate_ids) > 0:
                    random_pick = torch.randint(len(candidate_ids), (1,), device=self.device)
                    best_goal_idx = candidate_ids[random_pick].item()
                    break
                best_start_idx = torch.randint(patches.shape[0], (1,), device=self.device).item()
                best_goal_idx = (best_start_idx + 1) % patches.shape[0]

            start_pos[out_i, :2] = patches[best_start_idx, :2]
            goal_pos[out_i, :2] = patches[best_goal_idx, :2]
            start_pos[out_i, 2] = patches[best_start_idx, 2] + default_root_z[out_i]
            goal_pos[out_i, 2] = patches[best_goal_idx, 2]

        return start_pos, goal_pos

    def _sample_start_and_goal_from_reserved_clear_zones(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        origins = self.scene.env_origins[env_ids]
        maze_cfg = self._get_maze_terrain_cfg()
        start_clear_size = torch.tensor(maze_cfg.start_clear_size, device=self.device)
        goal_clear_size = torch.tensor(maze_cfg.goal_clear_size, device=self.device)

        start_center = origins[:, :2]
        goal_center = start_center + self._reserved_goal_offset_from_start(maze_cfg).unsqueeze(0)

        start_xy = self._sample_xy_in_clear_zone(start_center, start_clear_size, len(env_ids))
        goal_xy = self._sample_xy_in_clear_zone(goal_center, goal_clear_size, len(env_ids))

        start_z = origins[:, 2] + self.robot.data.default_root_state[env_ids, 2]
        goal_z = origins[:, 2]
        start_pos = torch.cat([start_xy, start_z.unsqueeze(-1)], dim=-1)
        goal_pos = torch.cat([goal_xy, goal_z.unsqueeze(-1)], dim=-1)
        return start_pos, goal_pos

    def _sample_start_and_goal_in_terrain_box(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        origins = self.scene.env_origins[env_ids]
        n = len(env_ids)
        device = self.device

        # Last-resort generic fallback. It does not guarantee obstacle-free points.
        terrain_size = (
            self.cfg.scene.terrain_generator.size if self.cfg.scene.terrain_generator is not None else (10.0, 10.0)
        )
        half_x = 0.5 * float(terrain_size[0]) - self.cfg.nav.start_goal_border_margin
        half_y = 0.5 * float(terrain_size[1]) - self.cfg.nav.start_goal_border_margin

        if self.cfg.nav.randomize_start:
            start_xy = torch.empty(n, 2, device=device)
            start_xy[:, 0] = origins[:, 0] + (2.0 * torch.rand(n, device=device) - 1.0) * half_x
            start_xy[:, 1] = origins[:, 1] + (2.0 * torch.rand(n, device=device) - 1.0) * half_y
        else:
            start_xy = self.robot.data.root_pos_w[env_ids, :2]

        goal_xy = torch.empty(n, 2, device=device)
        for i in range(n):
            goal_xy[i] = start_xy[i]
            for _ in range(int(self.cfg.nav.path_start_goal_sample_tries)):
                candidate = torch.empty(2, device=device)
                candidate[0] = origins[i, 0] + (2.0 * torch.rand((), device=device) - 1.0) * half_x
                candidate[1] = origins[i, 1] + (2.0 * torch.rand((), device=device) - 1.0) * half_y
                if torch.norm(candidate - start_xy[i]) >= self.cfg.nav.min_start_goal_separation:
                    goal_xy[i] = candidate
                    break

        start_z = self.robot.data.default_root_state[env_ids, 2]
        goal_z = torch.zeros(n, device=device)
        start_pos = torch.cat([start_xy, start_z.unsqueeze(-1)], dim=-1)
        goal_pos = torch.cat([goal_xy, goal_z.unsqueeze(-1)], dim=-1)
        return start_pos, goal_pos

    def _get_navigation_flat_patches(self) -> torch.Tensor | None:
        terrain = getattr(self.scene, "terrain", None)
        if terrain is None or terrain.terrain_origins is None:
            return None
        flat_patches = terrain.flat_patches
        if not flat_patches:
            return None
        return flat_patches.get(self.cfg.nav.flat_patch_sampling_name)

    def _has_maze_reserved_clear_zones(self) -> bool:
        return self._get_maze_terrain_cfg() is not None

    def _get_maze_terrain_cfg(self):
        terrain_generator = self.cfg.scene.terrain_generator
        if terrain_generator is None or not terrain_generator.sub_terrains:
            return None
        for sub_cfg in terrain_generator.sub_terrains.values():
            if hasattr(sub_cfg, "start_clear_size") and hasattr(sub_cfg, "goal_clear_size"):
                return sub_cfg
        return None

    def _reserved_goal_offset_from_start(self, maze_cfg) -> torch.Tensor:
        xmin, xmax, ymin, ymax = maze_cfg.maze_region
        if maze_cfg.start_pos is None:
            start_xy = torch.tensor(
                [xmin + 0.5 * maze_cfg.start_clear_size[0], ymin + 0.5 * maze_cfg.start_clear_size[1]],
                device=self.device,
            )
        else:
            start_xy = torch.tensor(maze_cfg.start_pos, device=self.device)
        if maze_cfg.goal_pos is None:
            goal_xy = torch.tensor(
                [xmax - 0.5 * maze_cfg.goal_clear_size[0], ymax - 0.5 * maze_cfg.goal_clear_size[1]],
                device=self.device,
            )
        else:
            goal_xy = torch.tensor(maze_cfg.goal_pos, device=self.device)
        return goal_xy - start_xy

    def _sample_xy_in_clear_zone(
        self, centers: torch.Tensor, clear_size: torch.Tensor, num_samples: int
    ) -> torch.Tensor:
        half_size = 0.5 * clear_size - float(self.cfg.nav.reserved_clear_zone_margin)
        half_size = torch.clamp(half_size, min=0.0)
        noise = (2.0 * torch.rand(num_samples, 2, device=self.device) - 1.0) * half_size.unsqueeze(0)
        return centers + noise

    def update_terrain_levels(self, env_ids):
        """Compatibility hook for the source reset path.

        Navigation reset updates the navigation curriculum independently before calling the source reset.
        If the source reset calls this hook because terrain curriculum is enabled, return the stored log only.
        """
        if not getattr(self, "_navigation_initialized", False):
            return {"Curriculum/terrain_levels": torch.tensor(0.0, device=self.device)}

        if self._skip_parent_navigation_curriculum_update:
            return self._pending_navigation_curriculum_log or self._navigation_curriculum_log_dict()

        return self._update_navigation_curricula(env_ids, update_terrain=True)

    def _update_navigation_curricula(self, env_ids: torch.Tensor, update_terrain: bool) -> dict[str, torch.Tensor]:
        valid = self.nav_episode_valid[env_ids]
        success = self.nav_success_buf[env_ids] & valid
        failure = valid & ~success
        can_update_terrain = (
            update_terrain
            and self.cfg.scene.terrain_generator is not None
            and getattr(self.scene, "terrain", None) is not None
            and self.scene.terrain.terrain_origins is not None
        )

        path_max_level = max(int(self.cfg.nav.num_path_difficulty_levels) - 1, 0)
        guidance_max_level = max(int(self.cfg.nav.num_path_guidance_levels) - 1, 0)
        terrain_promote_path_threshold = self._terrain_promote_path_level_threshold(path_max_level)
        path_level_before = self.path_difficulty_levels[env_ids].clone()
        guidance_level_before = self.path_guidance_levels[env_ids].clone()
        cooldown_before = self.curriculum_promotion_cooldown[env_ids].clone()
        can_promote = success & (cooldown_before <= 0)
        failed_at_min_guidance = failure & (guidance_level_before <= 0)

        valid_ids = env_ids[valid]
        if len(valid_ids) > 0:
            self.curriculum_promotion_cooldown[valid_ids] = torch.clamp(
                self.curriculum_promotion_cooldown[valid_ids] - 1,
                min=0,
            )

        success_ids = env_ids[success]
        failure_ids = env_ids[failure]
        min_guidance_failure_ids = env_ids[failed_at_min_guidance]
        if len(success_ids) > 0:
            self.path_failure_streak[success_ids] = 0
            self.path_guidance_failure_streak[success_ids] = 0
            self.terrain_failure_streak[success_ids] = 0
        if len(failure_ids) > 0:
            self.path_failure_streak[failure_ids] += 1
            self.path_guidance_failure_streak[failure_ids] += 1
            self.terrain_failure_streak[failure_ids] += 1
            self.path_guidance_success_streak[failure_ids] = 0
            self.path_success_streak[failure_ids] = 0
            self.terrain_success_streak[failure_ids] = 0

        guidance_success_candidate = (
            success
            & self.cfg.nav.path_guidance_level_up_on_success
            & (guidance_level_before < guidance_max_level)
        )
        path_success_candidate = (
            success
            & self.cfg.nav.path_level_up_on_guidance_mastery
            & (guidance_level_before >= guidance_max_level)
            & (path_level_before < path_max_level)
        )
        terrain_success_candidate = (
            success
            & can_update_terrain
            & self.cfg.nav.terrain_level_up_on_path_curriculum
            & (path_level_before >= terrain_promote_path_threshold)
            & (guidance_level_before >= int(self.cfg.nav.terrain_promote_guidance_level_threshold))
            & ~guidance_success_candidate
            & ~path_success_candidate
        )

        guidance_success_ids = env_ids[guidance_success_candidate]
        path_success_ids = env_ids[path_success_candidate]
        terrain_success_ids = env_ids[terrain_success_candidate]
        if len(guidance_success_ids) > 0:
            self.path_guidance_success_streak[guidance_success_ids] += 1
            self.path_success_streak[guidance_success_ids] = 0
            self.terrain_success_streak[guidance_success_ids] = 0
        if len(path_success_ids) > 0:
            self.path_success_streak[path_success_ids] += 1
            self.path_guidance_success_streak[path_success_ids] = 0
            self.terrain_success_streak[path_success_ids] = 0
        if len(terrain_success_ids) > 0:
            self.terrain_success_streak[terrain_success_ids] += 1
            self.path_guidance_success_streak[terrain_success_ids] = 0
            self.path_success_streak[terrain_success_ids] = 0

        staged_success = guidance_success_candidate | path_success_candidate | terrain_success_candidate
        non_staged_success_ids = env_ids[success & ~staged_success]
        if len(non_staged_success_ids) > 0:
            self.path_guidance_success_streak[non_staged_success_ids] = 0
            self.path_success_streak[non_staged_success_ids] = 0
            self.terrain_success_streak[non_staged_success_ids] = 0

        guidance_move_up = (
            guidance_success_candidate
            & can_promote
            & (self.path_guidance_success_streak[env_ids] >= int(self.cfg.nav.path_guidance_promote_successes))
        )
        guidance_up_ids = env_ids[guidance_move_up]
        if len(guidance_up_ids) > 0:
            self.path_guidance_levels[guidance_up_ids] += 1
            self.path_guidance_success_streak[guidance_up_ids] = 0
            self.curriculum_promotion_cooldown[guidance_up_ids] = int(
                self.cfg.nav.curriculum_promotion_cooldown_episodes
            )

        path_move_up = (
            path_success_candidate
            & can_promote
            & (self.path_success_streak[env_ids] >= int(self.cfg.nav.path_promote_successes))
        )
        path_up_ids = env_ids[path_move_up]
        if len(path_up_ids) > 0:
            self.path_difficulty_levels[path_up_ids] += 1
            self.path_guidance_levels[path_up_ids] = int(self.cfg.nav.post_path_promotion_guidance_level)
            self.path_success_streak[path_up_ids] = 0
            self.path_guidance_success_streak[path_up_ids] = 0
            self.curriculum_promotion_cooldown[path_up_ids] = int(
                self.cfg.nav.curriculum_promotion_cooldown_episodes
            )

        move_up = (
            terrain_success_candidate
            & can_promote
            & (self.terrain_success_streak[env_ids] >= int(self.cfg.nav.terrain_promote_successes))
        )

        guidance_move_down = (
            failure
            & self.cfg.nav.path_guidance_level_down_on_failure
            & (self.path_guidance_levels[env_ids] > 0)
            & (self.path_guidance_failure_streak[env_ids] >= int(self.cfg.nav.path_guidance_demote_failures))
        )
        guidance_down_ids = env_ids[guidance_move_down]
        if len(guidance_down_ids) > 0:
            self.path_guidance_levels[guidance_down_ids] -= 1
            self.path_guidance_failure_streak[guidance_down_ids] = 0
            self.path_failure_streak[guidance_down_ids] = 0
            self.path_guidance_success_streak[guidance_down_ids] = 0
            self.path_success_streak[guidance_down_ids] = 0
            self.terrain_success_streak[guidance_down_ids] = 0
            self.curriculum_promotion_cooldown[guidance_down_ids] = 0

        path_move_down = (
            failed_at_min_guidance
            & self.cfg.nav.path_level_down_on_failure
            & (self.path_difficulty_levels[env_ids] > 0)
            & (self.path_failure_streak[env_ids] >= int(self.cfg.nav.path_demote_failures))
        )
        path_down_ids = env_ids[path_move_down]
        if len(path_down_ids) > 0:
            self.path_difficulty_levels[path_down_ids] -= 1
            self.path_guidance_levels[path_down_ids] = 0
            self.path_failure_streak[path_down_ids] = 0
            self.path_guidance_failure_streak[path_down_ids] = 0
            self.terrain_failure_streak[path_down_ids] = 0
            self.path_guidance_success_streak[path_down_ids] = 0
            self.path_success_streak[path_down_ids] = 0
            self.terrain_success_streak[path_down_ids] = 0
            self.curriculum_promotion_cooldown[path_down_ids] = 0

        self.path_difficulty_levels.clamp_(0, path_max_level)
        self.path_guidance_levels.clamp_(0, guidance_max_level)

        move_down = (
            failed_at_min_guidance
            & can_update_terrain
            & self.cfg.nav.terrain_level_down_on_failure
            & (self.path_difficulty_levels[env_ids] <= int(self.cfg.nav.terrain_demote_path_level_threshold))
            & (self.terrain_failure_streak[env_ids] >= int(self.cfg.nav.terrain_demote_failures))
        )
        move_down &= ~(move_up | path_move_down)

        if can_update_terrain:
            self.scene.terrain.update_env_origins(env_ids, move_up, move_down)

        promoted_ids = env_ids[move_up]
        demoted_ids = env_ids[move_down]
        if len(promoted_ids) > 0:
            self.path_difficulty_levels[promoted_ids] = int(self.cfg.nav.post_terrain_promotion_path_level)
            self.path_guidance_levels[promoted_ids] = 0
            self.path_failure_streak[promoted_ids] = 0
            self.path_guidance_failure_streak[promoted_ids] = 0
            self.terrain_failure_streak[promoted_ids] = 0
            self.path_guidance_success_streak[promoted_ids] = 0
            self.path_success_streak[promoted_ids] = 0
            self.terrain_success_streak[promoted_ids] = 0
            self.curriculum_promotion_cooldown[promoted_ids] = int(
                self.cfg.nav.curriculum_promotion_cooldown_episodes
            )
        if len(demoted_ids) > 0:
            self.path_difficulty_levels[demoted_ids] = 0
            self.path_guidance_levels[demoted_ids] = 0
            self.path_failure_streak[demoted_ids] = 0
            self.path_guidance_failure_streak[demoted_ids] = 0
            self.terrain_failure_streak[demoted_ids] = 0
            self.path_guidance_success_streak[demoted_ids] = 0
            self.path_success_streak[demoted_ids] = 0
            self.terrain_success_streak[demoted_ids] = 0
            self.curriculum_promotion_cooldown[demoted_ids] = 0
        self.path_difficulty_levels.clamp_(0, path_max_level)
        self.path_guidance_levels.clamp_(0, guidance_max_level)

        if bool(getattr(self.cfg.nav, "fixed_route_prefix_curriculum", True)):
            # Keep the same route/start while guidance weakens and the task endpoint moves farther
            # along that route. Refresh only when the terrain patch changes.
            self._invalidate_navigation_routes(promoted_ids)
            self._invalidate_navigation_routes(demoted_ids)
        else:
            self._invalidate_navigation_routes(success_ids)
            self._invalidate_navigation_routes(min_guidance_failure_ids)
            self._invalidate_navigation_routes(promoted_ids)
            self._invalidate_navigation_routes(demoted_ids)
        self.nav_path_guidance_weight[env_ids] = self._path_guidance_weights_for_levels(
            self.path_guidance_levels[env_ids]
        )

        return self._navigation_curriculum_log_dict()

    def _terrain_curriculum_enabled(self) -> bool:
        return (
            self.cfg.scene.terrain_generator is not None
            and bool(getattr(self.cfg.scene.terrain_generator, "curriculum", False))
        )

    def set_learning_iteration(self, iteration: int, total_iterations: int | None = None) -> None:
        self._learning_iteration = max(int(iteration), 0)
        if total_iterations is not None:
            self._learning_total_iterations = max(int(total_iterations), 0)

    def _terrain_promote_path_level_threshold(self, path_max_level: int) -> int:
        path_max_level = max(int(path_max_level), 0)
        default_threshold = int(getattr(self.cfg.nav, "terrain_promote_path_level_threshold", path_max_level))
        start_threshold = int(
            getattr(self.cfg.nav, "terrain_promote_path_level_threshold_start", default_threshold)
        )
        final_threshold = int(
            getattr(self.cfg.nav, "terrain_promote_path_level_threshold_final", default_threshold)
        )
        if final_threshold < 0:
            final_threshold = path_max_level

        start_threshold = max(0, min(start_threshold, path_max_level))
        final_threshold = max(start_threshold, min(final_threshold, path_max_level))
        if final_threshold <= start_threshold:
            return start_threshold

        total_iterations = int(getattr(self, "_learning_total_iterations", 0))
        if total_iterations <= 0:
            return start_threshold

        progress = max(0.0, min(float(self._learning_iteration) / float(total_iterations), 1.0))
        tighten_start = float(getattr(self.cfg.nav, "terrain_promote_tighten_start_fraction", 0.5))
        tighten_end = float(getattr(self.cfg.nav, "terrain_promote_tighten_end_fraction", 0.8))
        tighten_start = max(0.0, min(tighten_start, 1.0))
        tighten_end = max(tighten_start + 1.0e-6, min(tighten_end, 1.0))

        if progress <= tighten_start:
            return start_threshold
        if progress >= tighten_end:
            return final_threshold

        ratio = (progress - tighten_start) / (tighten_end - tighten_start)
        threshold = start_threshold + int(round(float(final_threshold - start_threshold) * ratio))
        return max(start_threshold, min(threshold, final_threshold))

    def _navigation_curriculum_log_dict(self) -> dict[str, torch.Tensor]:
        if (
            getattr(self.scene, "terrain", None) is not None
            and self.scene.terrain.terrain_origins is not None
        ):
            terrain_level_mean = torch.mean(self.scene.terrain.terrain_levels.float())
        else:
            terrain_level_mean = torch.tensor(0.0, device=self.device)
        path_max_level = max(int(self.cfg.nav.num_path_difficulty_levels) - 1, 0)
        return {
            "Curriculum/terrain_levels": terrain_level_mean,
            "Curriculum/path_difficulty_levels": torch.mean(self.path_difficulty_levels.float()),
            "Curriculum/path_guidance_levels": torch.mean(self.path_guidance_levels.float()),
            "Curriculum/path_guidance_weight": torch.mean(self.nav_path_guidance_weight.float()),
            "Curriculum/terrain_promote_path_level_threshold": torch.tensor(
                float(self._terrain_promote_path_level_threshold(path_max_level)),
                device=self.device,
            ),
            "Curriculum/path_failure_streak": torch.mean(self.path_failure_streak.float()),
            "Curriculum/path_guidance_failure_streak": torch.mean(self.path_guidance_failure_streak.float()),
            "Curriculum/terrain_failure_streak": torch.mean(self.terrain_failure_streak.float()),
            "Curriculum/path_guidance_success_streak": torch.mean(self.path_guidance_success_streak.float()),
            "Curriculum/path_success_streak": torch.mean(self.path_success_streak.float()),
            "Curriculum/terrain_success_streak": torch.mean(self.terrain_success_streak.float()),
            "Curriculum/promotion_cooldown": torch.mean(self.curriculum_promotion_cooldown.float()),
            "Curriculum/active_navigation_routes": torch.mean(self.nav_route_active.float()),
            "Curriculum/navigation_route_refresh_required": torch.mean(self.nav_route_refresh_required.float()),
        }

    # --------------------------------------------------------------------------------------------
    # Small utilities
    # --------------------------------------------------------------------------------------------
    def _scale_and_clip_nav_action(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.clamp(actions.to(self.device), -1.0, 1.0)
        cmd = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        cmd[:, 0] = actions[:, 0] * self.cfg.nav.action_scale_lin_x
        cmd[:, 1] = actions[:, 1] * self.cfg.nav.action_scale_lin_y
        cmd[:, 2] = actions[:, 2] * self.cfg.nav.action_scale_ang_z
        cmd[:, 0] = torch.clamp(cmd[:, 0], *self.cfg.nav.lin_vel_x_range)
        cmd[:, 1] = torch.clamp(cmd[:, 1], *self.cfg.nav.lin_vel_y_range)
        cmd[:, 2] = torch.clamp(cmd[:, 2], *self.cfg.nav.ang_vel_z_range)
        return cmd

    def _check_goal_reached(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return reached

        dist = torch.norm(self.goal_pos_w[env_ids, :2] - self.robot.data.root_pos_w[env_ids, :2], dim=-1)
        self.current_goal_dist[env_ids] = dist
        final_dist = torch.norm(
            self.final_goal_pos_w[env_ids, :2] - self.robot.data.root_pos_w[env_ids, :2], dim=-1
        )
        self.current_final_goal_dist[env_ids] = final_dist
        waypoint_indices_all = self.nav_waypoint_indices[env_ids]
        waypoint_counts_all = self.nav_waypoint_counts[env_ids]
        tracking_final = waypoint_indices_all >= (waypoint_counts_all - 1)
        reached_envs = dist < self.cfg.nav.goal_reached_threshold
        final_target_reached = final_dist < self.cfg.nav.goal_reached_threshold

        if not torch.any(reached_envs | final_target_reached):
            return reached

        waypoint_reached_envs = reached_envs & ~final_target_reached
        reached_ids = env_ids[waypoint_reached_envs]
        final_reached = tracking_final[waypoint_reached_envs]
        final_reached_ids = reached_ids[final_reached]
        intermediate_reached_ids = reached_ids[~final_reached]

        if len(intermediate_reached_ids) > 0:
            self.waypoint_reached_step_buf[intermediate_reached_ids] += 1.0
            self.nav_waypoint_indices[intermediate_reached_ids] += 1
            next_indices = self.nav_waypoint_indices[intermediate_reached_ids]
            self.goal_pos_w[intermediate_reached_ids] = self.nav_waypoints_w[
                intermediate_reached_ids, next_indices
            ]
            self.current_goal_dist[intermediate_reached_ids] = torch.norm(
                self.goal_pos_w[intermediate_reached_ids, :2]
                - self.robot.data.root_pos_w[intermediate_reached_ids, :2],
                dim=-1,
            )

        direct_final_reached_ids = env_ids[final_target_reached]
        if len(final_reached_ids) > 0 or len(direct_final_reached_ids) > 0:
            final_success_ids = torch.cat((final_reached_ids, direct_final_reached_ids), dim=0).unique()
            new_success_ids = final_success_ids[~self.nav_success_buf[final_success_ids]]
            if len(new_success_ids) > 0:
                self.final_goal_reached_step_buf[new_success_ids] = 1.0
                self.goal_reached_buf[new_success_ids] = True
                self.nav_success_buf[new_success_ids] = True
                reached[new_success_ids] = True
        return reached

    def _navigation_terminal_log_dict(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        valid_env_ids = env_ids[self.nav_episode_valid[env_ids]]
        if len(valid_env_ids) == 0:
            return {}

        attempt_dist = self.nav_attempt_dist[valid_env_ids]
        final_dist = self.current_final_goal_dist[valid_env_ids]
        full_route_goal_dist = torch.norm(
            self.full_route_goal_pos_w[valid_env_ids, :2] - self.robot.data.root_pos_w[valid_env_ids, :2],
            dim=-1,
        )
        logs = {
            "Navigation/episode_start_to_final_distance": torch.mean(attempt_dist),
            "Navigation/episode_start_to_task_goal_distance": torch.mean(attempt_dist),
            "Navigation/final_distance_to_goal": torch.mean(final_dist),
            "Navigation/task_goal_distance": torch.mean(final_dist),
            "Navigation/full_route_goal_distance": torch.mean(full_route_goal_dist),
            "Navigation/progress_ratio": torch.mean(
                (attempt_dist - final_dist) / torch.clamp(attempt_dist, min=1.0e-6)
            ),
            "Navigation/task_goal_progress_ratio": torch.mean(
                (attempt_dist - final_dist) / torch.clamp(attempt_dist, min=1.0e-6)
            ),
            "Navigation/route_prefix_fraction": torch.mean(self.nav_route_prefix_fraction[valid_env_ids]),
            "Navigation/current_path_reference_distance": torch.mean(self.current_goal_dist[valid_env_ids]),
            "Navigation/current_distance_to_final_goal": torch.mean(final_dist),
            "Navigation/path_progress": torch.mean(self.nav_path_progress[valid_env_ids]),
            "Navigation/path_remaining": torch.mean(self.nav_path_remaining[valid_env_ids]),
            "Navigation/path_lateral_error": torch.mean(self.nav_path_lateral_error[valid_env_ids]),
            "Navigation/mean_abs_cmd_y": torch.mean(torch.abs(self.nav_cmd[valid_env_ids, 1])),
            "Navigation/mean_abs_cmd_y_rate": torch.mean(
                torch.abs(self.nav_cmd[valid_env_ids, 1] - self.prev_nav_cmd[valid_env_ids, 1])
            ),
            "Navigation/mean_abs_cmd_yaw_rate": torch.mean(
                torch.abs(self.nav_cmd[valid_env_ids, 2] - self.prev_nav_cmd[valid_env_ids, 2])
            ),
        }
        if self.cfg.nav.log_success_rate:
            logs["Navigation/success_rate"] = torch.mean(self.nav_success_buf[valid_env_ids].float())
        return logs

    def _navigation_curriculum_stage_log_dict(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "Navigation/start_to_final_distance": torch.mean(self.nav_attempt_dist[env_ids]),
            "Navigation/start_to_task_goal_distance": torch.mean(self.nav_attempt_dist[env_ids]),
            "Navigation/path_guidance_weight": torch.mean(self.nav_path_guidance_weight[env_ids]),
            "Navigation/path_difficulty_level": torch.mean(self.path_difficulty_levels[env_ids].float()),
            "Navigation/path_guidance_level": torch.mean(self.path_guidance_levels[env_ids].float()),
            "Navigation/path_reference_count": torch.mean(self.nav_waypoint_counts[env_ids].float()),
            "Navigation/route_prefix_fraction": torch.mean(self.nav_route_prefix_fraction[env_ids]),
            "Navigation/route_full_length": torch.mean(self.nav_route_full_length[env_ids]),
            "Navigation/path_metric_score": torch.mean(self.nav_path_metric_score[env_ids]),
            "Navigation/path_metric_length": torch.mean(self.nav_path_metric_length[env_ids]),
            "Navigation/path_metric_detour": torch.mean(self.nav_path_metric_detour[env_ids]),
            "Navigation/path_metric_turns": torch.mean(self.nav_path_metric_turns[env_ids]),
        }

    def _init_navigation_visualization_markers(self) -> None:
        if self._navigation_point_markers is not None:
            return

        marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/NavigationStartGoal",
            markers={
                "start_point": sim_utils.SphereCfg(
                    radius=0.12,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 0.1)),
                ),
                "goal_point": sim_utils.SphereCfg(
                    radius=0.14,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.0)),
                ),
                "planned_path_point": sim_utils.SphereCfg(
                    radius=0.035,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.35, 1.0)),
                ),
                "waypoint_point": sim_utils.SphereCfg(
                    radius=0.065,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0)),
                ),
                "current_waypoint_point": sim_utils.SphereCfg(
                    radius=0.09,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.0, 1.0)),
                ),
            },
        )
        self._navigation_point_markers = VisualizationMarkers(marker_cfg)

    def visualize_navigation_points(
        self,
        env_ids: torch.Tensor | list[int] | None = None,
        *,
        max_envs: int | None = None,
        z_offset: float = 0.15,
        path_z_offset: float = 0.08,
        waypoint_z_offset: float = 0.22,
        show_path: bool = True,
        show_waypoints: bool = True,
        max_path_points_per_env: int = 96,
        visible: bool = True,
    ) -> None:
        """Visualize navigation start, goal, planned global path, and active path references.

        Green markers show starts, red markers show final goals, blue markers show the dense
        planned route reference, yellow markers show sparse path-guidance references, and purple
        markers show the currently tracked reference.
        """
        if not visible:
            if self._navigation_point_markers is not None:
                self._navigation_point_markers.set_visibility(False)
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        if max_envs is not None:
            env_ids = env_ids[:max_envs]
        if len(env_ids) == 0:
            if self._navigation_point_markers is not None:
                self._navigation_point_markers.set_visibility(False)
            return

        self._init_navigation_visualization_markers()
        assert self._navigation_point_markers is not None

        translation_parts: list[torch.Tensor] = []
        marker_index_parts: list[torch.Tensor] = []

        start_points = self.start_marker_pos_w[env_ids].clone()
        goal_points = self.goal_marker_pos_w[env_ids].clone()
        start_points[:, 2] += z_offset
        goal_points[:, 2] += z_offset
        start_goal_points = torch.empty(2 * len(env_ids), 3, device=self.device)
        start_goal_points[0::2] = start_points
        start_goal_points[1::2] = goal_points
        start_goal_indices = torch.empty(2 * len(env_ids), dtype=torch.long, device=self.device)
        start_goal_indices[0::2] = 0
        start_goal_indices[1::2] = 1
        translation_parts.append(start_goal_points)
        marker_index_parts.append(start_goal_indices)

        if show_path:
            path_points = self._navigation_path_visualization_points(
                env_ids,
                z_offset=path_z_offset,
                max_points_per_env=max_path_points_per_env,
            )
            if path_points is not None:
                translation_parts.append(path_points)
                marker_index_parts.append(torch.full((path_points.shape[0],), 2, dtype=torch.long, device=self.device))

        if show_waypoints:
            waypoint_points, current_waypoint_points = self._navigation_waypoint_visualization_points(
                env_ids,
                waypoint_z_offset=waypoint_z_offset,
            )
            if waypoint_points is not None:
                translation_parts.append(waypoint_points)
                marker_index_parts.append(
                    torch.full((waypoint_points.shape[0],), 3, dtype=torch.long, device=self.device)
                )
            if current_waypoint_points is not None:
                translation_parts.append(current_waypoint_points)
                marker_index_parts.append(
                    torch.full((current_waypoint_points.shape[0],), 4, dtype=torch.long, device=self.device)
                )

        translations = torch.cat(translation_parts, dim=0)
        marker_indices = torch.cat(marker_index_parts, dim=0)

        self._navigation_point_markers.set_visibility(True)
        self._navigation_point_markers.visualize(translations=translations, marker_indices=marker_indices)

    def _navigation_path_visualization_points(
        self,
        env_ids: torch.Tensor,
        *,
        z_offset: float,
        max_points_per_env: int,
    ) -> torch.Tensor | None:
        points: list[torch.Tensor] = []
        for env_id_tensor in env_ids:
            env_id = int(env_id_tensor.item())
            metadata = self._get_maze_metadata_for_env(env_id)
            route = self._active_navigation_routes[env_id]
            if metadata is None or not self._route_matches_metadata(route, metadata):
                continue
            assert route is not None
            graph = metadata.get("graph")
            candidate = route.get("candidate")
            if graph is None or candidate is None:
                continue
            local_points = path_reference_points(
                graph,
                candidate.cell_path,
                start_xy=route["start_local_xy"],
                goal_xy=route["goal_local_xy"],
                include_cell_centers=True,
            )
            local_points = self._downsample_sequence(local_points, int(max_points_per_env))
            env_origin = self.scene.env_origins[env_id]
            z = env_origin[2] + float(z_offset)
            for local_xy in local_points:
                world_x, world_y = self._local_xy_to_world_xy(metadata, local_xy, env_origin)
                points.append(torch.stack((world_x, world_y, z)))
        if not points:
            return None
        return torch.stack(points, dim=0)

    def _navigation_waypoint_visualization_points(
        self,
        env_ids: torch.Tensor,
        *,
        waypoint_z_offset: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        waypoint_points: list[torch.Tensor] = []
        current_points: list[torch.Tensor] = []
        for env_id_tensor in env_ids:
            env_id = int(env_id_tensor.item())
            count = int(self.nav_waypoint_counts[env_id].item())
            if count <= 0:
                continue
            waypoints = self.nav_waypoints_w[env_id, :count].clone()
            waypoints[:, 2] += float(waypoint_z_offset)
            waypoint_points.extend(waypoints)

            current_index = int(self.nav_waypoint_indices[env_id].item())
            current_index = max(0, min(current_index, count - 1))
            current = self.nav_waypoints_w[env_id, current_index].clone()
            current[2] += float(waypoint_z_offset) + 0.08
            current_points.append(current)

        waypoint_tensor = torch.stack(waypoint_points, dim=0) if waypoint_points else None
        current_tensor = torch.stack(current_points, dim=0) if current_points else None
        return waypoint_tensor, current_tensor

    @staticmethod
    def _downsample_sequence(sequence: list, max_count: int) -> list:
        if max_count <= 0 or len(sequence) <= max_count:
            return sequence
        if max_count == 1:
            return [sequence[-1]]
        indices: list[int] = []
        last_index = len(sequence) - 1
        for out_i in range(max_count):
            index = int(round(float(out_i * last_index) / float(max_count - 1)))
            if not indices or index != indices[-1]:
                indices.append(index)
        return [sequence[index] for index in indices]

    @staticmethod
    def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
        return torch.atan2(torch.sin(x), torch.cos(x))

    @staticmethod
    def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
        # q is assumed [w, x, y, z]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    def _yaw_to_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        q = torch.zeros((yaw.shape[0], 4), device=self.device)
        q[:, 0] = torch.cos(0.5 * yaw)
        q[:, 3] = torch.sin(0.5 * yaw)
        return q
