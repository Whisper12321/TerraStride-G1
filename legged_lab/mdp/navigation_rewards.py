from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.g1.g1_navigation_env import G1NavigationEnv


_EPS = 1.0e-6


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _nav_dt(env: G1NavigationEnv) -> float:
    if hasattr(env, "nav_step_dt"):
        return float(env.nav_step_dt)
    return float(env.step_dt)


def _goal_distance(env: G1NavigationEnv) -> torch.Tensor:
    if hasattr(env, "current_goal_dist"):
        return env.current_goal_dist
    return torch.norm(env.goal_pos_w[:, :2] - env.robot.data.root_pos_w[:, :2], dim=-1)


def _final_goal_distance(env: G1NavigationEnv) -> torch.Tensor:
    if hasattr(env, "current_final_goal_dist"):
        return env.current_final_goal_dist
    if hasattr(env, "final_goal_pos_w"):
        return torch.norm(env.final_goal_pos_w[:, :2] - env.robot.data.root_pos_w[:, :2], dim=-1)
    return _goal_distance(env)


def _goal_radius(env: G1NavigationEnv, goal_radius: float | None = None) -> float:
    if goal_radius is None:
        return float(getattr(env.cfg.nav, "goal_reached_threshold", 0.5))
    return float(goal_radius)


def _at_goal(env: G1NavigationEnv, goal_radius: float | None = None) -> torch.Tensor:
    return _goal_distance(env) < _goal_radius(env, goal_radius)


def _at_final_goal(env: G1NavigationEnv, goal_radius: float | None = None) -> torch.Tensor:
    return _final_goal_distance(env) < _goal_radius(env, goal_radius)


def _navigation_success(env: G1NavigationEnv) -> torch.Tensor:
    return getattr(env, "nav_success_buf", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))


def _path_difficulty_fraction(env: G1NavigationEnv) -> torch.Tensor:
    if not hasattr(env, "path_difficulty_levels"):
        return torch.zeros(env.num_envs, device=env.device)
    cfg_nav = getattr(getattr(env, "cfg", None), "nav", None)
    num_levels = max(int(getattr(cfg_nav, "num_path_difficulty_levels", 1)), 1)
    if num_levels <= 1:
        return torch.zeros(env.num_envs, device=env.device)
    levels = env.path_difficulty_levels.float()
    return torch.clamp(levels / float(num_levels - 1), min=0.0, max=1.0)


def _tracking_final_waypoint(env: G1NavigationEnv) -> torch.Tensor:
    if not hasattr(env, "nav_waypoint_indices") or not hasattr(env, "nav_waypoint_counts"):
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return env.nav_waypoint_indices >= (env.nav_waypoint_counts - 1)


def _planar_speed(env: G1NavigationEnv) -> torch.Tensor:
    return torch.norm(env.robot.data.root_lin_vel_b[:, :2], dim=-1)


def _yaw_rate_abs(env: G1NavigationEnv) -> torch.Tensor:
    return torch.abs(env.robot.data.root_ang_vel_b[:, 2])


def _episode_time_s(env: G1NavigationEnv) -> torch.Tensor:
    if hasattr(env, "low_level_step_dt"):
        step_dt = float(env.low_level_step_dt)
    elif hasattr(env, "step_dt"):
        step_dt = float(env.step_dt)
    else:
        step_dt = _nav_dt(env)
    return env.episode_length_buf.float() * step_dt


def _path_subgoal_length_for_time_limit(env: G1NavigationEnv) -> torch.Tensor:
    # Prefer the dense active task path length. nav_path_total_length may be computed from
    # guidance-sparsified references, which can become a direct start-goal segment late in curriculum.
    if hasattr(env, "nav_path_metric_length"):
        metric_length = torch.clamp(env.nav_path_metric_length, min=0.0)
        if hasattr(env, "nav_path_total_length"):
            sparse_length = torch.clamp(env.nav_path_total_length, min=0.0)
            return torch.where(metric_length > 0.0, metric_length, sparse_length)
        return metric_length
    if hasattr(env, "nav_path_total_length"):
        return torch.clamp(env.nav_path_total_length, min=0.0)
    if hasattr(env, "nav_attempt_dist"):
        return torch.clamp(env.nav_attempt_dist, min=0.0)
    # Current navigation envs store the path-curriculum sub-goal in final_goal_pos_w.
    return _final_goal_distance(env)


def _goal_still_score(
    env: G1NavigationEnv,
    goal_radius: float | None = None,
    speed_sigma: float = 0.15,
    yaw_rate_sigma: float = 0.35,
) -> torch.Tensor:
    """Continuous score for being inside the final goal and nearly stationary."""
    speed_score = torch.exp(-torch.square(_planar_speed(env) / max(float(speed_sigma), _EPS)))
    yaw_score = torch.exp(-torch.square(_yaw_rate_abs(env) / max(float(yaw_rate_sigma), _EPS)))
    return _at_final_goal(env, goal_radius).float() * speed_score * yaw_score


def _goal_stable_bool(
    env: G1NavigationEnv,
    goal_radius: float | None = None,
    speed_threshold: float = 0.15,
    yaw_rate_threshold: float = 0.35,
) -> torch.Tensor:
    return (
        _at_final_goal(env, goal_radius)
        & (_planar_speed(env) < float(speed_threshold))
        & (_yaw_rate_abs(env) < float(yaw_rate_threshold))
    )


# --------------------------------------------------------------------------------------
# Task shaping helpers. Active navigation rewards use privileged path progress plus final-goal terms.
# --------------------------------------------------------------------------------------


def nav_waypoint_progress(
    env: G1NavigationEnv,
    max_progress_speed: float = 0.8,
    allow_negative: bool = False,
    max_negative_speed: float = 0.2,
) -> torch.Tensor:
    """Reward net progress toward the current waypoint.

    This is local route following, not a direct-to-final-goal shaping term. The reward is suppressed
    on steps where a waypoint switches, because the distance baseline changes at that instant.
    """
    progress = env.prev_goal_dist - env.current_goal_dist
    progress_rate = progress / max(_nav_dt(env), _EPS)
    if allow_negative:
        min_value = -float(max_negative_speed)
    else:
        min_value = 0.0
    progress_rate = torch.clamp(progress_rate, min=min_value, max=float(max_progress_speed))
    waypoint_switched = (
        getattr(env, "waypoint_reached_step_buf", torch.zeros(env.num_envs, device=env.device)) > 0.0
    ) | (getattr(env, "final_goal_reached_step_buf", torch.zeros(env.num_envs, device=env.device)) > 0.0)
    return torch.where(waypoint_switched, torch.zeros_like(progress_rate), progress_rate)


def nav_final_goal_progress(
    env: G1NavigationEnv,
    max_progress_speed: float = 0.4,
    allow_negative: bool = False,
    max_negative_speed: float = 0.1,
) -> torch.Tensor:
    """Weak global shaping toward the final goal.

    Keep this low-weight in mazes: it provides a deployment-like compass bias without requiring
    non-deployable path-reference observations.
    """
    if not hasattr(env, "prev_final_goal_dist"):
        return torch.zeros(env.num_envs, device=env.device)
    progress = env.prev_final_goal_dist - _final_goal_distance(env)
    progress_rate = progress / max(_nav_dt(env), _EPS)
    min_value = -float(max_negative_speed) if allow_negative else 0.0
    return torch.clamp(progress_rate, min=min_value, max=float(max_progress_speed))


def nav_path_progress(
    env: G1NavigationEnv,
    max_progress_speed: float = 0.8,
    allow_negative: bool = False,
    max_negative_speed: float = 0.2,
    use_guidance_weight: bool = True,
) -> torch.Tensor:
    """Reward progress along the privileged global path reference.

    This uses the projection of the robot onto the planned path instead of discrete hidden
    waypoint-hit events, so it provides smoother guidance and can be annealed away.
    """
    if not hasattr(env, "nav_path_progress") or not hasattr(env, "prev_nav_path_progress"):
        return torch.zeros(env.num_envs, device=env.device)
    progress = env.nav_path_progress - env.prev_nav_path_progress
    progress_rate = progress / max(_nav_dt(env), _EPS)
    min_value = -float(max_negative_speed) if allow_negative else 0.0
    reward = torch.clamp(progress_rate, min=min_value, max=float(max_progress_speed))
    if use_guidance_weight and hasattr(env, "nav_path_guidance_weight"):
        reward = reward * env.nav_path_guidance_weight
    return reward


def nav_path_lateral_deviation(
    env: G1NavigationEnv,
    deviation_scale: float = 1.0,
    max_deviation: float = 3.0,
    use_guidance_weight: bool = True,
) -> torch.Tensor:
    """Penalty magnitude for drifting away from the privileged path reference."""
    if not hasattr(env, "nav_path_lateral_error"):
        return torch.zeros(env.num_envs, device=env.device)
    deviation = torch.clamp(env.nav_path_lateral_error / max(float(deviation_scale), _EPS), max=float(max_deviation))
    if use_guidance_weight and hasattr(env, "nav_path_guidance_weight"):
        deviation = deviation * env.nav_path_guidance_weight
    return deviation


def nav_waypoint_reached(env: G1NavigationEnv) -> torch.Tensor:
    """One-shot bonus for crossing intermediate waypoints.

    RewardManager multiplies by nav dt, so return an event rate to make the configured weight the
    actual per-waypoint bonus.
    """
    events = getattr(env, "waypoint_reached_step_buf", torch.zeros(env.num_envs, device=env.device))
    return events / max(_nav_dt(env), _EPS)


def nav_final_goal_reached(env: G1NavigationEnv) -> torch.Tensor:
    """One-shot bonus for first reaching the final goal."""
    events = getattr(env, "final_goal_reached_step_buf", torch.zeros(env.num_envs, device=env.device))
    return events / max(_nav_dt(env), _EPS)


def nav_prefix_goal_progress(
    env: G1NavigationEnv,
    max_progress_speed: float = 0.4,
    allow_negative: bool = False,
    max_negative_speed: float = 0.1,
) -> torch.Tensor:
    """Progress toward the active path-curriculum prefix goal."""
    return nav_final_goal_progress(env, max_progress_speed, allow_negative, max_negative_speed)


def nav_prefix_goal_reached(
    env: G1NavigationEnv,
    min_multiplier: float = 1.0,
    max_multiplier: float = 1.0,
) -> torch.Tensor:
    """One-shot bonus for first reaching the active path-curriculum prefix goal.

    The multiplier can scale this event with path curriculum difficulty while keeping the
    success condition itself unchanged.
    """
    events = getattr(env, "final_goal_reached_step_buf", torch.zeros(env.num_envs, device=env.device))
    difficulty_fraction = _path_difficulty_fraction(env)
    multiplier = float(min_multiplier) + difficulty_fraction * (float(max_multiplier) - float(min_multiplier))
    return events * multiplier / max(_nav_dt(env), _EPS)


def nav_desired_speed(env: G1NavigationEnv, desired_speed: float = 0.6, speed_sigma: float = 0.3) -> torch.Tensor:
    """Reward moving near a nominal navigation speed, following HiPAN's high-level speed term."""
    return torch.exp(-torch.abs(float(desired_speed) - _planar_speed(env)) / max(float(speed_sigma), _EPS))


def nav_near_waypoint(env: G1NavigationEnv, distance_scale: float = 1.5) -> torch.Tensor:
    """Local dense shaping around the current waypoint."""
    return torch.exp(-_goal_distance(env) / max(float(distance_scale), _EPS))


def nav_near_final_goal(env: G1NavigationEnv, distance_scale: float = 3.0) -> torch.Tensor:
    """Weak dense shaping around the final target."""
    return torch.exp(-_final_goal_distance(env) / max(float(distance_scale), _EPS))


def nav_final_goal_region(env: G1NavigationEnv, goal_radius: float | None = None) -> torch.Tensor:
    """Binary reward for being inside the final goal region."""
    return (_at_final_goal(env, goal_radius) & ~_navigation_success(env)).float()


def nav_final_goal_hold_still(
    env: G1NavigationEnv,
    goal_radius: float | None = None,
    speed_sigma: float = 0.15,
    yaw_rate_sigma: float = 0.35,
) -> torch.Tensor:
    """Reward standing still only at the final goal, never at intermediate waypoints."""
    return _goal_still_score(env, goal_radius, speed_sigma, yaw_rate_sigma) * (~_navigation_success(env)).float()


def nav_final_goal_velocity_penalty(
    env: G1NavigationEnv,
    goal_radius: float | None = None,
    yaw_rate_weight: float = 0.2,
) -> torch.Tensor:
    """Penalize moving after entering the final goal region."""
    active = (_at_final_goal(env, goal_radius) & ~_navigation_success(env)).float()
    return active * (
        torch.square(_planar_speed(env)) + float(yaw_rate_weight) * torch.square(_yaw_rate_abs(env))
    )


def nav_final_goal_command_penalty(
    env: G1NavigationEnv,
    goal_radius: float | None = None,
    yaw_weight: float = 0.5,
) -> torch.Tensor:
    """Penalize commanding motion inside the final goal region."""
    cmd = env.nav_cmd
    active = (_at_final_goal(env, goal_radius) & ~_navigation_success(env)).float()
    return active * (
        torch.sum(torch.square(cmd[:, :2]), dim=-1) + float(yaw_weight) * torch.square(cmd[:, 2])
    )


def nav_timeout(env: G1NavigationEnv) -> torch.Tensor:
    """Timeout is bad unless the active path-curriculum sub-goal has already been reached."""
    events = (env.time_out_buf & ~_navigation_success(env)).float()
    return events / max(_nav_dt(env), _EPS)


def nav_path_subgoal_time_overrun(
    env: G1NavigationEnv,
    threshold_s: float = 0.0,
    expected_speed: float = 0.5,
    grace_time_s: float = 20.0,
    min_time_s: float = 20.0,
    ramp_time_s: float = 15.0,
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """Continuous penalty for taking too long to reach the active path-curriculum sub-goal.

    If ``threshold_s`` is positive it is used as a fixed time limit. Otherwise the limit is
    adaptive to the current prefix path: sub-goal path length / expected speed + grace time.
    The returned value is a dimensionless penalty in [0, max_penalty]; RewardManager applies
    the configured negative weight and multiplies by the high-level navigation dt.
    """
    if float(threshold_s) > 0.0:
        time_limit = torch.full((env.num_envs,), float(threshold_s), device=env.device)
    else:
        expected_speed = max(float(expected_speed), _EPS)
        time_limit = _path_subgoal_length_for_time_limit(env) / expected_speed + float(grace_time_s)
        time_limit = torch.clamp(time_limit, min=float(min_time_s))

    overrun_s = torch.clamp(_episode_time_s(env) - time_limit, min=0.0)
    if float(ramp_time_s) > 0.0:
        penalty = torch.clamp(overrun_s / float(ramp_time_s), max=float(max_penalty))
    else:
        penalty = (overrun_s > 0.0).float() * float(max_penalty)
    return penalty * (~_navigation_success(env)).float()


def nav_goal_time_overrun(
    env: G1NavigationEnv,
    threshold_s: float = 0.0,
    expected_speed: float = 0.5,
    grace_time_s: float = 20.0,
    min_time_s: float = 20.0,
    ramp_time_s: float = 15.0,
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """Backward-compatible alias for nav_path_subgoal_time_overrun."""
    return nav_path_subgoal_time_overrun(
        env,
        threshold_s=threshold_s,
        expected_speed=expected_speed,
        grace_time_s=grace_time_s,
        min_time_s=min_time_s,
        ramp_time_s=ramp_time_s,
        max_penalty=max_penalty,
    )


def nav_goal_reached(
    env: G1NavigationEnv,
    goal_radius: float | None = None,
    speed_sigma: float = 0.15,
    yaw_rate_sigma: float = 0.35,
) -> torch.Tensor:
    """Compatibility name for final-goal holding."""
    return nav_final_goal_hold_still(env, goal_radius, speed_sigma, yaw_rate_sigma)


def nav_goal_progress(
    env: G1NavigationEnv,
    max_progress_speed: float = 0.8,
    allow_negative: bool = False,
    max_negative_speed: float = 0.2,
) -> torch.Tensor:
    """Compatibility wrapper for current-waypoint progress."""
    return nav_waypoint_progress(env, max_progress_speed, allow_negative, max_negative_speed)


def nav_near_goal(env: G1NavigationEnv, distance_scale: float = 1.5) -> torch.Tensor:
    """Compatibility wrapper for current-waypoint proximity."""
    return nav_near_waypoint(env, distance_scale)


def nav_at_goal(env: G1NavigationEnv, goal_radius: float | None = None) -> torch.Tensor:
    """Compatibility wrapper for final-goal region reward."""
    return nav_final_goal_region(env, goal_radius)


def nav_hold_still(
    env: G1NavigationEnv,
    goal_radius: float | None = None,
    speed_sigma: float = 0.15,
    yaw_rate_sigma: float = 0.35,
) -> torch.Tensor:
    """Compatibility wrapper for final-goal holding."""
    return nav_final_goal_hold_still(env, goal_radius, speed_sigma, yaw_rate_sigma)


def nav_goal_velocity_penalty(
    env: G1NavigationEnv,
    goal_radius: float | None = None,
    yaw_rate_weight: float = 0.2,
) -> torch.Tensor:
    """Compatibility wrapper for final-goal velocity penalty."""
    return nav_final_goal_velocity_penalty(env, goal_radius, yaw_rate_weight)


def nav_goal_command_penalty(
    env: G1NavigationEnv,
    goal_radius: float | None = None,
    yaw_weight: float = 0.5,
) -> torch.Tensor:
    """Compatibility wrapper for final-goal command penalty."""
    return nav_final_goal_command_penalty(env, goal_radius, yaw_weight)


# --------------------------------------------------------------------------------------
# Command smoothness
# --------------------------------------------------------------------------------------


def nav_command_rate(
    env: G1NavigationEnv,
    lin_weight: float = 1.0,
    yaw_weight: float = 1.5,
) -> torch.Tensor:
    """Penalize large changes between consecutive high-level velocity commands."""
    delta = env.nav_cmd - env.prev_nav_cmd
    return float(lin_weight) * torch.sum(torch.square(delta[:, :2]), dim=-1) + float(yaw_weight) * torch.square(
        delta[:, 2]
    )


def nav_command_acceleration(
    env: G1NavigationEnv,
    lin_weight: float = 1.0,
    yaw_weight: float = 1.5,
) -> torch.Tensor:
    """Second-difference command smoothness term.

    If the environment does not expose prev_prev_nav_cmd yet, this term safely returns zero.
    """
    if not hasattr(env, "prev_prev_nav_cmd"):
        return torch.zeros(env.num_envs, device=env.device)
    acc = env.nav_cmd - 2.0 * env.prev_nav_cmd + env.prev_prev_nav_cmd
    return float(lin_weight) * torch.sum(torch.square(acc[:, :2]), dim=-1) + float(yaw_weight) * torch.square(acc[:, 2])


def nav_command_tracking_error(
    env: G1NavigationEnv,
    lin_weight: float = 1.0,
    yaw_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize high-level commands that the embedded locomotion layer is not realizing."""
    actual = torch.cat((env.robot.data.root_lin_vel_b[:, :2], env.robot.data.root_ang_vel_b[:, 2:3]), dim=-1)
    error = env.nav_cmd - actual
    return float(lin_weight) * torch.sum(torch.square(error[:, :2]), dim=-1) + float(yaw_weight) * torch.square(
        error[:, 2]
    )


# --------------------------------------------------------------------------------------
# Navigation safety
# --------------------------------------------------------------------------------------


def nav_undesired_collision(env: G1NavigationEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Count undesired body contacts over the contact history."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1).float()


def nav_is_terminated(env: G1NavigationEnv) -> torch.Tensor:
    """Penalize failure resets only, excluding timeouts and successful goal resets."""
    events = (env.reset_buf & ~env.time_out_buf & ~_navigation_success(env)).float()
    return events / max(_nav_dt(env), _EPS)


# --------------------------------------------------------------------------------------
# Backward-compatible wrappers for older navigation configs
# --------------------------------------------------------------------------------------


def nav_pos_soft(env: G1NavigationEnv, sigma: float = 3.0, reward_window_s: float | None = None) -> torch.Tensor:
    del reward_window_s
    return nav_near_goal(env, distance_scale=sigma)


def nav_pos_tight(env: G1NavigationEnv, sigma: float = 0.5, reward_window_s: float | None = None) -> torch.Tensor:
    del reward_window_s
    return nav_at_goal(env, goal_radius=sigma)


def nav_heading(env: G1NavigationEnv, sigma: float = 1.0, reward_window_s: float | None = None) -> torch.Tensor:
    del sigma, reward_window_s
    return torch.zeros(env.num_envs, device=env.device)


def nav_agile(env: G1NavigationEnv, vmax: float = 1.0, sigma_tight: float = 0.5) -> torch.Tensor:
    del vmax, sigma_tight
    return nav_goal_progress(env)


def nav_stall(env: G1NavigationEnv, sigma_soft: float = 2.0, static_speed_threshold: float = 0.10) -> torch.Tensor:
    far_from_goal = _goal_distance(env) > float(sigma_soft)
    static = _planar_speed(env) < float(static_speed_threshold)
    success = getattr(env, "nav_success_buf", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    waypoint_event = getattr(env, "waypoint_reached_step_buf", torch.zeros(env.num_envs, device=env.device)) > 0.0
    final_event = getattr(env, "final_goal_reached_step_buf", torch.zeros(env.num_envs, device=env.device)) > 0.0
    return (far_from_goal & static & ~success & ~waypoint_event & ~final_event).float()
