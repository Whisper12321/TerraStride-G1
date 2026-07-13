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

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.g1.g1_locomotion_env import G1LocomotionEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv | G1LocomotionEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv | G1LocomotionEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv | G1LocomotionEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv | G1LocomotionEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv | G1LocomotionEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv | G1LocomotionEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_vel_l2(env: BaseEnv | G1LocomotionEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv | G1LocomotionEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )

def joint_action_rate_l2(env: BaseEnv | G1LocomotionEnv) -> torch.Tensor:
    actions = env.action_buffer._circular_buffer.buffer
    num_joint_actions = getattr(env, "num_joint_actions", env.robot.data.default_joint_pos.shape[1])

    return torch.sum(
        torch.square(
            actions[:, -1, :num_joint_actions] - actions[:, -2, :num_joint_actions]
        ),
        dim=1,
    )

def gait_cycle_action_rate_l2(env: BaseEnv | G1LocomotionEnv) -> torch.Tensor:
    if not getattr(env.cfg.gait, "enable_variable_gait_cycle", False):
        return torch.zeros(env.num_envs, device=env.device)

    actions = env.action_buffer._circular_buffer.buffer
    num_joint_actions = getattr(env, "num_joint_actions", env.robot.data.default_joint_pos.shape[1])

    if actions.shape[-1] <= num_joint_actions:
        return torch.zeros(env.num_envs, device=env.device)

    gait_action_delta = actions[:, -1, num_joint_actions:] - actions[:, -2, num_joint_actions:]

    return torch.sum(torch.square(gait_action_delta), dim=1)


def undesired_contacts(env: BaseEnv | G1LocomotionEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv | G1LocomotionEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv | G1LocomotionEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv | G1LocomotionEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(
    env: BaseEnv | G1LocomotionEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv | G1LocomotionEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv | G1LocomotionEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward = (reward - threshold).clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv | G1LocomotionEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # zero_flag = (
    #     torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    # ) < 0.1
    zero_flag = 1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def body_orientation_l2(
    env: BaseEnv | G1LocomotionEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv | G1LocomotionEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv | G1LocomotionEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


# Regularization Reward
def ankle_torque(env: G1LocomotionEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def leg_joint_torque(env: G1LocomotionEnv) -> torch.Tensor:
    """Penalize squared torques on all leg joints."""
    leg_joint_ids = list(env.left_leg_ids) + list(env.right_leg_ids)
    return torch.sum(torch.square(env.robot.data.applied_torque[:, leg_joint_ids]), dim=1)


def ankle_action(env: G1LocomotionEnv) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, env.ankle_joint_ids]), dim=1)


def hip_roll_action(env: G1LocomotionEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[1], env.right_leg_ids[1]]]), dim=1)


def hip_yaw_action(env: G1LocomotionEnv) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)


def feet_y_distance(env: G1LocomotionEnv, nom_dist: float = 0.25) -> torch.Tensor:
    """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    
    feet_dist_lateral = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1])
    y_distance_b = torch.abs(feet_dist_lateral - nom_dist)
    # y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b


# Periodic gait-based reward function
def gait_clock(phase, air_ratio, delta_t):
    """
    Generate periodic gait clock signals for foot swing and stance phases.

    This function constructs two phase-dependent signals:
    - `I_frc`: active during swing phase (used for penalizing ground force)
    - `I_spd`: active during stance phase (used for penalizing foot speed)

    Transitions between swing and stance are smoothed within a margin of `delta_t`
    to create differentiable transitions.

    Parameters
    ----------
    phase : torch.Tensor
        Normalized gait phase in [0, 1], shape: [num_envs].
    air_ratio : torch.Tensor
        Proportion of the gait cycle spent in swing phase, shape: [num_envs].
    delta_t : float
        Transition width around phase boundaries for smooth interpolation.

    Returns
    -------
    I_frc : torch.Tensor
        Gait-based swing-phase clock signal, range [0, 1], shape: [num_envs].
    I_spd : torch.Tensor
        Gait-based stance-phase clock signal, range [0, 1], shape: [num_envs].

    Notes
    -----
    - The transitions at the boundaries (e.g., swing→stance) are linear interpolations.
    - Used in reward shaping to associate expected behavior with gait phases.
    """
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))

    trans_flag1 = phase < delta_t
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    trans_flag3 = phase > (1 - delta_t)

    I_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    I_spd = 1.0 - I_frc
    return I_frc, I_spd


def gait_feet_frc_perio(env: G1LocomotionEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


def gait_feet_spd_perio(env: G1LocomotionEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot speed during the support phase of the gait."""
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    return left_spd_score + right_spd_score


def gait_feet_frc_support_perio(env: G1LocomotionEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Reward that promotes proper support force during stance (support) phase."""
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


#-------------------------------------------------------自己添加------------------------------------------------------------------
def base_height_l2(
    env: BaseEnv | G1LocomotionEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    terrain_radius: float = 0.25,
    height_error_scale: float = 50.0,
) -> torch.Tensor:
    """Reward the root height relative to the local terrain under the robot."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w
    current_height = root_pos_w[:, 2]

    if sensor_cfg is not None:
        sensor = env.scene.sensors[sensor_cfg.name]
        ray_hits_w = sensor.data.ray_hits_w

        ray_hits_z = ray_hits_w[..., 2]
        valid_rays = torch.isfinite(ray_hits_w).all(dim=-1)

        root_xy = root_pos_w[:, :2].unsqueeze(1)
        ray_xy_dist = torch.linalg.norm(ray_hits_w[..., :2] - root_xy, dim=-1)
        valid_rays = valid_rays & torch.isfinite(ray_xy_dist)

        local_rays = valid_rays & (ray_xy_dist <= terrain_radius)
        local_count = local_rays.sum(dim=1)
        local_height_sum = torch.where(local_rays, ray_hits_z, torch.zeros_like(ray_hits_z)).sum(dim=1)
        local_height = local_height_sum / local_count.clamp(min=1).to(ray_hits_z.dtype)

        masked_dist = torch.where(valid_rays, ray_xy_dist, torch.full_like(ray_xy_dist, float("inf")))
        nearest_ray_ids = torch.argmin(masked_dist, dim=1)
        nearest_height = ray_hits_z.gather(1, nearest_ray_ids.unsqueeze(1)).squeeze(1)

        has_valid_ray = valid_rays.any(dim=1)
        terrain_height = torch.where(local_count > 0, local_height, nearest_height)
        terrain_height = torch.where(has_valid_ray, terrain_height, current_height - target_height)

        base_height = current_height - terrain_height
    else:
        base_height = current_height

    height_error = base_height - target_height
    height_error = torch.where(torch.isfinite(height_error), height_error, torch.zeros_like(height_error))
    return torch.exp(-height_error_scale * torch.square(height_error))


def swing_foot_tilt_penalty(
    env: BaseEnv | G1LocomotionEnv,
    delta_t: float = 0.02,
    left_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*left_ankle_roll.*"]),
    right_foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*right_ankle_roll.*"])
) -> torch.Tensor:
    """
    使用gait_clock和重力投影的摆动脚倾斜惩罚
    保持与您原有代码相同的参数格式
    """
    # 获取机器人资产
    asset: Articulation = env.scene[left_foot_cfg.name]
    
    # 获取步态相位掩码
    left_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]  # [num_envs]
    right_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]  # [num_envs]
    
    # 如果没有摆动脚，直接返回0
    if torch.sum(left_swing_mask) + torch.sum(right_swing_mask) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    # 检查body_ids
    if len(left_foot_cfg.body_ids) == 0 or len(right_foot_cfg.body_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device)
    
    left_foot_idx = left_foot_cfg.body_ids[0]
    right_foot_idx = right_foot_cfg.body_ids[0]
    
    # 获取脚部姿态
    left_foot_quat = asset.data.body_quat_w[:, left_foot_idx]  # [num_envs, 4]
    right_foot_quat = asset.data.body_quat_w[:, right_foot_idx]  # [num_envs, 4]
    
    # 重力向量
    gravity = asset.data.GRAVITY_VEC_W  # [3]
    
    # 计算重力在脚部坐标系中的投影
    left_gravity_foot = math_utils.quat_apply_inverse(left_foot_quat, gravity)  # [num_envs, 3]
    right_gravity_foot = math_utils.quat_apply_inverse(right_foot_quat, gravity)  # [num_envs, 3]
    
    # 计算倾斜度：重力XY分量的绝对值之和
    left_tilt = torch.sum(torch.abs(left_gravity_foot[:, :2]), dim=1)  # [num_envs]
    right_tilt = torch.sum(torch.abs(right_gravity_foot[:, :2]), dim=1)  # [num_envs]
    
    # 应用摆动掩码
    left_penalty = left_swing_mask * left_tilt
    right_penalty = right_swing_mask * right_tilt

    return left_penalty + right_penalty


def swing_foot_clearance_adaptive(
    env: BaseEnv | G1LocomotionEnv,
    sensor_cfg: SceneEntityCfg | None = SceneEntityCfg("height_scanner"),
    target_clearance: float | None = None,
    base_clearance: float = 0.08,
    max_clearance: float = 0.20,
    obstacle_margin: float = 0.03,
    obstacle_scale: float = 1.0,
    obstacle_threshold: float = 0.025,
    std: float = 0.04,
    terrain_radius: float = 0.20,
    obstacle_probe_radius: float = 0.16,
    min_lookahead_time: float = 0.12,
    max_lookahead_time: float = 0.45,
    foot_height_offset: float = 0.04,
    min_swing_progress: float = 0.10,
    max_swing_progress: float = 0.90,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward swing foot clearance relative to local terrain and upcoming height rises."""
    if target_clearance is not None:
        base_clearance = target_clearance

    foot_pos_w = env.robot.data.body_pos_w[:, env.feet_body_ids, :]
    terrain_height = torch.zeros(env.num_envs, 2, device=env.device, dtype=foot_pos_w.dtype)
    future_terrain_height = terrain_height

    if sensor_cfg is not None and sensor_cfg.name in env.scene.sensors:
        sensor = env.scene.sensors[sensor_cfg.name]
        ray_hits_w = sensor.data.ray_hits_w
        ray_hits_z = ray_hits_w[..., 2]
        ray_valid = torch.isfinite(ray_hits_w).all(dim=-1)[:, None, :]

        foot_xy = foot_pos_w[:, :, :2]
        ray_xy = ray_hits_w[:, None, :, :2]
        ray_xy_dist = torch.linalg.norm(ray_xy - foot_xy[:, :, None, :], dim=-1)
        valid_rays = ray_valid & torch.isfinite(ray_xy_dist)

        local_rays = valid_rays & (ray_xy_dist <= terrain_radius)
        local_count = local_rays.sum(dim=-1)
        local_height_sum = torch.where(
            local_rays, ray_hits_z[:, None, :], torch.zeros_like(ray_xy_dist)
        ).sum(dim=-1)
        local_height = local_height_sum / local_count.clamp(min=1).to(ray_hits_z.dtype)

        masked_dist = torch.where(valid_rays, ray_xy_dist, torch.full_like(ray_xy_dist, float("inf")))
        nearest_ray_ids = torch.argmin(masked_dist, dim=-1)
        nearest_height = torch.gather(ray_hits_z, 1, nearest_ray_ids)

        has_valid_ray = valid_rays.any(dim=-1)
        terrain_height = torch.where(local_count > 0, local_height, nearest_height)
        terrain_height = torch.where(has_valid_ray, terrain_height, torch.zeros_like(terrain_height))

    gait_phase = env.gait_phase
    phase_ratio = env.phase_ratio
    left_swing = gait_phase[:, 0] < phase_ratio[:, 0]
    right_swing = gait_phase[:, 1] < phase_ratio[:, 1]

    left_progress = torch.where(
        left_swing,
        gait_phase[:, 0] / phase_ratio[:, 0].clamp(min=1e-6),
        torch.zeros_like(gait_phase[:, 0]),
    )
    right_progress = torch.where(
        right_swing,
        gait_phase[:, 1] / phase_ratio[:, 1].clamp(min=1e-6),
        torch.zeros_like(gait_phase[:, 1]),
    )
    swing_progress = torch.stack((left_progress, right_progress), dim=1)
    swing_mask = torch.stack((left_swing, right_swing), dim=1)
    progress_mask = (swing_progress >= min_swing_progress) & (swing_progress <= max_swing_progress)

    if sensor_cfg is not None and sensor_cfg.name in env.scene.sensors:
        root_yaw_quat = math_utils.yaw_quat(env.robot.data.root_link_quat_w)
        cmd_vel_b = torch.cat(
            (
                env.command_generator.command[:, :2],
                torch.zeros(env.num_envs, 1, device=env.device, dtype=foot_pos_w.dtype),
            ),
            dim=1,
        )
        cmd_vel_w = math_utils.quat_apply(root_yaw_quat, cmd_vel_b)[:, :2]

        gait_cycle = env.gait_cycle.unsqueeze(1) if hasattr(env, "gait_cycle") else env.cfg.gait.gait_cycle
        remaining_swing_time = (1.0 - swing_progress).clamp(min=0.0, max=1.0) * phase_ratio * gait_cycle
        lookahead_time = remaining_swing_time.clamp(min=min_lookahead_time, max=max_lookahead_time)
        future_xy = foot_pos_w[:, :, :2] + cmd_vel_w[:, None, :] * lookahead_time.unsqueeze(-1)

        future_dist = torch.linalg.norm(ray_xy - future_xy[:, :, None, :], dim=-1)
        future_valid = ray_valid & torch.isfinite(future_dist)
        future_local = future_valid & (future_dist <= obstacle_probe_radius)
        future_count = future_local.sum(dim=-1)
        future_height = torch.where(
            future_local,
            ray_hits_z[:, None, :],
            torch.full_like(future_dist, -float("inf")),
        ).amax(dim=-1)
        future_terrain_height = torch.where(future_count > 0, future_height, terrain_height)

    foot_clearance = foot_pos_w[:, :, 2] - terrain_height - foot_height_offset
    terrain_rise = torch.clamp(future_terrain_height - terrain_height, min=0.0)
    obstacle_extra = torch.where(
        terrain_rise > obstacle_threshold,
        obstacle_scale * terrain_rise + obstacle_margin,
        torch.zeros_like(terrain_rise),
    )
    dynamic_clearance = torch.clamp(base_clearance + obstacle_extra, max=max_clearance)

    clearance_profile = dynamic_clearance * torch.sin(torch.pi * swing_progress).clamp(min=0.0)
    height_error = foot_clearance - clearance_profile
    reward = torch.exp(-torch.square(height_error) / (2.0 * std * std))

    cmd_norm = torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    cmd_mask = (cmd_norm > command_threshold).unsqueeze(1)
    active_mask = swing_mask & progress_mask & cmd_mask

    reward = torch.where(active_mask, reward, torch.zeros_like(reward))
    active_count = active_mask.sum(dim=1).clamp(min=1).to(reward.dtype)
    return reward.sum(dim=1) / active_count


def foothold_flatness_reward(
    env: BaseEnv | G1LocomotionEnv,
    lfoot_sensor_cfg: SceneEntityCfg = SceneEntityCfg("Lfoot_height_scanner"),
    rfoot_sensor_cfg: SceneEntityCfg = SceneEntityCfg("Rfoot_height_scanner"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_sensor", body_names=[".*ankle_roll.*"]
    ),
    height_std: float = 0.03,
    force_threshold: float = 1.0,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """
    奖励机器人脚接触地面时踩在更平缓的地形上。

    原理：
    - 分别读取左右脚高度扫描器的射线命中高度；
    - 分别计算左右脚附近地形高度方差；
    - 只有该脚接触地面时，才计算该脚的平整度奖励；
    - 方差越小，说明脚附近地形越平缓，奖励越高；
    - 摆动脚不参与该奖励。

    Returns:
        torch.Tensor: shape = [num_envs]
    """

    def _foot_flatness(sensor_name: str) -> torch.Tensor:
        sensor = env.scene.sensors[sensor_name]
        ray_hits_w = sensor.data.ray_hits_w
        ray_hits_z = ray_hits_w[..., 2]

        valid_mask = torch.isfinite(ray_hits_z)

        valid_count = valid_mask.sum(dim=1).clamp(min=1)
        valid_count_f = valid_count.to(ray_hits_z.dtype)

        height_sum = torch.where(
            valid_mask,
            ray_hits_z,
            torch.zeros_like(ray_hits_z),
        ).sum(dim=1)

        height_mean = height_sum / valid_count_f

        height_error = torch.where(
            valid_mask,
            ray_hits_z - height_mean.unsqueeze(1),
            torch.zeros_like(ray_hits_z),
        )

        height_var = torch.sum(torch.square(height_error), dim=1) / valid_count_f

        height_var = torch.where(
            torch.isfinite(height_var),
            height_var,
            torch.zeros_like(height_var),
        )

        return torch.exp(-height_var / (height_std ** 2))

    # 左右脚附近地形平整度，shape: [num_envs]
    left_score = _foot_flatness(lfoot_sensor_cfg.name)
    right_score = _foot_flatness(rfoot_sensor_cfg.name)

    # 读取左右脚接触力，shape: [num_envs, 2]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    contact_force = torch.norm(
        contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :],
        dim=-1,
    )

    contact_mask = contact_force > force_threshold

    left_contact = contact_mask[:, 0].to(left_score.dtype)
    right_contact = contact_mask[:, 1].to(right_score.dtype)

    # 只奖励接触地面的脚
    reward_sum = left_score * left_contact + right_score * right_contact
    contact_count = left_contact + right_contact

    # 单脚支撑时只看支撑脚，双脚支撑时取平均，无脚接触时奖励为0
    reward = reward_sum / contact_count.clamp(min=1.0)
    reward = reward * (contact_count > 0).to(reward.dtype)

    # 静止指令下不鼓励额外落脚策略
    moving_mask = (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
    ) > command_threshold

    return reward * moving_mask



def _get_filtered_foot_heights(
    env: BaseEnv | G1LocomotionEnv,
    sensor_name: str,
    outlier_threshold: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    获取单只脚高度扫描器的滤波后高度数据。

    Args:
        env: 环境对象
        sensor_name: 高度扫描器名称，例如 "Lfoot_height_scanner"
        outlier_threshold: 离群点阈值，单位 m。
            如果某个高度点与中位数差值超过该阈值，则认为是异常点。

    Returns:
        filtered_heights: shape = [num_envs, num_rays]
            滤波后的高度数据，无效点会被替换成中位数。
        valid_mask: shape = [num_envs, num_rays]
            有效高度点 mask。
    """
    sensor = env.scene.sensors[sensor_name]
    ray_hits_w = sensor.data.ray_hits_w
    ray_hits_z = ray_hits_w[..., 2]

    finite_mask = torch.isfinite(ray_hits_z)

    # 为了计算 median，先把无效点替换为 0
    safe_z = torch.where(
        finite_mask,
        ray_hits_z,
        torch.zeros_like(ray_hits_z),
    )

    # 计算每个环境的有效点均值，用于替换无效点
    valid_count = finite_mask.sum(dim=1).clamp(min=1)
    valid_mean = safe_z.sum(dim=1) / valid_count.to(ray_hits_z.dtype)

    safe_z = torch.where(
        finite_mask,
        ray_hits_z,
        valid_mean.unsqueeze(1),
    )

    # 用中位数作为局部地形中心，比 mean 更不怕异常值
    median_z = torch.median(safe_z, dim=1, keepdim=True)[0]

    # 剔除离中位数太远的异常高度点
    deviation = torch.abs(safe_z - median_z)
    valid_mask = finite_mask & (deviation < outlier_threshold)

    # 如果剔除后某个环境没有有效点，则退回使用 finite_mask
    valid_count_after_filter = valid_mask.sum(dim=1, keepdim=True)
    valid_mask = torch.where(
        valid_count_after_filter > 0,
        valid_mask,
        finite_mask,
    )

    # 最终把无效点替换成中位数，避免后面计算 min/max/var 被污染
    filtered_heights = torch.where(
        valid_mask,
        ray_hits_z,
        median_z.expand_as(ray_hits_z),
    )

    filtered_heights = torch.where(
        torch.isfinite(filtered_heights),
        filtered_heights,
        median_z.expand_as(ray_hits_z),
    )

    return filtered_heights, valid_mask

def foothold_edge_penalty(
    env: BaseEnv | G1LocomotionEnv,
    lfoot_sensor_cfg: SceneEntityCfg = SceneEntityCfg("Lfoot_height_scanner"),
    rfoot_sensor_cfg: SceneEntityCfg = SceneEntityCfg("Rfoot_height_scanner"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_sensor", body_names=[".*ankle_roll.*"]
    ),
    edge_height_threshold: float = 0.035,
    edge_height_margin: float = 0.04,
    outlier_threshold: float = 0.08,
    force_threshold: float = 1.0,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """
    只在脚接触地面时，强烈惩罚脚底附近存在高度突变。
    高度数据先经过中位数离群点滤波。
    """

    def _foot_edge_score(sensor_name: str) -> torch.Tensor:
        heights, valid_mask = _get_filtered_foot_heights(
            env,
            sensor_name,
            outlier_threshold=outlier_threshold,
        )

        z_for_max = torch.where(
            valid_mask,
            heights,
            torch.full_like(heights, -float("inf")),
        )
        z_for_min = torch.where(
            valid_mask,
            heights,
            torch.full_like(heights, float("inf")),
        )

        local_max_z = torch.max(z_for_max, dim=1)[0]
        local_min_z = torch.min(z_for_min, dim=1)[0]

        has_valid_ray = valid_mask.any(dim=1)

        height_range = local_max_z - local_min_z
        height_range = torch.where(
            has_valid_ray & torch.isfinite(height_range),
            height_range,
            torch.zeros_like(height_range),
        )

        edge_amount = height_range - edge_height_threshold

        edge_penalty = torch.clamp(
            edge_amount / edge_height_margin,
            min=0.0,
            max=2.0,
        )

        return edge_penalty

    left_penalty = _foot_edge_score(lfoot_sensor_cfg.name)
    right_penalty = _foot_edge_score(rfoot_sensor_cfg.name)

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    contact_force = torch.norm(
        contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :],
        dim=-1,
    )

    contact_mask = contact_force > force_threshold

    left_contact = contact_mask[:, 0].to(left_penalty.dtype)
    right_contact = contact_mask[:, 1].to(right_penalty.dtype)

    penalty_sum = left_penalty * left_contact + right_penalty * right_contact
    contact_count = left_contact + right_contact

    penalty = penalty_sum / contact_count.clamp(min=1.0)
    penalty = penalty * (contact_count > 0).to(penalty.dtype)

    moving_mask = (
        torch.norm(env.command_generator.command[:, :2], dim=1)
        + torch.abs(env.command_generator.command[:, 2])
    ) > command_threshold

    return penalty * moving_mask
