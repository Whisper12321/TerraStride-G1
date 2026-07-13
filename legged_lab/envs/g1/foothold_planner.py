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
# Copyright (c) 2025-2026, The D12-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the D12-Lab Project,
# and is distributed under the BSD-3-Clause license.


import isaaclab.sim as sim_utils
import torch
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat


class FootholdPlannerMixin:
    """落脚点规划相关接口和实现。"""

    def is_foothold_planner_enabled(self) -> bool:
        return bool(
            self.cfg.scene.height_scanner.enable_height_scan
            and self.cfg.foothold.enable_foothold_planner
        )

    def init_foothold_planner_buffers(self):
        if not self.is_foothold_planner_enabled():
            return

        # 假设射线为规则网格，根据高度扫描配置计算网格尺寸。
        self.nx = int(round(self.cfg.scene.height_scanner.size[0] / self.cfg.scene.height_scanner.resolution) + 1)
        self.ny = int(round(self.cfg.scene.height_scanner.size[1] / self.cfg.scene.height_scanner.resolution) + 1)
        self.desired_foothold = torch.zeros(self.num_envs, 3, device=self.device)
        self.desired_foothold_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.swing_time = torch.zeros(self.num_envs, 2, device=self.device)
        self.prev_gait_phase = self.gait_phase.clone()
        self.prev_support_foot_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.first_foothold_flag = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.ref_traj_points = torch.zeros(
            self.num_envs,
            int(self.cfg.gait.gait_cycle * self.cfg.gait.gait_air_ratio_l / self.step_dt),
            3,
            device=self.device,
        )
        self.foot_center_offset_local = torch.tensor(
            [self.cfg.foothold.offset_x, self.cfg.foothold.offset_y, self.cfg.foothold.offset_z],
            device=self.device,
        )
        self._prev_foothold_contact = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)

    def init_visualization_markers(self):
        point_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/Points",
            markers={
                "green_point": sim_utils.SphereCfg(
                    radius=0.06,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "blue_point": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
                "red_point": sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
                ),
                "yellow_point": sim_utils.SphereCfg(
                    radius=0.06,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ),
                "orange_point": sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
                ),
                "sky_point": sim_utils.SphereCfg(
                    radius=0.04,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
            },
        )
        self.num_cells = self.ny * self.nx
        self.vis_steepness_buffer = torch.zeros(self.num_envs * self.num_cells, 3, device=self.device)
        print(self.num_cells, self.ny, self.nx)

        self.single_planner_point_num = 0
        self.marker_indices = torch.cat(
            [
                torch.zeros(self.num_envs, dtype=torch.long),
                torch.ones(self.num_envs * self.single_planner_point_num, dtype=torch.long),
                torch.full((self.num_envs * self.num_cells,), 2, dtype=torch.long),
                torch.full((self.num_envs,), 3, dtype=torch.long),
                torch.full(
                    (
                        self.num_envs
                        * int(self.cfg.gait.gait_cycle * self.cfg.gait.gait_air_ratio_l / self.step_dt),
                    ),
                    4,
                    dtype=torch.long,
                ),
                torch.full((self.num_envs,), 5, dtype=torch.long),
            ]
        ).tolist()
        self.vis_foothold_points_buffer = torch.zeros(
            self.num_envs + self.num_envs * self.single_planner_point_num, 3, device=self.device
        )
        self.optimal_foothold_point_buffer = torch.zeros(self.num_envs, 3, device=self.device)
        self.ref_foot_traj_buffer = torch.zeros(
            self.num_envs * int(self.cfg.gait.gait_cycle * self.cfg.gait.gait_air_ratio_l / self.step_dt),
            3,
            device=self.device,
        )
        self.foot_center_pos_buffer = torch.zeros(self.num_envs, 3, device=self.device)
        self.point_markers = VisualizationMarkers(point_marker_cfg)

    def reset_foothold_planner(self, env_ids):
        if not self.is_foothold_planner_enabled():
            return

        self.first_foothold_flag[env_ids] = True
        self.desired_foothold_b[env_ids] = 0.0

    def update_foothold_planner(self, env_ids=None):
        if not self.is_foothold_planner_enabled():
            return
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        air_ratio = self.phase_ratio[env_ids]
        left_swing = self.gait_phase[env_ids, 0] < air_ratio[:, 0]
        right_swing = self.gait_phase[env_ids, 1] < air_ratio[:, 1]
        self.swing_time[env_ids, 0] = torch.where(
            left_swing,
            self.gait_phase[env_ids, 0] * self.cfg.gait.gait_cycle,
            0.0,
        )
        self.swing_time[env_ids, 1] = torch.where(
            right_swing,
            self.gait_phase[env_ids, 1] * self.cfg.gait.gait_cycle,
            0.0,
        )

        left_switch = left_swing & (self.prev_gait_phase[env_ids, 0] >= air_ratio[:, 0])
        right_switch = right_swing & (self.prev_gait_phase[env_ids, 1] >= air_ratio[:, 1])
        switch_envs_ids = env_ids[torch.nonzero(left_switch | right_switch).flatten()]
        swing_envs_ids = env_ids[torch.nonzero(left_swing | right_swing).flatten()]

        if left_switch.any():
            left_switch_envs = env_ids[torch.nonzero(left_switch).flatten()]
            left_foot_pos, _ = self._get_feet_center_pos_w(left_switch_envs)
            self.prev_support_foot_pos_w[left_switch_envs] = left_foot_pos
            self.first_foothold_flag[left_switch_envs] = False

        if right_switch.any():
            right_switch_envs = env_ids[torch.nonzero(right_switch).flatten()]
            _, right_foot_pos = self._get_feet_center_pos_w(right_switch_envs)
            self.prev_support_foot_pos_w[right_switch_envs] = right_foot_pos
            self.first_foothold_flag[right_switch_envs] = False

        if self.cfg.foothold.plan_mode == "on_switch":
            self.foothold_planner(switch_envs_ids)
        elif self.cfg.foothold.plan_mode == "on_swing":
            target_ids = swing_envs_ids
            mask = self.episode_length_buf[target_ids] % 10 == 0
            self.foothold_planner(target_ids[mask])
        else:
            self.foothold_planner(switch_envs_ids)

    def finalize_foothold_planner_step(self, env_ids=None):
        if self.is_foothold_planner_enabled():
            if env_ids is None:
                self.prev_gait_phase = self.gait_phase.clone()
            elif len(env_ids) > 0:
                self.prev_gait_phase[env_ids] = self.gait_phase[env_ids]

    def _get_feet_center_pos_w(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        left_ankle_pos_w = self.robot.data.body_state_w[env_ids, self.feet_body_ids[0], :3]
        right_ankle_pos_w = self.robot.data.body_state_w[env_ids, self.feet_body_ids[1], :3]

        left_ankle_quat_w = self.robot.data.body_state_w[env_ids, self.feet_body_ids[0], 3:7]
        right_ankle_quat_w = self.robot.data.body_state_w[env_ids, self.feet_body_ids[1], 3:7]

        offset = self.foot_center_offset_local.view(1, 3)
        left_center_w = left_ankle_pos_w + quat_apply(left_ankle_quat_w, offset.expand_as(left_ankle_pos_w))
        right_center_w = right_ankle_pos_w + quat_apply(right_ankle_quat_w, offset.expand_as(right_ankle_pos_w))
        return left_center_w, right_center_w

    def foothold_planner(self, envs_ids):
        """
        为指定环境计算机体坐标系下的期望落脚点 (x, y)。
        基于离散搜索，使用当前地形高度图（来自 height_scanner）。
        """
        n = len(envs_ids)
        if n == 0:
            return

        robot = self.robot
        cmd = self.command_generator.command[envs_ids, :]
        vx, vy, wz = cmd[:, 0].unsqueeze(1), cmd[:, 1].unsqueeze(1), cmd[:, 2].unsqueeze(1)

        T_swing = self.cfg.gait.gait_cycle * self.cfg.gait.gait_air_ratio_l
        leg_width = self.cfg.foothold.leg_width / 2

        phase = self.gait_phase[envs_ids, :]
        air_ratio = self.phase_ratio[envs_ids, :]

        left_swing = phase[:, 0] < air_ratio[:, 0]
        right_swing = phase[:, 1] < air_ratio[:, 1]

        support_is_left = right_swing.unsqueeze(1)

        left_foot_pos, right_foot_pos = self._get_feet_center_pos_w(envs_ids)

        foot_pos_w = torch.where(support_is_left, left_foot_pos, right_foot_pos)

        root_pos = robot.data.root_pos_w[envs_ids, :]
        root_quat = robot.data.root_quat_w[envs_ids, :]
        yaw_root_quat = yaw_quat(root_quat)

        lateral_offset = torch.where(support_is_left, -leg_width, leg_width)
        theta = wz * T_swing
        L_nom = vx * T_swing - torch.sin(theta) * lateral_offset
        W_nom = vy * T_swing + torch.cos(theta) * lateral_offset

        step_local = torch.cat([L_nom, W_nom, torch.zeros_like(L_nom)], dim=-1)

        step_world = quat_apply(yaw_root_quat, step_local)
        target_world = root_pos + step_world

        ray_hits = self.height_scanner.data.ray_hits_w[envs_ids]
        sensor_pos = self.height_scanner.data.pos_w[envs_ids]
        ray_hits_clean = self._clean_ray_hits_vectorized(ray_hits, sensor_pos)

        ray_hits_xy = ray_hits_clean[..., :2]
        ray_hits_z = ray_hits_clean[..., 2]

        if torch.isnan(ray_hits_z).any():
            print("ray_hits_z is nan: ", ray_hits_z)
        if torch.isinf(ray_hits_z).any():
            print("ray_hits_z is inf: ", ray_hits_z)
        if torch.isnan(ray_hits_xy).any():
            print("ray_hits_xy is nan: ", ray_hits_xy)
        if torch.isinf(ray_hits_xy).any():
            print("ray_hits_xy is inf: ", ray_hits_xy)

        xy_map = ray_hits_xy.reshape(n, self.ny, self.nx, 2)
        height_map = ray_hits_z.reshape(n, self.ny, self.nx)

        steepness_map = self._compute_steepness_map(height_map)
        edge_distance_map = self._compute_edge_distance_map(height_map, xy_map)

        num_cells = self.ny * self.nx
        xy_flat = xy_map.view(n, num_cells, 2)

        R_grid = int(self.cfg.foothold.world_window_radius / self.cfg.scene.height_scanner.resolution)
        K = 2 * R_grid + 1

        target_xy = target_world[:, :2]
        dist2 = torch.norm(xy_flat - target_xy.unsqueeze(1), dim=-1)
        nearest_idx = torch.argmin(dist2, dim=-1)
        iy_center = nearest_idx // self.nx
        ix_center = nearest_idx % self.nx

        offset_y = torch.arange(-R_grid, R_grid + 1, device=self.device)
        offset_x = torch.arange(-R_grid, R_grid + 1, device=self.device)
        iy_grid = iy_center.view(n, 1, 1) + offset_y.view(1, K, 1)
        ix_grid = ix_center.view(n, 1, 1) + offset_x.view(1, 1, K)
        iy_clamped = torch.clamp(iy_grid, 0, self.ny - 1)
        ix_clamped = torch.clamp(ix_grid, 0, self.nx - 1)

        flat_indices = iy_clamped * self.nx + ix_clamped
        flat_indices = flat_indices.view(n, -1)

        idx_expanded = flat_indices.unsqueeze(-1).expand(-1, -1, 2)
        candidate_xy = torch.gather(xy_flat, dim=1, index=idx_expanded)

        height_flat = height_map.view(n, self.num_cells)
        candidate_z = torch.gather(height_flat, dim=1, index=flat_indices)
        candidate_z = candidate_z.unsqueeze(-1)

        steepness_flat = steepness_map.view(n, self.num_cells)
        candidate_steepness = torch.gather(steepness_flat, dim=1, index=flat_indices)
        candidate_steepness = candidate_steepness.unsqueeze(-1)

        edge_dist_flat = edge_distance_map.view(n, self.num_cells)
        candidate_edge_dist = torch.gather(edge_dist_flat, dim=1, index=flat_indices)
        candidate_edge_dist = candidate_edge_dist.unsqueeze(-1)

        candidate_point_flat = torch.cat([candidate_xy, candidate_z], dim=-1)

        target_world[:, 2] = height_map[torch.arange(n, device=self.device), iy_center, ix_center]
        slope = self._compute_slope(envs_ids, target_world, xy_flat, height_flat)
        b_nom_x, b_nom_y, sigma_T, sigma_t = self._compute_b_nom(
            envs_ids, slope, support_is_left, L_nom, W_nom, T_swing
        )

        optimal_point = self._compute_optimal_foothold(
            envs_ids,
            root_pos,
            yaw_root_quat,
            candidate_point_flat,
            candidate_steepness,
            candidate_edge_dist,
            b_nom_x,
            b_nom_y,
            L_nom,
            W_nom,
            sigma_T,
            sigma_t,
        )
        if torch.isnan(optimal_point).any():
            print("optimal_point is nan: ", optimal_point)
        if torch.isinf(optimal_point).any():
            print("optimal_point is inf: ", optimal_point)

        self.desired_foothold[envs_ids] = optimal_point
        desired_rel_w = optimal_point - root_pos
        self.desired_foothold_b[envs_ids] = quat_apply_inverse(root_quat, desired_rel_w)

        ref_trajectory = self._compute_ref_trajectory(envs_ids)
        self.ref_traj_points[envs_ids] = ref_trajectory

        if torch.isnan(ref_trajectory).any():
            print("ref_trajectory is nan: ", ref_trajectory)
        if torch.isinf(ref_trajectory).any():
            print("ref_trajectory is inf: ", ref_trajectory)

        self.single_planner_point_num = (iy_clamped.numel() // n) * (ix_clamped.numel() // n)
        self.vis_foothold_points_buffer = torch.zeros(
            self.num_envs + self.num_envs * self.single_planner_point_num, 3, device=self.device
        )
        self.marker_indices = torch.cat(
            [
                torch.zeros(self.num_envs, dtype=torch.long),
                torch.ones(self.num_envs * self.single_planner_point_num, dtype=torch.long),
                torch.full((self.num_envs * self.num_cells,), 2, dtype=torch.long),
                torch.full((self.num_envs,), 3, dtype=torch.long),
                torch.full(
                    (
                        self.num_envs
                        * int(self.cfg.gait.gait_cycle * self.cfg.gait.gait_air_ratio_l / self.step_dt),
                    ),
                    4,
                    dtype=torch.long,
                ),
                torch.full((self.num_envs,), 5, dtype=torch.long),
            ]
        ).tolist()

        self.vis_foothold_points_buffer[envs_ids, 0] = target_xy[:, 0]
        self.vis_foothold_points_buffer[envs_ids, 1] = target_xy[:, 1]
        self.vis_foothold_points_buffer[envs_ids, 2] = height_map[torch.arange(n, device=self.device), iy_center, ix_center]

        start_indices = self.num_envs + envs_ids * self.single_planner_point_num
        row_indices = start_indices[:, None] + torch.arange(self.single_planner_point_num, device=self.device)
        row_indices = row_indices.reshape(-1)
        candidate_points_flat = candidate_point_flat.reshape(-1, 3)
        self.vis_foothold_points_buffer[row_indices, :] = candidate_points_flat

        self.optimal_foothold_point_buffer[envs_ids] = optimal_point

        target_indices = envs_ids[:, None] * num_cells + torch.arange(num_cells, device=self.device)
        target_indices = target_indices.reshape(-1)
        self.vis_steepness_buffer[target_indices, 0] = xy_flat.reshape(-1, 2)[:, 0]
        self.vis_steepness_buffer[target_indices, 1] = xy_flat.reshape(-1, 2)[:, 1]
        root_height_expanded = root_pos[:, 2].unsqueeze(1).expand(-1, num_cells).reshape(-1)
        self.vis_steepness_buffer[target_indices, 2] = steepness_flat.reshape(-1) / 6 + root_height_expanded

        traj_len = self.ref_traj_points.shape[1]
        starts = envs_ids * traj_len
        offsets = torch.arange(traj_len, device=self.device)
        indices = (starts[:, None] + offsets).reshape(-1)
        traj_data_flat = self.ref_traj_points[envs_ids].reshape(-1, 3)
        self.ref_foot_traj_buffer[indices] = traj_data_flat

        self.foot_center_pos_buffer[envs_ids, :] = foot_pos_w

        all_points = torch.cat(
            [
                self.vis_foothold_points_buffer,
                self.vis_steepness_buffer,
                self.optimal_foothold_point_buffer,
                self.ref_foot_traj_buffer,
                self.foot_center_pos_buffer,
            ],
            dim=0,
        )
        if n > 0:
            self.point_markers.set_visibility(True)
            self.point_markers.visualize(translations=all_points, marker_indices=self.marker_indices)
        else:
            self.point_markers.set_visibility(False)

    def _clean_ray_hits_vectorized(self, ray_hits, sensor_pos):
        """
        向量化清洗射线数据：用每个环境的有效点均值填充 NaN/Inf。
        """
        valid = torch.isfinite(ray_hits).all(dim=-1)
        valid_float = valid.unsqueeze(-1).float()

        sum_valid = (ray_hits * valid_float).sum(dim=1)
        count_valid = valid.sum(dim=1, keepdim=True).float()

        mean_valid = sum_valid / count_valid.clamp(min=1)

        safe_sensor_pos = sensor_pos.clone()
        sensor_valid = torch.isfinite(safe_sensor_pos).all(dim=-1)
        default_safe_pos = torch.tensor([0.0, 0.0, 0.9], device=safe_sensor_pos.device).expand_as(safe_sensor_pos)
        safe_sensor_pos = torch.where(sensor_valid.unsqueeze(-1), safe_sensor_pos, default_safe_pos)

        fill_values = torch.where(count_valid > 0, mean_valid, safe_sensor_pos)

        fill_values_valid = torch.isfinite(fill_values).all(dim=-1)
        if not torch.all(fill_values_valid):
            default_fill = torch.tensor([0.0, 0.0, 0.9], device=fill_values.device).expand_as(fill_values)
            fill_values = torch.where(fill_values_valid.unsqueeze(-1), fill_values, default_fill)

        ray_hits_clean = torch.where(valid.unsqueeze(-1), ray_hits, fill_values.unsqueeze(1))

        if not torch.isfinite(ray_hits_clean).all():
            default_clean = torch.tensor([0.0, 0.0, 0.9], device=ray_hits_clean.device).expand_as(ray_hits_clean)
            ray_hits_clean = torch.where(torch.isfinite(ray_hits_clean), ray_hits_clean, default_clean)

        return ray_hits_clean

    def _compute_b_nom(self, env_ids, slope, support_is_left, L_nom, W_nom, T_swing):
        g = self.cfg.foothold.gravity
        z0 = self.cfg.foothold.z0
        l = self.cfg.foothold.leg_width
        t_swing = torch.where(support_is_left.squeeze(1), self.swing_time[env_ids, 1], self.swing_time[env_ids, 0])
        sigma_T = (2 * g**0.5 * (torch.sqrt(slope * T_swing + z0) - z0**0.5)) / slope
        sigma_t = (2 * g**0.5 * (torch.sqrt(slope * t_swing + z0) - z0**0.5)) / slope

        exp_sigma = torch.exp(sigma_T).unsqueeze(-1)

        b_nom_x = L_nom / (exp_sigma - 1)
        b_nom_y = torch.where(
            support_is_left,
            -l / (1 + exp_sigma) - W_nom / (1 - exp_sigma),
            l / (1 + exp_sigma) - W_nom / (1 - exp_sigma),
        )

        if torch.isnan(b_nom_x).any():
            print("b_nom_x is nan: ", b_nom_x)
        if torch.isinf(b_nom_x).any():
            print("b_nom_x is inf: ", b_nom_x)
        if torch.isnan(b_nom_y).any():
            print("b_nom_y is nan: ", b_nom_y)
        if torch.isinf(b_nom_y).any():
            print("b_nom_y is inf: ", b_nom_y)
        if torch.isnan(sigma_T).any():
            print("sigma_T is nan: ", sigma_T)
        if torch.isinf(sigma_T).any():
            print("sigma_T is inf: ", sigma_T)
        if torch.isnan(sigma_t).any():
            print("sigma_t is nan: ", sigma_t)
        if torch.isinf(sigma_t).any():
            print("sigma_t is inf: ", sigma_t)

        return b_nom_x, b_nom_y, sigma_T, sigma_t

    def _compute_slope(self, env_ids, target_world, xy_flat, height_flat):
        prev_support_foot_pos_w_xy = self.prev_support_foot_pos_w[env_ids, :2].unsqueeze(1)
        dist2_to_target = torch.norm((xy_flat - prev_support_foot_pos_w_xy), dim=-1)
        nearest_idx = torch.argmin(dist2_to_target, dim=-1)

        nearest_terrain_z = height_flat[torch.arange(height_flat.shape[0]), nearest_idx]

        dh = torch.norm(-prev_support_foot_pos_w_xy.squeeze(1) + target_world[:, :2], dim=-1)
        dz = -nearest_terrain_z + target_world[:, 2]

        dz = torch.where(torch.abs(dz) < 1e-3, torch.sign(dz) * 1e-3, dz)
        dz = torch.where(torch.abs(dz) == 0, 1e-3, dz)

        slope = torch.where(dh > 1e-4, dz / dh, torch.full_like(dz, float("inf")))
        slope = torch.where(self.first_foothold_flag[env_ids], 1e-3, slope)
        slope = torch.clamp(slope, -1.0, 1.0)
        if torch.isnan(slope).any():
            print("slope is nan: ", slope)
        if torch.isinf(slope).any():
            print("slope is inf: ", slope)
        return slope

    def _compute_steepness_map(self, height_map):
        device = height_map.device
        sobel_x = torch.tensor(
            [[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]],
            device=device,
        )
        sobel_y = torch.tensor(
            [[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]],
            device=device,
        )
        height_batch = height_map.unsqueeze(1)
        gx = torch.nn.functional.conv2d(height_batch, sobel_x, padding=1)
        gy = torch.nn.functional.conv2d(height_batch, sobel_y, padding=1)
        steepness_raw = torch.sqrt(gx**2 + gy**2).squeeze(1)

        dx = self.cfg.scene.height_scanner.resolution
        dy = dx
        foot_grid_h = int(round(self.cfg.foothold.foot_h / dx))
        foot_grid_w = int(round(self.cfg.foothold.foot_w / dy))
        if foot_grid_h % 2 == 0:
            foot_grid_h += 1
        if foot_grid_w % 2 == 0:
            foot_grid_w += 1
        steepness = torch.nn.functional.avg_pool2d(
            steepness_raw.unsqueeze(1),
            kernel_size=(foot_grid_h, foot_grid_w),
            stride=1,
            padding=(foot_grid_h // 2, foot_grid_w // 2),
        ).squeeze(1)

        max_steepness = 1.0
        steepness = torch.clamp(steepness, 0.0, max_steepness) / max_steepness
        if torch.isnan(steepness).any():
            print("steepness is nan: ", steepness)
        if torch.isinf(steepness).any():
            print("steepness is inf: ", steepness)
        return steepness

    def _compute_edge_distance_map(self, height_map, xy_map):
        n, ny, nx = height_map.shape

        detect_kernel = 3
        height_batch = height_map.unsqueeze(1)

        local_max = torch.nn.functional.max_pool2d(
            height_batch,
            kernel_size=detect_kernel,
            stride=1,
            padding=detect_kernel // 2,
        )
        local_min = -torch.nn.functional.max_pool2d(
            -height_batch,
            kernel_size=detect_kernel,
            stride=1,
            padding=detect_kernel // 2,
        )

        local_height_range = (local_max - local_min).squeeze(1)
        edge_mask = local_height_range > self.cfg.foothold.edge_height_thresh

        num_cells = ny * nx
        xy_flat = xy_map.view(n, num_cells, 2)
        edge_flat = edge_mask.view(n, num_cells)

        pairwise_dist = torch.cdist(xy_flat, xy_flat)

        inf_dist = torch.full_like(pairwise_dist, float("inf"))
        dist_to_edge = torch.where(
            edge_flat.unsqueeze(1),
            pairwise_dist,
            inf_dist,
        ).min(dim=-1).values

        no_edge_env = ~edge_flat.any(dim=-1)
        if no_edge_env.any():
            dist_to_edge[no_edge_env] = float("inf")

        edge_distance_map = dist_to_edge.view(n, ny, nx)
        return edge_distance_map

    def _compute_optimal_foothold(
        self,
        envs_id,
        root_pos,
        yaw_root_quat,
        candidate_point_flat,
        candidate_steepness,
        candidate_edge_dist,
        b_nom_x,
        b_nom_y,
        L_nom,
        W_nom,
        sigma_T,
        sigma_t,
    ):
        omega_init = self.cfg.foothold.gravity / self.cfg.foothold.z0
        x_dot_init = self.command_generator.command[envs_id, 0]
        epsinon_init = 1 / omega_init * x_dot_init

        epsinon_t = epsinon_init * torch.exp(sigma_t)

        u_T = candidate_point_flat[:, :, :3] - root_pos[:, :3].unsqueeze(1)
        n = len(envs_id)
        K = u_T.shape[1] if u_T.dim() == 3 else u_T.shape[0] // n

        if u_T.dim() == 3:
            quat_expanded = yaw_root_quat[:, None, :].expand(-1, K, -1).reshape(-1, 4)
            u_T_flat = u_T.reshape(-1, 3)
        else:
            quat_expanded = yaw_root_quat.repeat_interleave(K, dim=0)
            u_T_flat = u_T

        u_T_rot_flat = quat_apply_inverse(quat_expanded, u_T_flat)
        u_T_rot = u_T_rot_flat.reshape(n, K, 3)

        b_x = (epsinon_t * torch.exp(sigma_T - sigma_t)).unsqueeze(1) - u_T_rot[:, :, 0]
        b_y = (epsinon_t * torch.exp(sigma_T - sigma_t)).unsqueeze(1) - u_T_rot[:, :, 1]

        term1 = torch.cat([(u_T_rot[:, :, 0] - L_nom).unsqueeze(-1), (u_T_rot[:, :, 1] - W_nom).unsqueeze(-1)], dim=-1)
        term2 = torch.cat([(b_x - b_nom_x).unsqueeze(-1), (b_y - b_nom_y).unsqueeze(-1)], dim=-1)
        term3 = candidate_steepness
        term4 = candidate_point_flat[:, :, :3] - self.desired_foothold[envs_id, :3].unsqueeze(1)
        term5 = candidate_edge_dist

        term1 = torch.norm(term1, dim=-1)
        term2 = torch.norm(term2, dim=-1)
        term3 = term3.squeeze(-1)
        term4 = torch.norm(term4, dim=-1)
        term5 = term5.squeeze(-1)

        eps = 1e-6
        min_scale = 0.05
        half_foot_diag = 0.5 * (self.cfg.foothold.foot_h**2 + self.cfg.foothold.foot_w**2) ** 0.5
        d_safe = half_foot_diag + self.cfg.foothold.edge_safe_margin

        step_scale = torch.sqrt(L_nom.squeeze(-1) ** 2 + W_nom.squeeze(-1) ** 2).unsqueeze(1)
        step_scale = torch.clamp(step_scale, min=min_scale)

        dcm_scale = torch.sqrt(b_nom_x.squeeze(-1) ** 2 + b_nom_y.squeeze(-1) ** 2).unsqueeze(1)
        dcm_scale = torch.clamp(dcm_scale, min=min_scale)

        history_scale = torch.full_like(term4, 2.0 * self.cfg.foothold.world_window_radius)
        history_scale = torch.clamp(history_scale, min=min_scale)

        term1 = term1 / (step_scale + eps)
        term2 = term2 / (dcm_scale + eps)
        term4 = term4 / (history_scale + eps)
        term5 = torch.relu(d_safe - term5) / (d_safe + eps)

        alpha1 = self.cfg.foothold.alpha1
        alpha2 = self.cfg.foothold.alpha2
        alpha3 = self.cfg.foothold.alpha3
        alpha4 = self.cfg.foothold.alpha4
        alpha5 = self.cfg.foothold.alpha5

        obj = alpha1 * term1 + alpha2 * term2 + alpha3 * term3 + alpha4 * term4 + alpha5 * term5

        optimal_id = torch.argmin(obj, dim=-1)
        optimal_point = candidate_point_flat[torch.arange(len(envs_id)), optimal_id, :]

        return optimal_point

    def _compute_ref_trajectory(self, envs_id):
        def Bezier_curve(t, P0, P1, P2):
            orig_shape = t.shape
            t_flat = t.reshape(-1)

            t_flat = t_flat[:, None, None]

            B = (1 - t_flat) ** 2 * P0 + 2 * (1 - t_flat) * t_flat * P1 + t_flat**2 * P2

            if t.dim() == 0:
                return B[0]
            return B.reshape(orig_shape + (len(envs_id), 3)).transpose(0, 1)

        desired_foothold = self.desired_foothold[envs_id, :].clone()
        prev_support_foot_pos_w = self.prev_support_foot_pos_w[envs_id, :]

        ankle_offset = 0.04
        slope_height_threshold = 3e-2
        safety_margin = 0.05
        min_clearance = 0.10
        max_clearance = 0.35
        desired_foothold[:, 2] = desired_foothold[:, 2] + ankle_offset
        dz = desired_foothold[:, 2] - prev_support_foot_pos_w[:, 2]

        mask_up = dz > slope_height_threshold
        mask_down = dz < -slope_height_threshold

        start_xy = prev_support_foot_pos_w[:, :2]
        end_xy = desired_foothold[:, :2]
        weight_start = torch.where(mask_up, 1.0 / 3.0, torch.where(mask_down, 1.0 / 2.0, 0.5))
        weight_end = 1.0 - weight_start

        mid_xy = weight_start.unsqueeze(1) * start_xy + weight_end.unsqueeze(1) * end_xy
        middle_point_x = mid_xy[:, 0]
        middle_point_y = mid_xy[:, 1]

        mid_up = desired_foothold[:, 2] + safety_margin + 0.6 * dz
        mid_down = prev_support_foot_pos_w[:, 2] + safety_margin - 0.3 * dz
        mid_flat = prev_support_foot_pos_w[:, 2] + min_clearance
        middle_point_z = torch.where(mask_up, mid_up, torch.where(mask_down, mid_down, mid_flat))

        base_height = torch.max(desired_foothold[:, 2], prev_support_foot_pos_w[:, 2])
        lower = base_height + min_clearance
        upper = base_height + max_clearance
        middle_point_z = torch.clamp(middle_point_z, lower, upper)

        middel_point = torch.cat(
            [middle_point_x.unsqueeze(1), middle_point_y.unsqueeze(1), middle_point_z.unsqueeze(1)], dim=-1
        )

        T_swing = self.cfg.gait.gait_cycle * self.cfg.gait.gait_air_ratio_l
        dt = self.step_dt
        tt = torch.linspace(
            torch.tensor(0.0, device=self.device),
            torch.tensor(1.0, device=self.device),
            steps=int(T_swing / dt),
        ).to(self.device)
        ref_trajectory = Bezier_curve(tt, prev_support_foot_pos_w, middel_point, desired_foothold)

        return ref_trajectory
