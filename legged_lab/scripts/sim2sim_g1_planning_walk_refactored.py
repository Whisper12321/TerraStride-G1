"""
Clean sim2sim runner for Unitree G1 12-DOF policy trained in IsaacLab/LeggedLab.

Key alignment points with training:
1. IsaacLab policy joint order is assumed to be breadth-first / interleaved:
   [L0, R0, L1, R1, L2, R2, L3, R3, L4, R4, L5, R5]
2. MuJoCo joint order is assumed to be depth-first:
   [L0, L1, L2, L3, L4, L5, R0, R1, R2, R3, R4, R5]
3. Actor observation layout follows G1Env.compute_current_observations():
   ang_vel(3), projected_gravity(3), command(3), joint_pos(12), joint_vel(12),
   previous action(num_actions), sin_phase(1), cos_phase(1), phase_ratio(2),
   optional gait_cycle_obs(1), then frame stack, then height_scan(187).
4. Variable gait-cycle support is kept as a config switch. It must match the
   policy used during training. If training used enable_variable_gait_cycle=False,
   keep it False here too.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import mujoco
import mujoco_viewer
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

try:
    from joystick import JoyStickController
except Exception:  # keep the script usable without joystick.py
    JoyStickController = None


# -----------------------------------------------------------------------------
# Joint order conventions
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MUJOCO_MODEL_PATH = PROJECT_ROOT / "legged_lab/assets/g1_description/scene_navigation_locomotion_terrain_v4_hfield.xml"
DEFAULT_LOCOMOTION_POLICY_PATH = PROJECT_ROOT / "logs/g1_locomotion/2026-05-26_17-11-30/exported/policy.pt"

MUJOCO_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

ISAAC_POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

# MuJoCo depth-first -> IsaacLab/policy breadth-first.
MUJOCO_TO_ISAAC = np.array([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11], dtype=np.int64)
# IsaacLab/policy breadth-first -> MuJoCo depth-first.
ISAAC_TO_MUJOCO = np.array([0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11], dtype=np.int64)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class SimConfig:
    mujoco_model_path: str = str(DEFAULT_MUJOCO_MODEL_PATH)
    policy_path: str = str(DEFAULT_LOCOMOTION_POLICY_PATH)
    sim_duration_s: float = 600.0
    dt: float = 0.002
    decimation: int = 10
    render: bool = True
    use_joystick: bool = True
    use_keyboard: bool = False
    policy_device: str = "cpu"


@dataclass
class CameraConfig:
    enable_follow_camera: bool = False
    rotate_with_robot: bool = True
    track_body_name: str = "pelvis"

    world_distance: float = 28.0
    world_elevation: float = -89.0
    world_azimuth: float = 90.0
    world_lookat: tuple = (0.0, 0.0, 0.0)

    distance: float = 1.5

    # Relative to robot heading:
    # 0 follows behind the robot and looks forward.
    # 180 looks from the front toward the rear.
    # 90 / -90 are side views.
    relative_azimuth: float = 0.0

    elevation: float = -15.0
    lookat_offset: tuple = (0.0, 0.0, 0.15)


@dataclass
class DepthCameraConfig:
    enabled: bool = False
    parent_body_name: str = "pelvis"

    # Mirrors the IsaacLab G1 noisy_camera configuration used for navigation.
    width: int = 64
    height: int = 36
    target_horizontal_fov_deg: float = 100.0
    vertical_fov_deg: float = 85.0
    offset_pos: tuple = (0.05366, 0.01753, 0.47387)
    offset_rot_wxyz: tuple = (0.9659258262890683, 0.0, 0.25881904510252074, 0.0)

    min_distance: float = 0.1
    max_distance: float = 4.0
    crop_region: tuple[int, int, int, int] = (9, 9, 16, 16)
    depth_range: tuple[float, float] = (0.0, 4.0)
    update_every_sim_steps: int = 10
    window_scale: int = 8
    show_raw: bool = True
    show_processed: bool = True
    apply_gaussian_blur: bool = True
    blur_kernel_size: int = 3
    blur_sigma: float = 1.0

    # Match IsaacLab's ray-cast camera setup after the recent change: terrain only.
    # MuJoCo scene uses group 1 for terrain/walls/obstacles and group 2/3 for robot visual/collision.
    geomgroup: tuple[int, int, int, int, int, int] = (0, 1, 0, 0, 0, 0)

    @property
    def processed_height(self) -> int:
        up, down, _, _ = self.crop_region
        return self.height - up - down

    @property
    def processed_width(self) -> int:
        _, _, left, right = self.crop_region
        return self.width - left - right

    @property
    def depth_dim(self) -> int:
        return self.processed_height * self.processed_width


@dataclass
class VideoConfig:
    enabled: bool = False
    output_path: str = "outputs/sim2sim_multiview.mp4"
    fps: float = 30.0
    panel_width: int = 640
    panel_height: int = 480
    codec: str = "mp4v"
    stop_on_goal: bool = False


@dataclass
class HeightScanConfig:
    enabled: bool = True
    x_range: float = 1.6
    y_range: float = 1.0
    nx: int = 17
    ny: int = 11
    offset: float = 0.5
    min_height: float = 0.0
    max_height: float = 1.0
    ray_start_offset_z: float = 0.8
    geomgroup: Tuple[int, int, int, int, int, int] = (0, 1, 0, 0, 0, 0)
    # Your old script used transpose after reshape(17, 11). Keep it as default,
    # but verify once against IsaacLab RayCaster pattern order.
    transpose_to_isaac_order: bool = True
    update_every_policy_step: int = 1
    update_markers: bool = True

    @property
    def num_rays(self) -> int:
        return self.nx * self.ny if self.enabled else 0


@dataclass
class GaitConfig:
    gait_air_ratio_l: float = 0.43
    gait_air_ratio_r: float = 0.43
    gait_phase_offset_l: float = 0.43
    gait_phase_offset_r: float = 0.93
    gait_cycle: float = 0.8

    # Must match the exported low-level locomotion policy.
    enable_variable_gait_cycle: bool = False
    observe_gait_cycle: bool = False
    gait_cycle_scale_range: Tuple[float, float] = (0.75, 1.25)
    gait_cycle_smoothing: float = 0.1


@dataclass
class RobotConfig:
    num_joint_actions: int = 12
    action_scale: float = 0.25
    actor_obs_history_length: int = 10
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    initial_base_height: float = 0.85

    # MuJoCo/depth-first order: left leg then right leg.
    default_joint_pos_mujoco: np.ndarray = field(
        default_factory=lambda: np.array(
            [-0.3, 0.0, 0.0, 0.6, -0.3, 0.0,
             -0.3, 0.0, 0.0, 0.6, -0.3, 0.0],
            dtype=np.float64,
        )
    )
    kp_mujoco: np.ndarray = field(
        default_factory=lambda: np.array(
            [200.0, 150.0, 150.0, 200.0, 20.0, 20.0,
             200.0, 150.0, 150.0, 200.0, 20.0, 20.0],
            dtype=np.float64,
        )
    )
    kd_mujoco: np.ndarray = field(
        default_factory=lambda: np.array(
            [5.0, 5.0, 5.0, 5.0, 2.0, 2.0,
             5.0, 5.0, 5.0, 5.0, 2.0, 2.0],
            dtype=np.float64,
        )
    )
    tau_limit_mujoco: np.ndarray = field(
        default_factory=lambda: np.array(
            [88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
             88.0, 139.0, 88.0, 139.0, 50.0, 50.0],
            dtype=np.float64,
        )
    )


@dataclass
class NavigationConfig:
    enabled: bool = False
    policy_path: str = "auto"
    goal_xy: tuple[float, float] | None = None
    goal_site_name: str = "goal_marker"
    goal_reached_threshold: float = 0.6
    stop_on_goal: bool = False

    actor_obs_history_length: int = 4
    navigation_decimation: int = 5
    command_interpolation: str = "linear"
    command_smoothing_alpha: float = 0.6
    print_interval: int = 10

    action_scale_lin_x: float = 1.0
    action_scale_lin_y: float = 0.35
    action_scale_ang_z: float = 0.9
    lin_vel_x_range: tuple[float, float] = (-0.3, 0.8)
    lin_vel_y_range: tuple[float, float] = (-0.3, 0.3)
    ang_vel_z_range: tuple[float, float] = (-1.0, 1.0)

    goal_xy_scale: float = 0.1
    goal_distance_scale: float = 0.1
    heading_error_scale: float = 1.0
    previous_command_scale: float = 1.0
    base_velocity_scale: float = 1.0
    projected_gravity_scale: float = 1.0

    clip_observations: float = 100.0


@dataclass
class Sim2SimConfig:
    sim: SimConfig = field(default_factory=SimConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    gait: GaitConfig = field(default_factory=GaitConfig)
    height_scan: HeightScanConfig = field(default_factory=HeightScanConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    depth_camera: DepthCameraConfig = field(default_factory=DepthCameraConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)

    @property
    def num_gait_actions(self) -> int:
        return 1 if self.gait.enable_variable_gait_cycle else 0

    @property
    def num_actions(self) -> int:
        return self.robot.num_joint_actions + self.num_gait_actions

    @property
    def num_single_obs(self) -> int:
        # base actor obs: 3 + 3 + 3 + 12 + 12 + num_actions + 1 + 1 + 2
        n = 37 + self.num_actions
        if self.gait.enable_variable_gait_cycle and self.gait.observe_gait_cycle:
            n += 1
        return n

    @property
    def stacked_obs_dim(self) -> int:
        return self.robot.actor_obs_history_length * self.num_single_obs

    @property
    def policy_input_dim(self) -> int:
        return self.stacked_obs_dim + self.height_scan.num_rays

    @property
    def nav_low_dim(self) -> int:
        return 13

    @property
    def nav_actor_history_dim(self) -> int:
        return self.navigation.actor_obs_history_length * self.nav_low_dim

    @property
    def nav_policy_input_dim(self) -> int:
        return self.nav_actor_history_dim + self.depth_camera.depth_dim

    @property
    def nav_critic_input_dim(self) -> int:
        # Critic training used actor low-dim state plus 13 privileged path features per history frame.
        return self.navigation.actor_obs_history_length * 26 + self.depth_camera.depth_dim


@dataclass
class Command:
    vx: float = 0.0
    vy: float = 0.0
    dyaw: float = 0.0


class NumpyHistory:
    """Circular history matching IsaacLab CircularBuffer.buffer flatten order."""

    def __init__(self, history_length: int):
        self.history_length = int(history_length)
        self._buffer: Optional[np.ndarray] = None

    def append(self, obs: np.ndarray) -> None:
        obs = obs.astype(np.float32, copy=True)
        if self._buffer is None:
            self._buffer = np.tile(obs[None, :], (self.history_length, 1))
            return
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = obs

    def flatten(self) -> np.ndarray:
        if self._buffer is None:
            raise RuntimeError("Navigation history is empty.")
        return self._buffer.reshape(-1).astype(np.float32, copy=False)


class NavigationRuntime:
    def __init__(self, cfg: Sim2SimConfig):
        self.history = NumpyHistory(cfg.navigation.actor_obs_history_length)
        self.prev_prev_nav_cmd = np.zeros(3, dtype=np.float32)
        self.prev_nav_cmd = np.zeros(3, dtype=np.float32)
        self.nav_cmd = np.zeros(3, dtype=np.float32)
        self.last_raw_action = np.zeros(3, dtype=np.float32)
        self.goal_reached = False

    def update_from_action(self, action: np.ndarray, cfg: Sim2SimConfig) -> None:
        raw_cmd = scale_and_clip_navigation_action(action, cfg)
        alpha = float(np.clip(cfg.navigation.command_smoothing_alpha, 0.0, 1.0))
        self.prev_prev_nav_cmd = self.prev_nav_cmd.copy()
        self.prev_nav_cmd = self.nav_cmd.copy()
        self.nav_cmd = alpha * raw_cmd + (1.0 - alpha) * self.prev_nav_cmd
        self.last_raw_action = action.astype(np.float32, copy=True)

    def interpolated_command(self, low_level_substep: int, cfg: Sim2SimConfig) -> np.ndarray:
        if self.goal_reached:
            return np.zeros(3, dtype=np.float32)
        if cfg.navigation.command_interpolation == "linear":
            ratio = float(low_level_substep + 1) / float(cfg.navigation.navigation_decimation)
            return ((1.0 - ratio) * self.prev_nav_cmd + ratio * self.nav_cmd).astype(np.float32)
        return self.nav_cmd.astype(np.float32, copy=True)


# -----------------------------------------------------------------------------
# Math / MuJoCo helpers
# -----------------------------------------------------------------------------

def get_site_pos(model, data, site_name: str) -> np.ndarray:
    """Read a site world position from MuJoCo model/data."""
    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )
    if site_id < 0:
        raise RuntimeError(f"Cannot find site '{site_name}' in XML.")

    # data.site_xpos is reliable after mj_forward.
    mujoco.mj_forward(model, data)
    return data.site_xpos[site_id].copy()


def set_site_pos(model: mujoco.MjModel, site_name: str, pos_w: np.ndarray) -> None:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise RuntimeError(f"Cannot find site '{site_name}' in XML.")
    model.site_pos[site_id] = np.asarray(pos_w, dtype=np.float64)


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)


def get_yaw_rotation_from_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def get_mujoco_state(data: mujoco.MjData, num_joints: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return joint q/dq in MuJoCo order, base quaternion wxyz, base lin vel, base ang vel, gravity in body frame."""
    q = data.qpos[7:7 + num_joints].astype(np.float64).copy()
    dq = data.qvel[6:6 + num_joints].astype(np.float64).copy()

    quat_wxyz = data.qpos[3:7].astype(np.float64).copy()
    rot = R.from_quat(quat_wxyz_to_xyzw(quat_wxyz))

    lin_vel_body = rot.apply(data.qvel[:3], inverse=True).astype(np.float64)
    ang_vel_body = data.qvel[3:6].astype(np.float64).copy()
    projected_gravity = rot.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.float64)
    return q, dq, quat_wxyz, lin_vel_body, ang_vel_body, projected_gravity


def pd_control(target_q: np.ndarray, q: np.ndarray, kp: np.ndarray, target_dq: np.ndarray, dq: np.ndarray, kd: np.ndarray) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


# -----------------------------------------------------------------------------
# Height scan
# -----------------------------------------------------------------------------

def sample_height_scan(model: mujoco.MjModel, data: mujoco.MjData, cfg: Sim2SimConfig) -> np.ndarray:
    """Sample terrain heights and convert to IsaacLab-style height_scan observation."""
    hs = cfg.height_scan
    if not hs.enabled:
        return np.zeros(0, dtype=np.float32)

    robot_pos = data.qpos[:3].astype(np.float64).copy()
    robot_quat_wxyz = data.qpos[3:7].astype(np.float64).copy()
    ray_start_z = robot_pos[2] + hs.ray_start_offset_z

    x_lin = np.linspace(-hs.x_range / 2.0, hs.x_range / 2.0, hs.nx)
    y_lin = np.linspace(-hs.y_range / 2.0, hs.y_range / 2.0, hs.ny)
    grid_x, grid_y = np.meshgrid(x_lin, y_lin, indexing="ij")
    local_offsets = np.stack([grid_x, grid_y, np.zeros_like(grid_x)], axis=-1).reshape(-1, 3)

    r_yaw = get_yaw_rotation_from_quat_wxyz(robot_quat_wxyz)
    world_offsets = (r_yaw @ local_offsets.T).T
    ray_origins = robot_pos + world_offsets
    ray_origins[:, 2] = ray_start_z
    ray_dirs = np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float64), (ray_origins.shape[0], 1))

    geomgroup = np.array(hs.geomgroup, dtype=np.uint8)
    geom_ids = np.full(ray_origins.shape[0], -1, dtype=np.int32)
    hit_z = np.zeros(ray_origins.shape[0], dtype=np.float64)

    robot_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "pelvis",
    )

    for idx in range(ray_origins.shape[0]):
        dist = mujoco.mj_ray(
            model,
            data,
            ray_origins[idx],
            ray_dirs[idx],
            geomgroup,
            flg_static=1,
            bodyexclude=robot_body_id,
            geomid=geom_ids[idx:idx + 1],
        )
        hit_z[idx] = ray_origins[idx, 2] + dist * ray_dirs[idx, 2] if dist >= 0.0 else 0.0

        if hs.update_markers:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"marker{idx}")
            if body_id != -1:
                mocap_id = model.body_mocapid[body_id]
                if mocap_id != -1:
                    marker_pos = robot_pos + world_offsets[idx]
                    marker_pos[2] = hit_z[idx]
                    data.mocap_pos[mocap_id] = marker_pos
                    data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])

    # IsaacLab: sensor_z - ray_hit_z - height_scan_offset, then clamp [0, 1].
    height_scan = robot_pos[2] - hit_z - hs.offset
    if hs.transpose_to_isaac_order:
        height_scan = height_scan.reshape(hs.nx, hs.ny).T.flatten()
    height_scan = np.clip(height_scan, hs.min_height, hs.max_height).astype(np.float32)
    return height_scan


# -----------------------------------------------------------------------------
# Gait / observation / action adapter
# -----------------------------------------------------------------------------

class GaitState:
    def __init__(self, cfg: Sim2SimConfig):
        self.cfg = cfg
        self.clock = 0.0
        self.cycle = cfg.gait.gait_cycle
        self.phase = np.zeros(2, dtype=np.float32)
        self.phase_ratio = np.array([cfg.gait.gait_air_ratio_l, cfg.gait.gait_air_ratio_r], dtype=np.float32)
        self.phase_offset = np.array([cfg.gait.gait_phase_offset_l, cfg.gait.gait_phase_offset_r], dtype=np.float32)

    def update_cycle_from_action(self, action_isaac: np.ndarray) -> None:
        if not self.cfg.gait.enable_variable_gait_cycle:
            self.cycle = self.cfg.gait.gait_cycle
            return
        gait_action = math.tanh(float(action_isaac[self.cfg.robot.num_joint_actions]))
        min_scale, max_scale = self.cfg.gait.gait_cycle_scale_range
        scale = min_scale + 0.5 * (gait_action + 1.0) * (max_scale - min_scale)
        target_cycle = self.cfg.gait.gait_cycle * scale
        alpha = self.cfg.gait.gait_cycle_smoothing
        self.cycle = (1.0 - alpha) * self.cycle + alpha * target_cycle

    def step_phase(self) -> None:
        step_dt = self.cfg.sim.dt * self.cfg.sim.decimation
        self.clock = (self.clock + step_dt / self.cycle) % 1.0
        self.phase[0] = (self.clock + self.phase_offset[0]) % 1.0
        self.phase[1] = (self.clock + self.phase_offset[1]) % 1.0


def build_single_actor_obs(
    cfg: Sim2SimConfig,
    q_mujoco: np.ndarray,
    dq_mujoco: np.ndarray,
    ang_vel_body: np.ndarray,
    projected_gravity: np.ndarray,
    command: Command,
    previous_action_isaac: np.ndarray,
    gait: GaitState,
) -> np.ndarray:
    obs = np.zeros(cfg.num_single_obs, dtype=np.float32)

    q_isaac = q_mujoco[MUJOCO_TO_ISAAC]
    dq_isaac = dq_mujoco[MUJOCO_TO_ISAAC]
    default_isaac = cfg.robot.default_joint_pos_mujoco[MUJOCO_TO_ISAAC]

    obs[0:3] = ang_vel_body
    obs[3:6] = projected_gravity
    obs[6:9] = np.array([command.vx, command.vy, command.dyaw], dtype=np.float32)
    obs[9:21] = q_isaac - default_isaac
    obs[21:33] = dq_isaac

    idx = 33
    obs[idx:idx + cfg.num_actions] = previous_action_isaac
    idx += cfg.num_actions

    zero_flag = (math.sqrt(command.vx ** 2 + command.vy ** 2) + abs(command.dyaw)) < 0.1
    obs[idx] = 0.0 if zero_flag else math.sin(2.0 * math.pi * gait.phase[0])
    obs[idx + 1] = 1.0 if zero_flag else math.cos(2.0 * math.pi * gait.phase[0])
    obs[idx + 2:idx + 4] = gait.phase_ratio
    idx += 4

    if cfg.gait.enable_variable_gait_cycle and cfg.gait.observe_gait_cycle:
        obs[idx] = gait.cycle / cfg.gait.gait_cycle

    return np.clip(obs, -cfg.robot.clip_observations, cfg.robot.clip_observations)


def build_policy_input(cfg: Sim2SimConfig, obs_history: deque, height_scan: np.ndarray) -> np.ndarray:
    policy_input = np.zeros((1, cfg.policy_input_dim), dtype=np.float32)
    for i, obs in enumerate(obs_history):
        start = i * cfg.num_single_obs
        policy_input[0, start:start + cfg.num_single_obs] = obs
    if cfg.height_scan.enabled:
        policy_input[0, cfg.stacked_obs_dim:cfg.stacked_obs_dim + cfg.height_scan.num_rays] = height_scan
    return np.clip(policy_input, -cfg.robot.clip_observations, cfg.robot.clip_observations)


def action_isaac_to_target_q_mujoco(cfg: Sim2SimConfig, action_isaac: np.ndarray) -> np.ndarray:
    joint_action_isaac = action_isaac[:cfg.robot.num_joint_actions]
    joint_action_mujoco = joint_action_isaac[ISAAC_TO_MUJOCO]
    return cfg.robot.default_joint_pos_mujoco + cfg.robot.action_scale * joint_action_mujoco


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def scale_and_clip_navigation_action(action: np.ndarray, cfg: Sim2SimConfig) -> np.ndarray:
    nav = cfg.navigation
    action = np.clip(action.astype(np.float32, copy=False), -1.0, 1.0)
    cmd = np.zeros(3, dtype=np.float32)
    cmd[0] = action[0] * nav.action_scale_lin_x
    cmd[1] = action[1] * nav.action_scale_lin_y
    cmd[2] = action[2] * nav.action_scale_ang_z
    cmd[0] = np.clip(cmd[0], *nav.lin_vel_x_range)
    cmd[1] = np.clip(cmd[1], *nav.lin_vel_y_range)
    cmd[2] = np.clip(cmd[2], *nav.ang_vel_z_range)
    return cmd


def build_navigation_current_obs(
    cfg: Sim2SimConfig,
    data: mujoco.MjData,
    nav_runtime: NavigationRuntime,
    lin_vel_body: np.ndarray,
    ang_vel_body: np.ndarray,
    projected_gravity: np.ndarray,
    goal_xy: np.ndarray,
) -> np.ndarray:
    nav = cfg.navigation
    root_pos = data.qpos[:3].astype(np.float64)
    yaw = yaw_from_quat_wxyz(data.qpos[3:7].astype(np.float64))

    goal_vec_w = goal_xy.astype(np.float64) - root_pos[:2]
    goal_dist = float(np.linalg.norm(goal_vec_w))
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    goal_vec_b = np.array(
        [
            c * goal_vec_w[0] - s * goal_vec_w[1],
            s * goal_vec_w[0] + c * goal_vec_w[1],
        ],
        dtype=np.float32,
    )

    desired_heading = math.atan2(goal_vec_w[1], goal_vec_w[0])
    heading_error = wrap_to_pi(desired_heading - yaw)
    base_vel = np.array([lin_vel_body[0], lin_vel_body[1], ang_vel_body[2]], dtype=np.float32)

    obs = np.concatenate(
        [
            goal_vec_b * nav.goal_xy_scale,
            np.array([goal_dist * nav.goal_distance_scale], dtype=np.float32),
            np.array([heading_error * nav.heading_error_scale], dtype=np.float32),
            nav_runtime.prev_nav_cmd.astype(np.float32) * nav.previous_command_scale,
            base_vel * nav.base_velocity_scale,
            projected_gravity.astype(np.float32) * nav.projected_gravity_scale,
        ]
    )
    return np.clip(obs, -nav.clip_observations, nav.clip_observations).astype(np.float32)


def resolve_navigation_policy_path(path: str) -> str:
    if path and path != "auto":
        return os.path.abspath(path)
    candidates = glob.glob(os.path.abspath("logs/g1_navigation/*/model_*.pt"))
    candidates = [candidate for candidate in candidates if os.path.isfile(candidate)]
    if not candidates:
        raise FileNotFoundError(
            "No navigation checkpoint found under logs/g1_navigation/*/model_*.pt. "
            "Pass --nav-policy explicitly."
        )
    return max(candidates, key=os.path.getmtime)


def load_navigation_policy(cfg: Sim2SimConfig):
    path = resolve_navigation_policy_path(cfg.navigation.policy_path)
    device = torch.device(cfg.sim.policy_device)
    try:
        policy = torch.jit.load(path, map_location=device)
        policy.eval()
        print(f"[INFO] Loaded TorchScript navigation policy: {path}")
        return policy, path
    except Exception:
        pass

    from rsl_rl.modules import ActorCriticDepthCnnGru

    policy = ActorCriticDepthCnnGru(
        num_actor_obs=cfg.nav_policy_input_dim,
        num_critic_obs=cfg.nav_critic_input_dim,
        num_actions=3,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=0.7,
        noise_std_type="log",
        depth_shape=(1, cfg.depth_camera.processed_height, cfg.depth_camera.processed_width),
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
    ).to(device)
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    policy.reset()
    print(f"[INFO] Loaded RSL-RL navigation checkpoint: {path}")
    return policy, path


def infer_navigation_action(policy, nav_input: np.ndarray, device: torch.device) -> np.ndarray:
    obs_tensor = torch.from_numpy(nav_input[None, :]).to(device)
    with torch.no_grad():
        if hasattr(policy, "act_inference"):
            action_tensor = policy.act_inference(obs_tensor)
        else:
            action_tensor = policy(obs_tensor)
    return action_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)


class MujocoDepthCamera:
    """MuJoCo ray-cast depth camera aligned with IsaacLab PinholeCameraPattern ordering."""

    def __init__(self, model: mujoco.MjModel, cfg: Sim2SimConfig):
        self.model = model
        self.cfg = cfg
        self.dc = cfg.depth_camera
        self.parent_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.dc.parent_body_name,
        )
        if self.parent_body_id < 0:
            raise RuntimeError(f"Cannot find body '{self.dc.parent_body_name}' for depth camera.")

        self.cv2 = None
        if self.dc.show_raw or self.dc.show_processed:
            try:
                import cv2  # type: ignore
            except Exception as exc:
                raise RuntimeError("Depth camera visualization requires OpenCV (cv2).") from exc
            self.cv2 = cv2

        self.offset_pos = np.array(self.dc.offset_pos, dtype=np.float64)
        self.offset_rot = R.from_quat(quat_wxyz_to_xyzw(np.array(self.dc.offset_rot_wxyz, dtype=np.float64)))
        self.geomgroup = np.array(self.dc.geomgroup, dtype=np.uint8)
        self.local_ray_dirs = self._build_isaaclab_local_ray_directions()
        self.local_ray_plane_scale = self.local_ray_dirs[:, 0].copy()
        self.raw_depth_m = np.full((self.dc.height, self.dc.width), self.dc.max_distance, dtype=np.float32)
        self.processed_depth = np.ones(
            (self.dc.processed_height, self.dc.processed_width),
            dtype=np.float32,
        )
        self._printed_pose = False
        self._closed = False

    def _build_isaaclab_local_ray_directions(self) -> np.ndarray:
        """Match isaaclab.sensors.ray_caster.patterns.pinhole_camera_pattern exactly."""
        width = int(self.dc.width)
        height = int(self.dc.height)
        fx = width / (2.0 * math.tan(math.radians(self.dc.target_horizontal_fov_deg) * 0.5))
        fy = height / (2.0 * math.tan(math.radians(self.dc.vertical_fov_deg) * 0.5))
        cx = width / 2.0
        cy = height / 2.0

        u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64), indexing="xy")
        x = (u + 0.5 - cx) / fx
        y = (v + 0.5 - cy) / fy

        # IsaacLab converts camera coordinates (x right, y down, z forward)
        # into the world/robotics convention used here: x forward, y left, z up.
        dirs = np.stack([np.ones_like(x), -x, -y], axis=-1).reshape(-1, 3)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        return dirs.astype(np.float64)

    def _camera_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parent_pos = data.xpos[self.parent_body_id].astype(np.float64).copy()
        parent_quat_wxyz = data.xquat[self.parent_body_id].astype(np.float64).copy()
        parent_rot = R.from_quat(quat_wxyz_to_xyzw(parent_quat_wxyz))

        cam_pos = parent_pos + parent_rot.apply(self.offset_pos)
        cam_rot = parent_rot * self.offset_rot
        forward_w = cam_rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float64))
        up_w = cam_rot.apply(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        return cam_pos, forward_w, up_w

    def update(self, data: mujoco.MjData) -> np.ndarray:
        if self._closed:
            return self.processed_depth

        depth_m = self._raycast_depth_image(data)
        self.raw_depth_m = depth_m.astype(np.float32, copy=False)
        self.processed_depth = self._process_depth(depth_m)

        raw_norm = self._normalize_depth(depth_m)

        if self.dc.show_raw:
            self._show("mujoco_depth_raw", raw_norm)

        if self.dc.show_processed:
            self._show("mujoco_depth_processed", self.processed_depth)

        return self.processed_depth

    def render(self, data: mujoco.MjData) -> None:
        self.update(data)

    def flat_observation(self) -> np.ndarray:
        return self.processed_depth.reshape(-1).astype(np.float32, copy=True)

    def raw_normalized_image(self) -> np.ndarray:
        return self._normalize_depth(self.raw_depth_m).astype(np.float32, copy=True)

    def _raycast_depth_image(self, data: mujoco.MjData) -> np.ndarray:
        cam_pos, forward_w, _ = self._camera_pose(data)
        parent_quat_wxyz = data.xquat[self.parent_body_id].astype(np.float64).copy()
        parent_rot = R.from_quat(quat_wxyz_to_xyzw(parent_quat_wxyz))
        cam_rot = parent_rot * self.offset_rot
        ray_dirs_w = cam_rot.apply(self.local_ray_dirs)

        depth = np.full(self.local_ray_dirs.shape[0], self.dc.max_distance, dtype=np.float64)
        geom_id = np.array([-1], dtype=np.int32)
        for idx, ray_dir_w in enumerate(ray_dirs_w):
            geom_id[0] = -1
            ray_dist = mujoco.mj_ray(
                self.model,
                data,
                cam_pos,
                ray_dir_w,
                self.geomgroup,
                flg_static=1,
                bodyexclude=self.parent_body_id,
                geomid=geom_id,
            )
            if self.dc.min_distance <= ray_dist <= self.dc.max_distance:
                depth[idx] = ray_dist * self.local_ray_plane_scale[idx]

        depth = np.nan_to_num(depth, nan=self.dc.max_distance, posinf=self.dc.max_distance, neginf=self.dc.max_distance)
        depth = np.clip(depth, self.dc.min_distance, self.dc.max_distance)

        if not self._printed_pose:
            print(
                "[DEPTH] MuJoCo ray camera pose "
                f"pos={np.round(cam_pos, 4).tolist()} "
                f"forward={np.round(forward_w, 4).tolist()}"
            )
            print(
                "[DEPTH] IsaacLab-aligned ray pattern: "
                f"shape={self.dc.height}x{self.dc.width}, "
                f"hfov={self.dc.target_horizontal_fov_deg:.1f}, "
                f"vfov={self.dc.vertical_fov_deg:.1f}, "
                "indexing=xy, flatten=row-major, output=(1,18,32)."
            )
            self._printed_pose = True

        return depth.reshape(self.dc.height, self.dc.width)

    def _crop(self, image: np.ndarray) -> np.ndarray:
        up, down, left, right = self.dc.crop_region
        end_down = image.shape[0] - down
        end_right = image.shape[1] - right
        return image[up:end_down, left:end_right]

    def _process_depth(self, depth_m: np.ndarray) -> np.ndarray:
        cropped = self._crop(depth_m)
        if self.dc.apply_gaussian_blur:
            cropped = self._gaussian_blur(cropped)
        return self._normalize_depth(cropped).astype(np.float32, copy=False)

    def _normalize_depth(self, depth_m: np.ndarray) -> np.ndarray:
        depth_m = np.clip(depth_m, self.dc.min_distance, self.dc.max_distance)
        depth_norm = (depth_m - self.dc.depth_range[0]) / (self.dc.depth_range[1] - self.dc.depth_range[0])
        return np.clip(depth_norm, 0.0, 1.0)

    def _gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        kernel_size = int(self.dc.blur_kernel_size)
        if kernel_size <= 1:
            return image
        if kernel_size % 2 == 0:
            kernel_size += 1
        radius = kernel_size // 2
        coords = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel_1d = np.exp(-(coords ** 2) / (2.0 * self.dc.blur_sigma ** 2))
        kernel_1d /= np.sum(kernel_1d)
        kernel = np.outer(kernel_1d, kernel_1d)
        padded = np.pad(image, ((radius, radius), (radius, radius)), mode="reflect")
        blurred = np.zeros_like(image, dtype=np.float64)
        for r in range(kernel_size):
            for c in range(kernel_size):
                blurred += kernel[r, c] * padded[r:r + image.shape[0], c:c + image.shape[1]]
        return blurred

    def _show(self, name: str, image: np.ndarray) -> None:
        image_u8 = (image * 255.0).astype(np.uint8)
        if self.dc.window_scale > 1:
            image_u8 = self.cv2.resize(
                image_u8,
                (image_u8.shape[1] * self.dc.window_scale, image_u8.shape[0] * self.dc.window_scale),
                interpolation=self.cv2.INTER_NEAREST,
            )
        self.cv2.imshow(name, image_u8)
        self.cv2.waitKey(1)

    def close(self) -> None:
        if self._closed:
            return
        if self.cv2 is not None:
            for name in ("mujoco_depth_raw", "mujoco_depth_processed"):
                try:
                    self.cv2.destroyWindow(name)
                except Exception:
                    pass
        self._closed = True


def get_follow_camera_body_id(model, cfg) -> int:
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        cfg.camera.track_body_name,
    )

    if body_id < 0:
        raise RuntimeError(
            f"Cannot find body '{cfg.camera.track_body_name}' for follow camera."
        )
    return body_id


def setup_follow_camera_view(cam, cfg, body_id: int) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = body_id
    cam.distance = cfg.camera.distance
    cam.elevation = cfg.camera.elevation
    cam.azimuth = cfg.camera.relative_azimuth


def setup_follow_camera(model, viewer, cfg):
    body_id = get_follow_camera_body_id(model, cfg)
    setup_follow_camera_view(viewer.cam, cfg, body_id)
    return body_id


def setup_world_camera_view(cam, cfg) -> None:
    """Fixed top-down overview camera for the full maze."""
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array(cfg.camera.world_lookat, dtype=np.float64)
    cam.distance = cfg.camera.world_distance
    cam.elevation = cfg.camera.world_elevation
    cam.azimuth = cfg.camera.world_azimuth


def setup_world_camera(viewer, cfg) -> None:
    setup_world_camera_view(viewer.cam, cfg)


def configure_scene_option(vopt) -> None:
    vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    vopt.geomgroup[1] = True   # terrain, walls, obstacles
    vopt.geomgroup[2] = True   # robot visual mesh
    vopt.geomgroup[3] = False  # robot collision
    vopt.geomgroup[5] = False  # height scan markers
    vopt.sitegroup[:] = True   # start, goal, origin markers


def configure_viewer_visuals(viewer) -> None:
    configure_scene_option(viewer.vopt)


def yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    """Return yaw angle in radians from MuJoCo quaternion [w, x, y, z]."""
    w, x, y, z = quat_wxyz

    # yaw around world z
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def update_follow_camera_view(cam, data, cfg, body_id) -> None:
    """Update camera to follow robot position and optionally rotate with robot yaw."""
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = body_id
    cam.distance = cfg.camera.distance
    cam.elevation = cfg.camera.elevation

    offset = np.array(cfg.camera.lookat_offset, dtype=np.float64)
    cam.lookat[:] = data.xpos[body_id] + offset

    if cfg.camera.rotate_with_robot:
        # MuJoCo root quaternion: [w, x, y, z]
        base_quat = data.qpos[3:7]
        yaw = yaw_from_quat_wxyz(base_quat)
        yaw_deg = np.degrees(yaw)

        # relative_azimuth=0 follows behind the robot and looks forward.
        cam.azimuth = yaw_deg + cfg.camera.relative_azimuth
    else:
        cam.azimuth = cfg.camera.relative_azimuth


def render_main_viewer(viewer) -> None:
    """Restore the GLFW context before rendering after MuJoCo offscreen depth renders."""
    try:
        import glfw  # type: ignore

        if getattr(viewer, "window", None) is not None:
            glfw.make_context_current(viewer.window)
    except Exception:
        pass
    viewer.render()


class MultiViewVideoRecorder:
    """Record world overview, third-person follow, and depth camera panels."""

    def __init__(
        self,
        model: mujoco.MjModel,
        cfg: Sim2SimConfig,
        depth_camera: MujocoDepthCamera,
        follow_body_id: int,
    ):
        self.model = model
        self.cfg = cfg
        self.vc = cfg.video
        self.depth_camera = depth_camera
        self.follow_body_id = follow_body_id

        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise RuntimeError("Video recording requires OpenCV (cv2).") from exc
        self.cv2 = cv2

        os.makedirs(os.path.dirname(os.path.abspath(self.vc.output_path)), exist_ok=True)

        self.panel_width = max(1, int(self.vc.panel_width))
        self.panel_height = max(1, int(self.vc.panel_height))
        self.fps = max(1.0, float(self.vc.fps))
        self.frame_interval_steps = max(1, int(round(1.0 / (self.fps * cfg.sim.dt))))

        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), self.panel_width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), self.panel_height)

        self.scene_option = mujoco.MjvOption()
        configure_scene_option(self.scene_option)

        self.world_cam = mujoco.MjvCamera()
        setup_world_camera_view(self.world_cam, cfg)
        self.follow_cam = mujoco.MjvCamera()
        setup_follow_camera_view(self.follow_cam, cfg, follow_body_id)

        self.world_renderer = mujoco.Renderer(
            model,
            height=self.panel_height,
            width=self.panel_width,
        )
        self.follow_renderer = mujoco.Renderer(
            model,
            height=self.panel_height,
            width=self.panel_width,
        )

        frame_size = (self.panel_width * 3, self.panel_height)
        fourcc = self.cv2.VideoWriter_fourcc(*self.vc.codec)
        self.writer = self.cv2.VideoWriter(self.vc.output_path, fourcc, self.fps, frame_size)
        if not self.writer.isOpened():
            self.world_renderer.close()
            self.follow_renderer.close()
            raise RuntimeError(f"Cannot open video writer for '{self.vc.output_path}'.")

        self.frames_written = 0
        print(
            "[VIDEO] recording multi-view video "
            f"path={self.vc.output_path}, fps={self.fps:.1f}, "
            f"codec={self.vc.codec}, "
            f"panel={self.panel_width}x{self.panel_height}, "
            f"every={self.frame_interval_steps} sim steps"
        )

    def maybe_write(self, sim_step: int, data: mujoco.MjData) -> None:
        if sim_step % self.frame_interval_steps != 0:
            return
        self.write(data)

    def write(self, data: mujoco.MjData) -> None:
        self.world_renderer.update_scene(data, camera=self.world_cam, scene_option=self.scene_option)
        world_rgb = self.world_renderer.render()

        update_follow_camera_view(self.follow_cam, data, self.cfg, self.follow_body_id)
        self.follow_renderer.update_scene(data, camera=self.follow_cam, scene_option=self.scene_option)
        follow_rgb = self.follow_renderer.render()

        self.depth_camera.update(data)
        depth_rgb = self._depth_to_rgb(self.depth_camera.raw_normalized_image())

        panels = [
            self._label(world_rgb, "World overview"),
            self._label(follow_rgb, "Third-person forward"),
            self._label(depth_rgb, "Depth camera"),
        ]
        frame_rgb = np.concatenate(panels, axis=1)
        frame_bgr = self.cv2.cvtColor(frame_rgb, self.cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)
        self.frames_written += 1

    def _depth_to_rgb(self, depth: np.ndarray) -> np.ndarray:
        depth_u8 = (np.clip(depth, 0.0, 1.0) * 255.0).astype(np.uint8)
        depth_u8 = self.cv2.resize(
            depth_u8,
            (self.panel_width, self.panel_height),
            interpolation=self.cv2.INTER_NEAREST,
        )
        return np.repeat(depth_u8[:, :, None], 3, axis=2)

    def _label(self, image_rgb: np.ndarray, text: str) -> np.ndarray:
        if image_rgb.shape[0] != self.panel_height or image_rgb.shape[1] != self.panel_width:
            image_rgb = self.cv2.resize(
                image_rgb,
                (self.panel_width, self.panel_height),
                interpolation=self.cv2.INTER_AREA,
            )
        image_rgb = image_rgb.copy()
        self.cv2.rectangle(image_rgb, (8, 8), (310, 42), (0, 0, 0), thickness=-1)
        self.cv2.putText(
            image_rgb,
            text,
            (18, 32),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=self.cv2.LINE_AA,
        )
        return image_rgb

    def close(self) -> None:
        self.writer.release()
        self.world_renderer.close()
        self.follow_renderer.close()
        print(f"[VIDEO] saved {self.frames_written} frames to {self.vc.output_path}")


# -----------------------------------------------------------------------------
# Main sim loop
# -----------------------------------------------------------------------------

def run_mujoco(policy: torch.jit.ScriptModule, cfg: Sim2SimConfig) -> None:
    print_config_summary(cfg)

    model = mujoco.MjModel.from_xml_path(cfg.sim.mujoco_model_path)
    model.opt.timestep = cfg.sim.dt
    data = mujoco.MjData(model)

    start_pos = get_site_pos(model, data, "start_marker")
    # Use the start marker from the XML for the initial xy position.
    data.qpos[0] = start_pos[0]
    data.qpos[1] = start_pos[1]
    data.qpos[2] = cfg.robot.initial_base_height
    data.qpos[7:7 + cfg.robot.num_joint_actions] = cfg.robot.default_joint_pos_mujoco
    mujoco.mj_forward(model, data)

    navigation_goal_xy: Optional[np.ndarray] = None
    if cfg.navigation.enabled:
        if cfg.navigation.goal_xy is None:
            goal_pos = get_site_pos(model, data, cfg.navigation.goal_site_name)
            navigation_goal_xy = goal_pos[:2].astype(np.float64)
        else:
            navigation_goal_xy = np.array(cfg.navigation.goal_xy, dtype=np.float64)
            goal_site_pos = np.array([navigation_goal_xy[0], navigation_goal_xy[1], 0.08], dtype=np.float64)
            set_site_pos(model, cfg.navigation.goal_site_name, goal_site_pos)
            mujoco.mj_forward(model, data)
        print(f"[NAV] goal_xy={np.round(navigation_goal_xy, 3).tolist()}")

    world_viewer: Optional[mujoco_viewer.MujocoViewer] = None
    follow_viewer: Optional[mujoco_viewer.MujocoViewer] = None
    follow_camera_body_id = None
    if cfg.sim.render:
        world_viewer = mujoco_viewer.MujocoViewer(
            model,
            data,
            title="sim2sim_world_overview",
            width=1100,
            height=900,
        )
        setup_world_camera(world_viewer, cfg)
        configure_viewer_visuals(world_viewer)

        if cfg.camera.enable_follow_camera:
            follow_viewer = mujoco_viewer.MujocoViewer(
                model,
                data,
                title="sim2sim_third_person_follow",
                width=900,
                height=700,
            )
            follow_camera_body_id = setup_follow_camera(model, follow_viewer, cfg)
            configure_viewer_visuals(follow_viewer)

    depth_camera: Optional[MujocoDepthCamera] = None
    if cfg.depth_camera.enabled:
        depth_camera = MujocoDepthCamera(model, cfg)

    video_recorder: Optional[MultiViewVideoRecorder] = None
    recording_goal_xy: Optional[np.ndarray] = None
    if cfg.video.enabled:
        if depth_camera is None:
            depth_camera = MujocoDepthCamera(model, cfg)
        if follow_camera_body_id is None:
            follow_camera_body_id = get_follow_camera_body_id(model, cfg)
        if cfg.video.stop_on_goal:
            if cfg.navigation.goal_xy is None:
                goal_pos = get_site_pos(model, data, cfg.navigation.goal_site_name)
                recording_goal_xy = goal_pos[:2].astype(np.float64)
            else:
                recording_goal_xy = np.array(cfg.navigation.goal_xy, dtype=np.float64)
        video_recorder = MultiViewVideoRecorder(
            model=model,
            cfg=cfg,
            depth_camera=depth_camera,
            follow_body_id=follow_camera_body_id,
        )

    navigation_policy = None
    navigation_runtime: Optional[NavigationRuntime] = None
    nav_policy_device = torch.device(cfg.sim.policy_device)
    if cfg.navigation.enabled:
        if depth_camera is None:
            raise RuntimeError("Navigation requires the MuJoCo depth camera. Enable cfg.depth_camera.enabled.")
        navigation_policy, _ = load_navigation_policy(cfg)
        navigation_runtime = NavigationRuntime(cfg)

    command = Command()
    joystick = None
    if cfg.sim.use_joystick and not cfg.navigation.enabled:
        if JoyStickController is None:
            print("[WARN] joystick.py not found; falling back to zero command.")
        else:
            joystick = JoyStickController()
            joystick.start()
    elif cfg.sim.use_joystick and cfg.navigation.enabled:
        print("[NAV] navigation enabled; joystick velocity command is ignored.")

    policy.eval()
    policy_device = torch.device(cfg.sim.policy_device)
    policy.to(policy_device)

    gait = GaitState(cfg)
    previous_action_isaac = np.zeros(cfg.num_actions, dtype=np.float32)
    target_q_mujoco = cfg.robot.default_joint_pos_mujoco.copy()
    target_dq_mujoco = np.zeros(cfg.robot.num_joint_actions, dtype=np.float64)

    obs_history = deque(maxlen=cfg.robot.actor_obs_history_length)
    for _ in range(cfg.robot.actor_obs_history_length):
        obs_history.append(np.zeros(cfg.num_single_obs, dtype=np.float32))

    height_scan = np.zeros(cfg.height_scan.num_rays, dtype=np.float32)
    policy_step_count = 0
    total_steps = int(cfg.sim.sim_duration_s / cfg.sim.dt)

    try:
        for sim_step in tqdm(range(total_steps), desc="sim2sim"):
            if joystick is not None:
                joystick._update_cmd_vel()
                cmd_vel = joystick.get_cmd_vel()
                command.vx = float(cmd_vel["vx"])
                command.vy = float(cmd_vel["vy"])
                command.dyaw = float(cmd_vel["wz"])

            q_mujoco, dq_mujoco, _, lin_vel_body, ang_vel_body, projected_gravity = get_mujoco_state(
                data, cfg.robot.num_joint_actions
            )

            if sim_step % cfg.sim.decimation == 0:
                # Match training order: in G1Env.step(), gait is updated before observations are computed.
                gait.step_phase()

                if cfg.navigation.enabled:
                    goal_dist = float(np.linalg.norm(navigation_goal_xy - data.qpos[:2]))
                    if goal_dist <= cfg.navigation.goal_reached_threshold:
                        if not navigation_runtime.goal_reached:
                            print(f"[NAV] goal reached: dist={goal_dist:.3f}")
                        navigation_runtime.goal_reached = True
                        if cfg.navigation.stop_on_goal:
                            if video_recorder is not None:
                                video_recorder.write(data)
                                video_recorder.close()
                                video_recorder = None
                            break

                    if (
                        not navigation_runtime.goal_reached
                        and policy_step_count % cfg.navigation.navigation_decimation == 0
                    ):
                        depth_obs = depth_camera.update(data)
                        nav_current_obs = build_navigation_current_obs(
                            cfg=cfg,
                            data=data,
                            nav_runtime=navigation_runtime,
                            lin_vel_body=lin_vel_body,
                            ang_vel_body=ang_vel_body,
                            projected_gravity=projected_gravity,
                            goal_xy=navigation_goal_xy,
                        )
                        navigation_runtime.history.append(nav_current_obs)
                        nav_input = np.concatenate(
                            [navigation_runtime.history.flatten(), depth_obs.reshape(-1).astype(np.float32)]
                        ).astype(np.float32)
                        nav_input = np.clip(
                            nav_input,
                            -cfg.navigation.clip_observations,
                            cfg.navigation.clip_observations,
                        )
                        nav_action = infer_navigation_action(navigation_policy, nav_input, nav_policy_device)
                        if nav_action.shape[0] != 3:
                            raise RuntimeError(
                                f"Navigation policy output dim mismatch: got {nav_action.shape[0]}, expected 3."
                            )
                        navigation_runtime.update_from_action(nav_action, cfg)

                        nav_step = policy_step_count // cfg.navigation.navigation_decimation
                        if cfg.navigation.print_interval > 0 and nav_step % cfg.navigation.print_interval == 0:
                            print(
                                "[NAV] "
                                f"step={nav_step} "
                                f"goal_dist={goal_dist:.2f} "
                                f"action={np.round(nav_action, 3).tolist()} "
                                f"cmd={np.round(navigation_runtime.nav_cmd, 3).tolist()}"
                            )

                    nav_substep = policy_step_count % cfg.navigation.navigation_decimation
                    nav_cmd = navigation_runtime.interpolated_command(nav_substep, cfg)
                    command.vx = float(nav_cmd[0])
                    command.vy = float(nav_cmd[1])
                    command.dyaw = float(nav_cmd[2])

                single_obs = build_single_actor_obs(
                    cfg=cfg,
                    q_mujoco=q_mujoco,
                    dq_mujoco=dq_mujoco,
                    ang_vel_body=ang_vel_body,
                    projected_gravity=projected_gravity,
                    command=command,
                    previous_action_isaac=previous_action_isaac,
                    gait=gait,
                )
                obs_history.append(single_obs)

                if cfg.height_scan.enabled:
                    if policy_step_count % cfg.height_scan.update_every_policy_step == 0:
                        height_scan = sample_height_scan(model, data, cfg)

                policy_input = build_policy_input(cfg, obs_history, height_scan)
                with torch.no_grad():
                    action_tensor = policy(torch.from_numpy(policy_input).to(policy_device))
                action_isaac = action_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)

                expected = cfg.num_actions
                if action_isaac.shape[0] != expected:
                    raise RuntimeError(
                        f"Policy output dim mismatch: got {action_isaac.shape[0]}, expected {expected}. "
                        f"Check enable_variable_gait_cycle={cfg.gait.enable_variable_gait_cycle}."
                    )

                action_isaac = np.clip(action_isaac, -cfg.robot.clip_actions, cfg.robot.clip_actions)
                gait.update_cycle_from_action(action_isaac)
                target_q_mujoco = action_isaac_to_target_q_mujoco(cfg, action_isaac)
                previous_action_isaac = action_isaac.copy()
                policy_step_count += 1

            tau = pd_control(
                target_q=target_q_mujoco,
                q=q_mujoco,
                kp=cfg.robot.kp_mujoco,
                target_dq=target_dq_mujoco,
                dq=dq_mujoco,
                kd=cfg.robot.kd_mujoco,
            )
            tau = np.clip(tau, -cfg.robot.tau_limit_mujoco, cfg.robot.tau_limit_mujoco)
            data.ctrl[:cfg.robot.num_joint_actions] = tau

            mujoco.mj_step(model, data)
            if video_recorder is not None:
                if cfg.video.stop_on_goal:
                    recording_goal_dist = float(np.linalg.norm(recording_goal_xy - data.qpos[:2]))
                    if recording_goal_dist <= cfg.navigation.goal_reached_threshold:
                        video_recorder.write(data)
                        video_recorder.close()
                        video_recorder = None
                        print(f"[VIDEO] recording stopped at goal: dist={recording_goal_dist:.3f}")
                    else:
                        video_recorder.maybe_write(sim_step, data)
                else:
                    video_recorder.maybe_write(sim_step, data)
            if world_viewer is not None:
                render_main_viewer(world_viewer)
            if follow_viewer is not None:
                update_follow_camera_view(follow_viewer.cam, data, cfg, follow_camera_body_id)
                render_main_viewer(follow_viewer)
            if (
                depth_camera is not None
                and not cfg.navigation.enabled
                and sim_step % cfg.depth_camera.update_every_sim_steps == 0
            ):
                depth_camera.render(data)
    finally:
        if joystick is not None:
            joystick.stop()
        if video_recorder is not None:
            video_recorder.close()
        if depth_camera is not None:
            depth_camera.close()
        if follow_viewer is not None:
            follow_viewer.close()
        if world_viewer is not None:
            world_viewer.close()


def print_config_summary(cfg: Sim2SimConfig) -> None:
    print("\n===== Sim2Sim config summary =====")
    print(f"policy input dim       : {cfg.policy_input_dim}")
    print(f"single obs dim         : {cfg.num_single_obs}")
    print(f"history length         : {cfg.robot.actor_obs_history_length}")
    print(f"height dim             : {cfg.height_scan.num_rays}")
    print(f"navigation enabled     : {cfg.navigation.enabled}")
    if cfg.navigation.enabled:
        goal_text = "site:" + cfg.navigation.goal_site_name if cfg.navigation.goal_xy is None else str(cfg.navigation.goal_xy)
        print(f"navigation policy      : {cfg.navigation.policy_path}")
        print(f"navigation goal        : {goal_text}")
        print(f"navigation input dim   : {cfg.nav_policy_input_dim}")
        print(f"navigation decimation  : {cfg.navigation.navigation_decimation}")
    print(f"world overview viewer  : {cfg.sim.render}")
    print(f"third-person viewer    : {cfg.sim.render and cfg.camera.enable_follow_camera}")
    print(f"third-person azimuth   : {cfg.camera.relative_azimuth:.1f}")
    if cfg.sim.render:
        print(
            "world camera           : "
            f"lookat={tuple(float(v) for v in cfg.camera.world_lookat)}, "
            f"distance={cfg.camera.world_distance:.1f}, "
            f"elevation={cfg.camera.world_elevation:.1f}, "
            f"azimuth={cfg.camera.world_azimuth:.1f}"
        )
    print(f"depth camera enabled   : {cfg.depth_camera.enabled}")
    if cfg.depth_camera.enabled:
        print(
            "depth camera           : "
            f"{cfg.depth_camera.width}x{cfg.depth_camera.height}, "
            f"processed={cfg.depth_camera.processed_height}x{cfg.depth_camera.processed_width}, "
            f"vfov={cfg.depth_camera.vertical_fov_deg:.1f}, "
            f"target_hfov={cfg.depth_camera.target_horizontal_fov_deg:.1f}, "
            f"crop={cfg.depth_camera.crop_region}, "
            f"range=({cfg.depth_camera.min_distance:.2f}, {cfg.depth_camera.max_distance:.2f}), "
            f"blur={cfg.depth_camera.apply_gaussian_blur}"
        )
    print(f"record video enabled   : {cfg.video.enabled}")
    if cfg.video.enabled:
        print(
            "record video           : "
            f"{cfg.video.output_path}, fps={cfg.video.fps:.1f}, "
            f"codec={cfg.video.codec}, "
            f"panel={cfg.video.panel_width}x{cfg.video.panel_height}, "
            f"stop_on_goal={cfg.video.stop_on_goal}"
        )
    print(f"num actions            : {cfg.num_actions}")
    print(f"variable gait cycle    : {cfg.gait.enable_variable_gait_cycle}")
    print(f"observe gait cycle     : {cfg.gait.observe_gait_cycle}")
    print(f"MuJoCo->Isaac map      : {MUJOCO_TO_ISAAC.tolist()}")
    print(f"Isaac->MuJoCo map      : {ISAAC_TO_MUJOCO.tolist()}")
    print("==================================\n")


def make_config_from_args() -> Sim2SimConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=SimConfig.mujoco_model_path)
    parser.add_argument("--policy", type=str, default=SimConfig.policy_path)
    parser.add_argument("--navigation", action="store_true", help="Enable high-level navigation policy.")
    parser.add_argument("--nav-policy", type=str, default=NavigationConfig.policy_path)
    parser.add_argument("--goal-x", type=float, default=None, help="Navigation final goal x in MuJoCo world frame.")
    parser.add_argument("--goal-y", type=float, default=None, help="Navigation final goal y in MuJoCo world frame.")
    parser.add_argument("--goal-site", type=str, default=NavigationConfig.goal_site_name)
    parser.add_argument("--goal-threshold", type=float, default=NavigationConfig.goal_reached_threshold)
    parser.add_argument("--nav-decimation", type=int, default=NavigationConfig.navigation_decimation)
    parser.add_argument("--nav-print-interval", type=int, default=NavigationConfig.print_interval)
    parser.add_argument("--stop-on-goal", action="store_true")
    parser.add_argument("--duration", type=float, default=SimConfig.sim_duration_s)
    parser.add_argument("--device", type=str, default=SimConfig.policy_device)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-joystick", action="store_true")
    parser.add_argument(
        "--multi-view",
        action="store_true",
        help="Open world overview, third-person follow, and depth camera views.",
    )
    parser.add_argument(
        "--variable-gait",
        action=argparse.BooleanOptionalAction,
        default=GaitConfig.enable_variable_gait_cycle,
        help="Must match the low-level policy training config.",
    )
    parser.add_argument(
        "--observe-gait-cycle",
        action=argparse.BooleanOptionalAction,
        default=GaitConfig.observe_gait_cycle,
        help="Must match the low-level policy training config.",
    )
    parser.add_argument("--height-update-every", type=int, default=1)
    parser.add_argument("--no-height", action="store_true")
    parser.add_argument("--no-height-transpose", action="store_true")
    parser.add_argument(
        "--camera-follow",
        action=argparse.BooleanOptionalAction,
        default=CameraConfig.enable_follow_camera,
        help="Open an additional third-person camera that follows robot pelvis.",
    )
    parser.add_argument("--third-person-view", action="store_true", help="Alias for enabling the follow viewer.")
    parser.add_argument(
        "--third-person-azimuth",
        type=float,
        default=CameraConfig.relative_azimuth,
        help="Relative third-person azimuth. 0 follows behind and looks forward.",
    )
    parser.add_argument("--third-person-distance", type=float, default=CameraConfig.distance)
    parser.add_argument("--world-distance", type=float, default=CameraConfig.world_distance)
    parser.add_argument("--world-elevation", type=float, default=CameraConfig.world_elevation)
    parser.add_argument("--world-azimuth", type=float, default=CameraConfig.world_azimuth)
    parser.add_argument("--show-depth-camera", action="store_true", help="Show MuJoCo depth camera windows.")
    parser.add_argument("--record-video", action="store_true", help="Save a side-by-side world/follow/depth video.")
    parser.add_argument("--stop-recording-on-goal", action="store_true", help="Close the video once the robot reaches the goal.")
    parser.add_argument("--video-path", type=str, default=VideoConfig.output_path)
    parser.add_argument("--video-fps", type=float, default=VideoConfig.fps)
    parser.add_argument("--video-codec", type=str, default=VideoConfig.codec, help="FourCC codec, e.g. mp4v or MJPG.")
    parser.add_argument("--video-panel-width", type=int, default=VideoConfig.panel_width)
    parser.add_argument("--video-panel-height", type=int, default=VideoConfig.panel_height)
    parser.add_argument("--depth-update-every", type=int, default=DepthCameraConfig.update_every_sim_steps)
    parser.add_argument("--depth-window-scale", type=int, default=DepthCameraConfig.window_scale)
    parser.add_argument("--depth-width", type=int, default=DepthCameraConfig.width)
    parser.add_argument("--depth-height", type=int, default=DepthCameraConfig.height)
    parser.add_argument("--depth-vfov", type=float, default=DepthCameraConfig.vertical_fov_deg)
    parser.add_argument("--depth-hfov", type=float, default=DepthCameraConfig.target_horizontal_fov_deg)
    parser.add_argument("--depth-no-blur", action="store_true", help="Disable IsaacLab-style Gaussian blur.")
    parser.add_argument("--depth-no-raw", action="store_true", help="Hide raw 64x36 depth window.")
    parser.add_argument("--depth-no-processed", action="store_true", help="Hide cropped 18x32 depth window.")
    args = parser.parse_args()

    cfg = Sim2SimConfig()
    cfg.sim.mujoco_model_path = args.xml
    cfg.sim.policy_path = args.policy
    cfg.sim.sim_duration_s = args.duration
    cfg.sim.policy_device = args.device
    cfg.sim.render = not args.no_render
    cfg.sim.use_joystick = not args.no_joystick

    cfg.navigation.enabled = args.navigation
    cfg.navigation.policy_path = args.nav_policy
    cfg.navigation.goal_site_name = args.goal_site
    cfg.navigation.goal_reached_threshold = max(0.0, float(args.goal_threshold))
    if args.goal_x is not None or args.goal_y is not None:
        if args.goal_x is None or args.goal_y is None:
            parser.error("--goal-x and --goal-y must be provided together.")
        cfg.navigation.goal_xy = (float(args.goal_x), float(args.goal_y))
    cfg.navigation.navigation_decimation = max(1, args.nav_decimation)
    cfg.navigation.print_interval = int(args.nav_print_interval)
    cfg.navigation.stop_on_goal = args.stop_on_goal

    cfg.gait.enable_variable_gait_cycle = args.variable_gait
    cfg.gait.observe_gait_cycle = args.observe_gait_cycle

    cfg.height_scan.enabled = not args.no_height
    cfg.height_scan.update_every_policy_step = max(1, args.height_update_every)
    cfg.height_scan.transpose_to_isaac_order = not args.no_height_transpose

    cfg.camera.enable_follow_camera = args.camera_follow or args.third_person_view or args.multi_view
    cfg.camera.relative_azimuth = float(args.third_person_azimuth)
    cfg.camera.distance = max(0.2, float(args.third_person_distance))
    cfg.camera.world_distance = float(args.world_distance)
    cfg.camera.world_elevation = float(args.world_elevation)
    cfg.camera.world_azimuth = float(args.world_azimuth)

    show_depth_camera = args.show_depth_camera or args.multi_view
    cfg.video.enabled = bool(args.record_video)
    cfg.video.output_path = args.video_path
    cfg.video.fps = float(args.video_fps)
    if len(args.video_codec) != 4:
        parser.error("--video-codec must be exactly 4 characters, e.g. mp4v or MJPG.")
    cfg.video.codec = args.video_codec
    cfg.video.panel_width = max(1, int(args.video_panel_width))
    cfg.video.panel_height = max(1, int(args.video_panel_height))
    cfg.video.stop_on_goal = bool(args.stop_recording_on_goal)

    cfg.depth_camera.enabled = show_depth_camera or args.navigation or cfg.video.enabled
    cfg.depth_camera.update_every_sim_steps = max(1, args.depth_update_every)
    cfg.depth_camera.window_scale = max(1, args.depth_window_scale)
    cfg.depth_camera.width = max(1, args.depth_width)
    cfg.depth_camera.height = max(1, args.depth_height)
    cfg.depth_camera.vertical_fov_deg = float(args.depth_vfov)
    cfg.depth_camera.target_horizontal_fov_deg = float(args.depth_hfov)
    cfg.depth_camera.apply_gaussian_blur = not args.depth_no_blur
    cfg.depth_camera.show_raw = show_depth_camera and not args.depth_no_raw
    cfg.depth_camera.show_processed = show_depth_camera and not args.depth_no_processed
    return cfg


if __name__ == "__main__":
    sim2sim_cfg = make_config_from_args()
    loaded_policy = torch.jit.load(sim2sim_cfg.sim.policy_path, map_location=sim2sim_cfg.sim.policy_device)
    run_mujoco(loaded_policy, sim2sim_cfg)
