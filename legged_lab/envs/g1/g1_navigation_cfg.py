from __future__ import annotations

from typing import Literal

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from legged_lab.mdp import navigation_rewards as navigation_mdp
from legged_lab.envs.g1.g1_locomotion_cfg import G1LocomotionEnvCfg
from legged_lab.assets.g1_description.g1 import G1_CFG 
from legged_lab.envs.base.base_config import (
    BaseSceneCfg,
    HeightScannerCfg,
    LfootHeightScannerCfg,
    RfootHeightScannerCfg,
    NoisyCameraCfg,
)
from legged_lab.terrains import PLANNING_TERRAIN_CFG  



@configclass
class NavigationObservationCfg:
    """High-level navigation observation configuration.

    This is independent from the low-level locomotion actor/critic observation history.
    The low-level policy still uses G1LocomotionEnv.compute_observations().
    """

    # Independent high-level history buffers.
    actor_obs_history_length: int = 4
    critic_obs_history_length: int = 4


    # Navigation-side sensor usage. These do not create sensors by themselves.
    # The depth image uses scene.noisy_camera, following the InstinctLab ray-cast camera path.
    use_height_scan: bool = False
    use_lidar: bool = False
    use_depth_camera: bool = True

    # Sensor preprocessing for navigation observations.
    height_scan_min: float = 0.0
    height_scan_max: float = 1.0
    lidar_clip: float = 10.0
    depth_data_type: str = "distance_to_image_plane_noised"
    depth_clip_min: float = 0.0
    depth_clip_max: float = 1.0

@configclass
class NavigationObsScalesCfg:
    """Navigation-level observation scales."""

    goal_xy: float = 0.1
    goal_distance: float = 0.1
    heading_error: float = 1.0
    previous_command: float = 1.0
    base_velocity: float = 1.0
    projected_gravity: float = 1.0
    path_xy: float = 0.1
    path_distance: float = 0.1
    path_lateral_error: float = 0.2
    path_metric_length: float = 0.1
    path_metric_turns: float = 0.1


@configclass
class NavigationNoiseCfg:
    """Navigation-only observation noise.

    This is separate from cfg.noise used by the low-level locomotion observation path.
    Noise is applied only to the high-level navigation actor observation.
    The navigation critic observation uses the same deployable fields without actor-side noise.
    """

    add_noise: bool = True

    # Noise for low-dimensional navigation observations.
    goal_xy: float = 0.02
    goal_distance: float = 0.02
    heading_error: float = 0.02
    previous_command: float = 0.01
    base_velocity: float = 0.02
    projected_gravity: float = 0.01

    # Noise for navigation-side sensors.
    height_scan: float = 0.02
    lidar: float = 0.0
    depth: float = 0.05


@configclass
class NavigationTaskCfg:
    """High-level navigation task settings."""

    # Frozen low-level locomotion policy.
    # "auto" searches exported locomotion policies and loads the newest TorchScript policy.pt.
    locomotion_policy_path: str = "auto"

    # Frequency relation:
    # low-level locomotion policy frequency = 1 / (sim.dt * sim.decimation)
    # navigation frequency = locomotion frequency / navigation_decimation
    # Default is 1/10 of the low-level control sampling frequency.
    navigation_decimation: int = 5

    # Interpolation from previous high-level command to current high-level command.
    command_interpolation: Literal["linear", "hold"] = "linear"
    command_smoothing_alpha: float = 0.6

    # High-level action is assumed in [-1, 1] and scaled into velocity commands.
    action_scale_lin_x: float = 1.0
    action_scale_lin_y: float = 0.35
    action_scale_ang_z: float = 0.9

    # Final command clipping range. Keep compatible with locomotion training command ranges.
    lin_vel_x_range: tuple[float, float] = (-0.3, 0.8)
    lin_vel_y_range: tuple[float, float] = (-0.3, 0.3)
    ang_vel_z_range: tuple[float, float] = (-1.0, 1.0)

    # Terrain-level curriculum trigger.
    # Success first weakens privileged path guidance and increases path difficulty. Terrain level
    # increases only after the path-aware curriculum is mastered on the current terrain.
    terrain_level_up_on_path_curriculum: bool = True

    # Demotion policy:
    # Failures first strengthen privileged path guidance and reduce path difficulty. Terrain is
    # demoted only when the path curriculum is already easy and the same env still fails repeatedly.
    terrain_level_down_on_failure: bool = True
    terrain_demote_failures: int = 4
    terrain_demote_path_level_threshold: int = 0

    # Goal success / termination.
    goal_reached_threshold: float = 0.6
    terminate_on_goal: bool = True

    # Start-goal sampling and path curriculum.
    randomize_start: bool = True
    num_path_difficulty_levels: int = 6
    num_path_guidance_levels: int = 4
    # Keep a stable route per env and grow the current task endpoint along that route.
    # The actor still observes only the current task endpoint, not intermediate waypoints.
    fixed_route_prefix_curriculum: bool = True
    fixed_route_candidate_level: int = -1
    fixed_route_candidate_count: int = 30
    path_reference_stride_levels: tuple[int, ...] = (1, 2, 4, 1_000_000)
    path_guidance_reward_scales: tuple[float, ...] = (1.0, 0.65, 0.3, 0.0)
    path_lookahead_distance: float = 1.0
    max_navigation_waypoints: int = 64
    path_start_goal_clearance: float = 0.45
    path_start_goal_sample_tries: int = 24
    path_guidance_level_up_on_success: bool = True
    path_guidance_level_down_on_failure: bool = True
    path_level_up_on_guidance_mastery: bool = True
    path_level_down_on_failure: bool = True
    path_guidance_promote_successes: int = 2
    path_promote_successes: int = 2
    curriculum_promotion_cooldown_episodes: int = 1
    path_guidance_demote_failures: int = 4
    path_demote_failures: int = 3
    post_path_promotion_guidance_level: int = 0

    # More conservative thresholds: stable but terrain levels may advance slowly.
    # terrain_promote_path_level_threshold: int = 5
    # terrain_promote_guidance_level_threshold: int = 3
    # terrain_promote_successes: int = 4

    # Terrain promotion is staged after guidance/path curricula. The path-level threshold is tightened
    # during training by set_learning_iteration().
    terrain_promote_path_level_threshold: int = 2
    terrain_promote_path_level_threshold_start: int = 2
    terrain_promote_path_level_threshold_final: int = -1
    terrain_promote_tighten_start_fraction: float = 0.02
    terrain_promote_tighten_end_fraction: float = 0.10
    terrain_promote_guidance_level_threshold: int = 2
    terrain_promote_successes: int = 2

    post_terrain_promotion_path_level: int = 0
    min_start_goal_separation: float = 1.0
    start_goal_border_margin: float = 1.5
    flat_patch_sampling_name: str = "navigation"
    reserved_clear_zone_margin: float = 0.5

    # Debug / logging.
    debug_vis: bool = True
    log_success_rate: bool = True


@configclass
class NavigationRewardCfg:
    """Navigation rewards with privileged path guidance and deployable actor observations."""

    undesired_contacts = RewTerm(
        func=navigation_mdp.nav_undesired_collision,
        weight=-2.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*knee.*", "pelvis", ".*hip_yaw.*", ".*hip_roll.*", ".*hip_pitch.*"]
            ),
            "threshold": 1.0,
        },
    )

    termination_penalty = RewTerm(func=navigation_mdp.nav_is_terminated, weight=-5)

    # Privileged training guidance. The actor does not observe the path; the reward scale is
    # annealed by path_guidance_levels until the final level uses only deployable goal rewards.
    path_progress = RewTerm(
        func=navigation_mdp.nav_path_progress,
        weight=1.0,
        params={"max_progress_speed": 0.8, "allow_negative": False, "use_guidance_weight": True},
    )
    path_lateral_deviation = RewTerm(
        func=navigation_mdp.nav_path_lateral_deviation,
        weight=-0.15,
        params={"deviation_scale": 1.0, "max_deviation": 3.0, "use_guidance_weight": True},
    )

    # Deployment objective: the policy observes only the current task endpoint relative pose.
    # Keep the direct endpoint progress weak in mazes so detours are not over-penalized.
    final_goal_progress = RewTerm(
        func=navigation_mdp.nav_final_goal_progress,
        weight=0.10,
        params={"max_progress_speed": 0.6, "allow_negative": False},
    )
    final_goal_reached = RewTerm(
        func=navigation_mdp.nav_prefix_goal_reached,
        weight=8.0,
        params={"min_multiplier": 1.0, "max_multiplier": 2.0},
    )
    desired_speed = RewTerm(
        func=navigation_mdp.nav_desired_speed,
        weight=0.25,
        params={"desired_speed": 0.6, "speed_sigma": 0.3},
    )

    # Smooth high-level planning commands.
    command_rate = RewTerm(
        func=navigation_mdp.nav_command_rate,
        weight=-0.2,
        params={"lin_weight": 1.5, "yaw_weight": 2.5},
    )
    command_acceleration = RewTerm(
        func=navigation_mdp.nav_command_acceleration,
        weight=-0.2,
        params={"lin_weight": 1.5, "yaw_weight": 2.5},
    )
    command_tracking = RewTerm(
        func=navigation_mdp.nav_command_tracking_error,
        weight=-0.25,
        params={"lin_weight": 1.0, "yaw_weight": 1.5},
    )

    path_subgoal_time_overrun = RewTerm(
        func=navigation_mdp.nav_path_subgoal_time_overrun,
        weight=-0.35,
        params={
            "threshold_s": 0.0,
            "expected_speed": 0.5,
            "grace_time_s": 10.0,
            "min_time_s": 20.0,
            "ramp_time_s": 15.0,
            "max_penalty": 1.0,
        },
    )

    timeout_penalty = RewTerm(func=navigation_mdp.nav_timeout, weight=-3.5)


@configclass
class G1NavigationEnvCfg(G1LocomotionEnvCfg):
    # 基础属性
    device: str = "cuda:0" # 指定环境运行的设备
    # 场景配置
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=150.0, # 每个训练回合最长持续60s
        num_envs=4096, # 并行训练环境数量
        env_spacing=2.5, # 并行环境间距
        robot=G1_CFG, # G1_CFG
        terrain_type="generator", # 使用 “粗糙地形生成器” 生成训练地形
        terrain_generator=PLANNING_TERRAIN_CFG, #GRAVEL_TERRAINS_CFG, #使用 “粗糙地形生成器” 生成训练地形
        # terrain_type="plane",
        # terrain_generator= None,
        max_init_terrain_level=1, # 初始地形的最大难度等级
        #高度传感器配置
        height_scanner=HeightScannerCfg(
            enable_height_scan=True, # 启用高度扫描功能
            prim_body_name="pelvis", # 指定传感器安装位置
            resolution=0.1, # 扫描分辨率为0.1米/点
            size=(1.6, 1.0), # 扫描区域物理尺寸，前后方向+-0.8m，左右方向+-0.5m
            debug_vis=False, # 启用调试可视化功能
            drift_range=(-0.0, 0.0),  # (0.3, 0.3) 扫描位置随机漂移范围
        ),

        Lfoot_height_scanner=LfootHeightScannerCfg(
            enable_height_scan=True,
            prim_body_name="left_ankle_roll_link",
            resolution=0.01,
            size=(0.2, 0.08),
            debug_vis=False,
            drift_range=(-0.0, 0.0),  # (0.3, 0.3)
        ),
        Rfoot_height_scanner=RfootHeightScannerCfg(
            enable_height_scan=True,
            prim_body_name="right_ankle_roll_link",
            resolution=0.01,
            size=(0.2, 0.08),
            debug_vis=False,
            drift_range=(-0.0, 0.0),  # (0.3, 0.3)
        ),

        noisy_camera=NoisyCameraCfg(
            enable_camera=True,
            prim_body_name="pelvis",
            # Camera parameters
            width= 64,
            height=36,
            horizontal_fov_deg=100.0,
            vertical_fov_deg=85.0,
            offset_pos=(0.05366, 0.01753, 0.47387),
            offset_rot=(0.9659258262890683, 0.0, 0.25881904510252074, 0.0),
            offset_convention="world",
            min_distance=0.1,
            max_distance=4.0,
            mesh_prim_paths=["/World/ground/"],
            crop_region=(9, 9, 16, 16),
            depth_range=(0.0, 4.0),
            aux_mesh_and_link_names={
                "torso_link_23dof_rev_1_0": None,
                },
            debug_vis=False,
        ),

    )
    nav: NavigationTaskCfg = NavigationTaskCfg()
    nav_obs: NavigationObservationCfg = NavigationObservationCfg()
    nav_obs_scales: NavigationObsScalesCfg = NavigationObsScalesCfg()
    nav_noise: NavigationNoiseCfg = NavigationNoiseCfg()
    reward: NavigationRewardCfg = NavigationRewardCfg()


@configclass
class G1NavigationDepthCnnGruPolicyCfg(RslRlPpoActorCriticCfg):
    """Policy config for navigation CNN+GRU actor-critic."""

    class_name: str = "ActorCriticDepthCnnGru"
    init_noise_std: float = 0.7
    noise_std_type: Literal["scalar", "log"] = "log"
    activation: str = "elu"

    # Fusion MLP after concatenating proprio/task and depth-GRU features.
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]

    # Flattened depth observations are reshaped to this image shape inside the network.
    depth_shape: tuple[int, int, int] = (1, 18, 32)

    # Depth CNN encoders.
    actor_cnn_channels: tuple[int, ...] = (16, 32, 64)
    critic_cnn_channels: tuple[int, ...] = (16, 32, 64)
    actor_depth_feature_dim: int = 256
    critic_depth_feature_dim: int = 256

    # Recurrent depth-feature encoders.
    actor_gru_hidden_dim: int = 256
    critic_gru_hidden_dim: int = 256
    actor_gru_num_layers: int = 1
    critic_gru_num_layers: int = 1

    # Low-dimensional proprio/task-history encoders.
    actor_proprio_hidden_dims: tuple[int, ...] = (256, 256)
    critic_proprio_hidden_dims: tuple[int, ...] = (256, 256)


@configclass
class G1NavigationAgentCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for high-level navigation policy.

    One rollout step corresponds to one high-level navigation step, not one low-level locomotion step.
    """

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 16
    max_iterations = 100000
    empirical_normalization = False

    policy: G1NavigationDepthCnnGruPolicyCfg = G1NavigationDepthCnnGruPolicyCfg()

    # 策略网络配置
    # policy = RslRlPpoActorCriticCfg(
    #     class_name="ActorCritic", # Actor-Critic框架
    #     init_noise_std=1.0, # 初始动作噪声标准差，用于训练初期探索，噪声越大，探索性越强
    #     noise_std_type="log", # 噪声类型 标量 "scalar"
    #     actor_hidden_dims=[512, 256, 128], # actor网络隐藏层维度
    #     critic_hidden_dims=[512, 256, 128], # critic网络隐藏层维度
    #     activation="elu", # 激活函数使用Elu
    # )


    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )

    clip_actions = None
    save_interval = 500
    runner_class_name = "OnPolicyRunner"
    experiment_name = "g1_navigation"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "g1_navigation"
    wandb_project = "g1_navigation"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"


g1NavigationEnvCfg = G1NavigationEnvCfg
g1NavigationAgentCfg = G1NavigationAgentCfg
