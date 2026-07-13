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

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.utils import configclass
import isaaclab.envs.mdp as isaac_mdp
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

from legged_lab.mdp import locomotion_rewards as locomotion_mdp
from legged_lab.assets.g1_description.g1 import G1_CFG 
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    LfootHeightScannerCfg,
    RfootHeightScannerCfg,
    NoisyCameraCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import ROUGH_TERRAINS_CFG, PLANNING_TERRAIN_CFG  



@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.43
    gait_air_ratio_r: float = 0.43
    gait_phase_offset_l: float = 0.43
    gait_phase_offset_r: float = 0.93
    gait_cycle: float = 0.8

    enable_variable_gait_cycle: bool = False # 实施变步频
    gait_cycle_scale_range: tuple = (0.75, 1.25)
    gait_cycle_smoothing: float = 0.1
    observe_gait_cycle: bool = False # 将gait_cycle添加进obs中


@configclass
class FootholdCfg:
    leg_width: float = 0.25
    world_window_radius: float = 0.2 # 落脚点规划范围半径 (m)
    gravity: float = 9.81 # 重力加速度
    z0: float = 1 # 质心初始相对高度 (m)
    enable_foothold_planner: bool = False
    plan_mode = "on_switch"

    alpha1 = 13.0 # 用于控制支撑脚与目标点的距离
    alpha2 = 2.0 # 用于控制DCM偏量
    alpha3 = 15.0 # 用于控制候选点的陡峭度
    alpha4 = 0.0 # 用于控制前后两次规划的落脚点不要相隔太远
    alpha5 = 12.0     # 边缘距离软约束权重
     
     
    edge_height_thresh = 0.03   # 判定为“边缘/高度突变”的局部高度差阈值，单位m
    edge_safe_margin = 0.04     # 脚掌外额外安全裕度，单位 m

    foot_h = 0.10 # 计算陡峭度机器人脚长度，比实际稍小
    foot_w = 0.10 # 计算陡峭度机器人脚宽度，比实际稍小

    offset_x = 0.05 # 踝关节与脚掌中心偏置，单位 m
    offset_y = 0.0 # 踝关节与脚掌中心偏置，单位 m
    offset_z = 0.0 # 踝关节与脚掌中心偏置，单位 m


@configclass
class LocomotionRewardCfg:
    ##task reward:
    # 追踪XY轴线性速度（指数奖励）
    track_lin_vel_xy_exp = RewTerm(func=locomotion_mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=locomotion_mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=locomotion_mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=locomotion_mdp.ang_vel_xy_l2, weight=-0.05)
    energy = RewTerm(func=locomotion_mdp.energy, weight=-5e-4)
    dof_acc_l2 = RewTerm(func=locomotion_mdp.joint_acc_l2, weight=-1.25e-7)
    dof_vel_l2 = RewTerm(func=locomotion_mdp.joint_vel_l2, weight=-1e-3)

    # 将原本action_rate_l2拆成两部分，关节和步频惩罚
    # action_rate_l2 = RewTerm(func=locomotion_mdp.action_rate_l2, weight=-0.005)
    joint_action_rate_l2 = RewTerm(func=locomotion_mdp.joint_action_rate_l2, weight=-0.005)
    gait_cycle_action_rate_l2 = RewTerm(func=locomotion_mdp.gait_cycle_action_rate_l2, weight=-0.02)

    undesired_contacts = RewTerm(
        func=locomotion_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*knee.*", "pelvis", ".*hip_yaw.*", ".*hip_roll.*", ".*hip_pitch.*"]
            ),
            "threshold": 1.0,
        },
    )
    body_orientation_l2 = RewTerm(
        func=locomotion_mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")}, weight=-1.5
    )
    flat_orientation_l2 = RewTerm(func=locomotion_mdp.flat_orientation_l2, weight=-1.5)
    termination_penalty = RewTerm(func=locomotion_mdp.is_terminated, weight=-200.0)
    feet_slide = RewTerm(
        func=locomotion_mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )
    feet_force = RewTerm(
        func=locomotion_mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=locomotion_mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=locomotion_mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=isaac_mdp.joint_pos_limits, weight=-4.0)
    joint_deviation_hip = RewTerm(
        func=locomotion_mdp.joint_deviation_l1,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*hip_yaw_joint",
                    ".*hip_roll_joint",
                ],
            )
        },
    )

    joint_deviation_legs = RewTerm(
        func=locomotion_mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*hip_pitch_joint",
                    ".*knee.*",
                    ".*ankle_pitch.*",
                    ".*ankle_roll.*",
                ],
            )
        },
    )

    gait_feet_frc_perio = RewTerm(func=locomotion_mdp.gait_feet_frc_perio, weight=1.0)
    gait_feet_spd_perio = RewTerm(func=locomotion_mdp.gait_feet_spd_perio, weight=1.0)
    gait_feet_frc_support_perio = RewTerm(func=locomotion_mdp.gait_feet_frc_support_perio, weight=0.6)
    

    # ankle_torque = RewTerm(func=locomotion_mdp.ankle_torque, weight=-0.0005)
    leg_joint_torque = RewTerm(func=locomotion_mdp.leg_joint_torque, weight=-1.0e-7)
    # ankle_action = RewTerm(func=locomotion_mdp.ankle_action, weight=-0.001)
    # hip_roll_action = RewTerm(func=locomotion_mdp.hip_roll_action, weight=-0.1)
    # hip_yaw_action = RewTerm(func=locomotion_mdp.hip_yaw_action, weight=-0.1)
    feet_y_distance = RewTerm(func=locomotion_mdp.feet_y_distance, weight=-2.0, params={"nom_dist": 0.25})


    #-----------------新增奖励函数-----------------#
    # 机器人基座相对局部地形高度奖励。多地形下作为弱约束，避免压过步态和速度奖励。
    base_height = RewTerm(
        func=locomotion_mdp.base_height_l2,
        weight=0.4,
        params={
            "target_height": 0.76,  # G1 12DOF URDF/MJCF 中 pelvis 的默认站立高度
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "terrain_radius": 0.25,
            "height_error_scale": 50.0,
        }
    )

    # 鼓励机器人落脚选择更平坦的地面
    foothold_flatness = RewTerm(
        func=locomotion_mdp.foothold_flatness_reward,
        weight=0.1,
        params={
            "lfoot_sensor_cfg": SceneEntityCfg("Lfoot_height_scanner"),
            "rfoot_sensor_cfg": SceneEntityCfg("Rfoot_height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=[".*ankle_roll.*"],
            ),
            "height_std": 0.03,
            "force_threshold": 1.0,
            "command_threshold": 0.1,
        },
    )

    # 对于踩到楼梯边缘的情况给于惩罚，视情况加不加
    # foothold_edge_penalty = RewTerm(
    #     func=locomotion_mdp.foothold_edge_penalty,
    #     weight=-1.0,
    #     params={
    #         "lfoot_sensor_cfg": SceneEntityCfg("Lfoot_height_scanner"),
    #         "rfoot_sensor_cfg": SceneEntityCfg("Rfoot_height_scanner"),
    #         "contact_sensor_cfg": SceneEntityCfg(
    #             "contact_sensor",
    #             body_names=[".*ankle_roll.*"],
    #         ),
    #         "edge_height_threshold": 0.06,
    #         "edge_height_margin": 0.04,
    #         "outlier_threshold": 2.0,
    #         "force_threshold": 1.0,
    #         "command_threshold": 0.1,
    #     },  
    # )



    # 惩罚摆动脚倾斜（点积版本）
    # swing_foot_tilt_penalty = RewTerm(
    #     func=locomotion_mdp.swing_foot_tilt_penalty,
    #     weight=-0.05,  # 建议从-0.2开始
    #     params={
    #         "delta_t": 0.05,
    #         "left_foot_cfg": SceneEntityCfg("robot", body_names=[".*left_ankle_roll.*"]),
    #         "right_foot_cfg": SceneEntityCfg("robot", body_names=[".*right_ankle_roll.*"])
    #     }
    # )


    # swing_foot_clearance = RewTerm(
    #     func=locomotion_mdp.swing_foot_clearance_adaptive,
    #     weight=0.2,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("height_scanner"),
    #         "base_clearance": 0.08,
    #         "max_clearance": 0.20,
    #         "obstacle_margin": 0.03,
    #         "obstacle_scale": 1.0,
    #         "obstacle_threshold": 0.025,
    #         "std": 0.04,
    #         "terrain_radius": 0.20,
    #         "obstacle_probe_radius": 0.18,
    #         "min_lookahead_time": 0.15,
    #         "max_lookahead_time": 0.35,
    #         "foot_height_offset": 0.04,
    #         "min_swing_progress": 0.15,
    #         "max_swing_progress": 0.85,
    #         "command_threshold": 0.05,
    #     },
    # )

@configclass
class G1LocomotionEnvCfg:
    # 基础属性
    device: str = "cuda:0" # 指定环境运行的设备
    # 场景配置
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0, # 每个训练回合最长持续20s
        num_envs=4096, # 并行训练环境数量
        env_spacing=2.5, # 并行环境间距
        robot=G1_CFG, # G1_CFG
        terrain_type="generator", # 使用 “粗糙地形生成器” 生成训练地形
        terrain_generator=PLANNING_TERRAIN_CFG, #GRAVEL_TERRAINS_CFG, #使用 “粗糙地形生成器” 生成训练地形
        # terrain_type="plane",
        # terrain_generator= None,
        max_init_terrain_level=2, # 初始地形的最大难度等级
        #高度传感器配置
        height_scanner=HeightScannerCfg(
            enable_height_scan=True, # 启用高度扫描功能
            prim_body_name="pelvis", # 指定传感器安装位置
            resolution=0.1, # 扫描分辨率为0.1米/点
            size=(1.6, 1.0), # 扫描区域物理尺寸，前后方向+-0.8m，左右方向+-0.5m
            debug_vis=True, # 启用调试可视化功能
            drift_range=(-0.0, 0.0),  # (0.3, 0.3) 扫描位置随机漂移范围
        ),

        Lfoot_height_scanner=LfootHeightScannerCfg(
            enable_height_scan=True,
            prim_body_name="left_ankle_roll_link",
            resolution=0.01,
            size=(0.2, 0.08),
            debug_vis=True,
            drift_range=(-0.0, 0.0),  # (0.3, 0.3)
        ),
        Rfoot_height_scanner=RfootHeightScannerCfg(
            enable_height_scan=True,
            prim_body_name="right_ankle_roll_link",
            resolution=0.01,
            size=(0.2, 0.08),
            debug_vis=True,
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
            depth_vis=True,
        ),

    )
    # 机器人配置
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10, # 演员网络决策使用的历史观测长度
        critic_obs_history_length=5, # 评论家网络决策使用的历史观测长度
        action_scale=0.25,
        terminate_contacts_body_names=[".*knee.*", "pelvis", ".*hip_pitch.*"], # 接触地面即判定终止的部位
        feet_body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        base_tilt_limit=0.8,  # base倾斜角度阈值，单位rad，0.8rad约45.8度
    )
    reward = LocomotionRewardCfg()
    gait = GaitCfg() # 步态参数
    foothold = FootholdCfg()
    # 归一化配置
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0, # 线速度观测的缩放系数（1.0表示不缩放）
            ang_vel=1.0, # 角速度观测的缩放系数
            projected_gravity=1.0, # 投影重力向量的缩放系数
            commands=1.0, # 指令（如目标速度）的缩放系数
            joint_pos=1.0, # 关节位置的缩放系数
            joint_vel=1.0, # 关节速度的缩放系数
            actions=1.0, # 动作的缩放系数
            height_scan=1.0, # 高度扫描数据的缩放系数
        ),
        clip_observations=100.0, # 观测值的裁剪阈值（超过100的观测值被截断，避免异常值影响）
        clip_actions=100.0, # 动作的裁剪阈值（超过100的动作被截断）
        height_scan_offset=0.5,  # 高度扫描数据的偏移量
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0), # 指令重采样的时间范围
        rel_standing_envs=0.2, # 20%的环境会收到"站立"指令
        rel_heading_envs=1.0, # 100%的环境会收到航向指令
        heading_command=True, # 启用航向控制（机器人需跟随航向指令调整方向）
        heading_control_stiffness=0.5, # 航向控制的刚度（数值越大，转向响应越灵敏）
        debug_vis=True, # 启用指令可视化（在仿真界面显示当前目标速度/航向）
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.3, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True, # 启用噪声添加
        noise_scales=NoiseScalesCfg(
            lin_vel=0.0, # 线速度观测的噪声比例
            ang_vel=0.2, # 角速度观测的噪声比例
            projected_gravity=0.05, # 投影重力向量的噪声比例
            joint_pos=0.08, # 关节位置观测的噪声比例
            joint_vel=1.0, # 关节速度观测的噪声比例
            height_scan=0.0, # 高度扫描数据的噪声比例
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg( # 随机化机器人刚体材料属性（启动时执行一次）
            physics_material=EventTerm(
                func=isaac_mdp.randomize_rigid_body_material, # 调用的随机化函数
                mode="startup", # 执行时机：环境启动时
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"), # 作用于机器人所有部件
                    "static_friction_range": (0.6, 1.0), # 静摩擦系数范围
                    "dynamic_friction_range": (0.4, 0.8), # 动摩擦系数范围
                    "restitution_range": (0.0, 0.005), # 恢复系数范围（接近0，几乎无弹性）
                    "num_buckets": 64, # 随机参数的离散化数量（64种组合）
                },
            ),
            add_base_mass=EventTerm(
                func=isaac_mdp.randomize_rigid_body_mass, # 调用质量随机化函数
                mode="startup", # 执行时机：环境启动时
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"), # 仅作用于基座
                    "mass_distribution_params": (-5.0, 5.0), # 质量随机增减范围
                    "operation": "add", # 操作类型：在基础质量上叠加随机值
                },
            ),
            reset_base=EventTerm( # 重置时随机化机器人初始状态 （每个回合重置时执行）
                func=isaac_mdp.reset_root_state_uniform, # 重置根节点状态的函数
                mode="reset", # 执行时机：回合重置时
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}, # 初始位姿随机范围
                    "velocity_range": { # 初始速度随机范围
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm( # 重置时随机化关节初始状态（每个回合重置时执行）
                func=isaac_mdp.reset_joints_by_scale, # 按比例重置关节的函数
                mode="reset", # 执行时机：回合重置时
                params={
                    "position_range": (0.5, 1.5), # 关节位置在默认值的0.5~1.5倍范围内随机
                    "velocity_range": (0.0, 0.0), # 关节初始速度固定为0
                },
            ),
            push_robot=EventTerm( # 训练中随机推机器人（模拟外部干扰）
                func=isaac_mdp.push_by_setting_velocity, # 通过设置速度推机器人的函数
                mode="interval",  # 执行时机：按时间间隔随机触发
                interval_range_s=(10.0, 15.0), # 触发间隔：10~15秒一次
                params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},  # 推力对应的速度变化范围
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}), # 动作延迟配置（当前禁用）
    )
    sim: SimCfg = SimCfg(dt=0.002, decimation=10, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)) # decimation=4, 观测数据的抽取倍数（每4步输出一次观测，即50Hz，降低计算量）；physxPhysX物理引擎参数（GPU最大刚体接触片数量，确保复杂地形接触计算）


@configclass
class G1LocomotionAgentCfg(RslRlOnPolicyRunnerCfg):
    # 基础训练配置
    seed = 42 # 随机种子
    device = "cuda:0" # 指定训练使用的设备
    num_steps_per_env = 24 # 每个并行环境在一次迭代中收集的轨迹步数（用于构建训练数据）
    max_iterations = 50000 # 训练的最大迭代次数
    empirical_normalization = False # 不启用经验归一化
    # 策略网络配置
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic", # Actor-Critic框架
        init_noise_std=1.0, # 初始动作噪声标准差，用于训练初期探索，噪声越大，探索性越强
        noise_std_type="log", # 噪声类型 标量 "scalar"
        actor_hidden_dims=[512, 256, 128], # actor网络隐藏层维度
        critic_hidden_dims=[512, 256, 128], # critic网络隐藏层维度
        activation="elu", # 激活函数使用Elu
    )
    # ppo算法参数
    # algorithm = RslRlPpoAlgorithmCfg( 
    #     class_name="PPO", #"AMPPPO"
    #     value_loss_coef=1.0, # 价值损失在总损失中的权重
    #     use_clipped_value_loss=True, # 启用价值损失裁剪
    #     clip_param=0.2, # ppo的核心裁剪参数，限制策略更新的幅度（确保新策略与旧策略 “接近”）
    #     entropy_coef=0.005, # 熵奖励系数，鼓励策略保持探索性（熵越大，动作分布越均匀，探索性越强）
    #     num_learning_epochs=5, # 每次收集数据后，用 5 个 epoch 训练网络（重复利用数据提升效率）
    #     num_mini_batches=4, # 将收集的轨迹数据分成 4 个 mini-batch，分批次训练（减轻内存压力）
    #     learning_rate=1.0e-3, # 学习率为 0.001（控制参数更新速度）
    #     schedule="adaptive", # 学习率调度策略为 “自适应”
    #     gamma=0.99, # 折扣因子
    #     lam=0.95, # GAE系数
    #     desired_kl=0.01, # 期望的 KL 散度（新旧策略分布的差异），用于自适应调整训练强度
    #     max_grad_norm=1.0, # 梯度裁剪的最大范数
    #     normalize_advantage_per_mini_batch=False, # 不按 mini-batch 归一化优势估计
    #     symmetry_cfg=None,  # RslRlSymmetryCfg() 对称性增强
    #     rnd_cfg=None,  # RslRlRndCfg() RND 探索机制
    # )
    algorithm = RslRlPpoAlgorithmCfg( 
        class_name="PPO", #"AMPPPO"
        value_loss_coef=1.0, # 价值损失在总损失中的权重
        use_clipped_value_loss=True, # 启用价值损失裁剪
        clip_param=0.2, # ppo的核心裁剪参数，限制策略更新的幅度（确保新策略与旧策略 “接近”）
        entropy_coef=0.01, # 熵奖励系数，鼓励策略保持探索性（熵越大，动作分布越均匀，探索性越强）
        num_learning_epochs=5, # 每次收集数据后，用 5 个 epoch 训练网络（重复利用数据提升效率）
        num_mini_batches=4, # 将收集的轨迹数据分成 4 个 mini-batch，分批次训练（减轻内存压力）
        learning_rate=1.0e-3, # 学习率为 0.001（控制参数更新速度）
        schedule="adaptive", # 学习率调度策略为 “自适应”
        gamma=0.99, # 折扣因子
        lam=0.95, # GAE系数
        desired_kl=0.01, # 期望的 KL 散度（新旧策略分布的差异），用于自适应调整训练强度
        max_grad_norm=1.0, # 梯度裁剪的最大范数
        normalize_advantage_per_mini_batch=False, # 不按 mini-batch 归一化优势估计
        symmetry_cfg=None,  # RslRlSymmetryCfg() 对称性增强
        rnd_cfg=None,  # RslRlRndCfg() RND 探索机制
    )

    # 训练流程配置
    clip_actions = None # 不裁剪动作
    save_interval = 500 # 每 100 次迭代保存一次模型 checkpoint
    runner_class_name = "OnPolicyRunner" # "AmpOnPolicyRunner" 使用基础的 “在线策略训练器”，可选"AmpOnPolicyRunner"
    experiment_name = "g1_locomotion" # 实验名称
    run_name = ""
    logger = "tensorboard" # 使用 TensorBoard 记录训练日志（如奖励、损失等指标）
    neptune_project = "g1_locomotion" # 指定 Neptune 项目名称
    wandb_project = "g1_locomotion" # 指定 wandb 项目名称
    resume = False # 不从之前的 checkpoint 恢复训练（从头开始训练）
    load_run = ".*" # 加载历史运行或模型的匹配模式（因resume=False，当前无效）
    load_checkpoint = "model_.*.pt" # 加载历史运行或模型的匹配模式（因resume=False，当前无效）


g1LocomotionEnvCfg = G1LocomotionEnvCfg
g1LocomotionAgentCfg = G1LocomotionAgentCfg
