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
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from legged_lab.sensors.noisy_camera.noisy_grouped_raycaster_camera import NoisyGroupedRayCasterCamera


from legged_lab.envs.g1.foothold_planner import FootholdPlannerMixin
from legged_lab.envs.g1.g1_locomotion_cfg import G1LocomotionEnvCfg
from legged_lab.utils.env_utils.scene import SceneCfg
from rsl_rl.env import VecEnv


class G1LocomotionEnv(FootholdPlannerMixin, VecEnv):
    """机器人环境类，继承自向量环境基类，用于强化学习训练"""
    def __init__(
        self,
        cfg: (
            G1LocomotionEnvCfg
        ),
        headless,
    ):
        self.cfg: (

            G1LocomotionEnvCfg
        )
        """初始化环境

        Args:
            cfg: 环境配置对象，包含仿真、机器人、奖励等所有参数
            headless: 是否无头模式（不显示图形界面）
        """

        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        # 配置仿真参数
        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        # 创建场景
        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        # 获取场景中的机器人和传感器
        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

        # 如果启用高度扫描，获取高度扫描仪
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        # 获取噪声深度相机
        if self.cfg.scene.noisy_camera.enable_camera:
            self.noisy_camera: NoisyGroupedRayCasterCamera = self.scene.sensors["noisy_camera"] 


        # 配置速度指令生成器
        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = UniformVelocityCommand(cfg=command_cfg, env=self)
        self.reward_manager = RewardManager(self.cfg.reward, self)

        self.init_buffers()
        if self.is_foothold_planner_enabled():
            self.init_visualization_markers()

        # 初始化事件管理器
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

    def init_buffers(self):
        """初始化各种数据缓冲区，用于存储状态、观测、动作等数据"""
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)

        self.num_joint_actions = self.robot.data.default_joint_pos.shape[1]
        self.num_gait_actions = 1 if self.cfg.gait.enable_variable_gait_cycle else 0
        self.num_actions = self.num_joint_actions + self.num_gait_actions

        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        # 新增：将 action_scale 从列表转换为张量并扩展维度
        self.action_scale = torch.tensor(
            self.cfg.robot.action_scale, 
            device=self.device, 
            dtype=torch.float32
        ).unsqueeze(0)  # 形状变为 [1, num_joints]，支持广播

        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"], self.num_envs, device=self.device
        )
        # 动作延迟缓冲区（用于模拟动作执行延迟）
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )
        # 如果启用动作延迟，初始化每个环境的延迟时间
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))
        # 配置场景实体（机器人关节、身体等）
        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        # 配置终止接触的身体部位
        self.termination_contact_cfg.resolve(self.scene)
        # 配置脚部接触传感器
        self.feet_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.feet_body_names, preserve_order=True
        )
        self.feet_cfg.resolve(self.scene)
        # 获取脚部身体ID
        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
        )

        # 获取膝部身体ID
        self.knee_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_knee_link", "right_knee_link"], preserve_order=True
        )

        # 获取左腿关节ID
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                ".*left_hip_pitch_joint",
                ".*left_hip_roll_joint",
                ".*left_hip_yaw_joint",
                ".*left_knee_joint",
                ".*left_ankle_pitch_joint",
                ".*left_ankle_roll_joint",
            ],
            preserve_order=True,
        )
        # 获取右腿关节ID
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=[
                ".*right_hip_pitch_joint",
                ".*right_hip_roll_joint",
                ".*right_hip_yaw_joint",
                ".*right_knee_joint",
                ".*right_ankle_pitch_joint",
                ".*right_ankle_roll_joint",
            ],
            preserve_order=True,
        )

        # 获取踝关节ID
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=[".*left_ankle_pitch_joint", ".*right_ankle_pitch_joint", ".*left_ankle_roll_joint", ".*right_ankle_roll_joint"],
            preserve_order=True,
        )


        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise
        # 初始化episode长度缓冲区
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # 步态参数缓冲区
        self.gait_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        # 步态周期
        self.gait_cycle = torch.full(
            (self.num_envs,), self.cfg.gait.gait_cycle, dtype=torch.float, device=self.device, requires_grad=False
        )
        # 相位比例
        self.phase_ratio = torch.tensor(
            [self.cfg.gait.gait_air_ratio_l, self.cfg.gait.gait_air_ratio_r], dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        # 相位偏移
        self.phase_offset = torch.tensor(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        # 变步频
        self.gait_clock = torch.zeros(self.num_envs, device=self.device)
        # 动作缓冲区
        self.action = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        # 添加：初始化上一步动作缓冲区（与当前动作形状相同）
        self.prev_action = torch.zeros_like(self.action)  # 关键：新增prev_action属性
        
        # 平均脚部力和速度缓冲区
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.init_obs_buffer()

        self.init_foothold_planner_buffers()
            
    def compute_current_observations(self):
        """计算当前时刻的观测数据

        Returns:
            current_actor_obs: 演员网络的观测
            current_critic_obs: 评论家网络的观测
        """
        robot = self.robot
        # 获取接触传感器的历史接触力数据（世界坐标系）
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        # 提取机器人状态信息
        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5

        sin_phase = torch.sin(2 * torch.pi * self.gait_phase[:,0]).unsqueeze(1) # 2
        cos_phase = torch.cos(2 * torch.pi * self.gait_phase[:,0]).unsqueeze(1) 
        zero_flag = (torch.norm(self.command_generator.command[:,:2],dim=1) + torch.abs(self.command_generator.command[:,2])) < 0.1
        sin_phase[zero_flag,:] = 0.
        cos_phase[zero_flag,:] = 1.
        # 拼接演员网络的观测向量
        current_actor_obs = torch.cat(
            [
                # root_lin_vel * self.obs_scales.lin_vel,  # 3
                ang_vel * self.obs_scales.ang_vel,  # 3
                projected_gravity * self.obs_scales.projected_gravity,  # 3
                command * self.obs_scales.commands,  # 3
                joint_pos * self.obs_scales.joint_pos,  # 12
                joint_vel * self.obs_scales.joint_vel,  # 12
                action * self.obs_scales.actions,  # 12
                sin_phase, #torch.sin(2 * torch.pi * self.gait_phase),  # 2
                cos_phase, #torch.cos(2 * torch.pi * self.gait_phase),  # 2
                self.phase_ratio,  # 2
            ],
            dim=-1,
        )
      
        current_obs = torch.cat(
            [
                root_lin_vel * self.obs_scales.lin_vel,  # 3
                ang_vel * self.obs_scales.ang_vel,  # 3
                projected_gravity * self.obs_scales.projected_gravity,  # 3
                command * self.obs_scales.commands,  # 3
                joint_pos * self.obs_scales.joint_pos,  # 12
                joint_vel * self.obs_scales.joint_vel,  # 12
                action * self.obs_scales.actions,  # 12
                sin_phase, #torch.sin(2 * torch.pi * self.gait_phase),  # 2
                cos_phase, #torch.cos(2 * torch.pi * self.gait_phase),  # 2
                self.phase_ratio,  # 2
            ],
            dim=-1,
        )
        
        # 评论家观测在演员观测基础上增加脚部接触状态
        current_critic_obs = torch.cat([current_obs, feet_contact], dim=-1)

        # 变步频观测
        if self.cfg.gait.enable_variable_gait_cycle and self.cfg.gait.observe_gait_cycle:
            gait_cycle_obs = (self.gait_cycle / self.cfg.gait.gait_cycle).unsqueeze(1)
            current_actor_obs = torch.cat([current_actor_obs, gait_cycle_obs], dim=-1)
            current_critic_obs = torch.cat([current_critic_obs, gait_cycle_obs], dim=-1)


        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        """计算完整观测（包含历史观测序列）"""
        # 获取当前时刻的观测
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        # 如果启用噪声，为演员观测添加随机噪声（[-noise_scale, noise_scale]范围内的均匀噪声）
      
        
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        # 将当前观测加入循环缓冲区（存储历史观测）
        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        # 将历史观测序列展平为一维向量
        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)

        # 如果启用高度扫描，添加高度扫描数据
        if self.cfg.scene.height_scanner.enable_height_scan:
            # 计算地面高度（传感器位置z坐标 - 射线击中地面的z坐标 - 偏移量），并缩放
            # height_scan = (
            #                       self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
            #                       - self.height_scanner.data.ray_hits_w[..., 2]
            #                       - self.cfg.normalization.height_scan_offset
            #               ) * self.obs_scales.height_scan
            #-------------------------------------------------------------------------------
            # 1. 提取原始数据（保留多射线维度，不做均值压缩）
            sensor_z = self.height_scanner.data.pos_w[:, 2].unsqueeze(1)  # [num_envs, 1]：每个环境1个传感器z坐标
            ray_hits_w = self.height_scanner.data.ray_hits_w  # [num_envs, num_rays, 3]：每个环境的每条射线的世界坐标
            ray_hits_z = ray_hits_w[..., 2]  # 【关键】保留多射线维度：[num_envs, num_rays]（不unsqueeze，直接2维，对应每个射线点）
            
            # 2. 处理无效数据（NaN/无穷大）：按环境填充，保留有效射线的原始值
            valid_mask = torch.isfinite(ray_hits_z)  # [num_envs, num_rays]：标记有效射线点
            if not torch.all(valid_mask):
                # 对每个环境单独填充（避免跨环境污染，保留单环境内射线的真实性）
                for env_idx in range(ray_hits_z.shape[0]):
                    env_valid_mask = valid_mask[env_idx]  # 当前环境的有效射线标记
                    env_valid_values = ray_hits_z[env_idx][env_valid_mask]  # 当前环境的有效射线z值
                    if len(env_valid_values) > 0:
                        # 用当前环境有效射线的均值填充无效射线（不改变有效射线的原始值）
                        ray_hits_z[env_idx][~env_valid_mask] = torch.mean(env_valid_values)
                    else:
                        # 兜底：当前环境无有效射线，用传感器z坐标填充（符合物理逻辑）
                        ray_hits_z[env_idx] = sensor_z[env_idx].item()
            # print("ray_hits_z = ",ray_hits_z.shape)
            # 3. 计算高度差（保留多射线维度，广播机制自动兼容）
            # sensor_z [num_envs,1] → 广播为 [num_envs, num_rays]，每条射线对应同一个环境的传感器z
            height_diff = sensor_z - ray_hits_z - self.cfg.normalization.height_scan_offset  # 结果：[num_envs, num_rays]
            
            # 4. 异常值限制（关键！避免单个射线点的极端值污染观测）
            # 按实际地形调整范围（比如±1米，确保每个射线点的高度差合理）
            height_diff = torch.clamp(height_diff, min=0.0, max=1.0)
            
            # 5. 缩放（将高度差映射到观测所需范围，比如±1）
            height_scan = height_diff * self.obs_scales.height_scan  # 结果：[num_envs, num_rays]
            # print("height_scan=",height_scan.reshape(11,17))
            #-------------------------------------------------------------------------
            # 将高度扫描数据添加到评论家观测
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            # 为高度扫描数据添加噪声（如果启用）
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            # 将带噪声的高度扫描数据添加到演员观测
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)


        # 裁剪观测值到指定范围（防止极端值影响训练）
        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)
        return actor_obs, critic_obs

    def reset(self, env_ids):
        """重置指定环境的状态

        Args:
            env_ids: 需要重置的环境索引
        """
        if len(env_ids) == 0:
            return

        # 重置脚部力和速度的平均缓冲区
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

        #重置变步频
        self.gait_clock[env_ids] = 0.0
        self.gait_cycle[env_ids] = self.cfg.gait.gait_cycle

        self.extras["log"] = dict()  # 重置日志信息
        # 如果启用地形难度递进，更新地形等级
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        # 重置场景中指定环境的实体
        self.scene.reset(env_ids)
        # 如果有"reset"模式的事件（如随机化初始状态），应用事件
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        # 重置奖励管理器并获取奖励相关的额外信息
        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf  # 记录超时信息

        # 重置指令生成器（重新生成速度指令）
        self.command_generator.reset(env_ids)
        # 重置观测缓冲区、动作缓冲区和episode长度计数器
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.reset_foothold_planner(env_ids)

        # 将重置后的状态写入仿真器并向前计算（更新物理状态）
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        """执行一步环境交互

        Args:
            actions: 智能体输出的动作（未延迟、未裁剪）

        Returns:
            actor_obs: 演员网络的观测
            reward_buf: 每个环境的奖励值
            self.reset_buf: 每个环境的重置标志（True表示需要重置）
            self.extras: 额外信息（日志、观测等）
        """
        # 1. 新增：先缓存当前动作到prev_action（在更新当前动作之前）
        self.prev_action = self.action.clone()  # 关键：保存上一步动作
        # 计算延迟后的动作（模拟动作执行延迟）
        delayed_actions = self.action_buffer.compute(actions)
        # 裁剪动作到指定范围并移动到目标设备
        self.action = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)

        # 将动作转换为关节目标位置（动作缩放 + 默认关节位置）
        # 修正：使用扩展后的 action_scale 张量，并扩展 default_joint_pos 维度
        default_joint_pos = self.robot.data.default_joint_pos  # 形状 [1, num_joints]

        joint_actions = self.action[:, :self.num_joint_actions]
        processed_actions = joint_actions * self.action_scale + default_joint_pos  # 广播为 [num_envs, num_joints]


        # 重置脚部力和速度的平均缓冲区
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device, requires_grad=False
        )

        # 执行多次物理仿真步（根据减速倍数，如decimation=4表示4次物理步对应1次环境步）
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1  # 累加物理仿真步数
            self.robot.set_joint_position_target(processed_actions)  # 设置关节目标位置
            self.scene.write_data_to_sim()  # 将状态写入仿真器
            self.sim.step(render=False)  # 执行物理仿真步（不渲染）
            self.scene.update(dt=self.physics_dt)  # 更新场景状态

            # 累加脚部接触力的模长（用于计算平均力）
            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
            )
            # 累加脚部线速度的模长（用于计算平均速度）
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        # 计算物理步内的平均脚部力和速度
        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        # 如果不是无头模式，渲染当前帧
        if not self.headless:
            self.sim.render()

        # 累加episode长度（环境步）
        self.episode_length_buf += 1
        # 更新步态参数（相位等）
        self._update_gait_cycle_from_action()
        self._calculate_gait_para()

        # 计算新的速度指令（根据时间步长更新）
        self.command_generator.compute(self.step_dt)
        # 如果有"interval"模式的事件（如周期性随机化），应用事件
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.update_foothold_planner()

        # 检查是否需要重置环境（碰撞或超时）
        self.reset_buf, self.time_out_buf = self.check_reset()
        # 计算当前步的奖励
        reward_buf = self.reward_manager.compute(self.step_dt)
        # 找出需要重置的环境索引
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # 重置指定环境
        self.reset(self.reset_env_ids)

        # 计算当前观测
        actor_obs, critic_obs = self.compute_observations()
        # 存储评论家观测到额外信息中
        self.extras["observations"] = {"critic": critic_obs}

        self.finalize_foothold_planner_step()

        if torch.isnan(reward_buf).any():
            print("reward is nan: ", reward_buf)
        if torch.isinf(reward_buf).any():
            print("reward is inf: ", reward_buf)
        
        if torch.isnan(self.action).any():
            print("action is nan: ", self.action)
        if torch.isinf(self.action).any():
            print("action is inf: ", self.action)

        return actor_obs, reward_buf, self.reset_buf, self.extras


    def check_reset(self):
        """检查是否需要重置环境

        Returns:
            reset_buf: 布尔数组，True表示对应环境需要重置
            time_out_buf: 布尔数组，True表示对应环境因超时而重置
        """
        # 获取接触力历史数据
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        # 检测是否有终止性接触（接触力模长超过阈值）
        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],  # 终止接触的身体部位
                    dim=-1,
                ),
                dim=1,  # 在时间维度上取最大值
            )[0]
            > 1.0,  # 接触力阈值
            dim=1,  # 在身体部位维度上取逻辑或
        )
        # 检测base倾斜角度是否超过阈值。projected_gravity_b的xy范数等于sin(倾斜角)。
        if self.cfg.robot.base_tilt_limit > 0.0:
            base_tilt = torch.linalg.norm(self.robot.data.projected_gravity_b[:, :2], dim=1)
            reset_buf |= base_tilt > np.sin(self.cfg.robot.base_tilt_limit)

        # 检测是否超时（episode长度达到最大值）
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        # 合并重置条件（接触终止或超时）
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        """初始化观测缓冲区和噪声参数"""
        if self.add_noise:
            # 计算观测噪声的缩放向量（与观测维度一一对应）
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            # 为不同观测分量设置噪声缩放
            # noise_vec[:3] = noise_scales.lin_vel * self.obs_scales.lin_vel  # 线速度噪声
            noise_vec[:3] = noise_scales.ang_vel * self.obs_scales.ang_vel  # 角速度噪声
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity  # 投影重力噪声
            noise_vec[6:9] = 0  # 指令不添加噪声（或单独设置）
            # 关节位置噪声
            idx = 9
            noise_vec[idx : idx + self.num_joint_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            # 关节速度噪声
            idx += self.num_joint_actions
            noise_vec[idx : idx + self.num_joint_actions] = noise_scales.joint_vel * self.obs_scales.joint_vel
            # 动作和步态相关观测不添加噪声
            idx += self.num_joint_actions
            noise_vec[idx : idx + self.num_actions] = 0.0 #动作

            idx += self.num_actions
            noise_vec[idx : idx + 6] = 0.0 #相位相关
            self.noise_scale_vec = noise_vec  # 存储噪声缩放向量

            # 如果启用高度扫描，初始化高度扫描的噪声缩放
            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                        self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                        - self.height_scanner.data.ray_hits_w[..., 2]
                        - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        # 初始化演员和评论家的观测循环缓冲区（存储历史观测）
        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def update_terrain_levels(self, env_ids):
        """更新地形等级（用于课程学习，逐步增加难度）

        Args:
            env_ids: 需要更新地形的环境索引

        Returns:
            extras: 包含地形等级信息的字典（用于日志）
        """
        # 计算机器人根节点与环境原点的平面距离（x,y方向）
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        # 判断是否需要移动地形（距离超过地形生成器大小的一半）
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        # 判断是否需要向下移动地形（距离小于预期移动距离的一半）
        move_down = (
                distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up  # 避免同时向上和向下移动

        # 更新环境原点（实现地形的动态移动，营造无限地形的效果）
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        # 记录地形等级的平均值（用于日志）
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())
        return extras

    def get_observations(self):
        """获取当前观测（用于评估或测试）

        Returns:
            actor_obs: 演员网络的观测
            self.extras: 额外信息
        """
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras


    @staticmethod
    def seed(seed: int = -1) -> int:
        """
        设置环境随机种子，确保实验可复现

        Args:
            seed: 随机种子值，-1表示使用随机种子

        Returns:
            int: 实际使用的种子值
        """
        try:
            # 设置Omniverse Replicator的全局种子（用于生成合成数据）
            import omni.replicator.core as rep  # type: ignore
            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            # 若未安装Replicator模块，则忽略
            pass
        # 设置PyTorch和NumPy的随机种子
        return torch_utils.set_seed(seed)


    def _update_gait_cycle_from_action(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        if not self.cfg.gait.enable_variable_gait_cycle:
            self.gait_cycle[env_ids] = self.cfg.gait.gait_cycle
            return

        gait_action = self.action[env_ids, self.num_joint_actions]
        gait_action = torch.tanh(gait_action)

        min_scale, max_scale = self.cfg.gait.gait_cycle_scale_range
        scale = min_scale + 0.5 * (gait_action + 1.0) * (max_scale - min_scale)

        target_cycle = self.cfg.gait.gait_cycle * scale
        alpha = self.cfg.gait.gait_cycle_smoothing
        self.gait_cycle[env_ids] = (1.0 - alpha) * self.gait_cycle[env_ids] + alpha * target_cycle


    def _calculate_gait_para(self, env_ids=None) -> None:
        """
       根据仿真时间和相位偏移更新步态参数
       计算每个环境中两条腿的当前步态相位（0-1范围）
       """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return

        # # 计算归一化时间：当前episode长度 × 每步时间 / 步态周期
        # t = self.episode_length_buf * self.step_dt / self.gait_cycle

        # # 计算两条腿的步态相位（添加相位偏移并对1取模，确保在[0,1)范围内）
        # # 相位偏移通常设置为0.5，表示两腿交替运动（如走路）
        # self.gait_phase[:, 0] = (t + self.phase_offset[:, 0]) % 1.0  # 第一条腿的相位
        # self.gait_phase[:, 1] = (t + self.phase_offset[:, 1]) % 1.0  # 第二条腿的相位

        # 变步频设置
        self.gait_clock[env_ids] = (self.gait_clock[env_ids] + self.step_dt / self.gait_cycle[env_ids]) % 1.0

        self.gait_phase[env_ids, 0] = (self.gait_clock[env_ids] + self.phase_offset[env_ids, 0]) % 1.0
        self.gait_phase[env_ids, 1] = (self.gait_clock[env_ids] + self.phase_offset[env_ids, 1]) % 1.0
