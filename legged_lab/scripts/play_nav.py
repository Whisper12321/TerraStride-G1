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

"""Play a trained G1 navigation policy with a frozen locomotion policy.

The high-level navigation policy outputs velocity commands. The navigation environment then
evaluates the frozen low-level locomotion policy internally for several control steps, so this
script only needs to call env.step() with high-level actions.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

import legged_lab.utils.cli_args as cli_args  # isort: skip


SUPPORTED_TASKS = {"g1_navigation"}


parser = argparse.ArgumentParser(description="Play a trained G1 autonomous navigation policy.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Navigation task name.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of navigation environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--navigation_policy_path",
    type=str,
    default=None,
    help="Path to a navigation TorchScript policy or RSL-RL checkpoint.",
)
parser.add_argument(
    "--locomotion_policy_path",
    type=str,
    default="/home/zstu58/zyd/g1_obstacle/logs/g1_locomotion/2026-05-26_17-11-30/exported/policy.pt",
    help="Path to the frozen low-level locomotion TorchScript policy.pt. Defaults to env cfg.",
)
parser.add_argument("--max_episode_length_s", type=float, default=60.0, help="Episode length used for play.")
parser.add_argument("--terrain_rows", type=int, default=1, help="Override generated terrain row count.")
parser.add_argument("--terrain_cols", type=int, default=1, help="Override generated terrain column count.")
parser.add_argument(
    "--terrain_difficulty",
    type=float,
    default=0.3,
    help="Force terrain generator difficulty in [0, 1]. 0 is easiest, 1 is hardest.",
)
parser.add_argument(
    "--min_path_cell_separation",
    type=int,
    default=4,
    help="Minimum Manhattan cell separation used when building path candidates. Larger values bias longer routes.",
)
parser.add_argument(
    "--path_level",
    type=int,
    default=4,
    help="Force navigation path difficulty level. 0 is easiest; max is num_path_difficulty_levels - 1.",
)
parser.add_argument(
    "--path_guidance_level",
    type=int,
    default=0,
    help="Force privileged path-guidance level. 0 is strongest; the highest level removes path reward guidance.",
)
parser.add_argument(
    "--waypoint_level",
    type=int,
    default=None,
    help="Deprecated alias for --path_guidance_level.",
)
parser.add_argument(
    "--full_terrain",
    action="store_true",
    default=False,
    help="Keep the configured full terrain grid. By default a single-env play run uses one terrain patch.",
)
parser.add_argument(
    "--show_depth",
    action="store_true",
    default=False,
    help="Show the first env normalized depth image in an OpenCV window.",
)
parser.add_argument(
    "--show_depth_rays",
    action="store_true",
    default=False,
    help="Show ray-caster hit points for the depth camera in the Isaac viewport.",
)
parser.add_argument(
    "--no_start_goal_vis",
    action="store_true",
    default=False,
    help="Disable start/goal marker visualization.",
)
parser.add_argument(
    "--vis_max_envs",
    type=int,
    default=8,
    help="Maximum number of env start/goal marker pairs to visualize.",
)
parser.add_argument(
    "--no_path_vis",
    action="store_true",
    default=False,
    help="Disable planned global path and path-reference visualization.",
)
parser.add_argument(
    "--path_vis_max_points",
    type=int,
    default=96,
    help="Maximum dense planned path points to visualize per environment.",
)
parser.add_argument("--print_interval", type=int, default=50, help="Print navigation status every N nav steps.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.task not in SUPPORTED_TASKS:
    raise ValueError(f"play_nav.py supports tasks {sorted(SUPPORTED_TASKS)}, got {args_cli.task!r}.")

# The navigation observation can use the ray-cast depth camera, so request camera support before app launch.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def _sync_seed_to_scene_and_terrain(env_cfg, seed: int) -> None:
    env_cfg.scene.seed = seed
    terrain_generator = env_cfg.scene.terrain_generator
    if terrain_generator is None:
        return

    terrain_generator.seed = seed
    for sub_cfg in terrain_generator.sub_terrains.values():
        if hasattr(sub_cfg, "seed"):
            sub_cfg.seed = seed


def _make_navigation_env_cfg(env_cfg, agent_cfg):
    env_cfg.noise.add_noise = False
    env_cfg.nav_noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = args_cli.max_episode_length_s
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    _sync_seed_to_scene_and_terrain(env_cfg, int(agent_cfg.seed))
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    # The low-level locomotion policy is loaded by G1NavigationEnv during construction.
    if args_cli.locomotion_policy_path is not None:
        env_cfg.nav.locomotion_policy_path = os.path.abspath(args_cli.locomotion_policy_path)

    if env_cfg.scene.terrain_generator is not None:
        if args_cli.terrain_difficulty is not None:
            difficulty = max(0.0, min(float(args_cli.terrain_difficulty), 1.0))
            env_cfg.scene.terrain_generator.difficulty_range = (difficulty, difficulty)
        if args_cli.terrain_rows is not None:
            env_cfg.scene.terrain_generator.num_rows = args_cli.terrain_rows
        if args_cli.terrain_cols is not None:
            env_cfg.scene.terrain_generator.num_cols = args_cli.terrain_cols
        if args_cli.min_path_cell_separation is not None:
            separation = max(1, int(args_cli.min_path_cell_separation))
            for sub_cfg in env_cfg.scene.terrain_generator.sub_terrains.values():
                if hasattr(sub_cfg, "min_path_candidate_cell_separation"):
                    sub_cfg.min_path_candidate_cell_separation = separation
        if not args_cli.full_terrain and args_cli.num_envs == 1:
            env_cfg.scene.terrain_generator.num_rows = 1
            env_cfg.scene.terrain_generator.num_cols = 1
            env_cfg.scene.max_init_terrain_level = 0

        env_cfg.scene.terrain_generator.curriculum = False

    env_cfg.scene.height_scanner.debug_vis = False
    env_cfg.scene.Lfoot_height_scanner.debug_vis = False
    env_cfg.scene.Rfoot_height_scanner.debug_vis = False
    env_cfg.scene.noisy_camera.debug_vis = args_cli.show_depth_rays
    env_cfg.scene.noisy_camera.depth_vis = args_cli.show_depth
    return env_cfg


def _env_supports_path_guidance(env) -> bool:
    return hasattr(env, "path_guidance_levels") and hasattr(env.cfg.nav, "num_path_guidance_levels")


def _apply_navigation_play_overrides(env, env_ids: torch.Tensor | None = None) -> bool:
    supports_guidance = _env_supports_path_guidance(env)
    forced_guidance_level = None
    if supports_guidance:
        forced_guidance_level = args_cli.path_guidance_level
        if args_cli.waypoint_level is not None:
            forced_guidance_level = args_cli.waypoint_level

    if args_cli.path_level is None and forced_guidance_level is None:
        return False

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if len(env_ids) == 0:
        return False

    if args_cli.path_level is not None:
        max_path_level = max(int(env.cfg.nav.num_path_difficulty_levels) - 1, 0)
        path_level = max(0, min(int(args_cli.path_level), max_path_level))
        env.path_difficulty_levels[env_ids] = path_level

    if supports_guidance and forced_guidance_level is not None:
        max_guidance_level = max(int(env.cfg.nav.num_path_guidance_levels) - 1, 0)
        guidance_level = max(0, min(int(forced_guidance_level), max_guidance_level))
        env.path_guidance_levels[env_ids] = guidance_level

    if hasattr(env, "_invalidate_navigation_routes"):
        env._invalidate_navigation_routes(env_ids)
    env._reset_navigation_task(env_ids)
    env._reset_nav_obs_buffers(env_ids)
    env.nav_cmd[env_ids] = 0.0
    env.prev_nav_cmd[env_ids] = 0.0
    env.prev_prev_nav_cmd[env_ids] = 0.0
    env.interp_cmd[env_ids] = 0.0
    if hasattr(env, "command_generator"):
        env.command_generator.command[env_ids, :3] = 0.0
    env.scene.write_data_to_sim()
    env.sim.forward()
    if hasattr(env, "_update_path_reference_metrics"):
        env._update_path_reference_metrics(env_ids)
        env.prev_nav_path_progress[env_ids] = env.nav_path_progress[env_ids]
    return True


def _default_policy_log_root(agent_cfg) -> str:
    return os.path.abspath(os.path.join("logs", agent_cfg.experiment_name))


def _load_navigation_policy_from_runner(
    env, agent_cfg, checkpoint_path: str
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], None]]:
    runner_class: type[OnPolicyRunner] | type[AmpOnPolicyRunner] = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint_path, load_optimizer=False)
    return runner.get_inference_policy(device=env.device), runner.alg.policy.reset


def _try_load_torchscript_policy(policy_path: str, device: str) -> torch.jit.ScriptModule | None:
    try:
        policy = torch.jit.load(policy_path, map_location=device)
    except Exception:
        return None
    policy.eval()
    return policy


def _resolve_navigation_policy(
    env, agent_cfg
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], None] | None, str]:
    if args_cli.navigation_policy_path is not None:
        policy_path = os.path.abspath(args_cli.navigation_policy_path)
        scripted_policy = _try_load_torchscript_policy(policy_path, env.device)
        if scripted_policy is not None:
            return scripted_policy, None, policy_path
        policy, reset_policy = _load_navigation_policy_from_runner(env, agent_cfg, policy_path)
        return policy, reset_policy, policy_path

    log_root_path = _default_policy_log_root(agent_cfg)
    checkpoint_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    policy, reset_policy = _load_navigation_policy_from_runner(env, agent_cfg, checkpoint_path)
    return policy, reset_policy, checkpoint_path


def _print_navigation_status(env, step_count: int) -> None:
    terrain_level = 0.0
    terrain = getattr(env.scene, "terrain", None)
    if terrain is not None and getattr(terrain, "terrain_levels", None) is not None:
        terrain_level = torch.mean(terrain.terrain_levels.float()).item()

    fields = [
        "[NAV]",
        f"task={args_cli.task}",
        f"step={step_count}",
        f"mean_ref_dist={env.current_goal_dist.mean().item():.3f}",
        f"mean_final_dist={env.current_final_goal_dist.mean().item():.3f}",
        f"path_level={env.path_difficulty_levels.float().mean().item():.2f}",
    ]
    if _env_supports_path_guidance(env):
        fields.extend(
            [
                f"path_guidance_level={env.path_guidance_levels.float().mean().item():.2f}",
                f"path_guidance_weight={env.nav_path_guidance_weight.float().mean().item():.2f}",
            ]
        )
    fields.extend(
        [
            f"path_refs={env.nav_waypoint_counts.float().mean().item():.1f}",
            f"path_remaining={env.nav_path_remaining.float().mean().item():.2f}",
            f"path_lateral_error={env.nav_path_lateral_error.float().mean().item():.2f}",
            f"terrain_level={terrain_level:.2f}",
            f"success_rate={env.nav_success_buf.float().mean().item():.3f}",
        ]
    )
    print(" ".join(fields))


def _show_depth_image(env) -> None:
    import cv2

    output = env.noisy_camera.data.output
    data_type = (
        "distance_to_image_plane_noised"
        if "distance_to_image_plane_noised" in output
        else "distance_to_image_plane"
    )
    depth = output[data_type][0, ..., 0]
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    if data_type.endswith("_noised"):
        depth = depth.clamp(0.0, 1.0)
    else:
        max_distance = float(getattr(env.noisy_camera.cfg, "max_distance", 2.5))
        if max_distance <= 0.0 or max_distance > 100.0:
            max_distance = 2.5
        depth = (depth / max_distance).clamp(0.0, 1.0)

    depth_u8 = (depth * 255).to(torch.uint8).cpu().numpy()
    depth_u8 = cv2.resize(
        depth_u8,
        (depth_u8.shape[1] * 8, depth_u8.shape[0] * 8),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imshow("navigation_depth", depth_u8)
    cv2.waitKey(1)


def play_nav():
    env_cfg, agent_cfg = task_registry.get_cfgs(args_cli.task)
    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)

    env_cfg = _make_navigation_env_cfg(env_cfg, agent_cfg)
    terrain_seed = (
        env_cfg.scene.terrain_generator.seed if env_cfg.scene.terrain_generator is not None else None
    )
    print(f"[INFO] Using seed: scene={env_cfg.scene.seed} terrain={terrain_seed}")
    camera_cfg = env_cfg.scene.noisy_camera
    if camera_cfg.enable_camera:
        print(
            "[INFO] Depth camera: "
            f"parent={camera_cfg.prim_body_name} "
            f"fov=({camera_cfg.horizontal_fov_deg:.1f}, {camera_cfg.vertical_fov_deg:.1f}) "
            f"range=({camera_cfg.min_distance:.2f}, {camera_cfg.max_distance:.2f}) "
            f"crop={camera_cfg.crop_region} "
            f"depth_range={camera_cfg.depth_range}"
        )
    env_class = task_registry.get_task_class(args_cli.task)
    env = env_class(env_cfg, args_cli.headless)

    navigation_policy, reset_navigation_policy, navigation_policy_path = _resolve_navigation_policy(env, agent_cfg)
    print(f"[INFO] Loaded navigation policy: {navigation_policy_path}")
    print(f"[INFO] Loaded locomotion policy: {env.cfg.nav.locomotion_policy_path}")
    if args_cli.show_depth:
        if args_cli.headless:
            print("[WARN] --show_depth needs a GUI display; depth window is disabled in headless mode.")
        elif not env.cfg.scene.noisy_camera.enable_camera:
            print("[WARN] --show_depth requested but scene.noisy_camera.enable_camera is False.")
        else:
            print("[INFO] Showing depth image in OpenCV window: navigation_depth")

    _apply_navigation_play_overrides(env)
    obs, _ = env.get_observations()
    if not args_cli.headless and not args_cli.no_start_goal_vis:
        env.visualize_navigation_points(
            max_envs=args_cli.vis_max_envs,
            show_path=not args_cli.no_path_vis,
            show_waypoints=not args_cli.no_path_vis,
            max_path_points_per_env=args_cli.path_vis_max_points,
        )

    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            nav_actions = navigation_policy(obs)
            obs, _, dones, _ = env.step(nav_actions)
            if reset_navigation_policy is not None:
                reset_navigation_policy(dones)
            if torch.any(dones):
                reset_ids = torch.nonzero(dones, as_tuple=False).flatten()
                if _apply_navigation_play_overrides(env, reset_ids):
                    obs, _ = env.get_observations()
            step_count += 1

            if not args_cli.headless and not args_cli.no_start_goal_vis:
                env.visualize_navigation_points(
                    max_envs=args_cli.vis_max_envs,
                    show_path=not args_cli.no_path_vis,
                    show_waypoints=not args_cli.no_path_vis,
                    max_path_points_per_env=args_cli.path_vis_max_points,
                )
            if args_cli.show_depth and not args_cli.headless and env.cfg.scene.noisy_camera.enable_camera:
                _show_depth_image(env)
            if args_cli.print_interval > 0 and step_count % args_cli.print_interval == 0:
                _print_navigation_status(env, step_count)


if __name__ == "__main__":
    play_nav()
    simulation_app.close()
