# G1 Obstacle Navigation and Locomotion

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.3.2-blue)](https://mujoco.org/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](LICENSE)

This repository is built on top of LeggedLab, IsaacLab, and RSL-RL. It focuses on Unitree G1 12-DOF lower-body locomotion, high-level navigation, perception-aware obstacle traversal, and MuJoCo sim2sim validation.

## Demo

<p align="center">
  <img src="docs/g1_multiview_hq_speed_2x.gif" width="900" alt="G1 obstacle navigation demo">
</p>

Main features:

- Low-level G1 velocity-tracking locomotion policy.
- High-level navigation policy that outputs velocity commands and runs a frozen low-level locomotion policy internally.
- Height-scan and depth-camera perception inputs.
- MuJoCo maze-style sim2sim scene with start and goal markers, walls, rough terrain, full-corridor stairs, and full-corridor ramps.
- Multi-view sim2sim visualization and recording: world overview, forward third-person follow camera, and raw grayscale depth-camera view.
- Optional automatic stop when the robot reaches the target.

## Repository Structure

```text
.
|-- Exported_policy/                         # Exported TorchScript policies
|-- docs/                                    # Documentation images and demo assets
|-- legged_lab/
|   |-- assets/g1_description/               # G1 MJCF/URDF assets and MuJoCo scenes
|   |-- envs/g1/                             # G1 locomotion/navigation environments
|   |-- mdp/                                 # Locomotion/navigation rewards
|   |-- scripts/                             # Training, playback, and sim2sim scripts
|   |-- sensors/                             # Sensor-related modules
|   |-- terrains/                            # IsaacLab terrain configurations
|   `-- utils/                               # Task registry, CLI helpers, and utilities
|-- rsl_rl/                                  # Bundled RSL-RL implementation
|-- logs/                                    # Training logs and checkpoints, ignored by git
`-- outputs/                                 # Videos and other generated outputs, ignored by git
```

Registered tasks:

| Task | Description |
| --- | --- |
| `g1_locomotion` | Low-level G1 velocity-tracking locomotion policy |
| `g1_navigation` | High-level navigation policy with perception inputs |

## Installation

Recommended environment:

- Ubuntu 22.04
- Python 3.10
- NVIDIA GPU with a working CUDA driver
- Isaac Sim 4.5.0
- IsaacLab 2.1.0
- MuJoCo 3.3.2

Install Isaac Sim and IsaacLab first by following the official IsaacLab installation guide. After confirming that the active Python environment can import `isaaclab`, install this project:

```bash
pip install -e .
pip install -e rsl_rl
pip install opencv-python
```

For MuJoCo rendering on a headless server, you may need:

```bash
export MUJOCO_GL=egl
```

## Training

Train the low-level locomotion policy:

```bash
python legged_lab/scripts/train.py \
  --task=g1_locomotion \
  --headless \
  --logger=tensorboard \
  --num_envs=4096
```

Training outputs are saved under:

```text
logs/g1_locomotion/<time_stamp>/
```

Train the high-level navigation policy:

```bash
python legged_lab/scripts/train.py \
  --task=g1_navigation \
  --headless \
  --logger=tensorboard \
  --num_envs=1024
```

The navigation environment uses a frozen low-level locomotion policy. If automatic policy discovery fails, specify the low-level policy path in the config or playback command.

## IsaacLab Playback

Play a low-level locomotion policy:

```bash
python legged_lab/scripts/play.py \
  --task=g1_locomotion \
  --num_envs=1
```

Play a high-level navigation policy:

```bash
python legged_lab/scripts/play_nav.py \
  --task=g1_navigation \
  --num_envs=1 \
  --navigation_policy_path logs/g1_navigation/<run>/model_<iter>.pt \
  --locomotion_policy_path logs/g1_locomotion/<run>/exported/policy.pt
```

Show the depth camera during navigation playback:

```bash
python legged_lab/scripts/play_nav.py \
  --task=g1_navigation \
  --num_envs=1 \
  --show_depth \
  --show_depth_rays
```

## MuJoCo Sim2Sim

Main script:

```text
legged_lab/scripts/sim2sim_g1_planning_walk_refactored.py
```

Default MuJoCo scene:

```text
legged_lab/assets/g1_description/scene_navigation_locomotion_terrain_v4_hfield.xml
```

The scene contains:

- `start_marker`: start position.
- `goal_marker`: target position.
- `route_start_stair_*`: full-corridor stairs near the robot start area.
- `route_gate_ramp_*`: full-corridor up/down ramp in the middle route.
- `rough_regular_*`: rough hfield terrain that does not overlap the stair or ramp sections.
- Walls, obstacles, and corridor constraints that force the robot through the designed terrain.

Run low-level locomotion in MuJoCo with manual or joystick commands:

```bash
python legged_lab/scripts/sim2sim_g1_planning_walk_refactored.py \
  --policy logs/g1_locomotion/<run>/exported/policy.pt
```

Enable high-level navigation and let the robot move toward the goal:

```bash
python legged_lab/scripts/sim2sim_g1_planning_walk_refactored.py \
  --navigation \
  --policy logs/g1_locomotion/<run>/exported/policy.pt \
  --nav-policy logs/g1_navigation/<run>/model_<iter>.pt
```

If `--nav-policy` is omitted, the script searches for the newest checkpoint matching `logs/g1_navigation/*/model_*.pt`.

## Multi-View Visualization and Video Recording

Open all three views:

```bash
python legged_lab/scripts/sim2sim_g1_planning_walk_refactored.py \
  --navigation \
  --multi-view \
  --policy logs/g1_locomotion/<run>/exported/policy.pt
```

The three views are:

- World overview: top-down view of the full terrain, robot, start marker, and goal marker.
- Third-person follow view: close camera that follows the robot and looks forward.
- Depth-camera view: raw grayscale depth image.

Record the three-view combined video and stop the program when the robot reaches the goal:

```bash
python legged_lab/scripts/sim2sim_g1_planning_walk_refactored.py \
  --navigation \
  --multi-view \
  --record-video \
  --stop-on-goal \
  --video-path outputs/g1_multiview.mp4 \
  --third-person-azimuth 0 \
  --third-person-distance 1.5
```

Record a higher-resolution MP4:

```bash
python legged_lab/scripts/sim2sim_g1_planning_walk_refactored.py \
  --navigation \
  --multi-view \
  --record-video \
  --stop-on-goal \
  --video-path outputs/g1_multiview_hq.mp4 \
  --video-codec mp4v \
  --video-panel-width 1280 \
  --video-panel-height 720 \
  --video-fps 30 \
  --third-person-azimuth 0 \
  --third-person-distance 1.5
```

The final video width is the sum of the three horizontal panels: `3 * --video-panel-width`. For example, `1280x720` panels produce a `3840x720` video.

Record without opening the real-time viewers:

```bash
python legged_lab/scripts/sim2sim_g1_planning_walk_refactored.py \
  --navigation \
  --no-render \
  --record-video \
  --stop-on-goal \
  --video-path outputs/g1_multiview.mp4 \
  --third-person-azimuth 0 \
  --third-person-distance 1.5
```

Useful camera and recording arguments:

| Argument | Description |
| --- | --- |
| `--third-person-azimuth 0` | Make the follow camera look toward the robot's forward direction |
| `--third-person-distance 1.5` | Set the follow-camera distance to 1.5 m |
| `--world-distance 28` | Set the world overview camera distance |
| `--world-elevation -89` | Use an almost top-down world view |
| `--video-panel-width 1280` | Width of each video panel |
| `--video-panel-height 720` | Height of each video panel |
| `--video-codec mp4v` | MP4 FourCC codec |
| `--stop-on-goal` | Stop the entire program once the goal is reached |
| `--stop-recording-on-goal` | Stop only the video recording once the goal is reached |

## Terrain Editing

The main MuJoCo terrain XML is:

```text
legged_lab/assets/g1_description/scene_navigation_locomotion_terrain_v4_hfield.xml
```

To adjust the stair height:

- Edit `pos.z` and `size.z` of `route_start_stair_up_*` and `route_start_stair_down_*`.
- For MuJoCo box geoms, `size.z` is half height, so the top surface is approximately `pos.z + size.z`.
- The current start-corridor stair height reaches about 0.21 m.

To adjust the ramp height or length:

- Edit `route_gate_ramp_up`, `route_gate_ramp_platform`, and `route_gate_ramp_down`.
- `quat` controls the slope angle, while `size` and `pos` control the ramp length, width, and height.

For a convincing locomotion demonstration, stairs and ramps should fill the full corridor width and should not overlap rough hfield terrain.

## License

This project follows the BSD-3-Clause License. It contains code derived from RSL-RL, IsaacLab, LeggedLab, and TienKung-Lab. See [LICENSE](LICENSE) for details.

## Acknowledgement

This project builds on the following open-source projects and resources:

- [IsaacLab](https://isaac-sim.github.io/IsaacLab)
- [LeggedLab](https://github.com/Hellod035/LeggedLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [MuJoCo](https://mujoco.org/)
- [Unitree G1 Description](https://www.unitree.com/g1/)
- [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)
