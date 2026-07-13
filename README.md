# TerraStride-G1

Vision-guided humanoid locomotion across diverse terrain.

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.3.2-blue)](https://mujoco.org/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](LICENSE)

TerraStride-G1 is a LeggedLab/IsaacLab project for Unitree G1 12-DOF lower-body locomotion and navigation. The repository contains training tasks, multi-terrain MuJoCo scenes, and sim2sim tools for evaluating terrain-adaptive locomotion with depth and height-scan perception.

## Demo (Speed 2x Normal Speed)

<p align="center">
  <img src="docs/g1_multiview_hq_speed_2x.gif" width="900" alt="G1 multi-terrain locomotion demo">
</p>

Scope:

- `g1_locomotion`: low-level velocity tracking for the G1 lower body.
- `g1_navigation`: high-level navigation that produces velocity commands for the frozen locomotion policy.
- Height-scan and depth-camera observations for terrain-aware navigation.
- MuJoCo scene with start/goal markers, walls, rough ground, full-corridor stairs, and full-corridor ramps.
- Multi-view MuJoCo recording: world overview, forward third-person follow view, and raw grayscale depth view.
- Goal-triggered stop for sim2sim evaluation and video capture.

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

The current codebase has been tested with:

| Component | Version |
| --- | --- |
| OS | Ubuntu 22.04 |
| Python | 3.10 |
| Isaac Sim | 4.5.0 |
| IsaacLab | 2.1.0 |
| MuJoCo | 3.3.2 |
| RSL-RL | Bundled in `rsl_rl/` |

Use the same Isaac Sim and IsaacLab versions unless you have checked the API changes in your target release.

### 1. Install System Packages

Install basic build and rendering dependencies:

```bash
sudo apt update
sudo apt install -y git git-lfs build-essential cmake \
  libgl1 libglib2.0-0 libx11-6 libegl1 mesa-utils
```

Initialize Git LFS if you plan to manage large assets outside normal Git history:

```bash
git lfs install
```

### 2. Create a Python Environment

Create a clean Conda environment:

```bash
conda create -n terrastride-g1 python=3.10 -y
conda activate terrastride-g1
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install Isaac Sim and IsaacLab

Install Isaac Sim 4.5.0 and IsaacLab 2.1.0 in the same Python environment. Follow the IsaacLab installation guide for your Isaac Sim installation method, then install the matching IsaacLab release.

A typical IsaacLab source installation looks like:

```bash
git clone --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

After installation, verify that IsaacLab is importable from the active environment:

```bash
python -c "import isaaclab; print('IsaacLab import OK')"
```

If this command fails, fix the Isaac Sim / IsaacLab installation before installing this project.

### 4. Clone and Install This Repository

Clone the repository, then install the project package and bundled RSL-RL package in editable mode:

```bash
git clone https://github.com/<your-user>/TerraStride-G1.git
cd TerraStride-G1

python -m pip install -e .
python -m pip install -e rsl_rl
python -m pip install opencv-python tensorboard
```

The root `setup.py` installs the project package and runtime dependencies including `mujoco==3.3.2`, `mujoco-python-viewer`, `pynput`, and `matplotlib`.

### 5. Configure MuJoCo Rendering

For desktop machines with a display server, MuJoCo usually works without extra configuration.

For headless servers, use EGL:

```bash
export MUJOCO_GL=egl
```

You can add this line to `~/.bashrc` or your job script if you always run on a headless GPU server.

Verify MuJoCo and OpenCV imports:

```bash
python -c "import mujoco, cv2; print('MuJoCo', mujoco.__version__); print('OpenCV import OK')"
```

### 6. Prepare Policy Checkpoints

Training logs and checkpoints are intentionally ignored by Git through `.gitignore`:

```text
logs/
outputs/
```

Prepare these artifacts locally before running navigation or sim2sim:

| Artifact | Expected path example | How to obtain |
| --- | --- | --- |
| Low-level locomotion policy | `logs/g1_locomotion/<run>/exported/policy.pt` | Train `g1_locomotion`, or provide a compatible exported policy |
| High-level navigation checkpoint | `logs/g1_navigation/<run>/model_<iter>.pt` | Train `g1_navigation`, or provide a compatible checkpoint |
| Demo videos | `outputs/` | Generated by the sim2sim recording commands |

When running scripts, pass policy paths explicitly unless you use the repository-relative defaults:

```bash
--policy logs/g1_locomotion/<run>/exported/policy.pt
--nav-policy logs/g1_navigation/<run>/model_<iter>.pt
```

### 7. Verify the Project Installation

Run a Python import check:

```bash
python -c "import torch, mujoco, cv2, isaaclab; import legged_lab.envs; print('TerraStride-G1 import OK')"
```

Check that the main task registration file is syntactically valid:

```bash
python -m py_compile \
  legged_lab/envs/__init__.py \
  legged_lab/envs/g1/g1_locomotion_cfg.py \
  legged_lab/envs/g1/g1_locomotion_env.py \
  legged_lab/envs/g1/g1_navigation_cfg.py \
  legged_lab/envs/g1/g1_navigation_env.py
```

After these checks pass, proceed to training or playback.

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

The navigation environment uses a frozen low-level locomotion policy. If policy discovery fails, pass the low-level policy path explicitly.

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

Keep stairs and ramps full-width if the evaluation should force the robot to traverse them. Avoid overlap between these route geoms and rough hfield terrain.

## Troubleshooting

IsaacLab import fails:

- Confirm that the active Conda environment is the same one used to install Isaac Sim and IsaacLab.
- Run `python -c "import isaaclab"` before installing this repository.
- Keep Isaac Sim 4.5.0 and IsaacLab 2.1.0 unless you have verified compatibility with newer APIs.

MuJoCo rendering fails on a server:

- Set `MUJOCO_GL=egl`.
- Make sure the machine has a working NVIDIA driver and EGL support.
- For video-only runs, add `--no-render`, but offscreen rendering must still work.

Robot does not move in MuJoCo:

- Without `--navigation`, the sim2sim script expects joystick or keyboard velocity commands.
- If `--no-joystick` is used without `--navigation`, the command is zero and the robot will stand still.
- For autonomous navigation, always pass `--navigation` and a valid `--nav-policy`.

Navigation policy cannot be found:

- The script searches `logs/g1_navigation/*/model_*.pt` only when `--nav-policy` is omitted.
- Since `logs/` is ignored by Git, cloned repositories do not include trained checkpoints.
- Train `g1_navigation` locally or provide a checkpoint explicitly with `--nav-policy`.

Video output is not clear enough:

- Increase `--video-panel-width` and `--video-panel-height`.
- For MP4 output, use `--video-codec mp4v`.
- If `ffmpeg` is available, post-process the generated video with H.264/CRF for better compression quality.

## Reproducibility Checklist

Before reporting or comparing results, record:

- Git commit hash of this repository.
- Isaac Sim, IsaacLab, MuJoCo, Python, PyTorch, and CUDA versions.
- Low-level policy path passed to `--policy`.
- Navigation checkpoint path passed to `--nav-policy`.
- MuJoCo XML scene path passed to `--xml`.
- Full sim2sim command used for evaluation or video recording.

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
