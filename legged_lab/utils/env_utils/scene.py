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

from typing import TYPE_CHECKING
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg

from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from legged_lab.sensors.camera import TiledCameraCfg
from legged_lab.terrains.ray_caster_cfg import RayCasterCfg
from legged_lab.sensors.noisy_camera import NoisyGroupedRayCasterCameraCfg

from legged_lab.utils.noise import (
    CropAndResizeCfg,
    GaussianBlurNoiseCfg,
    DepthNormalizationCfg,
)



if TYPE_CHECKING:
    from legged_lab.envs.base.base_env_config import BaseSceneCfg


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    def __init__(self, config: "BaseSceneCfg", physics_dt, step_dt):
        super().__init__(num_envs=config.num_envs, env_spacing=config.env_spacing)

        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type=config.terrain_type,
            terrain_generator=config.terrain_generator,
            max_init_terrain_level=config.max_init_terrain_level,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        self.robot: ArticulationCfg = config.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, update_period=physics_dt
        )

        self.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        self.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=(
                    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
                ),
            ),
        )

        if config.height_scanner.enable_height_scan:
            self.height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.height_scanner.prim_body_name,
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=patterns.GridPatternCfg(
                    resolution=config.height_scanner.resolution, size=config.height_scanner.size
                ),
                debug_vis=config.height_scanner.debug_vis,
                mesh_prim_paths=["/World/ground"],
                update_period=step_dt,
                drift_range=config.height_scanner.drift_range,
            )

        if config.Lfoot_height_scanner.enable_height_scan:
            self.Lfoot_height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.Lfoot_height_scanner.prim_body_name,
                offset=RayCasterCfg.OffsetCfg(pos=(0.03, 0.0, 20.0)),
                # attach_yaw_only=True,
                ray_alignment='yaw',
                pattern_cfg=patterns.GridPatternCfg(
                    resolution=config.Lfoot_height_scanner.resolution, size=config.Lfoot_height_scanner.size
                ),
                debug_vis=config.Lfoot_height_scanner.debug_vis,
                mesh_prim_paths=["/World/ground"],
                update_period=step_dt,
                drift_range=config.Lfoot_height_scanner.drift_range,
            )
        if config.Rfoot_height_scanner.enable_height_scan:
            self.Rfoot_height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.Rfoot_height_scanner.prim_body_name,
                offset=RayCasterCfg.OffsetCfg(pos=(0.03, 0.0, 20.0)),
                # attach_yaw_only=True,
                ray_alignment='yaw',
                pattern_cfg=patterns.GridPatternCfg(
                    resolution=config.Rfoot_height_scanner.resolution, size=config.Rfoot_height_scanner.size
                ),
                debug_vis=config.Rfoot_height_scanner.debug_vis,
                mesh_prim_paths=["/World/ground"],
                update_period=step_dt,
                drift_range=config.Rfoot_height_scanner.drift_range,
            )


        if config.lidar.enable_lidar:
            self.lidar = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.lidar.prim_body_name,
                offset=RayCasterCfg.OffsetCfg(pos=config.lidar.offset, rot=config.lidar.rotation),
                ray_alignment="yaw",
                pattern_cfg=config.lidar.pattern_cfg,
                debug_vis=config.lidar.debug_vis,
                mesh_prim_paths=config.lidar.mesh_prim_paths,
                max_distance=config.lidar.max_distance,
            )

        if config.depth_camera.enable_depth_camera:
            self.depth_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.depth_camera.prim_body_name,
                offset=config.depth_camera.offset,
                height=config.depth_camera.height,
                width=config.depth_camera.width,
                data_types=config.depth_camera.data_types,
                spawn=config.depth_camera.spawn,
                debug_vis=config.depth_camera.debug_vis,
                visualizer_cfg=config.depth_camera.visualizer_cfg,
            )

        if config.noisy_camera.enable_camera:
            noisy_camera_mesh_prim_paths = config.noisy_camera.mesh_prim_paths
            if noisy_camera_mesh_prim_paths is None:
                noisy_camera_mesh_prim_paths = ["/World/ground/", "/World/envs/env_.*/Robot/(?!.*pelvis).*"]
            self.noisy_camera = NoisyGroupedRayCasterCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.noisy_camera.prim_body_name,
                mesh_prim_paths=noisy_camera_mesh_prim_paths,
                attach_yaw_only=False,
                pattern_cfg=PinholeCameraPatternCfg(
                    focal_length=1.0,
                    horizontal_aperture=2 * math.tan(math.radians(config.noisy_camera.horizontal_fov_deg) / 2),
                    vertical_aperture=2 * math.tan(math.radians(config.noisy_camera.vertical_fov_deg) / 2),
                    width=config.noisy_camera.width,
                    height=config.noisy_camera.height,
                ),
                debug_vis=config.noisy_camera.debug_vis,
                data_types=["distance_to_image_plane"],
                update_period=0.02,
                depth_clipping_behavior="max",
                offset=NoisyGroupedRayCasterCameraCfg.OffsetCfg(
                    pos=config.noisy_camera.offset_pos,
                    rot=config.noisy_camera.offset_rot,
                    convention=config.noisy_camera.offset_convention,
                ),
                aux_mesh_and_link_names=config.noisy_camera.aux_mesh_and_link_names,
                min_distance=config.noisy_camera.min_distance,
                max_distance=config.noisy_camera.max_distance,
                # noise
                noise_pipeline={
                    "crop_and_resize": CropAndResizeCfg(crop_region=config.noisy_camera.crop_region),
                    "gaussian_blur": GaussianBlurNoiseCfg(kernel_size=3, sigma=1),
                    "depth_normalization": DepthNormalizationCfg(
                        depth_range=config.noisy_camera.depth_range,
                        normalize=True,
                        output_range=(0.0, 1.0),
                    ),
                },
                data_histories={"distance_to_image_plane_noised": 37},
            )
