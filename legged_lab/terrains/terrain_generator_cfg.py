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

"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import FlatPatchSamplingCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from . import maze_terrains_cfg


GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        )
    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_30": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,           # 该地形类型占主地形的比例（20%）
            step_height_range=(0.05, 0.30),  # 台阶高度范围（单位：米）
            step_width=0.3,           # 台阶宽度（单位：米）
            platform_width=3.0,       # 平台宽度（单位：米）
            border_width=1.0,         # 地形边界宽度（单位：米）
            holes=False,              # 是否包含孔洞（False表示无孔洞）
        ),
        "pyramid_stairs_inv_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10,          # 金字塔斜坡地形比例（5%）
            slope_range=(0.0, 0.4),    # 坡度范围（0.0-0.4，对应0°-21.8°）
            platform_width=2.0,       # 平台宽度（单位：米）
            border_width=0.25,         # 边界宽度（单位：米）
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,          # 倒置金字塔斜坡地形比例（5%）
            slope_range=(0.0, 0.4),    # 坡度范围（与正金字塔斜坡相同）
            platform_width=2.0,       # 平台宽度（单位：米）
            border_width=0.25,         # 边界宽度（单位：米）
        ),

        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.10, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
        # ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20, noise_range=(0.00, 0.02), noise_step=0.02, border_width=0.25
        ),
        # "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.15, amplitude_range=(0.0, 0.2), num_waves=5.0),
        "high_platform": terrain_gen.MeshPitTerrainCfg(
            proportion=0.10, pit_depth_range=(0.0, 0.3), platform_width=2.0, double_pit=True
        ),
        # "star": terrain_gen.MeshStarTerrainCfg(
        #     proportion=0.15, num_bars=6, bar_width_range=(0.05, 0.05), bar_height_range=(0.0, 0.25), platform_width=1.0
        # ),
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.10, gap_width_range=(0.1, 0.4), platform_width=2.0
        )
    },
)

PLANNING_TERRAIN_CFG = TerrainGeneratorCfg(
    # 单个 sub-terrain patch 的尺寸，单位 m。
    # 这里每个迷宫 patch 是 24m × 24m。
    # 注意：最终总地形尺寸约为：
    # x方向: size[0] * num_rows = 24 * 10 = 240m
    # y方向: size[1] * num_cols = 24 * 20 = 480m
    size=(24.0, 24.0),

    # terrain patch 的行数。
    # curriculum=True 时，difficulty 通常沿 row 方向递增。
    # 这里有 10 行，因此会生成 10 个主要难度层级。
    num_rows=10,

    # terrain patch 的列数。
    # 同一行的 difficulty 接近，但如果 randomize_layout_per_difficulty=True，
    # 同一难度附近也可以生成不同的迷宫布局。
    # 这里每个难度层级有 20 个不同 patch。
    num_cols=20,

    # 整个 terrain grid 外围额外边界宽度。
    # 设为 0 表示不额外生成外边界区域。
    border_width=0.0,

    # 外围边界高度。
    # border_width=0 时，这个参数基本不起作用。
    border_height=0.0,

    # 是否开启课程学习。
    # True 表示 TerrainGenerator 会给不同 row 传入不同 difficulty。
    # difficulty 越大，你的迷宫会更密、走廊更窄、障碍物更大/更高/更多。
    curriculum=True,

    # difficulty 的范围。
    # 这里表示 difficulty 从 0.0 到 1.0。
    # difficulty=0.0 是 easy，difficulty=1.0 是 hard。
    difficulty_range=(0.0, 1.0),

    # 是否使用地形缓存。
    # 调试阶段建议 False，否则修改参数后可能仍然加载旧地形。
    use_cache=False,

    # TerrainGenerator 的随机种子。
    # 控制整体地形生成过程中的随机性。
    seed=42,

    # 子地形类型集合。
    # 这里只放了 "maze_wall"，所以所有 patch 都是迷宫地形。
    sub_terrains={
        "maze_wall": maze_terrains_cfg.MeshMazeWallTerrainCfg(
            # 当前子地形在 sub_terrains 中的采样比例。
            # 这里只有一种地形，所以设为 1.0。
            proportion=1.0,

            # 迷宫生成区域，格式：
            # (xmin, xmax, ymin, ymax)
            # 坐标是在单个 24m × 24m patch 的局部坐标中。
            # 这里表示迷宫主体生成在 x=[1,23], y=[1,23]，
            # patch 四周各留 1m 空白边界。
            maze_region=(1.0, 23.0, 1.0, 23.0),

            # 墙体固定高度，单位 m。
            # 这里固定为 2m，不随 difficulty 改变。
            # 也就是说墙始终是“不可跨越导航边界”。
            wall_height=2.0,

            # 墙体厚度，单位 m。
            # 越厚碰撞越稳定，但走廊有效宽度会变小。
            wall_thickness=0.15,

            # x 方向 cell 数量课程范围。
            # difficulty=0.0 时约为 5 个 cell，cell 大、走廊宽、迷宫简单。
            # difficulty=1.0 时约为 11 个 cell，cell 小、走廊窄、迷宫复杂。
            # 对 maze_region 宽度 22m 来说：
            # 5 cells  -> cell 宽约 4.4m
            # 11 cells -> cell 宽约 2.0m
            num_cells_x_range=(5, 11),

            # y 方向 cell 数量课程范围。
            # 作用同 num_cells_x_range。
            num_cells_y_range=(5, 11),

            # DFS 连通迷宫生成后，额外打通墙的概率范围。
            # difficulty=0.0 时为 0.5：
            #   额外打通很多墙，墙更稀疏，路径更多。
            # difficulty=1.0 时为 0.03：
            #   额外打通很少墙，更接近标准 DFS maze，墙更密、路径更绕。
            # 这是控制“墙疏密”的核心参数。
            maze_extra_connection_prob_range=(0.5, 0.03),

            # 将同一行内的随机 difficulty 扰动量化成同一个难度等级。
            # raw difficulty 仍用于 layout seed，因此同一难度行的迷宫难度一致、布局不同。
            # 如果修改 num_rows，建议同步修改这个值。
            difficulty_quantization_levels=10,
            layout_seed_uses_raw_difficulty=True,

            # 是否启用 box 障碍物。
            enable_box_obstacles=True,

            # box 障碍物数量课程范围。
            # difficulty=0.0 时每个 patch 约 3 个 box。
            # difficulty=1.0 时每个 patch 约 8 个 box。
            num_box_obstacles_range=(3, 8),

            # 低难度下 box 障碍物的水平尺寸范围，单位 m。
            # 每个 box 的 x/y 尺寸会在这个范围内随机采样。
            # easy 时障碍物小，不容易堵路。
            box_size_range_easy=(0.12, 0.25),

            # 高难度下 box 障碍物的水平尺寸范围，单位 m。
            # hard 时障碍物更大，占据更多通道空间。
            box_size_range_hard=(0.40, 0.60),

            # 低难度下 box 障碍物高度范围，单位 m。
            # easy 时主要是低矮小障碍。
            box_height_range_easy=(0.10, 0.30),

            # 高难度下 box 障碍物高度范围，单位 m。
            # hard 时障碍物更高，更像需要绕行的物体。
            box_height_range_hard=(0.60, 1.20),

            # 是否启用圆柱障碍物。
            enable_cylinder_obstacles=True,

            # 圆柱障碍物数量课程范围。
            # difficulty=0.0 时每个 patch 约 2 个 cylinder。
            # difficulty=1.0 时每个 patch 约 6 个 cylinder。
            num_cylinder_obstacles_range=(2, 6),

            # 低难度下圆柱半径范围，单位 m。
            # 实际圆柱直径是 2 * radius。
            cylinder_radius_range_easy=(0.06, 0.12),

            # 高难度下圆柱半径范围，单位 m。
            # hard 时最大直径约 0.70m。
            cylinder_radius_range_hard=(0.25, 0.35),

            # 低难度下圆柱高度范围，单位 m。
            cylinder_height_range_easy=(0.10, 0.30),

            # 高难度下圆柱高度范围，单位 m。
            cylinder_height_range_hard=(0.60, 1.20),

            # 被 DFS 打通的通道开口保护距离，单位 m。
            # 用来防止障碍物、粗糙 patch、台阶重新堵住关键通路。
            # 值越大，通道越安全，但可放置障碍物空间越少。
            # 如果发现障碍物经常放置失败，可以适当降低，比如 0.25~0.30。
            passage_clearance=0.35,

            # 障碍物与墙、起点区、终点区、其他障碍物之间的默认安全距离，单位 m。
            # 值越大越不容易贴墙，但高难度时也越容易导致障碍物放不下。
            obstacle_clearance=0.18,

            # 是否优先把障碍物放在 cell 内部。
            # 建议 True。
            # 这样比在整个 maze_region 随机撒点更稳定，
            # 高难度下不容易因为和墙/通道 keepout 冲突而被跳过。
            obstacle_place_in_cells=True,

            # 障碍物采样失败时允许逐步降低 clearance。
            # 最低不会低于这个值。
            # 作用：尽量保证高难度下障碍物数量不被“清除”。
            obstacle_min_clearance=0.02,

            # 每个障碍物最多尝试采样次数。
            # 高难度墙密、障碍物大时可以适当提高。
            obstacle_max_tries=300,

            # 障碍物 footprint 相对于 cell 尺寸的最大比例。
            # 例如 0.45 表示障碍物水平尺寸最多约占 cell 尺寸的 45%。
            # 作用：高难度 cell 变小时，自动限制障碍物尺寸，避免堵死整条通道。
            obstacle_footprint_cell_ratio=0.45,

            # 如果障碍物最终没有成功放够数量，是否打印 warning。
            # 调试时建议 True；正式训练时可以 False，避免输出太多。
            warn_on_obstacle_drop=False,

            # 起点附近的留空区域大小，单位 m。
            # 格式为 (x方向尺寸, y方向尺寸)。
            # 该区域内不会生成墙、障碍物、粗糙 patch 或台阶。
            start_clear_size=(2.5, 2.5),

            # 终点附近的留空区域大小，单位 m。
            # 作用同 start_clear_size，避免目标点被堵住。
            goal_clear_size=(2.5, 2.5),

            # 是否启用粗糙地形 patch。
            # 当前开启。
            # 注意：粗糙 patch 会占据较大区域，高难度下可能和墙/通道保护区冲突。
            enable_rough_patches=True,

            # 每个 patch 中生成的粗糙地形 patch 数量。
            num_rough_patches=3,

            # 粗糙地形 patch 的目标尺寸，单位 m。
            # 如果 rough_patch_place_in_cells=True，
            # 高难度 cell 较小时，实际尺寸会被 cell ratio 裁剪。
            rough_patch_size=(2.0, 2.0),

            # 粗糙地形内部小块的尺寸，单位 m。
            # 越小越细碎，但 mesh 数量更多。
            rough_block_size=0.25,

            # 粗糙地形高度范围，单位 m。
            # 建议不要太高，否则人形机器人脚底碰撞容易不稳定。
            rough_height_range=(0.015, 0.05),

            # 粗糙 patch 与墙、障碍物、通道保护区之间的默认距离。
            rough_clearance=0.20,

            # 是否优先把粗糙 patch 放在 cell 内部。
            # 建议 True，可减少与墙体冲突。
            rough_patch_place_in_cells=True,

            # 粗糙 patch 采样失败时允许降低到的最小 clearance。
            rough_min_clearance=0.02,

            # 每个粗糙 patch 的最大采样尝试次数。
            rough_max_tries=300,

            # 粗糙 patch 相对于 cell 尺寸的最大比例。
            # 高难度 cell 变小时，rough patch 会被自动裁小。
            rough_patch_cell_ratio=0.75,

            # 是否启用台阶区域。
            # 当前关闭，因此台阶相关参数不会真正生效。
            enable_stairs=False,

            # 每个 maze patch 中生成的台阶区域数量。
            # enable_stairs=False 时不会使用。
            num_stair_regions=1,

            # 台阶步数。
            # enable_stairs=False 时不会使用。
            stair_num_steps=4,

            # 单级台阶的水平宽度，单位 m。
            # enable_stairs=False 时不会使用。
            stair_step_width=0.32,

            # 单级台阶高度，单位 m。
            # enable_stairs=False 时不会使用。
            stair_step_height=0.04,

            # 台阶整体宽度，单位 m。
            # enable_stairs=False 时不会使用。
            stair_width=0.9,

            # 台阶区域与墙、障碍物、通道保护区之间的安全距离。
            # enable_stairs=False 时不会使用。
            stair_clearance=0.25,

            # 当前子地形自身的随机种子。
            # 控制 DFS 迷宫、障碍物、粗糙 patch、台阶等随机过程。
            seed=42,

            # Navigation reset candidates. IsaacLab samples these on ground-level flat patches,
            # so starts/goals are not placed inside walls or obstacle volumes.
            flat_patch_sampling={
                "navigation": FlatPatchSamplingCfg(
                    num_patches=256,
                    patch_radius=[0.05, 0.35, 0.55],
                    x_range=(-1.0, 21.0),
                    y_range=(-1.0, 21.0),
                    z_range=(-0.05, 0.06),
                    max_height_diff=0.04,
                ),
            },

            # 是否根据 difficulty 派生不同 local seed。
            # True 表示不同 difficulty 的 patch 会有不同布局。
            # 如果 False，则相同 seed 下布局更容易重复。
            randomize_layout_per_difficulty=True,

            # difficulty 参与 seed 扰动时的缩放因子。
            # 数值越大，不同 difficulty 产生的 seed 差异越明显。
            seed_difficulty_scale=1000000,

            # Path-aware navigation curriculum metadata.
            # The terrain still ignores dynamic obstacles for global planning; obstacles remain local disturbances.
            build_path_metadata=True,
            path_candidate_levels=6,
            max_path_candidates_per_level=96,
            min_path_candidate_cell_separation=2,
        )
    },
)
