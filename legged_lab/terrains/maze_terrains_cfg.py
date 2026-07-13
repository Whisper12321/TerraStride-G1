from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from . import maze_terrains


@configclass
class MeshMazeWallTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a connected maze-wall mesh terrain with curriculum.

    Curriculum behavior:
        - ``wall_height`` is fixed and does not use difficulty.
        - ``num_cells_x_range`` / ``num_cells_y_range`` control corridor width.
          Easy: fewer cells -> wider corridors.
          Hard: more cells -> narrower corridors.
        - ``maze_extra_connection_prob_range`` controls wall sparsity.
          Easy: larger value -> more extra openings -> sparser walls.
          Hard: smaller value -> fewer openings -> denser walls.
        - obstacle easy/hard ranges control obstacle size and height.
        - path candidates are grouped by graph difficulty for navigation curriculum.
    """

    function = maze_terrains.maze_wall_terrain

    # ---------------- Maze region ----------------
    maze_region: tuple[float, float, float, float] = (1.0, 23.0, 1.0, 23.0)
    """Maze generation region: (xmin, xmax, ymin, ymax) in local terrain frame."""

    # Legacy fixed cell counts. Used if num_cells_*_range is None.
    num_cells_x: int = 8
    num_cells_y: int = 8

    # Difficulty-controlled effective cell counts.
    # Low difficulty -> fewer cells -> wider corridors.
    # High difficulty -> more cells -> narrower corridors.
    num_cells_x_range: tuple[float, float] | None = (6, 12)
    num_cells_y_range: tuple[float, float] | None = (6, 12)

    # ---------------- Wall parameters ----------------
    wall_height: float = 2.0
    """Fixed wall height. It is intentionally not controlled by difficulty."""

    # Legacy compatibility only. Not used when wall_height is set.
    wall_height_range: tuple[float, float] = (2.0, 2.0)

    wall_thickness: float = 0.15
    """Fixed wall thickness."""

    # Optional wall-thickness curriculum. Keep None unless you explicitly want it.
    wall_thickness_range: tuple[float, float] | None = None

    # Legacy compatibility only. DFS generation does not use this to create walls.
    wall_probability_range: tuple[float, float] = (0.4, 0.4)

    # Wall sparsity curriculum.
    # Easy -> 0.35: many extra openings, sparse walls.
    # Hard -> 0.05: few extra openings, dense maze.
    maze_extra_connection_prob_range: tuple[float, float] | None = (0.35, 0.05)

    # Legacy fallback if maze_extra_connection_prob_range is None.
    maze_extra_connection_prob: float = 0.12

    # Curriculum/layout decoupling.
    # If set, difficulty is quantized before controlling maze density, obstacle size/count,
    # roughness, etc. The raw difficulty can still drive the layout seed so same-level
    # patches have matching difficulty but different layouts.
    difficulty_quantization_levels: int | None = None
    layout_seed_uses_raw_difficulty: bool = True

    maze_start_cell: tuple[int, int] | None = (0, 0)
    passage_clearance: float = 0.35

    ground_height: float = 0.02
    border_wall: bool = True

    # ---------------- Obstacles ----------------
    enable_box_obstacles: bool = True
    # ---------------- 障碍物数量课程 ----------------
    num_box_obstacles_range: tuple[int, int] = (2, 8)
    """Box 障碍物数量课程范围。
    difficulty=0 时使用第一个值，difficulty=1 时使用第二个值。
    """

    # Legacy fixed ranges. Used only if easy/hard ranges are None.
    box_size_range: tuple[float, float] = (0.20, 0.40)
    box_height_range: tuple[float, float] = (0.25, 0.70)

    # Difficulty-controlled box size and height.
    box_size_range_easy: tuple[float, float] | None = (0.15, 0.30)
    box_size_range_hard: tuple[float, float] | None = (0.35, 0.65)
    box_height_range_easy: tuple[float, float] | None = (0.20, 0.45)
    box_height_range_hard: tuple[float, float] | None = (0.80, 1.50)

    enable_cylinder_obstacles: bool = True
    # ---------------- 障碍物数量课程 ----------------
    num_cylinder_obstacles_range: tuple[int, int] = (1, 6)
    """Cylinder 障碍物数量课程范围。
    difficulty=0 时使用第一个值，difficulty=1 时使用第二个值。
    """

    # Legacy fixed ranges. Used only if easy/hard ranges are None.
    cylinder_radius_range: tuple[float, float] = (0.12, 0.25)
    cylinder_height_range: tuple[float, float] = (0.25, 0.80)

    # Difficulty-controlled cylinder radius and height.
    cylinder_radius_range_easy: tuple[float, float] | None = (0.08, 0.16)
    cylinder_radius_range_hard: tuple[float, float] | None = (0.20, 0.35)
    cylinder_height_range_easy: tuple[float, float] | None = (0.20, 0.45)
    cylinder_height_range_hard: tuple[float, float] | None = (0.80, 1.50)

    obstacle_clearance: float = 0.18

    # Obstacle placement robustness.
    # True: sample obstacles inside maze cells instead of the entire maze region.
    # This prevents dense passage keepouts from causing high-difficulty obstacles to be skipped.
    obstacle_place_in_cells: bool = True

    # If placement is difficult, clearance is gradually relaxed down to this value.
    obstacle_min_clearance: float = 0.02

    # Maximum sampling attempts per obstacle. Increase if you still see dropped obstacles.
    obstacle_max_tries: int = 300

    # Maximum obstacle footprint relative to the current cell size.
    # This caps high-difficulty obstacle width/diameter so it still fits in narrow corridors.
    obstacle_footprint_cell_ratio: float = 0.45

    # Print placement counts when some obstacles or rough patches cannot be placed.
    warn_on_obstacle_drop: bool = False

    # ---------------- Rough terrain patches ----------------
    enable_rough_patches: bool = False
    num_rough_patches: int = 3
    rough_patch_size: tuple[float, float] = (2.0, 2.0)
    rough_block_size: float = 0.25
    rough_height_range: tuple[float, float] = (0.015, 0.05)
    rough_clearance: float = 0.20

    # Rough patch placement robustness.
    # True: place rough patches inside cells and clamp their size to fit current cell size.
    rough_patch_place_in_cells: bool = True
    rough_min_clearance: float = 0.02
    rough_max_tries: int = 300
    rough_patch_cell_ratio: float = 0.75

    # ---------------- Stair regions ----------------
    enable_stairs: bool = True
    num_stair_regions: int = 1
    stair_num_steps: int = 4
    stair_step_width: float = 0.32
    stair_step_height: float = 0.04
    stair_width: float = 0.9
    stair_clearance: float = 0.25

    # ---------------- Start / goal clear areas ----------------
    start_clear_size: tuple[float, float] = (2.5, 2.5)
    goal_clear_size: tuple[float, float] = (2.5, 2.5)

    start_pos: tuple[float, float] | None = None
    goal_pos: tuple[float, float] | None = None

    seed: int = 42
    randomize_layout_per_difficulty: bool = True
    seed_difficulty_scale: int = 1000000

    # ---------------- Path-planning metadata ----------------
    build_path_metadata: bool = True
    path_candidate_levels: int = 6
    max_path_candidates_per_level: int = 96
    min_path_candidate_cell_separation: int = 2
    generated_maze_metadata: list | None = None
