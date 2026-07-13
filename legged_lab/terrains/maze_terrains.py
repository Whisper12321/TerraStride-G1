"""Functions to generate connected maze terrains with difficulty-controlled curriculum.

This module is intended to live in the user's own workspace, not inside Isaac Lab source.
It generates:
    1. base ground
    2. connected DFS maze walls
    3. optional box/cylinder obstacles
    4. optional rough patches
    5. optional stair regions

Curriculum design:
    - wall height is fixed by ``cfg.wall_height``
    - difficulty controls wall sparsity via ``maze_extra_connection_prob_range``
    - difficulty controls corridor width via ``num_cells_x_range`` / ``num_cells_y_range``
    - difficulty controls obstacle size and height via easy/hard range pairs
    - obstacles are sampled inside maze cells with adaptive clearance so high-difficulty objects are not silently dropped
"""


from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf  # noqa: F401
import torch  # noqa: F401
import trimesh
from typing import TYPE_CHECKING

from isaaclab.utils import *  # noqa: F401, F403

from legged_lab.utils.path_planning import build_grid_maze_graph, generate_path_candidates, summarize_graph

if TYPE_CHECKING:
    from . import maze_terrains_cfg


Rect = tuple[float, float, float, float]  # xmin, xmax, ymin, ymax
Color = tuple[int, int, int, int]
GENERATED_MAZE_METADATA: list[dict] = []


def get_generated_maze_metadata() -> list[dict]:
    return GENERATED_MAZE_METADATA


def maze_wall_terrain(
    difficulty: float,
    cfg: maze_terrains_cfg.MeshMazeWallTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a connected maze-wall terrain with optional obstacles and terrain features.

    Difficulty controls:
        - wall sparsity: ``maze_extra_connection_prob_range``
          Easy -> more extra openings -> sparser walls.
          Hard -> fewer extra openings -> denser walls.
        - corridor width: ``num_cells_x_range`` / ``num_cells_y_range``
          Easy -> fewer cells -> wider corridors.
          Hard -> more cells -> narrower corridors.
        - obstacle size / height: easy/hard obstacle ranges.

    Wall height is intentionally fixed and does not use difficulty.
    """

    raw_difficulty = float(np.clip(difficulty, 0.0, 1.0))
    difficulty = _quantize_difficulty(
        raw_difficulty,
        getattr(cfg, "difficulty_quantization_levels", None),
    )
    meshes_list: list[trimesh.Trimesh] = []
    occupied_rects: list[Rect] = []
    navigation_blocking_rects: list[Rect] = []

    # -----------------------------
    # 1. Resolve basic parameters
    # -----------------------------
    terrain_length, terrain_width = cfg.size
    xmin, xmax, ymin, ymax = cfg.maze_region

    maze_length = xmax - xmin
    maze_width = ymax - ymin

    if maze_length <= 0.0 or maze_width <= 0.0:
        raise ValueError(f"Invalid maze_region={cfg.maze_region}. Expected xmin < xmax and ymin < ymax.")

    # Corridor width curriculum: fewer cells -> wider corridors, more cells -> narrower corridors.
    num_cells_x = _resolve_int_curriculum(
        cfg=cfg,
        attr_range="num_cells_x_range",
        attr_legacy="num_cells_x",
        difficulty=difficulty,
        min_value=1,
    )
    num_cells_y = _resolve_int_curriculum(
        cfg=cfg,
        attr_range="num_cells_y_range",
        attr_legacy="num_cells_y",
        difficulty=difficulty,
        min_value=1,
    )

    if num_cells_x <= 0 or num_cells_y <= 0:
        raise ValueError(f"num_cells_x and num_cells_y must be positive. Received: {num_cells_x}, {num_cells_y}")

    # Wall height is fixed. If the new parameter does not exist, fall back to the old range's first value.
    wall_height = float(getattr(cfg, "wall_height", getattr(cfg, "wall_height_range", (0.8, 0.8))[0]))

    # Wall thickness can be fixed or optionally controlled by difficulty.
    if getattr(cfg, "wall_thickness_range", None) is not None:
        wall_t = _lerp_range(cfg.wall_thickness_range, difficulty)
    else:
        wall_t = float(cfg.wall_thickness)

    ground_h = float(cfg.ground_height)
    if getattr(cfg, "randomize_layout_per_difficulty", True):
        seed_difficulty = raw_difficulty if getattr(cfg, "layout_seed_uses_raw_difficulty", True) else difficulty
        local_seed = int(
            cfg.seed
            + abs(hash(round(float(seed_difficulty), 6))) % getattr(cfg, "seed_difficulty_scale", 1000000)
        )
    else:
        local_seed = cfg.seed

    rng = np.random.default_rng(local_seed)

    # -----------------------------
    # 2. Base ground
    # -----------------------------
    ground_mesh = make_box(
        center=(0.5 * terrain_length, 0.5 * terrain_width, -0.5 * ground_h),
        size=(terrain_length, terrain_width, ground_h),
        color=(120, 120, 120, 255),
    )
    meshes_list.append(ground_mesh)

    # -----------------------------
    # 3. Start / goal clear areas
    # -----------------------------
    if cfg.start_pos is None:
        start_xy = np.array(
            [xmin + 0.5 * cfg.start_clear_size[0], ymin + 0.5 * cfg.start_clear_size[1]], dtype=np.float64
        )
    else:
        start_xy = np.array(cfg.start_pos, dtype=np.float64)

    if cfg.goal_pos is None:
        goal_xy = np.array(
            [xmax - 0.5 * cfg.goal_clear_size[0], ymax - 0.5 * cfg.goal_clear_size[1]], dtype=np.float64
        )
    else:
        goal_xy = np.array(cfg.goal_pos, dtype=np.float64)

    def _in_clear_zone(x: float, y: float) -> bool:
        return in_clear_zone(x, y, start_xy, goal_xy, cfg.start_clear_size, cfg.goal_clear_size)

    occupied_rects.append(_clear_rect(start_xy, cfg.start_clear_size))
    occupied_rects.append(_clear_rect(goal_xy, cfg.goal_clear_size))

    # -----------------------------
    # 4. Border walls
    # -----------------------------
    wall_color = (80, 80, 80, 255)

    def add_wall(center: tuple[float, float, float], size: tuple[float, float, float]) -> None:
        meshes_list.append(make_box(center=center, size=size, color=wall_color))
        rect = rect_from_center_size(center[:2], size[:2], extra=0.02)
        occupied_rects.append(rect)
        navigation_blocking_rects.append(rect)

    if cfg.border_wall:
        add_wall(
            center=((xmin + xmax) * 0.5, ymax - 0.5 * wall_t, 0.5 * wall_height),
            size=(maze_length, wall_t, wall_height),
        )
        add_wall(
            center=((xmin + xmax) * 0.5, ymin + 0.5 * wall_t, 0.5 * wall_height),
            size=(maze_length, wall_t, wall_height),
        )
        add_wall(
            center=(xmax - 0.5 * wall_t, (ymin + ymax) * 0.5, 0.5 * wall_height),
            size=(wall_t, maze_width, wall_height),
        )
        add_wall(
            center=(xmin + 0.5 * wall_t, (ymin + ymax) * 0.5, 0.5 * wall_height),
            size=(wall_t, maze_width, wall_height),
        )

    # -----------------------------
    # 5. Connected internal maze walls
    # -----------------------------
    cell_x = maze_length / num_cells_x
    cell_y = maze_width / num_cells_y

    # Wall sparsity curriculum: low difficulty -> larger extra connection probability -> sparser walls.
    if getattr(cfg, "maze_extra_connection_prob_range", None) is not None:
        extra_connection_prob = _lerp_range(cfg.maze_extra_connection_prob_range, difficulty)
    else:
        extra_connection_prob = float(getattr(cfg, "maze_extra_connection_prob", 0.0))
    extra_connection_prob = float(np.clip(extra_connection_prob, 0.0, 1.0))

    vertical_walls, horizontal_walls, vertical_passages, horizontal_passages = generate_connected_maze_walls(
        num_cells_x=num_cells_x,
        num_cells_y=num_cells_y,
        rng=rng,
        extra_connection_prob=extra_connection_prob,
        start_cell=getattr(cfg, "maze_start_cell", None),
    )

    maze_metadata = None
    if bool(getattr(cfg, "build_path_metadata", True)):
        graph = build_grid_maze_graph(
            num_cells_x=num_cells_x,
            num_cells_y=num_cells_y,
            maze_region=cfg.maze_region,
            vertical_walls=vertical_walls,
            horizontal_walls=horizontal_walls,
        )
        maze_metadata = {
            "raw_difficulty": raw_difficulty,
            "difficulty": difficulty,
            "local_seed": local_seed,
            "num_cells_x": num_cells_x,
            "num_cells_y": num_cells_y,
            "cell_size": (cell_x, cell_y),
            "maze_region": cfg.maze_region,
            "origin_xy": None,
            "graph": graph,
            "graph_summary": summarize_graph(graph),
            "path_candidates_by_level": generate_path_candidates(
                graph,
                num_levels=int(getattr(cfg, "path_candidate_levels", 6)),
                max_candidates_per_level=int(getattr(cfg, "max_path_candidates_per_level", 96)),
                min_cell_separation=int(getattr(cfg, "min_path_candidate_cell_separation", 2)),
            ),
        }

    passage_clearance = float(getattr(cfg, "passage_clearance", 0.20))
    _add_passage_keepout_rects(
        occupied_rects=occupied_rects,
        vertical_passages=vertical_passages,
        horizontal_passages=horizontal_passages,
        xmin=xmin,
        ymin=ymin,
        cell_x=cell_x,
        cell_y=cell_y,
        opening_size=(max(cell_x * 0.35, 0.4), max(cell_y * 0.35, 0.4)),
        clearance=passage_clearance,
    )

    # Vertical walls: along y, located on x cell boundaries.
    for ix, iy in vertical_walls:
        x = xmin + ix * cell_x
        y = ymin + (iy + 0.5) * cell_y
        if _in_clear_zone(x, y):
            continue
        wall_len_y = max(cell_y - 0.25 * wall_t, 0.1)
        add_wall(center=(x, y, 0.5 * wall_height), size=(wall_t, wall_len_y, wall_height))

    # Horizontal walls: along x, located on y cell boundaries.
    for ix, iy in horizontal_walls:
        x = xmin + (ix + 0.5) * cell_x
        y = ymin + iy * cell_y
        if _in_clear_zone(x, y):
            continue
        wall_len_x = max(cell_x - 0.25 * wall_t, 0.1)
        add_wall(center=(x, y, 0.5 * wall_height), size=(wall_len_x, wall_t, wall_height))

    # -----------------------------
    # 6. Optional box / cylinder obstacles
    # -----------------------------
    # Build cell-center candidates for obstacle/feature placement.
    # This avoids the previous failure mode where dense maze walls + passage keepouts
    # consumed most of the free random sampling area and high-difficulty obstacles were skipped.
    cell_candidates = _make_cell_candidates(
        xmin=xmin,
        ymin=ymin,
        cell_x=cell_x,
        cell_y=cell_y,
        num_cells_x=num_cells_x,
        num_cells_y=num_cells_y,
        start_xy=start_xy,
        goal_xy=goal_xy,
        start_clear_size=cfg.start_clear_size,
        goal_clear_size=cfg.goal_clear_size,
    )

    if getattr(cfg, "enable_box_obstacles", False):
        _add_box_obstacles(
            meshes_list,
            occupied_rects,
            navigation_blocking_rects,
            rng,
            cfg,
            start_xy,
            goal_xy,
            difficulty,
            cell_candidates,
            cell_x,
            cell_y,
        )

    if getattr(cfg, "enable_cylinder_obstacles", False):
        _add_cylinder_obstacles(
            meshes_list,
            occupied_rects,
            navigation_blocking_rects,
            rng,
            cfg,
            start_xy,
            goal_xy,
            difficulty,
            cell_candidates,
            cell_x,
            cell_y,
        )

    # -----------------------------
    # 7. Optional rough patches
    # -----------------------------
    if getattr(cfg, "enable_rough_patches", False):
        _add_rough_patches(
            meshes_list,
            occupied_rects,
            navigation_blocking_rects,
            rng,
            cfg,
            start_xy,
            goal_xy,
            cell_candidates,
            cell_x,
            cell_y,
        )

    # -----------------------------
    # 8. Optional stair regions
    # -----------------------------
    if getattr(cfg, "enable_stairs", False):
        _add_stair_regions(meshes_list, occupied_rects, navigation_blocking_rects, rng, cfg, start_xy, goal_xy)

    # -----------------------------
    # 9. Terrain origin
    # -----------------------------
    origin = np.array([start_xy[0], start_xy[1], 0.0], dtype=np.float32)
    if maze_metadata is not None:
        maze_metadata["origin_xy"] = (float(origin[0]), float(origin[1]))
        maze_metadata["occupied_rects"] = tuple(occupied_rects)
        maze_metadata["navigation_blocking_rects"] = tuple(navigation_blocking_rects)
        _append_maze_metadata(cfg, maze_metadata)
    return meshes_list, origin


# -------------------------------------------------------------------------------------------------
# Connected maze generation
# -------------------------------------------------------------------------------------------------


def generate_connected_maze_walls(
    num_cells_x: int,
    num_cells_y: int,
    rng: np.random.Generator,
    extra_connection_prob: float = 0.0,
    start_cell: tuple[int, int] | None = None,
) -> tuple[
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
]:
    """Generate connected maze walls with DFS.

    Returns:
        vertical_walls: walls between cell (ix-1, iy) and (ix, iy), ix in [1, nx-1].
        horizontal_walls: walls between cell (ix, iy-1) and (ix, iy), iy in [1, ny-1].
        vertical_passages: removed vertical walls, used to protect open passages from obstacles.
        horizontal_passages: removed horizontal walls, used to protect open passages from obstacles.
    """
    nx = int(num_cells_x)
    ny = int(num_cells_y)

    vertical_walls = {(ix, iy) for ix in range(1, nx) for iy in range(ny)}
    horizontal_walls = {(ix, iy) for ix in range(nx) for iy in range(1, ny)}

    carved_vertical: set[tuple[int, int]] = set()
    carved_horizontal: set[tuple[int, int]] = set()

    if start_cell is None:
        start_cell = (0, 0)
    sx = int(np.clip(start_cell[0], 0, nx - 1))
    sy = int(np.clip(start_cell[1], 0, ny - 1))

    visited = np.zeros((nx, ny), dtype=bool)
    stack = [(sx, sy)]
    visited[sx, sy] = True

    while stack:
        cx, cy = stack[-1]
        neighbors: list[tuple[int, int]] = []
        if cx > 0 and not visited[cx - 1, cy]:
            neighbors.append((cx - 1, cy))
        if cx < nx - 1 and not visited[cx + 1, cy]:
            neighbors.append((cx + 1, cy))
        if cy > 0 and not visited[cx, cy - 1]:
            neighbors.append((cx, cy - 1))
        if cy < ny - 1 and not visited[cx, cy + 1]:
            neighbors.append((cx, cy + 1))

        if not neighbors:
            stack.pop()
            continue

        next_x, next_y = neighbors[int(rng.integers(0, len(neighbors)))]

        if next_x == cx + 1:
            wall = (cx + 1, cy)
            vertical_walls.discard(wall)
            carved_vertical.add(wall)
        elif next_x == cx - 1:
            wall = (cx, cy)
            vertical_walls.discard(wall)
            carved_vertical.add(wall)
        elif next_y == cy + 1:
            wall = (cx, cy + 1)
            horizontal_walls.discard(wall)
            carved_horizontal.add(wall)
        elif next_y == cy - 1:
            wall = (cx, cy)
            horizontal_walls.discard(wall)
            carved_horizontal.add(wall)

        visited[next_x, next_y] = True
        stack.append((next_x, next_y))

    # Extra openings create loops and reduce wall density while preserving connectivity.
    extra_connection_prob = float(np.clip(extra_connection_prob, 0.0, 1.0))
    if extra_connection_prob > 0.0:
        for wall in list(vertical_walls):
            if rng.random() < extra_connection_prob:
                vertical_walls.remove(wall)
                carved_vertical.add(wall)
        for wall in list(horizontal_walls):
            if rng.random() < extra_connection_prob:
                horizontal_walls.remove(wall)
                carved_horizontal.add(wall)

    return vertical_walls, horizontal_walls, carved_vertical, carved_horizontal


# -------------------------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------------------------
def _lerp_int_range(value_range: tuple[int, int], difficulty: float) -> int:
    value = value_range[0] + difficulty * (value_range[1] - value_range[0])
    return int(round(value))


def _append_maze_metadata(cfg, metadata: dict) -> None:
    if not hasattr(cfg, "generated_maze_metadata") or getattr(cfg, "generated_maze_metadata") is None:
        setattr(cfg, "generated_maze_metadata", [])
    generated = getattr(cfg, "generated_maze_metadata")
    metadata["call_index"] = len(GENERATED_MAZE_METADATA)
    generated.append(metadata)
    GENERATED_MAZE_METADATA.append(metadata)

def _quantize_difficulty(difficulty: float, levels: int | None) -> float:
    if levels is None or int(levels) <= 1:
        return float(difficulty)
    num_levels = int(levels)
    level = min(int(np.floor(float(difficulty) * num_levels)), num_levels - 1)
    return float(level / max(num_levels - 1, 1))


def _lerp_range(value_range: tuple[float, float], difficulty: float) -> float:
    return float(value_range[0] + difficulty * (value_range[1] - value_range[0]))


def _lerp_tuple_range(
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    difficulty: float,
) -> tuple[float, float]:
    return (
        _lerp_range((start_range[0], end_range[0]), difficulty),
        _lerp_range((start_range[1], end_range[1]), difficulty),
    )


def _resolve_int_curriculum(
    cfg,
    attr_range: str,
    attr_legacy: str,
    difficulty: float,
    min_value: int = 1,
) -> int:
    value_range = getattr(cfg, attr_range, None)
    if value_range is None:
        return max(int(getattr(cfg, attr_legacy)), min_value)
    value = int(round(_lerp_range(value_range, difficulty)))
    return max(value, min_value)


def _resolve_obstacle_range(
    cfg,
    legacy_attr: str,
    easy_attr: str,
    hard_attr: str,
    difficulty: float,
) -> tuple[float, float]:
    easy_range = getattr(cfg, easy_attr, None)
    hard_range = getattr(cfg, hard_attr, None)
    if easy_range is not None and hard_range is not None:
        return _lerp_tuple_range(easy_range, hard_range, difficulty)
    return getattr(cfg, legacy_attr)


def make_box(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    color: Color | None = None,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(
        extents=size,
        transform=trimesh.transformations.translation_matrix(center),
    )
    if color is not None:
        mesh.visual.vertex_colors = color
    return mesh


def make_cylinder(
    center: tuple[float, float, float],
    radius: float,
    height: float,
    color: Color | None = None,
) -> trimesh.Trimesh:
    mesh = trimesh.creation.cylinder(
        radius=radius,
        height=height,
        sections=24,
        transform=trimesh.transformations.translation_matrix(center),
    )
    if color is not None:
        mesh.visual.vertex_colors = color
    return mesh


def rect_from_center_size(
    center_xy: tuple[float, float],
    size_xy: tuple[float, float],
    extra: float = 0.0,
) -> Rect:
    cx, cy = center_xy
    sx, sy = size_xy
    return (cx - 0.5 * sx - extra, cx + 0.5 * sx + extra, cy - 0.5 * sy - extra, cy + 0.5 * sy + extra)


def _clear_rect(center_xy: np.ndarray, size_xy: tuple[float, float]) -> Rect:
    return rect_from_center_size((float(center_xy[0]), float(center_xy[1])), size_xy, extra=0.1)


def rects_overlap(a: Rect, b: Rect) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1] or a[3] <= b[2] or a[2] >= b[3])


def rect_is_free(rect: Rect, occupied_rects: list[Rect]) -> bool:
    return all(not rects_overlap(rect, occ) for occ in occupied_rects)


def sample_position_in_region(
    rng: np.random.Generator,
    region: tuple[float, float, float, float],
    margin: float,
) -> tuple[float, float]:
    xmin, xmax, ymin, ymax = region
    if xmax - xmin <= 2.0 * margin or ymax - ymin <= 2.0 * margin:
        raise ValueError(f"Sampling margin {margin} is too large for region {region}.")
    x = rng.uniform(xmin + margin, xmax - margin)
    y = rng.uniform(ymin + margin, ymax - margin)
    return float(x), float(y)


def sample_free_position(
    rng: np.random.Generator,
    region: tuple[float, float, float, float],
    size_xy: tuple[float, float],
    occupied_rects: list[Rect],
    clearance: float,
    max_tries: int = 120,
) -> tuple[float, float] | None:
    margin = 0.5 * max(size_xy) + clearance
    for _ in range(max_tries):
        x, y = sample_position_in_region(rng, region, margin=margin)
        rect = rect_from_center_size((x, y), size_xy, extra=clearance)
        if rect_is_free(rect, occupied_rects):
            return x, y
    return None



def sample_free_position_adaptive(
    rng: np.random.Generator,
    region: tuple[float, float, float, float],
    size_xy: tuple[float, float],
    occupied_rects: list[Rect],
    clearance: float,
    min_clearance: float = 0.02,
    max_tries: int = 300,
) -> tuple[tuple[float, float] | None, float]:
    """Sample a free position while gradually relaxing clearance.

    This is used to keep high-difficulty terrains from silently dropping obstacles
    when the requested obstacle count is high or the corridors become narrow.
    """
    clearance = float(max(clearance, min_clearance))
    min_clearance = float(max(min_clearance, 0.0))
    clearance_levels = np.linspace(clearance, min_clearance, num=5)
    tries_per_level = max(max_tries // len(clearance_levels), 1)
    for clear in clearance_levels:
        pos = sample_free_position(
            rng=rng,
            region=region,
            size_xy=size_xy,
            occupied_rects=occupied_rects,
            clearance=float(clear),
            max_tries=tries_per_level,
        )
        if pos is not None:
            return pos, float(clear)
    return None, min_clearance


def _make_cell_candidates(
    xmin: float,
    ymin: float,
    cell_x: float,
    cell_y: float,
    num_cells_x: int,
    num_cells_y: int,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    start_clear_size: tuple[float, float],
    goal_clear_size: tuple[float, float],
) -> list[tuple[float, float]]:
    """Return cell centers that are not inside start/goal clear zones."""
    candidates: list[tuple[float, float]] = []
    for ix in range(num_cells_x):
        for iy in range(num_cells_y):
            x = xmin + (ix + 0.5) * cell_x
            y = ymin + (iy + 0.5) * cell_y
            if in_clear_zone(x, y, start_xy, goal_xy, start_clear_size, goal_clear_size):
                continue
            candidates.append((float(x), float(y)))
    return candidates


def _sample_cell_position(
    rng: np.random.Generator,
    cell_candidates: list[tuple[float, float]],
    cell_x: float,
    cell_y: float,
    size_xy: tuple[float, float],
    occupied_rects: list[Rect],
    clearance: float,
    max_offset_ratio: float = 0.25,
    min_clearance: float = 0.02,
    max_tries: int = 300,
) -> tuple[tuple[float, float] | None, float]:
    """Sample inside maze cells instead of the full maze region.

    Obstacles placed near cell centers are much less likely to block passage openings.
    The function also relaxes clearance gradually so objects are not silently dropped
    in high-difficulty settings.
    """
    if len(cell_candidates) == 0:
        return None, min_clearance

    clearance_levels = np.linspace(float(clearance), float(min_clearance), num=5)
    tries_per_level = max(max_tries // len(clearance_levels), 1)
    sx, sy = size_xy
    max_dx_base = max(0.5 * cell_x - 0.5 * sx, 0.0)
    max_dy_base = max(0.5 * cell_y - 0.5 * sy, 0.0)
    # Keep the object relatively central in the cell so it doesn't close doorways.
    max_dx_base = min(max_dx_base, max_offset_ratio * cell_x)
    max_dy_base = min(max_dy_base, max_offset_ratio * cell_y)

    for clear in clearance_levels:
        for _ in range(tries_per_level):
            cx, cy = cell_candidates[int(rng.integers(0, len(cell_candidates)))]
            dx = rng.uniform(-max_dx_base, max_dx_base) if max_dx_base > 0.0 else 0.0
            dy = rng.uniform(-max_dy_base, max_dy_base) if max_dy_base > 0.0 else 0.0
            x = float(cx + dx)
            y = float(cy + dy)
            rect = rect_from_center_size((x, y), size_xy, extra=float(clear))
            if rect_is_free(rect, occupied_rects):
                return (x, y), float(clear)
    return None, float(min_clearance)


def _clamp_xy_size_to_cell(
    size_xy: tuple[float, float],
    cell_x: float,
    cell_y: float,
    max_ratio: float,
    min_size: float = 0.05,
) -> tuple[float, float]:
    """Limit obstacle footprint so high-difficulty obstacles still fit inside a cell."""
    sx, sy = size_xy
    sx = min(float(sx), max(float(min_size), float(cell_x) * float(max_ratio)))
    sy = min(float(sy), max(float(min_size), float(cell_y) * float(max_ratio)))
    return sx, sy


def in_clear_zone(
    x: float,
    y: float,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    start_clear_size: tuple[float, float],
    goal_clear_size: tuple[float, float],
) -> bool:
    sx, sy = start_clear_size
    gx, gy = goal_clear_size
    in_start = abs(x - start_xy[0]) <= 0.5 * sx and abs(y - start_xy[1]) <= 0.5 * sy
    in_goal = abs(x - goal_xy[0]) <= 0.5 * gx and abs(y - goal_xy[1]) <= 0.5 * gy
    return in_start or in_goal


def _add_passage_keepout_rects(
    occupied_rects: list[Rect],
    vertical_passages: set[tuple[int, int]],
    horizontal_passages: set[tuple[int, int]],
    xmin: float,
    ymin: float,
    cell_x: float,
    cell_y: float,
    opening_size: tuple[float, float],
    clearance: float,
) -> None:
    """Protect only the center of each carved passage.

    The previous implementation protected an entire cell-width/cell-height strip for every passage.
    In dense mazes this occupied most of the free area, so high-difficulty obstacles were often skipped.
    This version protects only a compact rectangle around the passage opening.
    """
    opening_x, opening_y = opening_size
    for ix, iy in vertical_passages:
        x = xmin + ix * cell_x
        y = ymin + (iy + 0.5) * cell_y
        occupied_rects.append(rect_from_center_size((x, y), (opening_x, opening_y), extra=clearance))
    for ix, iy in horizontal_passages:
        x = xmin + (ix + 0.5) * cell_x
        y = ymin + iy * cell_y
        occupied_rects.append(rect_from_center_size((x, y), (opening_x, opening_y), extra=clearance))


# -------------------------------------------------------------------------------------------------
# Optional features
# -------------------------------------------------------------------------------------------------


def _add_box_obstacles(
    meshes_list: list[trimesh.Trimesh],
    occupied_rects: list[Rect],
    navigation_blocking_rects: list[Rect],
    rng: np.random.Generator,
    cfg,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    difficulty: float,
    cell_candidates: list[tuple[float, float]],
    cell_x: float,
    cell_y: float,
) -> None:
    if hasattr(cfg, "num_box_obstacles_range"):
        num_box_obstacles = _lerp_int_range(cfg.num_box_obstacles_range, difficulty)
    else:
        num_box_obstacles = int(getattr(cfg, "num_box_obstacles", 8))
    box_size_range = _resolve_obstacle_range(
        cfg,
        legacy_attr="box_size_range",
        easy_attr="box_size_range_easy",
        hard_attr="box_size_range_hard",
        difficulty=difficulty,
    )
    box_height_range = _resolve_obstacle_range(
        cfg,
        legacy_attr="box_height_range",
        easy_attr="box_height_range_easy",
        hard_attr="box_height_range_hard",
        difficulty=difficulty,
    )
    obstacle_clearance = float(getattr(cfg, "obstacle_clearance", 0.20))
    obstacle_min_clearance = float(getattr(cfg, "obstacle_min_clearance", 0.02))
    obstacle_max_tries = int(getattr(cfg, "obstacle_max_tries", 300))
    obstacle_footprint_ratio = float(getattr(cfg, "obstacle_footprint_cell_ratio", 0.45))
    obstacle_color = getattr(cfg, "box_obstacle_color", (165, 90, 45, 255))
    place_in_cells = bool(getattr(cfg, "obstacle_place_in_cells", True))

    placed = 0
    for _ in range(num_box_obstacles):
        sx = float(rng.uniform(box_size_range[0], box_size_range[1]))
        sy = float(rng.uniform(box_size_range[0], box_size_range[1]))
        sx, sy = _clamp_xy_size_to_cell((sx, sy), cell_x, cell_y, obstacle_footprint_ratio)
        h = float(rng.uniform(box_height_range[0], box_height_range[1]))

        if place_in_cells:
            pos, used_clearance = _sample_cell_position(
                rng=rng,
                cell_candidates=cell_candidates,
                cell_x=cell_x,
                cell_y=cell_y,
                size_xy=(sx, sy),
                occupied_rects=occupied_rects,
                clearance=obstacle_clearance,
                min_clearance=obstacle_min_clearance,
                max_tries=obstacle_max_tries,
            )
        else:
            pos, used_clearance = sample_free_position_adaptive(
                rng=rng,
                region=cfg.maze_region,
                size_xy=(sx, sy),
                occupied_rects=occupied_rects,
                clearance=obstacle_clearance,
                min_clearance=obstacle_min_clearance,
                max_tries=obstacle_max_tries,
            )
        if pos is None:
            continue

        x, y = pos
        if in_clear_zone(x, y, start_xy, goal_xy, cfg.start_clear_size, cfg.goal_clear_size):
            continue

        meshes_list.append(make_box(center=(x, y, 0.5 * h), size=(sx, sy, h), color=obstacle_color))
        rect = rect_from_center_size((x, y), (sx, sy), extra=used_clearance)
        occupied_rects.append(rect)
        navigation_blocking_rects.append(rect)
        placed += 1

    if getattr(cfg, "warn_on_obstacle_drop", False) and placed < num_box_obstacles:
        print(f"[maze_terrain] placed {placed}/{num_box_obstacles} box obstacles at difficulty={difficulty:.3f}")

def _add_cylinder_obstacles(
    meshes_list: list[trimesh.Trimesh],
    occupied_rects: list[Rect],
    navigation_blocking_rects: list[Rect],
    rng: np.random.Generator,
    cfg,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    difficulty: float,
    cell_candidates: list[tuple[float, float]],
    cell_x: float,
    cell_y: float,
) -> None:
    if hasattr(cfg, "num_cylinder_obstacles_range"):
        num_cylinder_obstacles = _lerp_int_range(cfg.num_cylinder_obstacles_range, difficulty)
    else:
        num_cylinder_obstacles = int(getattr(cfg, "num_cylinder_obstacles", 6))
    radius_range = _resolve_obstacle_range(
        cfg,
        legacy_attr="cylinder_radius_range",
        easy_attr="cylinder_radius_range_easy",
        hard_attr="cylinder_radius_range_hard",
        difficulty=difficulty,
    )
    height_range = _resolve_obstacle_range(
        cfg,
        legacy_attr="cylinder_height_range",
        easy_attr="cylinder_height_range_easy",
        hard_attr="cylinder_height_range_hard",
        difficulty=difficulty,
    )
    obstacle_clearance = float(getattr(cfg, "obstacle_clearance", 0.20))
    obstacle_min_clearance = float(getattr(cfg, "obstacle_min_clearance", 0.02))
    obstacle_max_tries = int(getattr(cfg, "obstacle_max_tries", 300))
    obstacle_footprint_ratio = float(getattr(cfg, "obstacle_footprint_cell_ratio", 0.45))
    obstacle_color = getattr(cfg, "cylinder_obstacle_color", (140, 80, 120, 255))
    place_in_cells = bool(getattr(cfg, "obstacle_place_in_cells", True))

    placed = 0
    for _ in range(num_cylinder_obstacles):
        radius = float(rng.uniform(radius_range[0], radius_range[1]))
        max_radius = 0.5 * min(cell_x, cell_y) * obstacle_footprint_ratio
        radius = min(radius, max(max_radius, 0.03))
        h = float(rng.uniform(height_range[0], height_range[1]))
        size_xy = (2.0 * radius, 2.0 * radius)

        if place_in_cells:
            pos, used_clearance = _sample_cell_position(
                rng=rng,
                cell_candidates=cell_candidates,
                cell_x=cell_x,
                cell_y=cell_y,
                size_xy=size_xy,
                occupied_rects=occupied_rects,
                clearance=obstacle_clearance,
                min_clearance=obstacle_min_clearance,
                max_tries=obstacle_max_tries,
            )
        else:
            pos, used_clearance = sample_free_position_adaptive(
                rng=rng,
                region=cfg.maze_region,
                size_xy=size_xy,
                occupied_rects=occupied_rects,
                clearance=obstacle_clearance,
                min_clearance=obstacle_min_clearance,
                max_tries=obstacle_max_tries,
            )
        if pos is None:
            continue

        x, y = pos
        if in_clear_zone(x, y, start_xy, goal_xy, cfg.start_clear_size, cfg.goal_clear_size):
            continue

        meshes_list.append(make_cylinder(center=(x, y, 0.5 * h), radius=radius, height=h, color=obstacle_color))
        rect = rect_from_center_size((x, y), size_xy, extra=used_clearance)
        occupied_rects.append(rect)
        navigation_blocking_rects.append(rect)
        placed += 1

    if getattr(cfg, "warn_on_obstacle_drop", False) and placed < num_cylinder_obstacles:
        print(f"[maze_terrain] placed {placed}/{num_cylinder_obstacles} cylinder obstacles at difficulty={difficulty:.3f}")

def _add_rough_patches(
    meshes_list: list[trimesh.Trimesh],
    occupied_rects: list[Rect],
    navigation_blocking_rects: list[Rect],
    rng: np.random.Generator,
    cfg,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    cell_candidates: list[tuple[float, float]],
    cell_x: float,
    cell_y: float,
) -> None:
    num_rough_patches = int(getattr(cfg, "num_rough_patches", 3))
    rough_patch_size = getattr(cfg, "rough_patch_size", (2.0, 2.0))
    rough_block_size = float(getattr(cfg, "rough_block_size", 0.25))
    rough_height_range = getattr(cfg, "rough_height_range", (0.015, 0.06))
    rough_clearance = float(getattr(cfg, "rough_clearance", 0.15))
    rough_min_clearance = float(getattr(cfg, "rough_min_clearance", 0.02))
    rough_max_tries = int(getattr(cfg, "rough_max_tries", 300))
    rough_patch_cell_ratio = float(getattr(cfg, "rough_patch_cell_ratio", 0.75))
    place_in_cells = bool(getattr(cfg, "rough_patch_place_in_cells", True))

    placed = 0
    for _ in range(num_rough_patches):
        sx, sy = _clamp_xy_size_to_cell(rough_patch_size, cell_x, cell_y, rough_patch_cell_ratio, min_size=0.25)
        if place_in_cells:
            pos, used_clearance = _sample_cell_position(
                rng=rng,
                cell_candidates=cell_candidates,
                cell_x=cell_x,
                cell_y=cell_y,
                size_xy=(sx, sy),
                occupied_rects=occupied_rects,
                clearance=rough_clearance,
                min_clearance=rough_min_clearance,
                max_tries=rough_max_tries,
            )
        else:
            pos, used_clearance = sample_free_position_adaptive(
                rng=rng,
                region=cfg.maze_region,
                size_xy=(sx, sy),
                occupied_rects=occupied_rects,
                clearance=rough_clearance,
                min_clearance=rough_min_clearance,
                max_tries=rough_max_tries,
            )
        if pos is None:
            continue

        x, y = pos
        if in_clear_zone(x, y, start_xy, goal_xy, cfg.start_clear_size, cfg.goal_clear_size):
            continue

        add_rough_patch(
            meshes_list=meshes_list,
            rng=rng,
            center_xy=(x, y),
            patch_size=(sx, sy),
            block_size=rough_block_size,
            height_range=rough_height_range,
        )
        rect = rect_from_center_size((x, y), (sx, sy), extra=used_clearance)
        occupied_rects.append(rect)
        navigation_blocking_rects.append(rect)
        placed += 1

    if getattr(cfg, "warn_on_obstacle_drop", False) and placed < num_rough_patches:
        print(f"[maze_terrain] placed {placed}/{num_rough_patches} rough patches")

def add_rough_patch(
    meshes_list: list[trimesh.Trimesh],
    rng: np.random.Generator,
    center_xy: tuple[float, float],
    patch_size: tuple[float, float],
    block_size: float,
    height_range: tuple[float, float],
) -> None:
    cx, cy = center_xy
    sx, sy = patch_size
    nx = max(int(sx / block_size), 1)
    ny = max(int(sy / block_size), 1)
    actual_block_x = sx / nx
    actual_block_y = sy / ny
    x0 = cx - 0.5 * sx
    y0 = cy - 0.5 * sy
    rough_color = (100, 90, 70, 255)

    for ix in range(nx):
        for iy in range(ny):
            x = x0 + (ix + 0.5) * actual_block_x
            y = y0 + (iy + 0.5) * actual_block_y
            h = float(rng.uniform(height_range[0], height_range[1]))
            meshes_list.append(
                make_box(
                    center=(x, y, 0.5 * h),
                    size=(actual_block_x, actual_block_y, h),
                    color=rough_color,
                )
            )


def _add_stair_regions(
    meshes_list: list[trimesh.Trimesh],
    occupied_rects: list[Rect],
    navigation_blocking_rects: list[Rect],
    rng: np.random.Generator,
    cfg,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
) -> None:
    num_stair_regions = int(getattr(cfg, "num_stair_regions", 2))
    stair_num_steps = int(getattr(cfg, "stair_num_steps", 4))
    stair_step_width = float(getattr(cfg, "stair_step_width", 0.32))
    stair_step_height = float(getattr(cfg, "stair_step_height", 0.04))
    stair_width = float(getattr(cfg, "stair_width", 1.0))
    stair_clearance = float(getattr(cfg, "stair_clearance", 0.20))
    directions = ("x", "-x", "y", "-y")
    stair_length = stair_num_steps * stair_step_width

    for _ in range(num_stair_regions):
        direction = str(rng.choice(directions))
        if direction in ("x", "-x"):
            size_xy = (stair_length, stair_width)
        else:
            size_xy = (stair_width, stair_length)

        pos = sample_free_position(
            rng=rng,
            region=cfg.maze_region,
            size_xy=size_xy,
            occupied_rects=occupied_rects,
            clearance=stair_clearance,
        )
        if pos is None:
            continue

        x, y = pos
        if in_clear_zone(x, y, start_xy, goal_xy, cfg.start_clear_size, cfg.goal_clear_size):
            continue

        add_stairs(
            meshes_list=meshes_list,
            center_xy=(x, y),
            direction=direction,
            num_steps=stair_num_steps,
            step_width=stair_step_width,
            step_height=stair_step_height,
            stair_width=stair_width,
        )
        rect = rect_from_center_size((x, y), size_xy, extra=stair_clearance)
        occupied_rects.append(rect)
        navigation_blocking_rects.append(rect)


def add_stairs(
    meshes_list: list[trimesh.Trimesh],
    center_xy: tuple[float, float],
    direction: str,
    num_steps: int,
    step_width: float,
    step_height: float,
    stair_width: float,
) -> None:
    cx, cy = center_xy
    total_length = num_steps * step_width
    step_color = (125, 120, 85, 255)

    for i in range(num_steps):
        h = (i + 1) * step_height
        offset = -0.5 * total_length + (i + 0.5) * step_width

        if direction == "x":
            x = cx + offset
            y = cy
            size = (step_width, stair_width, h)
        elif direction == "-x":
            x = cx - offset
            y = cy
            size = (step_width, stair_width, h)
        elif direction == "y":
            x = cx
            y = cy + offset
            size = (stair_width, step_width, h)
        elif direction == "-y":
            x = cx
            y = cy - offset
            size = (stair_width, step_width, h)
        else:
            raise ValueError(f"Unknown stair direction: {direction}")

        meshes_list.append(make_box(center=(x, y, 0.5 * h), size=size, color=step_color))
