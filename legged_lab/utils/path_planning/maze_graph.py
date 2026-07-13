from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np


Cell = tuple[int, int]


@dataclass(frozen=True)
class PathMetrics:
    path_length: float
    euclidean_distance: float
    detour_ratio: float
    turn_count: int
    negative_progress: float
    junction_count: int
    num_cells: int
    score: float = 0.0


@dataclass(frozen=True)
class PathCandidate:
    start_cell: Cell
    goal_cell: Cell
    cell_path: tuple[Cell, ...]
    metrics: PathMetrics


@dataclass
class MazeGraph:
    num_cells_x: int
    num_cells_y: int
    maze_region: tuple[float, float, float, float]
    cell_size: tuple[float, float]
    adjacency: dict[Cell, tuple[Cell, ...]]
    passage_centers: dict[tuple[Cell, Cell], tuple[float, float]]

    def cells(self) -> Iterable[Cell]:
        return self.adjacency.keys()

    def cell_center(self, cell: Cell) -> tuple[float, float]:
        xmin, _, ymin, _ = self.maze_region
        cell_x, cell_y = self.cell_size
        ix, iy = cell
        return xmin + (ix + 0.5) * cell_x, ymin + (iy + 0.5) * cell_y

    def edge_weight(self, a: Cell, b: Cell) -> float:
        ax, ay = self.cell_center(a)
        bx, by = self.cell_center(b)
        return math.hypot(bx - ax, by - ay)

    def degree(self, cell: Cell) -> int:
        count = 0
        for _ in self.adjacency.get(cell, ()):
            count += 1
        return count


def build_grid_maze_graph(
    *,
    num_cells_x: int,
    num_cells_y: int,
    maze_region: tuple[float, float, float, float],
    vertical_walls: set[tuple[int, int]],
    horizontal_walls: set[tuple[int, int]],
) -> MazeGraph:
    """Build a cell graph from DFS maze walls.

    Vertical wall ``(ix, iy)`` blocks cells ``(ix - 1, iy)`` and ``(ix, iy)``.
    Horizontal wall ``(ix, iy)`` blocks cells ``(ix, iy - 1)`` and ``(ix, iy)``.
    """
    xmin, xmax, ymin, ymax = maze_region
    cell_x = (xmax - xmin) / float(num_cells_x)
    cell_y = (ymax - ymin) / float(num_cells_y)

    adjacency: dict[Cell, list[Cell]] = {
        (ix, iy): [] for ix in range(num_cells_x) for iy in range(num_cells_y)
    }
    passage_centers: dict[tuple[Cell, Cell], tuple[float, float]] = {}

    def add_edge(a: Cell, b: Cell, passage: tuple[float, float]) -> None:
        adjacency[a].append(b)
        adjacency[b].append(a)
        passage_centers[(a, b)] = passage
        passage_centers[(b, a)] = passage

    for ix in range(num_cells_x - 1):
        for iy in range(num_cells_y):
            wall = (ix + 1, iy)
            if wall in vertical_walls:
                continue
            passage = (xmin + (ix + 1) * cell_x, ymin + (iy + 0.5) * cell_y)
            add_edge((ix, iy), (ix + 1, iy), passage)

    for ix in range(num_cells_x):
        for iy in range(num_cells_y - 1):
            wall = (ix, iy + 1)
            if wall in horizontal_walls:
                continue
            passage = (xmin + (ix + 0.5) * cell_x, ymin + (iy + 1) * cell_y)
            add_edge((ix, iy), (ix, iy + 1), passage)

    return MazeGraph(
        num_cells_x=num_cells_x,
        num_cells_y=num_cells_y,
        maze_region=maze_region,
        cell_size=(cell_x, cell_y),
        adjacency={cell: tuple(neighbors) for cell, neighbors in adjacency.items()},
        passage_centers=passage_centers,
    )


def astar_path(graph: MazeGraph, start: Cell, goal: Cell) -> list[Cell] | None:
    if start == goal:
        return [start]

    def heuristic(cell: Cell) -> float:
        cx, cy = graph.cell_center(cell)
        gx, gy = graph.cell_center(goal)
        return math.hypot(gx - cx, gy - cy)

    open_heap: list[tuple[float, int, Cell]] = []
    heapq.heappush(open_heap, (heuristic(start), 0, start))
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, float] = {start: 0.0}
    tie_breaker = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal:
            return _reconstruct_path(came_from, current)

        for neighbor in graph.adjacency.get(current, ()):
            tentative = g_score[current] + graph.edge_weight(current, neighbor)
            if tentative >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            tie_breaker += 1
            heapq.heappush(open_heap, (tentative + heuristic(neighbor), tie_breaker, neighbor))

    return None


def summarize_graph(graph: MazeGraph) -> dict[str, float]:
    all_distances: list[float] = []
    diameter = 0.0
    for cell in graph.cells():
        distances, _ = _dijkstra(graph, cell)
        finite = [dist for target, dist in distances.items() if target != cell and math.isfinite(dist)]
        if not finite:
            continue
        diameter = max(diameter, max(finite))
        all_distances.extend(finite)

    degrees = [len(neighbors) for neighbors in graph.adjacency.values()]
    dead_ends = sum(1 for degree in degrees if degree <= 1)
    junctions = sum(1 for degree in degrees if degree >= 3)
    num_cells = max(len(degrees), 1)
    return {
        "graph_diameter": diameter,
        "avg_shortest_path": float(np.mean(all_distances)) if all_distances else 0.0,
        "dead_end_ratio": dead_ends / float(num_cells),
        "junction_count": float(junctions),
        "junction_ratio": junctions / float(num_cells),
    }


def generate_path_candidates(
    graph: MazeGraph,
    *,
    num_levels: int,
    max_candidates_per_level: int,
    min_cell_separation: int = 2,
) -> dict[int, list[PathCandidate]]:
    """Generate start-goal candidates grouped by path difficulty quantiles."""
    candidates: list[PathCandidate] = []
    cells = list(graph.cells())
    for start in cells:
        distances, parents = _dijkstra(graph, start)
        for goal in cells:
            if goal <= start:
                continue
            if abs(goal[0] - start[0]) + abs(goal[1] - start[1]) < int(min_cell_separation):
                continue
            if not math.isfinite(distances.get(goal, float("inf"))):
                continue
            path = _path_from_parents(parents, start, goal)
            if path is None or len(path) < 2:
                continue
            metrics = compute_path_metrics(graph, path)
            candidates.append(PathCandidate(start, goal, tuple(path), metrics))
            reverse_path = list(reversed(path))
            reverse_metrics = compute_path_metrics(graph, reverse_path)
            candidates.append(PathCandidate(goal, start, tuple(reverse_path), reverse_metrics))

    if not candidates:
        return {level: [] for level in range(num_levels)}

    scored_metrics = _score_metrics([candidate.metrics for candidate in candidates])
    scored_candidates = [
        PathCandidate(candidate.start_cell, candidate.goal_cell, candidate.cell_path, metrics)
        for candidate, metrics in zip(candidates, scored_metrics)
    ]
    scored_candidates.sort(key=lambda candidate: candidate.metrics.score)

    grouped: dict[int, list[PathCandidate]] = {level: [] for level in range(num_levels)}
    if num_levels <= 1:
        grouped[0] = scored_candidates[:max_candidates_per_level]
        return grouped

    for rank, candidate in enumerate(scored_candidates):
        level = min(int(rank * num_levels / len(scored_candidates)), num_levels - 1)
        if len(grouped[level]) < int(max_candidates_per_level):
            grouped[level].append(candidate)
    return grouped


def compute_path_metrics(graph: MazeGraph, path: list[Cell] | tuple[Cell, ...]) -> PathMetrics:
    length = 0.0
    for a, b in zip(path[:-1], path[1:]):
        length += graph.edge_weight(a, b)

    sx, sy = graph.cell_center(path[0])
    gx, gy = graph.cell_center(path[-1])
    euclidean = max(math.hypot(gx - sx, gy - sy), 1.0e-6)
    turn_count = _turn_count(path)
    negative_progress = _negative_progress(graph, path)
    junction_count = 0
    for cell in path[1:-1]:
        neighbors = graph.adjacency.get(cell)
        if neighbors is not None and len(neighbors) >= 3:
            junction_count += 1
    return PathMetrics(
        path_length=length,
        euclidean_distance=euclidean,
        detour_ratio=length / euclidean,
        turn_count=turn_count,
        negative_progress=negative_progress,
        junction_count=junction_count,
        num_cells=len(path),
    )


def path_reference_points(
    graph: MazeGraph,
    cell_path: list[Cell] | tuple[Cell, ...],
    *,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    include_cell_centers: bool = True,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = [start_xy]
    for current, next_cell in zip(cell_path[:-1], cell_path[1:]):
        passage = graph.passage_centers.get((current, next_cell))
        if passage is not None:
            points.append(passage)
        if include_cell_centers:
            points.append(graph.cell_center(next_cell))
    points.append(goal_xy)
    return _dedupe_points(points)


def sparsify_reference_points(
    points: list[tuple[float, float]],
    *,
    stride: int,
    max_points: int,
) -> list[tuple[float, float]]:
    """Drop the first start point and keep sparse navigation targets, always ending at final goal."""
    if len(points) <= 1:
        return points
    if stride <= 0 or stride >= 1_000_000:
        sparse = [points[-1]]
    else:
        sparse = points[stride::stride]
        if not sparse or sparse[-1] != points[-1]:
            sparse.append(points[-1])

    if max_points > 0 and len(sparse) > max_points:
        indices = np.linspace(0, len(sparse) - 1, num=max_points, dtype=np.int64)
        sparse = [sparse[int(idx)] for idx in indices]
        sparse[-1] = points[-1]
    return _dedupe_points(sparse)


def _dijkstra(graph: MazeGraph, start: Cell) -> tuple[dict[Cell, float], dict[Cell, Cell]]:
    distances = {cell: float("inf") for cell in graph.cells()}
    distances[start] = 0.0
    parents: dict[Cell, Cell] = {}
    heap: list[tuple[float, int, Cell]] = [(0.0, 0, start)]
    tie_breaker = 0
    while heap:
        dist, _, current = heapq.heappop(heap)
        if dist > distances[current]:
            continue
        for neighbor in graph.adjacency.get(current, ()):
            next_dist = dist + graph.edge_weight(current, neighbor)
            if next_dist >= distances[neighbor]:
                continue
            distances[neighbor] = next_dist
            parents[neighbor] = current
            tie_breaker += 1
            heapq.heappush(heap, (next_dist, tie_breaker, neighbor))
    return distances, parents


def _reconstruct_path(came_from: dict[Cell, Cell], current: Cell) -> list[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _path_from_parents(parents: dict[Cell, Cell], start: Cell, goal: Cell) -> list[Cell] | None:
    if start == goal:
        return [start]
    if goal not in parents:
        return None
    path = [goal]
    current = goal
    while current != start:
        current = parents[current]
        path.append(current)
    path.reverse()
    return path


def _turn_count(path: list[Cell] | tuple[Cell, ...]) -> int:
    count = 0
    prev_dir: tuple[int, int] | None = None
    for a, b in zip(path[:-1], path[1:]):
        direction = (b[0] - a[0], b[1] - a[1])
        if prev_dir is not None and direction != prev_dir:
            count += 1
        prev_dir = direction
    return count


def _negative_progress(graph: MazeGraph, path: list[Cell] | tuple[Cell, ...]) -> float:
    goal_xy = graph.cell_center(path[-1])
    total = 0.0
    prev_dist: float | None = None
    for cell in path:
        xy = graph.cell_center(cell)
        dist = math.hypot(goal_xy[0] - xy[0], goal_xy[1] - xy[1])
        if prev_dist is not None and dist > prev_dist:
            total += dist - prev_dist
        prev_dist = dist
    return total


def _score_metrics(metrics: list[PathMetrics]) -> list[PathMetrics]:
    values = {
        "path_length": np.array([metric.path_length for metric in metrics], dtype=np.float64),
        "detour_ratio": np.array([metric.detour_ratio for metric in metrics], dtype=np.float64),
        "turn_count": np.array([metric.turn_count for metric in metrics], dtype=np.float64),
        "negative_progress": np.array([metric.negative_progress for metric in metrics], dtype=np.float64),
        "junction_count": np.array([metric.junction_count for metric in metrics], dtype=np.float64),
    }

    weights = {
        "path_length": 0.30,
        "detour_ratio": 0.25,
        "turn_count": 0.20,
        "negative_progress": 0.20,
        "junction_count": 0.05,
    }
    normalized: dict[str, np.ndarray] = {}
    for name, array in values.items():
        min_value = float(np.min(array))
        max_value = float(np.max(array))
        if max_value <= min_value:
            normalized[name] = np.zeros_like(array)
        else:
            normalized[name] = (array - min_value) / (max_value - min_value)

    scored: list[PathMetrics] = []
    for idx, metric in enumerate(metrics):
        score = sum(weights[name] * float(normalized[name][idx]) for name in weights)
        scored.append(
            PathMetrics(
                path_length=metric.path_length,
                euclidean_distance=metric.euclidean_distance,
                detour_ratio=metric.detour_ratio,
                turn_count=metric.turn_count,
                negative_progress=metric.negative_progress,
                junction_count=metric.junction_count,
                num_cells=metric.num_cells,
                score=score,
            )
        )
    return scored


def _dedupe_points(points: list[tuple[float, float]], eps: float = 1.0e-6) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for point in points:
        if deduped and math.hypot(point[0] - deduped[-1][0], point[1] - deduped[-1][1]) <= eps:
            continue
        deduped.append((float(point[0]), float(point[1])))
    return deduped
