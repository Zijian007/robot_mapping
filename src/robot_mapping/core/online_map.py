"""Persistent online map on top of FAST-LIO2 local ikd-tree.

Each keyframe owns its frame-end de-skewed body cloud. Observations are
projected using graph poses; voxel deduplication never changes the archive.
A growing 2D occupancy grid is updated incrementally for live preview and
navigation export.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from robot_mapping.core.grid import (
    GDK_FREE,
    GDK_OCCUPIED,
    GDK_UNKNOWN,
    OccupancyGridSpec,
)

from .spatial import GrowingOccupancy2D, WorldVoxelHash


from robot_mapping.ports import LioResult

if TYPE_CHECKING:
    from robot_mapping.backends.loop_types import LoopMatch, LoopSnapshot


class MappingPhase(str, Enum):
    IDLE = "idle"
    MAPPING = "mapping"


@dataclass(frozen=True)
class SE3Pose:
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    @property
    def xy(self) -> tuple[float, float]:
        return self.position[0], self.position[1]

    @property
    def yaw(self) -> float:
        x, y, z, w = self.orientation_xyzw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def pose_from_odom(odom: LioResult) -> SE3Pose:
    pos = odom["position"]
    quat = odom["orientation_xyzw"]
    return SE3Pose(
        position=(float(pos[0]), float(pos[1]), float(pos[2])),
        orientation_xyzw=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
    )


def yaw_delta(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(d)


class KeyframeGate:
    def __init__(self, dist_th: float = 0.5, angle_th: float = 0.2) -> None:
        self.dist_th = float(dist_th)
        self.angle_th = float(angle_th)
        self._last: SE3Pose | None = None

    def reset(self) -> None:
        self._last = None

    def accept(self, pose: SE3Pose) -> bool:
        if self._last is None:
            self._last = pose
            return True
        dx = pose.position[0] - self._last.position[0]
        dy = pose.position[1] - self._last.position[1]
        dist = math.hypot(dx, dy)
        if dist >= self.dist_th or yaw_delta(pose.yaw, self._last.yaw) >= self.angle_th:
            self._last = pose
            return True
        return False


@dataclass
class OnlineMapStats:
    keyframes: int = 0
    voxels: int = 0
    occupied_cells: int = 0
    free_cells: int = 0
    last_ingest_ms: float = 0.0
    loops: int = 0


@dataclass(frozen=True)
class KeyframeRecord:
    node_id: int
    lio_pose: SE3Pose
    local_points: np.ndarray


@dataclass(frozen=True)
class MapSnapshot:
    generation: int
    version: int
    points: np.ndarray
    grid: tuple[np.ndarray, OccupancyGridSpec] | None
    map_from_lio: np.ndarray


class OnlineMap:
    """Own immutable observations; publish rebuilt maps with a short locked swap.

    Ingestion never touches GTSAM. The loop worker alone owns its solver under
    _backend_lock, including reset/replay after an abandoned optimization.
    """

    def __init__(self, *, voxel_resolution=0.15, grid_resolution=0.05,
                 min_height=-0.25, max_height=1.8, margin_cells=5,
                 keyframe_dist_th=0.5, keyframe_angle_th=0.2,
                 free_radius_m=0.35, map_point_stride=1, start_mapping=True,
                 loop_config=None, pose_graph=None, loop_worker=None):
        from robot_mapping.backends.loop_types import LoopConfig
        self._voxel_resolution = float(voxel_resolution)
        self._grid_params = dict(resolution=grid_resolution, min_height=min_height,
                                 max_height=max_height, margin_cells=margin_cells,
                                 free_radius_m=free_radius_m)
        self.voxels = WorldVoxelHash(voxel_resolution)
        self.grid = GrowingOccupancy2D(**self._grid_params)
        self.gate = KeyframeGate(keyframe_dist_th, keyframe_angle_th)
        self.map_point_stride = max(1, int(map_point_stride))
        self._lock = threading.RLock()
        self._backend_lock = threading.Lock()
        self._generation = 0
        self._version = 0
        self._pose_revision = 0
        self._backend_generation = None
        self._phase = MappingPhase.MAPPING if start_mapping else MappingPhase.IDLE
        self._stats = OnlineMapStats()
        self._saved = False
        self._discarded = False
        self._keyframes = []
        self._opt_mats = []
        self._loop_edges = []
        self._last_lio_mat = None
        self._map_from_lio = np.eye(4)
        self._pose_graph = pose_graph
        self.loop_config = loop_config or LoopConfig(enabled=False)
        self._loop_worker = loop_worker

    @property
    def generation(self):
        with self._lock:
            return self._generation

    @property
    def ingest_token(self):
        with self._lock:
            return self._generation if self._phase is MappingPhase.MAPPING else None

    @property
    def phase(self):
        with self._lock:
            return self._phase

    def attach_loop_backend(self, pose_graph, loop_worker=None):
        with self._lock:
            self._pose_graph = pose_graph
            if loop_worker is not None:
                self._loop_worker = loop_worker

    def start_loop_worker(self):
        if self._loop_worker is not None:
            self._loop_worker.start()

    def stop_loop_worker(self):
        if self._loop_worker is not None:
            self._loop_worker.stop()

    def cancel_pending_loops(self):
        with self._lock:
            self._pose_revision += 1

    def stats(self):
        with self._lock:
            return OnlineMapStats(keyframes=len(self._keyframes), voxels=len(self.voxels),
                occupied_cells=self.grid.occupied_count(), free_cells=self.grid.free_count(),
                last_ingest_ms=self._stats.last_ingest_ms, loops=len(self._loop_edges))

    @property
    def loop_error(self):
        return self._loop_worker.last_error if self._loop_worker is not None else None

    def _clear_map_locked(self):
        self._generation += 1
        self._version += 1
        self._pose_revision = 0
        self.voxels = WorldVoxelHash(self._voxel_resolution)
        self.grid = GrowingOccupancy2D(**self._grid_params)
        self.gate.reset()
        self._stats = OnlineMapStats()
        self._keyframes = []
        self._opt_mats = []
        self._loop_edges = []
        self._last_lio_mat = None
        self._map_from_lio = np.eye(4)

    def set_map_alignment(self, map_from_lio, *, generation=None):
        """Set the fixed map frame before its first observation, never mid-map."""
        from .pose_math import is_se3
        mat = np.asarray(map_from_lio, dtype=np.float64)
        if not is_se3(mat):
            raise ValueError("invalid map alignment")
        with self._lock:
            if (self._keyframes or self._phase is not MappingPhase.MAPPING
                    or (generation is not None and generation != self._generation)):
                return False
            self._map_from_lio = mat.copy()
            return True

    def start_mapping(self, *, reset_lio=None):
        with self._lock:
            self._phase = MappingPhase.IDLE
            self._clear_map_locked()
            generation = self._generation
            self._saved = self._discarded = False
        if self._loop_worker is not None:
            self._loop_worker.reset()
        # Keep ingress closed until the frontend has reset under its own lock.
        if reset_lio is not None:
            reset_lio()
        with self._lock:
            if self._generation == generation:
                self._phase = MappingPhase.MAPPING

    def cancel_mapping(self):
        with self._lock:
            self._phase = MappingPhase.IDLE
            self._clear_map_locked()
            self._discarded = True
            self._saved = False
        if self._loop_worker is not None:
            self._loop_worker.reset()

    def mark_saved(self, snapshot=None):
        with self._lock:
            if snapshot is not None and (snapshot.generation != self._generation or
                                         snapshot.version != self._version):
                return False
            self._saved = True
            self._phase = MappingPhase.IDLE
            return True

    @property
    def was_saved(self):
        with self._lock:
            return self._saved

    @property
    def was_discarded(self):
        with self._lock:
            return self._discarded

    def should_persist(self):
        with self._lock:
            return not (self._discarded or self._saved) and len(self.voxels) > 0

    def freeze_for_save(self):
        with self._lock:
            if not self.should_persist():
                return None
            self._phase = MappingPhase.IDLE
            return MapSnapshot(self._generation, self._version,
                               self.voxels.points_array(), self.snapshot_grid(), self._map_from_lio.copy())

    def resume_after_save_error(self, snapshot):
        with self._lock:
            if snapshot.generation == self._generation and not self._discarded:
                self._phase = MappingPhase.MAPPING

    def is_save_snapshot_current(self, snapshot):
        with self._lock:
            return (snapshot.generation == self._generation and
                    snapshot.version == self._version and not self._discarded)

    def keyframe_count(self):
        with self._lock:
            return len(self._keyframes)

    def keyframe_local_points(self, idx):
        with self._lock:
            return self._keyframes[idx].local_points.copy()

    def keyframe_opt_matrix(self, idx):
        with self._lock:
            return self._opt_mats[idx].copy()

    def loop_snapshot(self) -> LoopSnapshot | None:
        from robot_mapping.backends.loop_types import LoopSnapshot
        with self._lock:
            if self._phase is not MappingPhase.MAPPING:
                return None
            return LoopSnapshot(self._generation, self._pose_revision,
                tuple(kf.local_points for kf in self._keyframes),
                tuple(m.copy() for m in self._opt_mats))

    def loop_snapshot_current(self, snapshot: LoopSnapshot) -> bool:
        with self._lock:
            return (self._phase is MappingPhase.MAPPING and
                    snapshot.generation == self._generation and
                    snapshot.pose_revision == self._pose_revision)

    @staticmethod
    def _lio_matrix(pose):
        from .pose_math import pose_parts_to_matrix
        return pose_parts_to_matrix(pose.position, pose.orientation_xyzw)

    def ingest_odom(self, odom: LioResult, *, generation: int | None = None) -> bool:
        """Only accept the same frame's de-skewed body points, never map snapshots."""
        if "scan_body_points" not in odom:
            raise RuntimeError("FAST-LIO2 缺少 scan_body_points，请重新编译 fastlio2 native")
        return self.ingest_local_points(pose_from_odom(odom), odom["scan_body_points"],
                                        generation=generation)

    def ingest_local_points(self, pose, points, *, generation=None):
        from .pose_math import relative_se3, is_se3
        t0 = time.perf_counter()
        lio_mat = self._lio_matrix(pose)
        if not is_se3(lio_mat):
            raise ValueError("non-finite or invalid keyframe pose")
        local = np.asarray(points, dtype=np.float64)
        if local.size == 0:
            return False
        if local.ndim != 2 or local.shape[1] < 3:
            raise ValueError("points must be (N, >=3)")
        local = local[np.isfinite(local[:, :3]).all(axis=1), :3][::self.map_point_stride].copy()
        if len(local) == 0:
            return False
        local.flags.writeable = False
        with self._lock:
            if self._phase is not MappingPhase.MAPPING:
                return False
            if generation is not None and generation != self._generation:
                return False
            if not self.gate.accept(pose):
                return False
            graph_mat = self._map_from_lio @ lio_mat if not self._opt_mats else (
                self._opt_mats[-1] @ relative_se3(self._last_lio_mat, lio_mat))
            node_id = len(self._keyframes)
            self._keyframes.append(KeyframeRecord(node_id, pose, local))
            self._opt_mats.append(graph_mat)
            self._last_lio_mat = lio_mat
            self._project(self.voxels, self.grid, local, graph_mat)
            self._version += 1
            self._saved = False
            self._stats.last_ingest_ms = (time.perf_counter() - t0) * 1000.
            current_generation = self._generation
        if self._loop_worker is not None:
            self._loop_worker.submit(node_id, generation=current_generation)
        return True

    def ingest_points_direct(self, pose, points):
        """Offline helper: points are in raw LIO world, not optimized map world."""
        from .pose_math import invert_se3, transform_points
        return self.ingest_local_points(pose, transform_points(invert_se3(self._lio_matrix(pose)), points))

    @staticmethod
    def _project(voxels, grid, local, mat):
        from .pose_math import transform_points
        world = transform_points(mat, local)
        voxels.add_points(world)
        # 2D evidence must retain original XY and Z. Projecting 15 cm voxel
        # centers onto 5 cm cells creates a 3-cell lattice and shifts heights.
        grid.ingest_points(world, robot_xy=(float(mat[0, 3]), float(mat[1, 3])))

    def accept_loop(self, match: LoopMatch) -> bool:
        from .pose_math import is_se3, relative_se3
        if self._pose_graph is None or not is_se3(match.T_match_query):
            return False
        cfg = self.loop_config
        if not all(math.isfinite(v) for v in (match.fitness, match.rmse_m,
                                               match.correction_m, match.correction_rad)):
            return False
        if (not cfg.min_inlier_ratio <= match.fitness <= 1. or
            not 0 <= match.rmse_m <= cfg.max_rmse_m or
            not 0 <= match.correction_m <= cfg.max_correction_m or
            not 0 <= match.correction_rad <= cfg.max_correction_rad):
            return False
        # _backend_lock serializes solver mutation; _lock protects the published map.
        # 1. Capture a consistent generation under _lock. No native work holds it.
        with self._backend_lock:
            with self._lock:
                if (self._phase is not MappingPhase.MAPPING or
                    match.generation != self._generation or
                    match.pose_revision != self._pose_revision):
                    return False
                n = len(self._keyframes)
                if not 0 <= match.match_id < match.query_id < n:
                    return False
                if any((e.match_id, e.query_id) == (match.match_id, match.query_id)
                       for e in self._loop_edges):
                    return False
                frames = list(self._keyframes)
                initial = [m.copy() for m in self._opt_mats]
                edges = list(self._loop_edges)
                generation = self._generation
            graph = self._pose_graph
            # 2. Update the solver and build a replacement map outside _lock.
            try:
                if self._backend_generation != generation:
                    graph.reset()
                for k in range(graph.size, n):
                    if k % 16 == 0 and not self._loop_match_current(match):
                        self._backend_generation = None
                        return False
                    rel = None if k == 0 else relative_se3(
                        self._lio_matrix(frames[k-1].lio_pose), self._lio_matrix(frames[k].lio_pose))
                    graph.add_keyframe(initial[k], relative_to_prev=rel)
                if self._backend_generation != generation:
                    for edge in edges:
                        graph.add_loop(edge.match_id, edge.query_id, edge.T_match_query)
                self._backend_generation = generation
                graph.add_loop(match.match_id, match.query_id, match.T_match_query)
                matrices = graph.optimized_matrices()
                voxels = WorldVoxelHash(self._voxel_resolution)
                grid = GrowingOccupancy2D(**self._grid_params)
                for index, (kf, mat) in enumerate(zip(frames, matrices)):
                    if index % 16 == 0 and not self._loop_match_current(match):
                        self._backend_generation = None
                        return False
                    self._project(voxels, grid, kf.local_points, mat)
                # 3. Catch up new observations, then publish in one locked swap.
                # A generation/pose change aborts this transaction; ingress keeps
                # using the previous complete map until publication.
                while True:
                    with self._lock:
                        if (self._generation != generation or
                            self._phase is not MappingPhase.MAPPING or
                            self._pose_revision != match.pose_revision):
                            self._backend_generation = None
                            return False
                        tail = self._keyframes[len(frames):]
                        if not tail:
                            self.voxels, self.grid = voxels, grid
                            self._opt_mats = matrices
                            self._loop_edges.append(match)
                            self._pose_revision += 1
                            self._version += 1
                            self._saved = False
                            return True
                    for kf in tail:
                        mat = matrices[-1] @ relative_se3(
                            self._lio_matrix(frames[-1].lio_pose), self._lio_matrix(kf.lio_pose))
                        frames.append(kf)
                        matrices.append(mat)
                        self._project(voxels, grid, kf.local_points, mat)
            except BaseException:
                self._backend_generation = None
                raise

    def _loop_match_current(self, match):
        with self._lock:
            return (self._phase is MappingPhase.MAPPING and
                    self._generation == match.generation and
                    self._pose_revision == match.pose_revision)

    def snapshot_grid(self):
        with self._lock:
            if self.grid.values is None or self.grid.spec is None:
                return None
            return self.grid.values.copy(), self.grid.spec

