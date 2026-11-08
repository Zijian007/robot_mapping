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

from .grid import OccupancyGridSpec
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
    grid_observations: int = 0
    removed_voxels: int = 0


@dataclass(frozen=True)
class KeyframeRecord:
    lio_pose: SE3Pose
    local_points: np.ndarray
    local_rays: np.ndarray | None = None


@dataclass(frozen=True)
class GridObservation:
    anchor_id: int
    anchor_from_tracking: np.ndarray
    local_rays: np.ndarray
    keyframe: bool = False


@dataclass(frozen=True)
class MapSnapshot:
    generation: int
    version: int
    points: np.ndarray
    grid: tuple[np.ndarray, OccupancyGridSpec] | None
    map_from_lio: np.ndarray
    tracking_from_footprint: np.ndarray
    grid_min_height: float
    grid_max_height: float


class OnlineMap:
    """Own immutable observations; publish rebuilt maps with a short locked swap.

    Ingestion never touches GTSAM. The loop worker alone owns its solver under
    _backend_lock, including reset/replay after an abandoned optimization.
    """

    def __init__(self, *, voxel_resolution=0.15, grid_resolution=0.05,
                 min_height=-0.25, max_height=1.8, margin_cells=5,
                 keyframe_dist_th=0.5, keyframe_angle_th=0.2,
                 free_radius_m=0.35, map_point_stride=1, start_mapping=True,
                 ray_clearing=False, height_resolution=.2, max_ray_range=30., max_rays=2048,
                 grid_update_interval_s=0.,
                 voxel_clearing=False, voxel_clear_observations=3,
                 loop_config=None, pose_graph=None, loop_worker=None):
        from robot_mapping.backends.loop_types import LoopConfig
        self._grid_params = dict(resolution=grid_resolution, min_height=min_height,
                                 max_height=max_height, margin_cells=margin_cells,
                                 free_radius_m=free_radius_m)
        self._ray_clearing = bool(ray_clearing)
        self._voxel_clearing = bool(voxel_clearing)
        if self._voxel_clearing and not self._ray_clearing:
            raise ValueError("3D voxel clearing requires ray_clearing")
        self._voxel_params = dict(resolution=voxel_resolution)
        self._voxel_type = WorldVoxelHash
        if self._voxel_clearing:
            from .ray_voxels import RayVoxelHash
            self._voxel_type = RayVoxelHash
            self._voxel_params.update(clear_observations=voxel_clear_observations,
                                      max_ray_range=max_ray_range, max_rays=max_rays)
        self._grid_update_interval_s = float(grid_update_interval_s)
        if not math.isfinite(self._grid_update_interval_s) or self._grid_update_interval_s < 0:
            raise ValueError("grid update interval must be finite and nonnegative")
        if self._ray_clearing:
            from .ray_grid import RayOccupancy2D
            self._grid_type = RayOccupancy2D
            self._grid_params.update(height_resolution=height_resolution,
                                     max_ray_range=max_ray_range, max_rays=max_rays)
        else:
            self._grid_type = GrowingOccupancy2D
        self.voxels = self._voxel_type(**self._voxel_params)
        self.grid = self._grid_type(**self._grid_params)
        self.gate = KeyframeGate(keyframe_dist_th, keyframe_angle_th)
        self.map_point_stride = max(1, int(map_point_stride))
        self._lock = threading.RLock()
        self._backend_lock = threading.Lock()
        self._generation = 0
        self._version = 0
        self._pose_revision = 0
        self._backend_generation = None
        self._phase = MappingPhase.MAPPING if start_mapping else MappingPhase.IDLE
        self._last_ingest_ms = 0.
        self._projection_timings = {"grid_ms": 0., "voxel_ms": 0.}
        self._saved = False
        self._discarded = False
        self._keyframes = []
        self._grid_observations = []
        self._last_grid_stamp = None
        self._last_input_stamp = None
        self._opt_mats = []
        self._loop_edges = []
        self._last_lio_mat = None
        self._map_from_lio = np.eye(4)
        self._tracking_from_footprint = np.eye(4)
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
                last_ingest_ms=self._last_ingest_ms, loops=len(self._loop_edges),
                grid_observations=len(self._grid_observations),
                removed_voxels=getattr(self.voxels, "removed_voxels", 0))

    @property
    def projection_timings(self):
        with self._lock:
            return dict(self._projection_timings)

    @property
    def loop_error(self):
        return self._loop_worker.last_error if self._loop_worker is not None else None

    def _clear_map_locked(self):
        self._generation += 1
        self._version += 1
        self._pose_revision = 0
        self.voxels = self._voxel_type(**self._voxel_params)
        self.grid = self._grid_type(**self._grid_params)
        self.gate.reset()
        self._last_ingest_ms = 0.
        self._projection_timings = {"grid_ms": 0., "voxel_ms": 0.}
        self._keyframes = []
        self._grid_observations = []
        self._last_grid_stamp = None
        self._last_input_stamp = None
        self._opt_mats = []
        self._loop_edges = []
        self._last_lio_mat = None
        self._map_from_lio = np.eye(4)
        self._tracking_from_footprint = np.eye(4)

    def set_map_alignment(self, map_from_lio, *, generation=None,
                          tracking_from_footprint=None, height_bounds=None):
        """Apply caller-selected coordinates, footprint extrinsic and map-Z bounds.

        No origin/heading policy is selected here. All geometry is fixed for
        this generation, including asynchronous loop reconstruction.
        """
        from .pose_math import is_se3
        mat = np.asarray(map_from_lio, dtype=np.float64)
        if not is_se3(mat):
            raise ValueError("invalid map alignment")
        base = np.eye(4) if tracking_from_footprint is None else np.asarray(tracking_from_footprint, dtype=np.float64)
        if not is_se3(base):
            raise ValueError("invalid tracking_from_footprint extrinsic")
        bounds = (self._grid_params["min_height"], self._grid_params["max_height"]) if height_bounds is None else height_bounds
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError("invalid map height bounds")
        with self._lock:
            if (self._keyframes or self._phase is not MappingPhase.MAPPING
                    or (generation is not None and generation != self._generation)):
                return False
            self._map_from_lio = mat.copy()
            self._tracking_from_footprint = base.copy()
            self.grid.min_height, self.grid.max_height = map(float, bounds)
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
            return not (self._discarded or self._saved) and bool(self._keyframes)

    def freeze_for_save(self):
        with self._lock:
            if not self.should_persist():
                return None
            self._phase = MappingPhase.IDLE
            return MapSnapshot(self._generation, self._version,
                               self.voxels.points_array(), self.snapshot_grid(), self._map_from_lio.copy(),
                               self._tracking_from_footprint.copy(),
                               self.grid.min_height, self.grid.max_height)

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

    def keyframe_local_rays(self, idx):
        with self._lock:
            rays = self._keyframes[idx].local_rays
            return None if rays is None else rays.copy()

    def grid_observation(self, idx):
        """Return one frozen observation with its current optimized map pose."""
        with self._lock:
            obs = self._grid_observations[idx]
            return (self._opt_mats[obs.anchor_id] @ obs.anchor_from_tracking,
                    obs.local_rays.copy())

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
        if odom.get("status") != "scan_to_map" or odom.get("lidar_update_valid") is not True:
            return False
        if "scan_body_points" not in odom:
            raise RuntimeError("FAST-LIO2 缺少 scan_body_points，请重新编译 fastlio2 native")
        return self.ingest_local_points(pose_from_odom(odom), odom["scan_body_points"],
                                        generation=generation, rays=odom.get("scan_body_rays"), stamp=odom.get("stamp"))

    def ingest_local_points(self, pose, points, *, generation=None, rays=None, stamp=None):
        """Return whether a graph keyframe was added; timed grid updates return False."""
        from .pose_math import relative_se3, is_se3
        t0 = time.perf_counter()
        with self._lock:
            self._projection_timings = {"grid_ms": 0., "voxel_ms": 0.}
        lio_mat = self._lio_matrix(pose)
        if not is_se3(lio_mat):
            raise ValueError("non-finite or invalid keyframe pose")
        if stamp is not None and not math.isfinite(stamp):
            raise ValueError("observation stamp must be finite")
        local = np.asarray(points, dtype=np.float64)
        if local.size == 0:
            return False
        if local.ndim != 2 or local.shape[1] < 3:
            raise ValueError("points must be (N, >=3)")
        local = local[np.isfinite(local[:, :3]).all(axis=1), :3][::self.map_point_stride].copy()
        if len(local) == 0:
            return False
        local.flags.writeable = False
        local_rays = None
        if self._ray_clearing:
            if rays is None:
                raise RuntimeError("ray clearing requires scan_body_rays from the same frame")
            local_rays = np.asarray(rays, dtype=np.float64)
            if local_rays.ndim != 2 or local_rays.shape[1] != 6 or not np.isfinite(local_rays).all():
                raise ValueError("scan_body_rays must be finite (N,6)")
        with self._lock:
            if self._phase is not MappingPhase.MAPPING:
                return False
            if generation is not None and generation != self._generation:
                return False
            if stamp is not None:
                if self._last_input_stamp is not None and stamp < self._last_input_stamp:
                    raise ValueError("observation timestamps moved backwards")
                if stamp == self._last_input_stamp:
                    return False
                self._last_input_stamp = stamp
            keyframe = self.gate.accept(pose)
            timed_update = (self._ray_clearing and self._grid_update_interval_s > 0 and stamp is not None
                and (self._last_grid_stamp is None or stamp-self._last_grid_stamp >= self._grid_update_interval_s))
            if not keyframe and not timed_update:
                return False
            if local_rays is not None:
                local_rays = local_rays.copy()
                local_rays.flags.writeable = False
            graph_mat = self._map_from_lio @ lio_mat if not self._opt_mats else (
                self._opt_mats[-1] @ relative_se3(self._last_lio_mat, lio_mat))
            if keyframe:
                node_id = len(self._keyframes)
                self._keyframes.append(KeyframeRecord(pose, local, local_rays))
                self._opt_mats.append(graph_mat)
                self._last_lio_mat = lio_mat
            self._projection_timings = self._project(self.voxels, self.grid, local if keyframe else None,
                          graph_mat, self._tracking_from_footprint, local_rays)
            if self._ray_clearing:
                relative = np.eye(4) if keyframe else relative_se3(self._last_lio_mat, lio_mat)
                relative.flags.writeable = False
                self._grid_observations.append(GridObservation(len(self._keyframes)-1, relative, local_rays, keyframe))
                self._last_grid_stamp = stamp
            self._version += 1
            self._saved = False
            self._last_ingest_ms = (time.perf_counter() - t0) * 1000.
            current_generation = self._generation
        if keyframe and self._loop_worker is not None:
            self._loop_worker.submit(node_id, generation=current_generation)
        return keyframe

    def ingest_points_direct(self, pose, points):
        """Offline helper: points are in raw LIO world, not optimized map world."""
        from .pose_math import invert_se3, transform_points
        return self.ingest_local_points(pose, transform_points(invert_se3(self._lio_matrix(pose)), points))

    def _project(self, voxels, grid, local, mat, tracking_from_footprint, rays=None):
        """Project one accepted observation, for both ingress and loop replay.

        Only keyframes supply geometry. Transform measured rays once and share
        them between the independent 2D and 3D evidence accumulators.
        """
        from .pose_math import transform_points
        start = time.perf_counter()
        if local is not None:
            world = transform_points(mat, local)
            voxels.add_points(world)
        voxel_ms = (time.perf_counter()-start)*1000.
        start = time.perf_counter()
        # 2D evidence must retain original XY and Z. Projecting 15 cm voxel
        # centers onto 5 cm cells creates a 3-cell lattice and shifts heights.
        if rays is None:
            grid.ingest_points(world, robot_xy=tuple((mat @ tracking_from_footprint)[:2, 3]))
            grid_ms = (time.perf_counter()-start)*1000.
        else:
            world_rays = self._project_grid(grid, rays, mat, tracking_from_footprint)
            grid_ms = (time.perf_counter()-start)*1000.
            if self._voxel_clearing:
                start = time.perf_counter()
                voxels.ingest_rays(world_rays)
                voxel_ms += (time.perf_counter()-start)*1000.
        return {"grid_ms": grid_ms, "voxel_ms": voxel_ms}

    def _replay_observation(self, voxels, grid, obs, frames, matrices, footprint):
        mat = matrices[obs.anchor_id] @ obs.anchor_from_tracking
        local = frames[obs.anchor_id].local_points if obs.keyframe else None
        self._project(voxels, grid, local, mat, footprint, obs.local_rays)

    @staticmethod
    def _project_grid(grid, rays, mat, tracking_from_footprint):
        from .pose_math import transform_points
        world_rays = np.column_stack((transform_points(mat, rays[:, :3]), transform_points(mat, rays[:, 3:])))
        grid.ingest_rays(world_rays, robot_xy=tuple((mat @ tracking_from_footprint)[:2, 3]))
        return world_rays

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
                observations = list(self._grid_observations)
                initial = [m.copy() for m in self._opt_mats]
                edges = list(self._loop_edges)
                generation = self._generation
                tracking_from_footprint = self._tracking_from_footprint.copy()
                grid_params = dict(self._grid_params, min_height=self.grid.min_height,
                                   max_height=self.grid.max_height)
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
                voxels = self._voxel_type(**self._voxel_params)
                grid = self._grid_type(**grid_params)
                for index, (kf, mat) in enumerate(zip(frames, matrices)):
                    if index % 16 == 0 and not self._loop_match_current(match):
                        self._backend_generation = None
                        return False
                    if not self._ray_clearing:
                        self._project(voxels, grid, kf.local_points, mat, tracking_from_footprint)
                for index, obs in enumerate(observations):
                    if index % 16 == 0 and not self._loop_match_current(match):
                        self._backend_generation = None
                        return False
                    self._replay_observation(voxels, grid, obs, frames, matrices, tracking_from_footprint)
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
                        grid_tail = self._grid_observations[len(observations):]
                        if not tail and not grid_tail:
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
                        if not self._ray_clearing:
                            self._project(voxels, grid, kf.local_points, mat, tracking_from_footprint)
                    for obs in grid_tail:
                        self._replay_observation(voxels, grid, obs, frames, matrices, tracking_from_footprint)
                        observations.append(obs)
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

