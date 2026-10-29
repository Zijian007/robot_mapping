"""Snapshot-based asynchronous FastVGICP detection and pose-graph correction."""
from __future__ import annotations

import math
import queue
import threading
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.spatial import cKDTree

from robot_mapping.core.pose_math import is_se3, relative_se3, se3_rotation_angle, se3_translation_norm, transform_points
from .capabilities import load_pygicp
from .loop_types import LoopConfig, LoopMatch, LoopSnapshot


AlignmentResult = tuple[np.ndarray, bool, float]
AlignFunction = Callable[..., AlignmentResult]

if TYPE_CHECKING:
    from robot_mapping.core.online_map import OnlineMap


class FastVGICPBackend:
    """Own one validated native binding; each call uses a fresh registration."""

    def __init__(self) -> None:
        self._registration_type = load_pygicp().FastVGICP

    def __call__(self, source_pts: np.ndarray, target_pts: np.ndarray,
                 init_T_target_source: np.ndarray, *, resolution: float,
                 max_correspondence_distance: float, num_threads: int = 2) -> AlignmentResult:
        reg = self._registration_type()
        src = np.asarray(source_pts, dtype=np.float64)
        tgt = np.asarray(target_pts, dtype=np.float64)
        for points in (src, tgt):
            if points.ndim != 2 or points.shape[1] < 3 or len(points) < 3:
                raise ValueError("point clouds must contain at least 3 finite XYZ points")
            if not np.isfinite(points[:, :3]).all():
                raise ValueError("point cloud contains non-finite coordinates")
        init = np.asarray(init_T_target_source, dtype=np.float64)
        if not is_se3(init):
            raise ValueError("invalid registration initial pose")
        reg.set_resolution(float(resolution))
        reg.set_max_correspondence_distance(float(max_correspondence_distance))
        reg.set_num_threads(int(num_threads))
        reg.set_input_target(tgt[:, :3])
        reg.set_input_source(src[:, :3])
        result = np.asarray(reg.align(init), dtype=np.float64)
        converged = bool(reg.has_converged()) and is_se3(result)
        # PCL compares squared distances to max_range here.
        mse = float(reg.get_fitness_score(float(max_correspondence_distance) ** 2))
        return result, converged and math.isfinite(mse), mse


def run_fast_vgicp(source_pts, target_pts, init_T_target_source, *, resolution,
                   max_correspondence_distance, num_threads=2) -> AlignmentResult:
    """One-shot registration; persistent workers construct FastVGICPBackend once."""
    return FastVGICPBackend()(source_pts, target_pts, init_T_target_source,
        resolution=resolution, max_correspondence_distance=max_correspondence_distance,
        num_threads=num_threads)


def overlap_metrics(source, target, transform, max_dist):
    if len(source) == 0 or len(target) == 0 or not is_se3(transform):
        return 0., math.inf, 0
    points = transform_points(transform, source)
    if not np.isfinite(points).all() or not np.isfinite(target).all():
        return 0., math.inf, 0
    distances, _ = cKDTree(np.asarray(target)[:, :3]).query(
        points, distance_upper_bound=float(max_dist), workers=1)
    valid = np.isfinite(distances)
    hits = int(valid.sum())
    rmse = float(np.sqrt(np.mean(distances[valid] ** 2))) if hits else math.inf
    return hits / len(points), rmse, hits


def estimate_inlier_ratio(source_pts, target_pts, T_target_source, max_dist):
    return overlap_metrics(source_pts, target_pts, T_target_source, max_dist)[0]


class AsyncLoopWorker:
    def __init__(self, api: OnlineMap, config: LoopConfig, *, align_fn: AlignFunction | None = None):
        self._api = api
        self.config = config
        self._align_fn = align_fn if align_fn is not None else FastVGICPBackend()
        self._q: queue.Queue[tuple[int, int] | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._busy = False
        self._lock = threading.Lock()
        self.last_match: LoopMatch | None = None
        self.last_error: str | None = None
        self.attempts = self.accepts = 0

    def start(self) -> None:
        if not self.config.enabled or self._thread is not None:
            return
        self.reset()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mapping_loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._api.cancel_pending_loops()
        self._q.put(None)
        if self._thread is not None:
            self._thread.join(timeout=30.)
            if self._thread.is_alive():
                raise RuntimeError("loop worker did not stop within 30 seconds")
            self._thread = None

    def reset(self) -> None:
        # In-flight work is invalidated by the map's generation token.
        with self._lock:
            while True:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self.last_match = self.last_error = None
            self.attempts = self.accepts = 0

    def submit(self, keyframe_id: int, *, generation: int) -> None:
        if self.config.enabled and not self._stop.is_set():
            self._q.put((generation, int(keyframe_id)))

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=.2)
            except queue.Empty:
                continue
            if item is None:
                break
            while True:
                try:
                    newer = self._q.get_nowait()
                except queue.Empty:
                    break
                if newer is None:
                    return
                item = newer
            with self._lock:
                self._busy = True
            try:
                self._process(item[1], generation=item[0])
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    self._busy = False

    def _process(self, query_id: int, *, generation: int | None = None) -> None:
        cfg = self.config
        snapshot = self._api.loop_snapshot()
        if snapshot is None or (generation is not None and generation != snapshot.generation):
            return
        if not cfg.min_keyframe_gap <= query_id < len(snapshot.clouds):
            return
        q_pose, source = snapshot.poses[query_id], snapshot.clouds[query_id]
        if len(source) < cfg.min_points:
            return
        best = None
        candidates = range(query_id - cfg.min_keyframe_gap + 1)
        candidates = sorted(candidates, key=lambda k: np.linalg.norm(
            snapshot.poses[k][:3, 3] - q_pose[:3, 3]))
        for cand in candidates:
            if self._stop.is_set() or not self._api.loop_snapshot_current(snapshot):
                return
            c_pose = snapshot.poses[cand]
            init = relative_se3(c_pose, q_pose)
            if se3_translation_norm(init) > cfg.search_radius_m:
                continue
            chunks = []
            # Never include temporally recent frames (especially the query)
            # in the historical submap used to validate a loop.
            for k in range(max(0, cand-cfg.neighbor_keyframes),
                           min(query_id-cfg.min_keyframe_gap+1, cand+cfg.neighbor_keyframes+1)):
                chunks.append(transform_points(relative_se3(c_pose, snapshot.poses[k]), snapshot.clouds[k]))
            target = np.concatenate(chunks, axis=0)
            if len(target) < cfg.min_points:
                continue
            self.attempts += 1
            measured, converged, mse = self._align_fn(source, target, init,
                resolution=cfg.vgicp_resolution,
                max_correspondence_distance=cfg.vgicp_max_correspondence_distance,
                num_threads=cfg.vgicp_num_threads)
            if not converged or not is_se3(measured) or not math.isfinite(mse) or mse < 0:
                continue
            correction = relative_se3(init, measured)
            corr_m, corr_r = se3_translation_norm(correction), se3_rotation_angle(correction)
            if corr_m > cfg.max_correction_m or corr_r > cfg.max_correction_rad:
                continue
            ratio, rmse, hits = overlap_metrics(source, target, measured,
                                                cfg.vgicp_max_correspondence_distance)
            if ratio < cfg.min_inlier_ratio or hits < cfg.min_points or rmse > cfg.max_rmse_m:
                continue
            match = LoopMatch(query_id, cand, measured.copy(), corr_m, corr_r, ratio,
                              rmse, snapshot.generation, snapshot.pose_revision)
            if best is None or (match.fitness, -match.rmse_m) > (best.fitness, -best.rmse_m):
                best = match
        if (best is not None and not self._stop.is_set() and
            self._api.loop_snapshot_current(snapshot) and self._api.accept_loop(best)):
            self.accepts += 1
            self.last_match = best
