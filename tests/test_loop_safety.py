"""Graph and concurrency regressions with no application/HAL imports."""

import threading
from dataclasses import replace

import numpy as np
import pytest

from robot_mapping.backends.capabilities import gtsam_available, pygicp_available
from robot_mapping.backends.loop_closure import AsyncLoopWorker
from robot_mapping.backends.loop_types import LoopConfig, LoopSnapshot
from robot_mapping.backends.pose_graph import PoseGraphBackend, PoseGraphNoise
from robot_mapping.core.online_map import OnlineMap, SE3Pose
from robot_mapping.core.pose_math import gravity_aligned_frame, transform_points

requires_gtsam = pytest.mark.skipif(not gtsam_available(), reason="validated GTSAM unavailable")
requires_vgicp = pytest.mark.skipif(not pygicp_available(), reason="patched VGICP unavailable")


def accept_test_loop(mapping, match_id, query_id, transform):
    """Submit a synthetic loop through the production acceptance path."""
    from robot_mapping.backends.loop_types import LoopMatch

    snapshot = mapping.loop_snapshot()
    return mapping.accept_loop(
        LoopMatch(
            query_id,
            match_id,
            transform,
            0.0,
            0.0,
            1.0,
            generation=snapshot.generation,
            pose_revision=snapshot.pose_revision,
        )
    )


def pose(x=0.0, y=0.0):
    return SE3Pose((x, y, 0.0), (0.0, 0.0, 0.0, 1.0))


def cloud(seed=3, size=300):
    return np.random.default_rng(seed).normal(size=(size, 3)) + [0, 0, 2]


def test_revisit_keeps_complete_cloud_even_without_new_voxels():
    om = OnlineMap(keyframe_dist_th=0.1)
    points = cloud()
    assert om.ingest_points_direct(pose(), points)
    assert om.ingest_points_direct(pose(1), points)
    assert len(om.keyframe_local_points(1)) == len(points)
    np.testing.assert_allclose(om.keyframe_local_points(1), points - [1, 0, 0])


def test_stale_frame_and_missing_scan_snapshot_are_rejected():
    om = OnlineMap()
    previous = om.generation
    om.start_mapping()
    assert not om.ingest_local_points(pose(8), cloud(), generation=previous)
    assert om.keyframe_count() == 0
    with pytest.raises(RuntimeError, match="scan_body_points"):
        om.ingest_odom(
            {
                "status": "scan_to_map",
                "lidar_update_valid": True,
                "position": [0, 0, 0],
                "orientation_xyzw": [0, 0, 0, 1],
            }
        )


def test_worker_stop_and_restart_do_not_reuse_queue_sentinel():
    api = DetectionAPI(cloud(), cloud())
    worker = AsyncLoopWorker(
        api,
        LoopConfig(enabled=True, min_keyframe_gap=1),
        align_fn=lambda *args, **kwargs: (np.eye(4), True, 0.0),
    )
    worker.start()
    worker.stop()
    worker.start()
    try:
        assert worker._thread.is_alive()
    finally:
        worker.stop()
    assert worker._thread is None


def test_start_keeps_ingress_closed_until_lio_reset_finishes():
    om = OnlineMap()
    attempts = []

    def reset():
        assert om.ingest_token is None
        attempts.append(om.ingest_local_points(pose(), cloud(), generation=om.generation))

    om.start_mapping(reset_lio=reset)
    assert attempts == [False]
    assert om.ingest_local_points(pose(), cloud(), generation=om.generation)


@requires_gtsam
def test_new_cloud_uses_corrected_map_pose_after_loop():
    om = OnlineMap(keyframe_dist_th=0.1, voxel_resolution=0.1, pose_graph=PoseGraphBackend())
    for x in (0.0, 1.0, 2.0):
        om.ingest_points_direct(pose(x), np.array([[10.05 + x, 0.05, 0.55]]))
    assert accept_test_loop(om, 0, 2, np.eye(4))
    assert om.ingest_points_direct(pose(3), np.array([[13.05, 0.05, 0.55]]))
    np.testing.assert_allclose(om.keyframe_local_points(3)[0], [10.05, 0.05, 0.55], atol=1e-7)
    expected = transform_points(om.keyframe_opt_matrix(3), om.keyframe_local_points(3))[0]
    assert np.min(np.linalg.norm(om.voxels.points_array() - expected, axis=1)) < 0.09
    assert abs(float(om.voxels.points_array()[:, 0].max()) - 13.05) > 0.3


class DetectionAPI:
    def __init__(self, source, target):
        self.snapshot = LoopSnapshot(0, 0, (target, source), (np.eye(4), np.eye(4)))
        self.matches = []

    def cancel_pending_loops(self):
        pass

    def loop_snapshot(self):
        return self.snapshot

    def loop_snapshot_current(self, snap):
        return snap is self.snapshot

    def accept_loop(self, match):
        self.matches.append(match)
        return True


@pytest.mark.parametrize(
    "mode",
    [
        "not_converged",
        "nan_matrix",
        "nan_error",
        "negative_error",
        "large_correction",
        "low_overlap",
        "high_rmse",
    ],
)
def test_invalid_loop_constraints_do_not_reach_backend(mode):
    target = cloud()
    source = target.copy()
    measured, converged, mse = np.eye(4), True, 0.001
    if mode == "not_converged":
        converged = False
    if mode == "nan_matrix":
        measured[0, 0] = np.nan
    if mode == "nan_error":
        mse = np.nan
    if mode == "negative_error":
        mse = -1
    if mode == "large_correction":
        measured[0, 3] = 3
    if mode == "low_overlap":
        source[20:] += [30, 0, 0]
    if mode == "high_rmse":
        target = np.array([[x * 4, y * 4, 0] for x in range(10) for y in range(10)], float)
        source = target + [0, 0, 0.6]
    api = DetectionAPI(source, target)
    worker = AsyncLoopWorker(
        api,
        LoopConfig(enabled=True, min_keyframe_gap=1, neighbor_keyframes=0),
        align_fn=lambda *a, **k: (measured, converged, mse),
    )
    worker._process(1)
    assert not api.matches


@requires_vgicp
def test_real_vgicp_low_overlap_is_rejected():
    rng = np.random.default_rng(13)
    target = rng.normal(size=(200, 3))
    source = np.concatenate((target[:20], rng.normal(size=(180, 3)) + [20, 0, 0]))
    api = DetectionAPI(source, target)
    worker = AsyncLoopWorker(
        api, LoopConfig(enabled=True, min_keyframe_gap=1, neighbor_keyframes=0, vgicp_num_threads=1)
    )
    worker._process(1)
    assert worker.attempts == 1
    assert not api.matches


@requires_gtsam
@requires_vgicp
@pytest.mark.parametrize("tilted", [False, True])
@pytest.mark.parametrize("translated", [False, True])
def test_actual_detection_optimizes_and_aligns_revisited_cloud(tilted, translated):
    cfg = LoopConfig(enabled=True, min_keyframe_gap=4, neighbor_keyframes=0, vgicp_num_threads=1)
    graph = PoseGraphBackend(PoseGraphNoise(odom_sigmas=(0.2,) * 6, loop_sigmas=(0.001,) * 6))
    om = OnlineMap(keyframe_dist_th=0.1, pose_graph=graph, loop_config=cfg)
    alignment = gravity_aligned_frame([-0.15, 0.01, 1.0], [1, 0, 0]) if tilted else np.eye(4)
    if translated:
        tracking_from_footprint = np.eye(4)
        tracking_from_footprint[:3, 3] = [-0.19, -0.205, -0.32]
        alignment[:3, 3] = -alignment[:3, :3] @ tracking_from_footprint[:3, 3]
        assert om.set_map_alignment(
            alignment,
            tracking_from_footprint=tracking_from_footprint,
            height_bounds=(-0.25 + alignment[2, 3], 1.8 + alignment[2, 3]),
        )
    else:
        assert om.set_map_alignment(alignment)
    height_bounds = (om.grid.min_height, om.grid.max_height)
    scene = cloud(seed=19, size=600)
    for raw_x, true_x in zip((0.0, 2.0, 4.0, 2.2, 0.5), (0.0, 2.0, 4.0, 2.0, 0.0)):
        om.ingest_local_points(pose(raw_x), scene - [true_x, 0, 0])
    worker = AsyncLoopWorker(om, cfg)
    worker._process(4)
    assert worker.accepts == 1
    assert om.stats().loops == 1
    assert (om.grid.min_height, om.grid.max_height) == height_bounds
    aligned = transform_points(om.keyframe_opt_matrix(4), om.keyframe_local_points(4))
    assert np.mean(np.linalg.norm(aligned - transform_points(alignment, scene), axis=1)) < 0.04
    assert om.ingest_local_points(pose(1.5), scene - [1.0, 0, 0])
    np.testing.assert_allclose(
        om.keyframe_opt_matrix(5)[:3, 3],
        alignment[:3, :3] @ [1.0, 0, 0] + alignment[:3, 3],
        atol=0.04,
    )


def test_correction_is_measured_near_query_not_world_origin():
    source = cloud()
    angle = 0.05
    rotation = np.eye(4)
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    api = DetectionAPI(source, transform_points(rotation, source))
    far = np.eye(4)
    far[0, 3] = 1000
    api.snapshot = replace(api.snapshot, poses=(far.copy(), far.copy()))
    worker = AsyncLoopWorker(
        api,
        LoopConfig(enabled=True, min_keyframe_gap=1, neighbor_keyframes=0),
        align_fn=lambda *a, **k: (rotation, True, 0.001),
    )
    worker._process(1)
    assert len(api.matches) == 1
    assert api.matches[0].correction_m == 0.0


@requires_gtsam
def test_inflight_old_loop_cannot_modify_restarted_map():
    cfg = LoopConfig(enabled=True, min_keyframe_gap=1, neighbor_keyframes=0)
    om = OnlineMap(keyframe_dist_th=0.1, pose_graph=PoseGraphBackend(), loop_config=cfg)
    scene = cloud()
    om.ingest_local_points(pose(), scene)
    om.ingest_local_points(pose(1), scene - [1, 0, 0])
    entered, release = threading.Event(), threading.Event()

    def align(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return np.eye(4), True, 0.001

    worker = AsyncLoopWorker(om, cfg, align_fn=align)
    thread = threading.Thread(target=worker._process, args=(1,))
    thread.start()
    try:
        assert entered.wait(2)
        om.start_mapping()
        om.ingest_local_points(pose(100), scene)
        om.ingest_local_points(pose(103), scene)
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert om.stats().loops == 0
    assert om.keyframe_opt_matrix(1)[0, 3] == 103


@requires_gtsam
def test_map_rebuild_does_not_hold_ingress_or_snapshot_lock(monkeypatch):
    om = OnlineMap(keyframe_dist_th=0.1, pose_graph=PoseGraphBackend())
    for x in (0.0, 1.0):
        om.ingest_local_points(pose(x), cloud())
    entered, release, read_ok = threading.Event(), threading.Event(), threading.Event()
    original = om._project

    def project(voxels, grid, points, matrix, tracking_from_footprint, rays=None):
        if threading.current_thread().name == "test_rebuild" and not entered.is_set():
            entered.set()
            assert release.wait(3)
        return original(voxels, grid, points, matrix, tracking_from_footprint, rays)

    monkeypatch.setattr(om, "_project", project)
    results, errors = [], []

    def optimize():
        try:
            results.append(accept_test_loop(om, 0, 1, np.eye(4)))
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=optimize, name="test_rebuild")
    thread.start()

    def observe():
        om.stats()
        assert om.snapshot_grid() is not None
        assert om.ingest_local_points(pose(2), cloud())
        read_ok.set()

    reader = threading.Thread(target=observe)
    try:
        assert entered.wait(2)
        reader.start()
        assert read_ok.wait(2), "map rebuild blocked the ingestion/snapshot lock"
    finally:
        release.set()
        thread.join(3)
        if reader.ident is not None:
            reader.join(3)
    assert not errors
    assert results == [True]
    assert om.keyframe_count() == 3
    assert len(om.loop_snapshot().poses) == 3


@requires_vgicp
def test_vgicp_releases_gil_during_actual_registration():
    import sys

    import pygicp

    reg = pygicp.FastVGICP()
    points = cloud(size=15000)
    reg.set_num_threads(1)
    reg.set_resolution(0.5)
    reg.set_input_source(points)
    reg.set_input_target(points + [0.2, -0.1, 0])
    entered, progressed = threading.Event(), threading.Event()

    # Raising the switch interval prevents a bytecode timeslice between the
    # event and align from passing the test without a native GIL release.
    def observe():
        entered.wait()
        progressed.set()

    thread = threading.Thread(target=observe)
    old = sys.getswitchinterval()
    try:
        sys.setswitchinterval(1.0)
        thread.start()
        entered.set()
        reg.align(np.eye(4))
        assert progressed.is_set(), "FastVGICP kept the Python GIL"
    finally:
        sys.setswitchinterval(old)
        entered.set()
        thread.join(2)


@requires_vgicp
def test_persistent_registration_loads_native_binding_once(monkeypatch):
    import robot_mapping.backends.loop_closure as loops

    original = loops.load_pygicp
    loads = []

    def load():
        loads.append(True)
        return original()

    monkeypatch.setattr(loops, "load_pygicp", load)
    backend = loops.FastVGICPBackend()
    points = cloud(size=300)
    for _ in range(2):
        transform, converged, mse = backend(
            points,
            points,
            np.eye(4),
            resolution=0.5,
            max_correspondence_distance=1.0,
            num_threads=1,
        )
        assert converged and mse < 0.02**2
        np.testing.assert_allclose(transform, np.eye(4), atol=0.02)
    assert len(loads) == 1


@requires_gtsam
def test_ray_evidence_rebuild_matches_online_and_does_not_clean_3d():
    om = OnlineMap(
        ray_clearing=True,
        keyframe_dist_th=0.0,
        min_height=0.0,
        max_height=2.0,
        pose_graph=PoseGraphBackend(),
    )
    for x in [2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0]:
        om.ingest_local_points(pose(), [[x, 0, 0.5]], rays=[[0, 0, 0.5, x, 0, 0.5]])
    before, spec = om.snapshot_grid()
    scores = om.grid._scores.copy()
    points = om.voxels.points_array().copy()
    assert accept_test_loop(om, 0, 6, np.eye(4))
    np.testing.assert_array_equal(om.snapshot_grid()[0], before)
    np.testing.assert_array_equal(om.grid._scores, scores)
    np.testing.assert_array_equal(om.voxels.points_array(), points)
    assert len(points) == 2  # the 3D person voxel is deliberately unchanged
    om.start_mapping()
    assert om.keyframe_count() == 0 and om.grid._scores is None


@requires_gtsam
@pytest.mark.parametrize("voxel_clearing", [False, True])
def test_loop_rebuild_catches_up_grid_only_observations(monkeypatch, voxel_clearing):
    mapping = OnlineMap(
        ray_clearing=True,
        grid_update_interval_s=1.0,
        keyframe_dist_th=0.5,
        voxel_clearing=voxel_clearing,
        min_height=0.0,
        pose_graph=PoseGraphBackend(),
    )
    mapping.ingest_local_points(
        pose(), [[2.025, 0.075, 0.525]], rays=[[0, 0.075, 0.525, 2.025, 0.075, 0.525]], stamp=0.0
    )
    mapping.ingest_local_points(
        pose(1.0), [[1.025, 0.075, 0.525]], rays=[[0, 0.075, 0.525, 1.025, 0.075, 0.525]], stamp=1.0
    )
    entered, release = threading.Event(), threading.Event()
    project = mapping._project_grid

    def delayed(grid, rays, matrix, footprint):
        if threading.current_thread().name == "grid-rebuild" and not entered.is_set():
            entered.set()
            assert release.wait(3.0)
        return project(grid, rays, matrix, footprint)

    monkeypatch.setattr(mapping, "_project_grid", delayed)
    results = []
    relative = np.eye(4)
    relative[0, 3] = 1.0
    thread = threading.Thread(
        target=lambda: results.append(accept_test_loop(mapping, 0, 1, relative)),
        name="grid-rebuild",
    )
    thread.start()
    try:
        assert entered.wait(3.0)
        for stamp in range(2, 7):
            assert not mapping.ingest_local_points(
                pose(1.0),
                [[3.125, 0.075, 0.525]],
                rays=[[0, 0.075, 0.525, 3.125, 0.075, 0.525]],
                stamp=float(stamp),
            )
        expected = mapping.grid.values.copy()
        expected_scores = mapping.grid._scores.copy()
        expected_voxels = mapping.voxels._cells.copy()
    finally:
        release.set()
        thread.join(5.0)
    assert not thread.is_alive()
    assert results == [True]
    assert mapping.keyframe_count() == 2
    assert mapping.stats().grid_observations == 7
    np.testing.assert_array_equal(mapping.grid.values, expected)
    np.testing.assert_array_equal(mapping.grid._scores, expected_scores)
    assert mapping.voxels._cells == expected_voxels
    assert len(expected_voxels) == (0 if voxel_clearing else 1)


@requires_gtsam
def test_3d_rebuild_replays_insertions_and_clearing_in_time_order():
    mapping = OnlineMap(
        ray_clearing=True, voxel_clearing=True, keyframe_dist_th=0.0, pose_graph=PoseGraphBackend()
    )
    for stamp, x in enumerate([2.025, 4.125, 4.125, 4.125, 2.025]):
        mapping.ingest_local_points(
            pose(), [[x, 0.075, 0.525]], rays=[[0, 0.075, 0.525, x, 0.075, 0.525]], stamp=stamp
        )
    assert mapping.stats().removed_voxels == 1
    expected = mapping.voxels._cells.copy()
    assert len(expected) == 2  # the final keyframe restores the old location
    assert accept_test_loop(mapping, 0, 4, np.eye(4))
    assert mapping.voxels._cells == expected
    assert mapping.stats().removed_voxels == 1
    np.testing.assert_allclose(mapping.keyframe_local_points(0), [[2.025, 0.075, 0.525]])
    mapping.start_mapping()
    assert len(mapping.voxels) == 0 and mapping.stats().removed_voxels == 0
