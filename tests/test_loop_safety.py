"""Graph and concurrency regressions with no application/HAL imports."""
from dataclasses import replace
import threading
import numpy as np
import pytest
from robot_mapping.core.online_map import OnlineMap, SE3Pose
from robot_mapping.core.pose_math import gravity_aligned_map_from_lio, transform_points
from robot_mapping.backends.loop_closure import AsyncLoopWorker, LoopConfig, LoopSnapshot
from robot_mapping.backends.pose_graph import PoseGraphBackend, PoseGraphNoise
from robot_mapping.backends.capabilities import gtsam_available, pygicp_available
requires_gtsam = pytest.mark.skipif(not gtsam_available(), reason="validated GTSAM unavailable")
requires_vgicp = pytest.mark.skipif(not pygicp_available(), reason="patched VGICP unavailable")



def accept_test_loop(mapping, match_id, query_id, transform):
    """Submit a synthetic loop through the production acceptance path."""
    from robot_mapping.backends.loop_types import LoopMatch
    snapshot = mapping.loop_snapshot()
    return mapping.accept_loop(LoopMatch(query_id, match_id, transform, 0., 0., 1.,
        generation=snapshot.generation, pose_revision=snapshot.pose_revision))

def pose(x=0., y=0.):
    return SE3Pose((x, y, 0.), (0., 0., 0., 1.))


def cloud(seed=3, size=300):
    return np.random.default_rng(seed).normal(size=(size, 3)) + [0, 0, 2]


def test_revisit_keeps_complete_cloud_even_without_new_voxels():
    om = OnlineMap(keyframe_dist_th=.1)
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
        om.ingest_odom({"position": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 1]})


def test_worker_stop_and_restart_do_not_reuse_queue_sentinel():
    api = DetectionAPI(cloud(), cloud())
    worker = AsyncLoopWorker(api, LoopConfig(enabled=True, min_keyframe_gap=1),
        align_fn=lambda *args, **kwargs: (np.eye(4), True, 0.))
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
    om = OnlineMap(keyframe_dist_th=.1, voxel_resolution=.1, pose_graph=PoseGraphBackend())
    for x in (0., 1., 2.):
        om.ingest_points_direct(pose(x), np.array([[10.05+x, .05, .55]]))
    assert accept_test_loop(om, 0, 2, np.eye(4))
    assert om.ingest_points_direct(pose(3), np.array([[13.05, .05, .55]]))
    np.testing.assert_allclose(om.keyframe_local_points(3)[0], [10.05, .05, .55], atol=1e-7)
    expected = transform_points(om.keyframe_opt_matrix(3), om.keyframe_local_points(3))[0]
    assert np.min(np.linalg.norm(om.voxels.points_array()-expected, axis=1)) < .09
    assert abs(float(om.voxels.points_array()[:, 0].max())-13.05) > .3


class DetectionAPI:
    def __init__(self, source, target):
        self.snapshot = LoopSnapshot(0, 0, (target, source), (np.eye(4), np.eye(4)))
        self.matches = []
    def cancel_pending_loops(self): pass
    def loop_snapshot(self): return self.snapshot
    def loop_snapshot_current(self, snap): return snap is self.snapshot
    def accept_loop(self, match):
        self.matches.append(match)
        return True


@pytest.mark.parametrize("mode", ["not_converged", "nan_matrix", "nan_error", "negative_error", "large_correction", "low_overlap", "high_rmse"])
def test_invalid_loop_constraints_do_not_reach_backend(mode):
    target = cloud()
    source = target.copy()
    measured, converged, mse = np.eye(4), True, .001
    if mode == "not_converged": converged = False
    if mode == "nan_matrix": measured[0, 0] = np.nan
    if mode == "nan_error": mse = np.nan
    if mode == "negative_error": mse = -1
    if mode == "large_correction": measured[0, 3] = 3
    if mode == "low_overlap": source[20:] += [30, 0, 0]
    if mode == "high_rmse":
        target = np.array([[x*4, y*4, 0] for x in range(10) for y in range(10)], float)
        source = target + [0, 0, .6]
    api = DetectionAPI(source, target)
    worker = AsyncLoopWorker(api, LoopConfig(enabled=True, min_keyframe_gap=1, neighbor_keyframes=0),
        align_fn=lambda *a, **k: (measured, converged, mse))
    worker._process(1)
    assert not api.matches


@requires_vgicp
def test_real_vgicp_low_overlap_is_rejected():
    rng = np.random.default_rng(13)
    target = rng.normal(size=(200, 3))
    source = np.concatenate((target[:20], rng.normal(size=(180, 3))+[20, 0, 0]))
    api = DetectionAPI(source, target)
    worker = AsyncLoopWorker(api, LoopConfig(enabled=True, min_keyframe_gap=1, neighbor_keyframes=0, vgicp_num_threads=1))
    worker._process(1)
    assert worker.attempts == 1
    assert not api.matches


@requires_gtsam
@requires_vgicp
@pytest.mark.parametrize("tilted", [False, True])
def test_actual_detection_optimizes_and_aligns_revisited_cloud(tilted):
    cfg = LoopConfig(enabled=True, min_keyframe_gap=4, neighbor_keyframes=0, vgicp_num_threads=1)
    graph = PoseGraphBackend(PoseGraphNoise(odom_sigmas=(.2,)*6, loop_sigmas=(.001,)*6))
    om = OnlineMap(keyframe_dist_th=.1, pose_graph=graph, loop_config=cfg)
    alignment = gravity_aligned_map_from_lio([-.15, .01, 1.], np.eye(3)) if tilted else np.eye(4)
    assert om.set_map_alignment(alignment)
    scene = cloud(seed=19, size=600)
    for raw_x, true_x in zip((0., 2., 4., 2.2, .5), (0., 2., 4., 2., 0.)):
        om.ingest_local_points(pose(raw_x), scene - [true_x, 0, 0])
    worker = AsyncLoopWorker(om, cfg)
    worker._process(4)
    assert worker.accepts == 1
    assert om.stats().loops == 1
    aligned = transform_points(om.keyframe_opt_matrix(4), om.keyframe_local_points(4))
    assert np.mean(np.linalg.norm(aligned-transform_points(alignment, scene), axis=1)) < .04
    assert om.ingest_local_points(pose(1.5), scene-[1., 0, 0])
    np.testing.assert_allclose(om.keyframe_opt_matrix(5)[:3, 3], alignment[:3, :3] @ [1., 0, 0], atol=.04)


def test_correction_is_measured_near_query_not_world_origin():
    source = cloud()
    angle = .05
    rotation = np.eye(4)
    rotation[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    api = DetectionAPI(source, transform_points(rotation, source))
    far = np.eye(4)
    far[0, 3] = 1000
    api.snapshot = replace(api.snapshot, poses=(far.copy(), far.copy()))
    worker = AsyncLoopWorker(api, LoopConfig(enabled=True, min_keyframe_gap=1, neighbor_keyframes=0),
        align_fn=lambda *a, **k: (rotation, True, .001))
    worker._process(1)
    assert len(api.matches) == 1
    assert api.matches[0].correction_m == 0.


@requires_gtsam
def test_inflight_old_loop_cannot_modify_restarted_map():
    cfg = LoopConfig(enabled=True, min_keyframe_gap=1, neighbor_keyframes=0)
    om = OnlineMap(keyframe_dist_th=.1, pose_graph=PoseGraphBackend(), loop_config=cfg)
    scene = cloud()
    om.ingest_local_points(pose(), scene)
    om.ingest_local_points(pose(1), scene-[1, 0, 0])
    entered, release = threading.Event(), threading.Event()
    def align(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return np.eye(4), True, .001
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
    om = OnlineMap(keyframe_dist_th=.1, pose_graph=PoseGraphBackend())
    for x in (0., 1.): om.ingest_local_points(pose(x), cloud())
    entered, release, read_ok = threading.Event(), threading.Event(), threading.Event()
    original = om._project
    def project(voxels, grid, points, matrix):
        if threading.current_thread().name == "test_rebuild" and not entered.is_set():
            entered.set()
            assert release.wait(3)
        return original(voxels, grid, points, matrix)
    monkeypatch.setattr(om, "_project", project)
    results, errors = [], []
    def optimize():
        try: results.append(accept_test_loop(om, 0, 1, np.eye(4)))
        except Exception as exc: errors.append(exc)
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
        if reader.ident is not None: reader.join(3)
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
    reg.set_resolution(.5)
    reg.set_input_source(points)
    reg.set_input_target(points+[.2, -.1, 0])
    entered, progressed = threading.Event(), threading.Event()
    # Raising the switch interval prevents a bytecode timeslice between the
    # event and align from passing the test without a native GIL release.
    def observe():
        entered.wait()
        progressed.set()
    thread = threading.Thread(target=observe)
    old = sys.getswitchinterval()
    try:
        sys.setswitchinterval(1.)
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
        transform, converged, mse = backend(points, points, np.eye(4),
            resolution=.5, max_correspondence_distance=1., num_threads=1)
        assert converged and mse < .02 ** 2
        np.testing.assert_allclose(transform, np.eye(4), atol=.02)
    assert len(loads) == 1
