from robot_mapping.runtime.timing import FrameTimings


def test_timing_window_drops_old_spikes_and_copies_samples():
    timing = FrameTimings(window=2)
    assert timing.summary()["frames"] == 0
    timing.add({"map_ms": 1000.0})
    sample = {"map_ms": 10.0}
    timing.add(sample)
    sample["map_ms"] = 999.0
    timing.add({"map_ms": 0.0})
    report = timing.summary()
    assert report["frames"] == 2
    assert report["last_ms"]["map_ms"] == 0.0
    assert report["stages_ms"]["map_ms"]["max"] == 10.0
    assert report["stages_ms"]["map_ms"]["p50"] == 5.0
    report["last_ms"]["map_ms"] = 1000.0
    assert timing.summary()["last_ms"]["map_ms"] == 0.0


def test_projection_timing_does_not_reuse_previous_observation():
    import numpy as np

    from robot_mapping.core.map_types import SE3Pose
    from robot_mapping.core.online_map import OnlineMap

    mapping = OnlineMap()
    pose = SE3Pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    points = np.array([[2.0, 0.0, 0.5]])
    assert mapping.ingest_local_points(pose, points)
    assert mapping.projection_timings["grid_ms"] > 0
    assert not mapping.ingest_local_points(pose, points)
    assert mapping.projection_timings == {"grid_ms": 0.0, "voxel_ms": 0.0}
