import numpy as np
import pytest

from robot_mapping.core.online_map import OnlineMap


def observation(stamp, distance=2., x=0., valid=True):
    return dict(status="scan_to_map", lidar_update_valid=valid, stamp=stamp,
        position=[x, 0, 0], orientation_xyzw=[0, 0, 0, 1],
        scan_body_points=np.array([[distance, 0, .5]]),
        scan_body_rays=np.array([[0, 0, .5, distance, 0, .5]]))


def cell(mapping, x):
    row, col = mapping.grid._world_to_rc(np.array([x]), np.array([0.]))
    return mapping.grid.values[row[0], col[0]]


def test_stationary_clear_uses_sensor_time_without_adding_graph_nodes_or_3d_points():
    mapping = OnlineMap(ray_clearing=True, grid_update_interval_s=1., min_height=.4, max_height=.6)
    assert mapping.ingest_odom(observation(100.))
    original_points = mapping.voxels.points_array().copy()
    for stamp in np.arange(100.1, 101., .1):
        assert not mapping.ingest_odom(observation(stamp))
    assert mapping.stats().grid_observations == 1
    assert not mapping.ingest_odom(observation(101.))
    assert cell(mapping, 2) == 100
    for stamp in range(102, 107):
        assert not mapping.ingest_odom(observation(stamp, 4.))
    assert cell(mapping, 2) == 0
    assert cell(mapping, 4) == 100
    assert mapping.keyframe_count() == 1
    assert mapping.stats().grid_observations == 7
    np.testing.assert_array_equal(mapping.voxels.points_array(), original_points)


def test_bad_updates_duplicates_and_stale_generations_do_not_add_evidence():
    mapping = OnlineMap(ray_clearing=True, grid_update_interval_s=1.)
    mapping.ingest_odom(observation(0.))
    assert not mapping.ingest_odom(observation(1., valid=False))
    assert not mapping.ingest_odom(observation(0.))
    assert mapping.stats().grid_observations == 1
    with pytest.raises(ValueError, match="backwards"):
        mapping.ingest_odom(observation(-1.))
    assert mapping.stats().grid_observations == 1
    old = mapping.generation
    mapping.start_mapping()
    assert mapping.stats().grid_observations == 0
    assert not mapping.ingest_odom(observation(999.), generation=old)
    assert mapping.ingest_odom(observation(0.))


def test_slow_motion_observation_uses_relative_pose_and_save_version():
    mapping = OnlineMap(ray_clearing=True, grid_update_interval_s=1.)
    mapping.ingest_odom(observation(0.))
    snapshot = mapping.freeze_for_save()
    assert not mapping.ingest_odom(observation(1., x=.2))
    mapping.resume_after_save_error(snapshot)
    assert not mapping.ingest_odom(observation(1., x=.2))
    assert not mapping.is_save_snapshot_current(snapshot)
    matrix, rays = mapping.grid_observation(1)
    np.testing.assert_allclose(matrix[:3, 3], [.2, 0, 0])
    np.testing.assert_allclose(rays, observation(1.)["scan_body_rays"])
    assert mapping.keyframe_count() == 1


def test_zero_interval_keeps_keyframe_only_behavior():
    mapping = OnlineMap(ray_clearing=True, grid_update_interval_s=0.)
    mapping.ingest_odom(observation(0.))
    mapping.ingest_odom(observation(100.))
    assert mapping.stats().grid_observations == 1
