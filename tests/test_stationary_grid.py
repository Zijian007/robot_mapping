import numpy as np
import pytest

from robot_mapping.core.online_map import OnlineMap


def observation(stamp, distance=2.0, x=0.0, valid=True):
    return dict(
        status="scan_to_map",
        lidar_update_valid=valid,
        stamp=stamp,
        position=[x, 0, 0],
        orientation_xyzw=[0, 0, 0, 1],
        scan_body_points=np.array([[distance, 0, 0.5]]),
        scan_body_rays=np.array([[0, 0, 0.5, distance, 0, 0.5]]),
    )


def cell(mapping, x):
    row, col = mapping.grid._world_to_rc(np.array([x]), np.array([0.0]))
    return mapping.grid.values[row[0], col[0]]


def test_stationary_clear_uses_sensor_time_without_adding_graph_nodes_or_3d_points():
    mapping = OnlineMap(
        ray_clearing=True, grid_update_interval_s=1.0, min_height=0.4, max_height=0.6
    )
    assert mapping.ingest_odom(observation(100.0))
    original_points = mapping.voxels.points_array().copy()
    for stamp in np.arange(100.1, 101.0, 0.1):
        assert not mapping.ingest_odom(observation(stamp))
    assert mapping.stats().grid_observations == 1
    assert not mapping.ingest_odom(observation(101.0))
    assert cell(mapping, 2) == 100
    for stamp in range(102, 107):
        assert not mapping.ingest_odom(observation(stamp, 4.0))
    assert cell(mapping, 2) == 0
    assert cell(mapping, 4) == 100
    assert mapping.keyframe_count() == 1
    assert mapping.stats().grid_observations == 7
    np.testing.assert_array_equal(mapping.voxels.points_array(), original_points)


def test_bad_updates_duplicates_and_stale_generations_do_not_add_evidence():
    mapping = OnlineMap(ray_clearing=True, grid_update_interval_s=1.0)
    mapping.ingest_odom(observation(0.0))
    assert not mapping.ingest_odom(observation(1.0, valid=False))
    assert not mapping.ingest_odom(observation(0.0))
    assert mapping.stats().grid_observations == 1
    with pytest.raises(ValueError, match="backwards"):
        mapping.ingest_odom(observation(-1.0))
    assert mapping.stats().grid_observations == 1
    old = mapping.generation
    mapping.start_mapping()
    assert mapping.stats().grid_observations == 0
    assert not mapping.ingest_odom(observation(999.0), generation=old)
    assert mapping.ingest_odom(observation(0.0))


def test_slow_motion_observation_uses_relative_pose_and_save_version():
    mapping = OnlineMap(ray_clearing=True, grid_update_interval_s=1.0)
    mapping.ingest_odom(observation(0.0))
    snapshot = mapping.freeze_for_save()
    assert not mapping.ingest_odom(observation(1.0, x=0.2))
    mapping.resume_after_save_error(snapshot)
    assert not mapping.ingest_odom(observation(1.0, x=0.2))
    assert not mapping.is_save_snapshot_current(snapshot)
    matrix, rays = mapping.grid_observation(1)
    np.testing.assert_allclose(matrix[:3, 3], [0.2, 0, 0])
    np.testing.assert_allclose(rays, observation(1.0)["scan_body_rays"])
    assert mapping.keyframe_count() == 1


def test_zero_interval_keeps_keyframe_only_behavior():
    mapping = OnlineMap(ray_clearing=True, grid_update_interval_s=0.0)
    mapping.ingest_odom(observation(0.0))
    mapping.ingest_odom(observation(100.0))
    assert mapping.stats().grid_observations == 1


def test_more_grid_rays_preserve_independent_voxel_clearing_across_reset():
    params = dict(
        ray_clearing=True,
        voxel_clearing=True,
        grid_update_interval_s=1.0,
        min_height=0.4,
        max_height=0.6,
        free_radius_m=0,
    )
    baseline = OnlineMap(**params, max_rays=1)
    improved = OnlineMap(**params, max_rays=128, voxel_max_rays=1)
    for _ in range(2):
        for mapping in (baseline, improved):
            mapping.start_mapping()
        for stamp in range(7):
            obs = observation(float(stamp))
            y = np.linspace(-1, 1, 80)
            points = np.column_stack(
                (np.full_like(y, 2 if stamp < 2 else 4), y, np.full_like(y, 0.5))
            )
            obs["scan_body_points"] = points
            obs["scan_body_rays"] = np.column_stack((np.tile([0, 0, 0.5], (len(y), 1)), points))
            baseline.ingest_odom(obs)
            improved.ingest_odom(obs)
            assert improved.voxels._cells == baseline.voxels._cells
            assert improved.voxels._misses == baseline.voxels._misses
        assert improved.stats().free_cells > baseline.stats().free_cells
        assert improved.grid.max_rays == 128
        assert improved.voxels.max_rays == 1
