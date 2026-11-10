import numpy as np
import pytest

from robot_mapping.core.ray_voxels import RayVoxelHash


def ray(x, y=0.075, z=0.525, origin_x=0.075):
    return np.array([[origin_x, y, z, x, y, z]])


def has(grid, x, y=0.075, z=0.525):
    return grid._key(x, y, z) in grid._cells


def test_stationary_repeated_clear_removes_person_but_preserves_wall():
    grid = RayVoxelHash()
    grid.add_points([[2.025, 0.075, 0.525], [4.125, 0.075, 0.525]])
    for _ in range(2):
        grid.ingest_rays(np.repeat(ray(4.125), 100, axis=0))
        assert has(grid, 2.025)
    grid.ingest_rays(ray(4.125))
    assert not has(grid, 2.025) and has(grid, 4.125)
    assert grid.removed_voxels == 1
    grid.add_points([[2.025, 0.075, 0.525]])
    grid.ingest_rays(ray(2.025))
    assert has(grid, 2.025)  # new observation can restore a removed voxel
    assert not grid._misses


def test_return_and_neighbours_reset_evidence_even_if_ray_not_selected():
    grid = RayVoxelHash(max_rays=1)
    grid.add_points([[2.025, 0.075, 0.525]])
    for _ in range(2):
        grid.ingest_rays(ray(4.125))
    grid.ingest_rays(np.vstack((ray(4.125), ray(2.175))))
    assert not grid._misses
    for _ in range(2):
        grid.ingest_rays(ray(4.125))
    assert has(grid, 2.025)


@pytest.mark.parametrize("z", [0.075, 1.125])
def test_other_heights_do_not_remove_low_obstacle_or_tabletop(z):
    grid = RayVoxelHash()
    grid.add_points([[2.025, 0.075, z]])
    for _ in range(10):
        grid.ingest_rays(ray(4.125))
    assert has(grid, 2.025, z=z)


def test_occlusion_range_rear_origin_and_grazing():
    grid = RayVoxelHash(max_ray_range=3.0)
    grid.add_points([[x, 0.075, 0.525] for x in [-2.025, 0.075, 2.025, 4.125]])
    for _ in range(5):
        grid.ingest_rays(ray(-10.0, origin_x=-0.525))
    assert not has(grid, -2.025)
    assert has(grid, 0.075) and has(grid, 2.025) and has(grid, 4.125)
    for _ in range(5):
        grid.ingest_rays(ray(1.125))  # foreground shields 2 m point
        grid.ingest_rays(ray(8.0, y=0.001))  # passes near voxel boundary only
    assert has(grid, 2.025)
    for _ in range(5):
        grid.ingest_rays(ray(8.0))
    assert not has(grid, 2.025) and has(grid, 4.125)  # range limited


def test_empty_invalid_and_clear_reset():
    grid = RayVoxelHash()
    grid.add_points([[2.025, 0.075, 0.525]])
    for _ in range(10):
        grid.ingest_rays(np.empty((0, 6)))
        grid.ingest_rays(np.zeros((1, 6)))
    assert has(grid, 2.025) and not grid._misses
    with pytest.raises(ValueError):
        grid.ingest_rays([[0, 0, 0, np.nan, 0, 0]])
    grid.ingest_rays(ray(4.125))
    grid.clear()
    assert not grid._cells and not grid._misses and grid.removed_voxels == 0


def test_only_accepted_sensor_observations_count_towards_3d_clearing():
    from robot_mapping.core.online_map import OnlineMap

    mapping = OnlineMap(ray_clearing=True, voxel_clearing=True, grid_update_interval_s=1.0)

    def observation(stamp, x=4.125, valid=True):
        return dict(
            status="scan_to_map",
            lidar_update_valid=valid,
            stamp=stamp,
            position=[0, 0, 0],
            orientation_xyzw=[0, 0, 0, 1],
            scan_body_points=np.array([[x, 0.075, 0.525]]),
            scan_body_rays=ray(x),
        )

    mapping.ingest_odom(observation(0.0, 2.025))
    mapping.ingest_odom(observation(1.0))
    for _ in range(10):
        mapping.ingest_odom(observation(1.0))
        mapping.ingest_odom(observation(2.0, valid=False))
    assert has(mapping.voxels, 2.025)
    mapping.ingest_odom(observation(3.0))
    assert has(mapping.voxels, 2.025)
    mapping.ingest_odom(observation(4.0))
    assert not has(mapping.voxels, 2.025)
    assert mapping.keyframe_count() == 1 and mapping.stats().grid_observations == 4
    assert mapping.freeze_for_save().points.shape == (0, 3)


def test_configuration_rejects_single_observation_clearing_and_missing_rays():
    from robot_mapping.config import settings_from_profile, validate_settings

    for profile in ({"online": {"voxel_clear_observations": 1}}, {"grid": {"ray_clearing": False}}):
        with pytest.raises(ValueError):
            validate_settings(settings_from_profile(profile, "unused.json"))
