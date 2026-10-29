"""Projection regressions migrated from the original mapping tool."""
import numpy as np
import pytest
from robot_mapping.core.online_map import OnlineMap, SE3Pose
from robot_mapping.core.pose_math import gravity_aligned_map_from_lio, transform_points


def pose(x=0):
    return SE3Pose((x, 0, 0), (0, 0, 0, 1))


def value_at(grid, x, y):
    spec = grid.spec
    return grid.values[int(np.floor((y-spec.origin_y)/spec.resolution)),
                       int(np.floor((x-spec.origin_x)/spec.resolution))]


def test_grid_preserves_subvoxel_xy_and_original_height():
    om = OnlineMap(voxel_resolution=.15, grid_resolution=.05, min_height=-.25)
    points = np.array([[1.06, 1.01, .4], [1.11, 1.01, .4], [1.16, 1.01, .4],
                       [2.01, 1.01, -.26]])
    assert om.ingest_local_points(pose(), points)
    assert om.grid.occupied_count() == 3
    for x in [1.06, 1.11, 1.16]:
        assert value_at(om.grid, x, 1.01) == 100
    # The low point's 3D voxel center is -0.225, but its true z is below the cut.
    assert len(om.voxels) == 2


def test_revisit_updates_2d_even_when_3d_voxel_already_exists():
    om = OnlineMap(voxel_resolution=.15, grid_resolution=.05, keyframe_dist_th=.1)
    assert om.ingest_points_direct(pose(), np.array([[1.06, 1.01, .4]]))
    assert om.ingest_points_direct(pose(.2), np.array([[1.11, 1.01, .4]]))
    assert len(om.voxels) == 1
    assert om.grid.occupied_count() == 2


def test_gravity_alignment_excludes_floor_without_raising_height_cut():
    theta = np.deg2rad(10)
    rotation = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])
    up = rotation.T @ np.array([0., 0., 9.81])
    alignment = gravity_aligned_map_from_lio(up, rotation)
    np.testing.assert_allclose(alignment[:3, :3], rotation, atol=1e-12)
    floor = np.array([[x, 1, -.32] for x in np.linspace(1, 5, 40)])
    obstacle = np.array([[2., 2., .3]])
    local = np.concatenate([floor, obstacle]) @ rotation
    om = OnlineMap(min_height=-.25)
    assert om.set_map_alignment(alignment, generation=om.generation)
    assert om.ingest_local_points(pose(), local)
    assert om.grid.occupied_count() == 1
    assert value_at(om.grid, 2, 2) == 100
    np.testing.assert_allclose(om.keyframe_local_points(0), local)
    np.testing.assert_allclose(om.keyframe_opt_matrix(0), alignment)
    np.testing.assert_allclose(om.freeze_for_save().map_from_lio, alignment)


def test_alignment_does_not_change_mid_map_or_cross_reset():
    om = OnlineMap()
    stale = om.generation
    om.start_mapping()
    alignment = gravity_aligned_map_from_lio([0, 1, 9.8], np.eye(3))
    assert not om.set_map_alignment(alignment, generation=stale)
    assert om.set_map_alignment(alignment, generation=om.generation)
    assert om.ingest_local_points(pose(), np.array([[2, 2, 1]]))
    assert not om.set_map_alignment(np.eye(4))
    om.start_mapping()
    assert om.ingest_local_points(pose(), np.array([[2, 2, 1]]))
    np.testing.assert_array_equal(om.keyframe_opt_matrix(0), np.eye(4))


@pytest.mark.parametrize("acc", [[0, 0, 0], [np.nan, 0, 1]])
def test_invalid_initial_acceleration_rejects_alignment(acc):
    with pytest.raises(ValueError):
        gravity_aligned_map_from_lio(acc, np.eye(3))
