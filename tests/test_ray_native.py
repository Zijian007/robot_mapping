"""Native and NumPy evidence must agree after every observation, including resets."""
import numpy as np
import pytest
from robot_mapping.backends.ray_native import kernels
from robot_mapping.core import ray_grid, ray_voxels

pytestmark = pytest.mark.skipif(kernels is None, reason='optional ray extension not built')

@pytest.mark.parametrize('resolution', [.05, .15, .2])
def test_native_matches_reference_after_each_observation(monkeypatch, resolution):
    rng = np.random.default_rng(812)
    grids = [ray_grid.RayOccupancy2D(resolution=.1, min_height=-.5, max_height=1.5,
             height_resolution=.2, max_rays=150, max_ray_range=6.) for _ in range(2)]
    voxels = [ray_voxels.RayVoxelHash(resolution, max_rays=150, max_ray_range=6.) for _ in range(2)]
    for step in range(15):
        origin = rng.uniform(-.5, .5, (500, 3))
        endpoint = rng.uniform(-4, 4, (500, 3))
        # Negative cells, vertical/zero rays, exact height limits, duplicates.
        endpoint[:10] = origin[:10]
        endpoint[10:20, :2] = origin[10:20, :2]
        endpoint[20:30, 2] = -.5
        endpoint[30:40, 2] = 1.5
        rays = np.column_stack((origin, endpoint))
        rays = np.vstack((rays, rays[40:80]))
        points = origin + (endpoint-origin)*.5
        for native, grid, voxel in zip((None, kernels), grids, voxels):
            monkeypatch.setattr(ray_grid, 'kernels', native)
            monkeypatch.setattr(ray_voxels, 'kernels', native)
            if step in (0, 5, 10):
                voxel.add_points(points)
            grid.ingest_rays(rays, robot_xy=(step*.1, -.3))
            voxel.ingest_rays(rays)
        np.testing.assert_array_equal(grids[0]._scores, grids[1]._scores)
        np.testing.assert_array_equal(grids[0]._values, grids[1]._values)
        assert voxels[0]._cells == voxels[1]._cells
        assert voxels[0]._misses == voxels[1]._misses
        assert voxels[0].removed_voxels == voxels[1].removed_voxels


def test_native_protection_resets_and_deletes_like_reference(monkeypatch):
    pair = [ray_voxels.RayVoxelHash(max_rays=1) for _ in range(2)]
    for v in pair: v.add_points([[2.025, .075, .525], [-2.025, .075, .525]])
    def ray(x): return [.075, .075, .525, x, .075, .525]
    sequence = [[ray(4.125)]]*2 + [[ray(4.125), ray(2.175)]] + [[ray(4.125)]]*3
    sequence += [[ray(-4.125)]]*3
    for rays in sequence:
        for native, v in zip((None, kernels), pair):
            monkeypatch.setattr(ray_voxels, 'kernels', native)
            v.ingest_rays(rays)
        assert pair[0]._cells == pair[1]._cells
        assert pair[0]._misses == pair[1]._misses
        assert pair[0].removed_voxels == pair[1].removed_voxels
    assert pair[0].removed_voxels == 2


def test_native_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        kernels.grid_free_keys(np.zeros((1, 3), dtype=np.int64), np.zeros((1, 2)), 0., 1., .2, 5, 10, 10)
    with pytest.raises(ValueError):
        kernels.voxel_evidence(np.zeros((1, 6)), np.zeros((1, 6)), np.zeros(2), kernels.VoxelIndex(), {}, .15, 30.)


def test_selection_keeps_stable_ties_and_group_order():
    rng = np.random.default_rng(42)
    groups = rng.integers(-3, 3, (4000, 5))
    distances = rng.integers(0, 5, 4000).astype(float)
    order = np.lexsort((distances, *groups[:, ::-1].T))
    expected = order[np.r_[True, (groups[order][1:] != groups[order][:-1]).any(axis=1)]]
    np.testing.assert_array_equal(kernels.select_grouped(groups, distances), expected)


def test_native_index_tracks_float32_insertion_deletion_and_clear(monkeypatch):
    pair = [ray_voxels.RayVoxelHash() for _ in range(2)]
    for cycle in range(2):
        for v in pair:
            v.clear()
            v.add_points(np.array([[2.025, .075, .525], [.45, .075, .525]], dtype=np.float32))
        for step in range(4):
            for native, v in zip((None, kernels), pair):
                monkeypatch.setattr(ray_voxels, 'kernels', native)
                v.ingest_rays([[.075, .075, .525, 4.125, .075, .525]])
            assert pair[0]._cells == pair[1]._cells
            assert pair[0]._misses == pair[1]._misses
        assert not pair[1]._cells
