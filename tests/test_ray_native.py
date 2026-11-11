"""Native and NumPy evidence must agree after every observation, including resets."""

import numpy as np
import pytest

from robot_mapping.backends.ray_native import kernels
from robot_mapping.core import ray_grid, ray_voxels, spatial

pytestmark = pytest.mark.skipif(kernels is None, reason="optional ray extension not built")


def voxel_pair(monkeypatch, factory=ray_voxels.RayVoxelHash, **kwargs):
    # Storage ownership is fixed at construction. Exercise a real Python oracle,
    # rather than toggling a module global on an already-native instance.
    result = []
    for backend in (None, kernels):
        with monkeypatch.context() as context:
            context.setattr(spatial, "kernels", backend)
            result.append(factory(**kwargs))
    return result


@pytest.mark.parametrize("resolution", [0.05, 0.15, 0.2])
def test_native_matches_reference_after_each_observation(monkeypatch, resolution):
    rng = np.random.default_rng(812)
    grids = [
        ray_grid.RayOccupancy2D(
            resolution=0.1,
            min_height=-0.5,
            max_height=1.5,
            height_resolution=0.2,
            max_rays=150,
            max_ray_range=6.0,
        )
        for _ in range(2)
    ]
    voxels = voxel_pair(monkeypatch, resolution=resolution, max_rays=150, max_ray_range=6.0)
    for step in range(15):
        origin = rng.uniform(-0.5, 0.5, (500, 3))
        endpoint = rng.uniform(-4, 4, (500, 3))
        # Negative cells, vertical/zero rays, exact height limits, duplicates.
        endpoint[:10] = origin[:10]
        endpoint[10:20, :2] = origin[10:20, :2]
        endpoint[20:30, 2] = -0.5
        endpoint[30:40, 2] = 1.5
        rays = np.column_stack((origin, endpoint))
        rays = np.vstack((rays, rays[40:80]))
        points = origin + (endpoint - origin) * 0.5
        for native, grid, voxel in zip((None, kernels), grids, voxels):
            monkeypatch.setattr(ray_grid, "kernels", native)
            monkeypatch.setattr(ray_voxels, "kernels", native)
            if step in (0, 5, 10):
                voxel.add_points(points)
            grid.ingest_rays(rays, robot_xy=(step * 0.1, -0.3))
            voxel.ingest_rays(rays)
        np.testing.assert_array_equal(grids[0]._scores, grids[1]._scores)
        np.testing.assert_array_equal(grids[0]._values, grids[1]._values)
        assert voxels[0]._cells == voxels[1]._cells
        assert voxels[0]._misses == voxels[1]._misses
        assert voxels[0].removed_voxels == voxels[1].removed_voxels


def test_native_protection_resets_and_deletes_like_reference(monkeypatch):
    pair = voxel_pair(monkeypatch, max_rays=1)
    for v in pair:
        v.add_points([[2.025, 0.075, 0.525], [-2.025, 0.075, 0.525]])

    def ray(x):
        return [0.075, 0.075, 0.525, x, 0.075, 0.525]

    sequence = [[ray(4.125)]] * 2 + [[ray(4.125), ray(2.175)]] + [[ray(4.125)]] * 3
    sequence += [[ray(-4.125)]] * 3
    for rays in sequence:
        for native, v in zip((None, kernels), pair):
            monkeypatch.setattr(ray_voxels, "kernels", native)
            v.ingest_rays(rays)
        assert pair[0]._cells == pair[1]._cells
        assert pair[0]._misses == pair[1]._misses
        assert pair[0].removed_voxels == pair[1].removed_voxels
    assert pair[0].removed_voxels == 2


def test_native_rejects_invalid_shapes():
    with pytest.raises(ValueError):
        kernels.grid_free_keys(
            np.zeros((1, 3), dtype=np.int64), np.zeros((1, 2)), 0.0, 1.0, 0.2, 5, 10, 10
        )
    with pytest.raises(ValueError):
        kernels.voxel_evidence(
            np.zeros((1, 6)), np.zeros((1, 6)), np.zeros(2), kernels.VoxelIndex(), {}, 0.15, 30.0
        )


def test_selection_keeps_stable_ties_and_group_order():
    rng = np.random.default_rng(42)
    groups = rng.integers(-3, 3, (4000, 5))
    distances = rng.integers(0, 5, 4000).astype(float)
    order = np.lexsort((distances, *groups[:, ::-1].T))
    expected = order[np.r_[True, (groups[order][1:] != groups[order][:-1]).any(axis=1)]]
    np.testing.assert_array_equal(kernels.select_grouped(groups, distances), expected)


def test_native_index_tracks_float32_insertion_deletion_and_clear(monkeypatch):
    pair = voxel_pair(monkeypatch)
    for cycle in range(2):
        for v in pair:
            v.clear()
            v.add_points(np.array([[2.025, 0.075, 0.525], [0.45, 0.075, 0.525]], dtype=np.float32))
        for step in range(4):
            for native, v in zip((None, kernels), pair):
                monkeypatch.setattr(ray_voxels, "kernels", native)
                v.ingest_rays([[0.075, 0.075, 0.525, 4.125, 0.075, 0.525]])
            assert pair[0]._cells == pair[1]._cells
            assert pair[0]._misses == pair[1]._misses
        assert not pair[1]._cells


@pytest.mark.parametrize("resolution", [0.05, 0.15, 0.2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_storage_preserves_insertion_order_centers_and_snapshot(
    monkeypatch, resolution, dtype
):
    pair = voxel_pair(monkeypatch, factory=spatial.WorldVoxelHash, resolution=resolution)
    rng = np.random.default_rng(71)
    # Cell boundaries, negative coordinates, duplicate cells and strided XYZ+fields.
    raw = rng.uniform(-4, 4, (600, 8)).astype(dtype)
    raw[:8, :3] = np.arange(-4, 4)[:, None] * resolution
    points = np.vstack((raw, raw[:100]))[:, ::2]
    for _ in range(2):
        added = [v.add_points(points) for v in pair]
        np.testing.assert_array_equal(*added)
        assert added[0].dtype == np.float32
        assert pair[0]._cells == pair[1]._cells
        arrays = [v.points_array() for v in pair]
        for arr in arrays:
            assert arr.dtype == np.float32 and arr.shape == (len(pair[0]), 3)

        def ordered(arr):
            return arr[np.lexsort(arr.T)]

        np.testing.assert_array_equal(ordered(arrays[0]), ordered(arrays[1]))
    # Returned buffers and private diagnostic snapshots cannot mutate the store.
    expected = len(pair[1])
    pair[1]._cells.clear()
    pair[1].points_array()[:] = 0
    assert len(pair[1]) == expected
    for v in pair:
        v.clear()
        assert len(v) == 0 and v.points_array().shape == (0, 3)


def test_float32_positive_observation_resets_the_stored_cell(monkeypatch):
    pair = voxel_pair(monkeypatch)
    point = np.array([[0.45, 0.075, 0.525]], dtype=np.float32)
    for v in pair:
        v.add_points(point)
        v.ingest_rays([[0.075, 0.075, 0.525, 4.125, 0.075, 0.525]])
        assert v._misses
        v.add_points(point)
        assert not v._misses
    assert pair[0]._cells == pair[1]._cells


def test_native_store_rejects_invalid_batch_without_partial_insertion():
    store = kernels.VoxelStore(0.15)
    for bad in (np.nan, np.inf, 1e20):
        with pytest.raises(ValueError):
            store.add_points([[1.0, 2.0, 3.0], [bad, 0.0, 0.0]])
        assert store.size == 0
    with pytest.raises(ValueError):
        store.ingest_rays(np.empty((0, 6)), np.empty((0, 6)), np.empty(0), 30.0, 1)
    with pytest.raises(ValueError):
        store.ingest_rays(np.zeros((1, 6)), np.zeros((1, 6)), np.zeros(2), 30.0, 3)


def test_native_store_serializes_concurrent_insertion_and_snapshots():
    from concurrent.futures import ThreadPoolExecutor

    store = kernels.VoxelStore(0.15)
    batches = [
        np.column_stack(
            (
                np.arange(i * 1000, (i + 1) * 1000) * 0.15 + 0.075,
                np.full(1000, 0.075),
                np.full(1000, 0.075),
            )
        )
        for i in range(4)
    ]

    def insert(batch):
        for _ in range(4):
            store.add_points(batch)
            assert store.points_array().shape[1] == 3

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(insert, batches))
    assert store.size == 4000
    assert not store.misses()


@pytest.mark.parametrize("max_rays", [1, 2, 19, 64])
@pytest.mark.parametrize("free_space_mode", ["height_complete", "projected"])
@pytest.mark.parametrize(
    "projection_params", [{}, {"projected_min_observations": 4, "projected_max_range": 1.25}]
)
def test_fused_grid_matches_reference_with_limits_growth_and_footprints(
    monkeypatch, max_rays, free_space_mode, projection_params
):
    rng = np.random.default_rng(206)
    pair = [
        ray_grid.RayOccupancy2D(
            resolution=0.15,
            min_height=-0.2,
            max_height=0.45,
            height_resolution=0.2,
            max_rays=max_rays,
            max_ray_range=3.0,
            free_space_mode=free_space_mode,
            **projection_params,
        )
        for _ in range(2)
    ]
    for step in range(25):
        # Expand in every direction; include round-to-even origins, exact height
        # bounds, partial top bins, duplicate groups and out-of-range returns.
        origin = rng.uniform(-0.5, 0.5, (200, 3)) + [np.sin(step) * 2, np.cos(step) * 2, 0]
        endpoint = origin + rng.uniform(-5, 5, (200, 3))
        origin[:8] = [0.125, -0.125, 0.25]
        endpoint[:8] = [[2.0, 0.0, z] for z in [-0.2, 0.0, 0.2, 0.45] * 2]
        endpoint[8:16, :2] = origin[8:16, :2]  # vertical rays
        endpoint[16:24] = origin[16:24]  # zero rays
        rays = np.vstack(
            (np.column_stack((origin, endpoint)), np.column_stack((origin, endpoint))[:30])
        )
        # Exercise non-contiguous float32 and float64 caller buffers.
        padded = np.zeros((len(rays), 12), dtype=np.float32 if step % 2 else np.float64)
        padded[:, ::2] = rays
        rays = padded[:, ::2]
        xy = None if step % 3 == 0 else (float(np.sin(step) * 2), float(np.cos(step) * 2))
        for backend, grid in zip((None, kernels), pair):
            with monkeypatch.context() as context:
                context.setattr(ray_grid, "kernels", backend)
                if step == 13:
                    grid.clear()
                grid.ingest_rays(rays, robot_xy=xy)
                # Empty and zero observations must not refresh the footprint.
                grid.ingest_rays(np.empty((0, 6)), robot_xy=(100.0, 100.0))
                grid.ingest_rays(np.zeros((1, 6)), robot_xy=(100.0, 100.0))
        assert pair[0].spec == pair[1].spec
        assert pair[0]._last_footprint_xy == pair[1]._last_footprint_xy
        np.testing.assert_array_equal(pair[0]._scores, pair[1]._scores)
        np.testing.assert_array_equal(pair[0].values, pair[1].values)
        if free_space_mode == "projected":
            np.testing.assert_array_equal(pair[0]._projected_counts, pair[1]._projected_counts)


def test_fused_grid_validates_in_place_buffers_before_mutation():
    grid = ray_grid.RayOccupancy2D(min_height=0.4, max_height=0.6)
    grid._ensure_seed(0.0, 0.0)
    rays = np.array([[0.0, 0.0, 0.5, 2.0, 0.0, 0.5]])
    distances = np.array([2.0])
    groups = grid._ray_groups(rays)
    scores, values = grid._scores, grid.values

    def apply(
        scores=scores,
        values=values,
        rays=rays,
        distances=distances,
        groups=groups,
        projected_counts=None,
    ):
        kernels.grid_update(
            rays,
            distances,
            groups,
            scores,
            values,
            grid._origin_x,
            grid._origin_y,
            grid.resolution,
            grid.min_height,
            grid.max_height,
            grid.height_resolution,
            grid.max_ray_range,
            grid.max_rays,
            grid.free_radius_m,
            None,
            None,
            projected_counts,
        )

    before = scores.copy(), values.copy()
    invalid_rays = rays.copy()
    invalid_rays[0, 5] = np.nan
    with pytest.raises(ValueError):
        apply(rays=invalid_rays)
    with pytest.raises(ValueError):
        apply(distances=np.array([0.0]))
    with pytest.raises(ValueError):
        apply(groups=np.zeros((1, 4), dtype=np.int64))
    with pytest.raises((TypeError, ValueError)):
        apply(scores=scores.astype(np.float64))
    with pytest.raises((TypeError, ValueError)):
        apply(values=values[:, ::-1])
    with pytest.raises((TypeError, ValueError)):
        apply(projected_counts=np.zeros(values.shape, dtype=np.float64))
    with pytest.raises(ValueError):
        apply(projected_counts=np.zeros((1, 1), dtype=np.int8))
    readonly = np.zeros(values.shape, dtype=np.int8)
    readonly.flags.writeable = False
    with pytest.raises(ValueError):
        apply(projected_counts=readonly)
    scores.flags.writeable = False
    with pytest.raises(ValueError):
        apply()
    scores.flags.writeable = True
    values.flags.writeable = False
    with pytest.raises(ValueError):
        apply()
    values.flags.writeable = True
    np.testing.assert_array_equal(scores, before[0])
    np.testing.assert_array_equal(values, before[1])
