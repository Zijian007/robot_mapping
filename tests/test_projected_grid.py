"""Planar free-space inference preserves height-aware obstacle evidence."""

import numpy as np
import pytest

from robot_mapping.core import ray_grid


def ray(x, z=0.5):
    return np.array([[0.0, 0.0, z, x, 0.0, z]])


def cell(grid, x):
    r, c = grid._world_to_rc(x, 0.0)
    return grid.values[r, c]


@pytest.fixture(params=["native", "python"])
def grid(request, monkeypatch):
    if request.param == "python":
        monkeypatch.setattr(ray_grid, "kernels", None)
    else:
        assert ray_grid.kernels is not None
    return ray_grid.RayOccupancy2D(
        min_height=0.0, max_height=2.0, free_space_mode="projected", free_radius_m=0
    )


def test_two_distinct_observations_needed_and_no_free_behind_endpoint(grid):
    grid.ingest_rays(np.repeat(ray(4), 1000, axis=0))
    assert cell(grid, 2) == -1
    grid.ingest_rays(ray(4))
    assert cell(grid, 2) == 0
    assert cell(grid, 4) == 100
    assert cell(grid, 4.1) == -1
    grid.clear()
    grid.ingest_rays(ray(4))
    assert cell(grid, 2) == -1


@pytest.mark.parametrize("height", [0.1, 1.1])
def test_low_obstacle_or_overhang_blocks_planar_beam_at_other_height(grid, height):
    for _ in range(2):
        grid.ingest_rays(ray(2, height))
    for _ in range(12):
        grid.ingest_rays(ray(4, 1.7))
    assert cell(grid, 1) == 0
    assert cell(grid, 2) == 100
    assert cell(grid, 3) == -1  # projected shadow, despite rays passing above/below


def test_current_hit_vetoes_old_free_even_before_hit_score_becomes_positive(grid):
    for _ in range(8):
        grid.ingest_rays(ray(4))
    assert cell(grid, 2) == 0
    grid.ingest_rays(np.vstack((ray(2), ray(4))), robot_xy=(2.0, 0.0))
    assert cell(grid, 2) == -1  # also takes precedence over footprint prior
    grid.ingest_rays(ray(4))
    assert cell(grid, 2) == -1
    grid.ingest_rays(ray(4))
    assert cell(grid, 2) == 0


def test_same_height_observations_can_clear_old_obstacle(grid):
    for _ in range(2):
        grid.ingest_rays(ray(2))
    for _ in range(6):
        grid.ingest_rays(ray(4))
    assert cell(grid, 2) == 0
    assert cell(grid, 4) == 100


def test_out_of_band_returns_do_not_infer_planar_free_space(grid):
    for _ in range(10):
        grid.ingest_rays(np.array([[0, 0, 0.5, 4, 0, -0.5], [0, 0, 0.5, 4, 0, 3.0]]))
    assert cell(grid, 1) == -1
    before = grid.values.copy()
    grid.ingest_rays(np.empty((0, 6)))
    np.testing.assert_array_equal(grid.values, before)


def test_projection_never_changes_height_evidence_or_occupied_mask(grid):
    strict = ray_grid.RayOccupancy2D(min_height=0.0, max_height=2.0, free_radius_m=0)
    for rays in [ray(2, 0.1)] * 2 + [ray(4, 1.7)] * 5 + [ray(4, 0.1)] * 6:
        grid.ingest_rays(rays)
        strict.ingest_rays(rays)
        np.testing.assert_array_equal(grid._scores, strict._scores)
        np.testing.assert_array_equal(grid.values == 100, strict.values == 100)


def test_projection_range_limit(grid):
    grid.max_ray_range = 3.0
    for _ in range(2):
        grid.ingest_rays(ray(10))
    assert cell(grid, 2) == 0
    assert cell(grid, 3) == -1


def test_tunable_observations_and_range_preserve_distant_hits(grid):
    grid.projected_min_observations = 4
    grid.projected_max_range = 5.0
    strict = ray_grid.RayOccupancy2D(min_height=0.0, max_height=2.0, free_radius_m=0)
    batch = np.repeat(ray(10), 100, axis=0)
    for i in range(4):
        grid.ingest_rays(batch)
        strict.ingest_rays(batch)
        assert cell(grid, 4) == (0 if i == 3 else -1)
        assert cell(grid, 5) == -1
        assert cell(grid, 8) == -1
        assert cell(grid, 10) == (100 if i >= 1 else -1)
        np.testing.assert_array_equal(grid._scores, strict._scores)


def test_projected_counter_saturates_without_signed_byte_overflow(grid):
    grid.projected_min_observations = 127
    for i in range(130):
        grid.ingest_rays(ray(0.5))
        assert cell(grid, 0.2) == (0 if i >= 126 else -1)
    assert grid._projected_counts.max() == 127


@pytest.mark.parametrize(
    "params",
    [
        {"projected_min_observations": 0},
        {"projected_min_observations": 128},
        {"projected_min_observations": 2.5},
        {"projected_min_observations": True},
        {"projected_max_range": 0},
        {"projected_max_range": float("nan")},
        {"projected_max_range": float("inf")},
    ],
)
def test_invalid_projection_tuning_rejected(params):
    with pytest.raises(ValueError):
        ray_grid.RayOccupancy2D(**params)
