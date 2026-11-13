"""Polar free space: clear each sensor sector up to its nearest in-band return only."""

import numpy as np
import pytest

from robot_mapping.core import ray_grid


def beam(x, y, z=0.5, origin=(0.0, 0.0, 0.5)):
    return np.array([[*origin, x, y, z]])


def value(grid, x, y=0.0):
    r, c = grid._world_to_rc(x, y)
    return grid.values[r, c]


def fan(distance, z=0.5, count=60, half_angle=0.05):
    """Several returns inside one 2-degree sector at the same range."""
    angles = np.linspace(-half_angle, half_angle, count) * np.pi / 180
    return np.column_stack(
        (
            np.zeros((count, 2)),
            np.full(count, 0.5),
            distance * np.cos(angles),
            distance * np.sin(angles),
            np.full(count, z),
        )
    )


@pytest.fixture(params=["native", "python"])
def make(request, monkeypatch):
    if request.param == "python":
        monkeypatch.setattr(ray_grid, "kernels", None)
    else:
        assert ray_grid.kernels is not None

    def build(**kwargs):
        params = dict(
            min_height=0.0,
            max_height=2.0,
            free_space_mode="polar",
            projected_min_observations=1,
            free_radius_m=0,
        )
        params.update(kwargs)
        return ray_grid.RayOccupancy2D(**params)

    return build


def test_single_observation_clears_up_to_the_return(make):
    grid = make()
    grid.ingest_rays(beam(4, 0))
    assert value(grid, 1) == 0
    assert value(grid, 3.8) == 0
    assert value(grid, 4.1) == -1  # nothing behind the return


def test_nearest_return_in_sector_occludes_farther_ones(make):
    grid = make(polar_bins=36)  # 10-degree sectors; bearings 0 and 6.8 degrees share one
    # A low obstacle at 2 m and a wall at 5 m in the same sector: the beam to the
    # wall passes above the low obstacle but must not clear behind it.
    rays = np.vstack((beam(2.0, 0.0, z=0.1), beam(5.0, 0.6, z=1.0)))
    grid.ingest_rays(rays)
    assert value(grid, 1.0) == 0
    assert value(grid, 1.5, 0.18) == 0  # the wall beam clears its own bearing up to 2 m
    assert value(grid, 3.0) == -1
    assert value(grid, 3.0, 0.36) == -1
    assert value(grid, 4.5, 0.54) == -1


def test_returns_within_margin_of_nearest_also_clear(make):
    # 10-degree sectors; both returns at bearings 1 and 6.5 degrees share one.
    grid = make(polar_bins=36, polar_closest_margin=0.3)
    rays = np.vstack((beam(2.0, 0.035), beam(2.2, 0.25)))
    grid.ingest_rays(rays)
    assert value(grid, 2.15, 0.27) == 0  # second return is 0.2 m behind the nearest, kept

    # With no margin the farther return is cut at the nearest range (2 m): it still
    # clears in front of that range along its own bearing, nothing beyond it.
    strict = make(polar_bins=36, polar_closest_margin=0.0)
    strict.ingest_rays(np.vstack((beam(2.0, 0.035), beam(3.0, 0.34))))
    assert value(strict, 1.5, 0.17) == 0
    assert value(strict, 2.6, 0.295) == -1
    assert value(strict, 2.2, 0.25) == -1


def test_sectors_are_independent(make):
    grid = make()
    rays = np.vstack((beam(1.0, 0.0), beam(0.0, 4.0)))  # different sectors
    grid.ingest_rays(rays)
    assert value(grid, 0.0, 3.5) == 0
    assert value(grid, 0.5, 0.0) == 0


def test_out_of_band_returns_do_not_clear_or_occlude(make):
    grid = make(min_height=0.2)
    grid.ingest_rays(np.vstack((beam(1.0, 0.0, z=0.05), beam(4.0, 0.0, z=0.5))))
    assert value(grid, 3.0) == 0  # floor return at 1 m neither occludes nor clears
    floor_only = make(min_height=0.2)
    floor_only.ingest_rays(beam(3.0, 0.0, z=0.05))
    assert value(floor_only, 2.0) == -1


def test_known_obstacle_at_any_height_stops_planar_clearing(make):
    grid = make()
    for _ in range(2):
        grid.ingest_rays(beam(2.0, 0.0, z=1.5))  # overhang at 2 m
    assert value(grid, 2.0) == 100
    for _ in range(3):
        grid.ingest_rays(beam(2.0, 1.0, z=0.5))  # other sector, keeps scores
    # A beam at another height in a new observation must not clear past the overhang.
    grid.ingest_rays(beam(4.0, 0.0, z=0.5))
    assert value(grid, 1.0) == 0
    assert value(grid, 3.0) == -1


def test_current_hit_vetoes_old_free(make):
    grid = make()
    grid.ingest_rays(beam(4.0, 0.0))
    assert value(grid, 2.0) == 0
    grid.ingest_rays(np.vstack((beam(2.0, 0.0), beam(4.0, 0.2))))
    assert value(grid, 2.0) == -1  # one hit: not yet occupied, no longer free


def test_min_observations_and_range_limit(make):
    grid = make(projected_min_observations=2, projected_max_range=3.0)
    grid.ingest_rays(beam(5.0, 0.0))
    assert value(grid, 2.0) == -1
    grid.ingest_rays(beam(5.0, 0.0))
    assert value(grid, 2.0) == 0
    assert value(grid, 3.5) == -1


def test_never_changes_height_evidence_or_occupied_mask(make):
    grid = make()
    strict = ray_grid.RayOccupancy2D(min_height=0.0, max_height=2.0, free_radius_m=0)
    for rays in [beam(2, 0, 0.1)] * 2 + [fan(4, 1.7)] * 5 + [fan(4, 0.1)] * 6:
        grid.ingest_rays(rays)
        strict.ingest_rays(rays)
        np.testing.assert_array_equal(grid._scores, strict._scores)
        np.testing.assert_array_equal(grid.values == 100, strict.values == 100)
        # Height-complete free space stays free in polar mode.
        assert not ((strict.values == 0) & (grid.values != 0)).any()


def test_dense_scan_fills_more_than_height_complete(make):
    angles = np.linspace(-np.pi, np.pi, 720, endpoint=False)
    walls = 4.0 / np.maximum(np.abs(np.cos(angles)), np.abs(np.sin(angles)))
    rays = np.column_stack(
        (
            np.zeros((720, 2)),
            np.full(720, 0.5),
            walls * np.cos(angles),
            walls * np.sin(angles),
            np.full(720, 0.6),
        )
    )
    polar = make()
    strict = ray_grid.RayOccupancy2D(min_height=0.0, max_height=2.0, free_radius_m=0)
    polar.ingest_rays(rays)
    strict.ingest_rays(rays)
    assert (polar.values == 0).sum() > 10 * max(1, (strict.values == 0).sum())
    inside = polar.values[polar._world_to_rc(0.0, 0.0)[0], :]
    assert (inside == 0).sum() > 100


def test_rebuild_after_clear_is_deterministic(make):
    grid = make(projected_min_observations=2)
    sequence = [fan(3.0), beam(2.0, 1.0), fan(3.0, 0.2), beam(4.0, -1.0)]
    for rays in sequence:
        grid.ingest_rays(rays)
    values, counts = grid.values.copy(), grid._projected_counts.copy()
    grid.clear()
    for rays in sequence:
        grid.ingest_rays(rays)
    np.testing.assert_array_equal(grid.values, values)
    np.testing.assert_array_equal(grid._projected_counts, counts)


@pytest.mark.parametrize(
    "params",
    [
        {"polar_bins": 4},
        {"polar_bins": 180.0},
        {"polar_bins": True},
        {"polar_closest_margin": -0.1},
        {"polar_closest_margin": float("nan")},
    ],
)
def test_invalid_polar_tuning_rejected(params):
    with pytest.raises(ValueError):
        ray_grid.RayOccupancy2D(free_space_mode="polar", **params)
