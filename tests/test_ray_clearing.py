"""Clearing evidence: elevation-binned ray selection and the end margin before returns."""

import numpy as np
import pytest

from robot_mapping.core import ray_grid

ORIGIN = (0.0, 0.0, 0.33)


def ray(x, z, y=0.0):
    return np.array([[*ORIGIN, x, y, z]])


def layer_score(grid, x, z, y=0.0):
    r, c = grid._world_to_rc(x, y)
    layer = int(np.floor((z - grid.min_height) / grid.height_resolution))
    return int(grid._scores[r, c, layer])


@pytest.fixture(params=["native", "python"])
def make(request, monkeypatch):
    if request.param == "python":
        monkeypatch.setattr(ray_grid, "kernels", None)
    else:
        assert ray_grid.kernels is not None

    def build(**kwargs):
        params = dict(min_height=0.07, max_height=2.0, height_resolution=0.1, free_radius_m=0)
        params.update(kwargs)
        return ray_grid.RayOccupancy2D(**params)

    return build


def test_ceiling_rays_at_different_elevations_all_clear(make):
    """A shallow ceiling ray clears the high layer of a former obstacle even when a
    steeper ceiling ray in the same azimuth ends nearer."""
    grid = make(score_miss=2, score_max=4)
    for _ in range(2):  # a person's head at (2, 0), layer 1.57-1.67 m
        grid.ingest_rays(ray(2.0, 1.62))
    assert layer_score(grid, 2.0, 1.62) > 0
    steep, shallow = ray(1.0, 2.9), ray(4.0, 2.9)  # both end on a 2.9 m ceiling
    for _ in range(2):
        grid.ingest_rays(np.vstack((steep, shallow)))
    # The shallow ray crosses x = 2 m at about 1.615 m.
    assert layer_score(grid, 2.0, 1.62) <= 0


def test_end_margin_keeps_grazed_overhang(make):
    """A ray rising into the underside of a slab crosses the slab's own height layer
    in the cells just before its return; the margin keeps that from clearing it."""
    for margin, expected_positive in ((0.0, False), (0.2, True)):
        grid = make(free_end_margin=margin, score_miss=2)
        grid.ingest_rays(ray(2.85, 0.72))  # slab hit, layer 0.67-0.77
        assert layer_score(grid, 2.85, 0.72) > 0
        # Ends on the underside at 3.0 m; crosses x = 2.85 m at 0.68 m (same layer).
        grid.ingest_rays(ray(3.0, 0.70))
        assert (layer_score(grid, 2.85, 0.72) > 0) is expected_positive, margin


def test_end_margin_does_not_apply_to_clipped_beams(make):
    grid = make(free_end_margin=0.5, max_ray_range=3.0)
    grid.ingest_rays(ray(10.0, 0.33))  # horizontal, no return within range
    assert layer_score(grid, 2.7, 0.33) < 0  # within 0.5 m of the clipped end, still cleared
    short = make(free_end_margin=0.5)
    short.ingest_rays(ray(3.0, 0.33))
    assert layer_score(short, 2.0, 0.33) < 0
    assert layer_score(short, 2.7, 0.33) == 0  # within 0.5 m of the return


@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan")])
def test_invalid_end_margin_rejected(value):
    with pytest.raises(ValueError):
        ray_grid.RayOccupancy2D(free_end_margin=value)
