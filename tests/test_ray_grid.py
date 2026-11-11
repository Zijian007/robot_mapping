import numpy as np
import pytest

from robot_mapping.core.ray_grid import RayOccupancy2D


def ray(x, z=0.5, origin=(0.0, 0.0, 0.5)):
    return np.array([[*origin, x, 0.0, z]])


def cell(grid, x, y=0.0):
    row, col = grid._world_to_rc(np.array([x]), np.array([y]))
    return grid.values[row[0], col[0]]


def test_person_clears_after_reobserving_empty_space_and_wall_stays():
    # One explicitly configured height bin: crossing it covers the whole band.
    grid = RayOccupancy2D(min_height=0.4, max_height=0.6)
    for _ in range(2):
        grid.ingest_rays(ray(2))
    assert cell(grid, 2) == 100
    for _ in range(5):
        grid.ingest_rays(ray(4))
    assert cell(grid, 2) == 0
    assert cell(grid, 4) == 100
    assert cell(grid, 4.1) == -1  # no clearing behind the measured wall


def test_repeated_points_are_not_independent_evidence_and_hits_win():
    grid = RayOccupancy2D(min_height=0.0, max_height=2.0)
    mixed = np.vstack((np.repeat(ray(2), 1000, axis=0), ray(4)))
    grid.ingest_rays(mixed)
    assert cell(grid, 2) == -1
    grid.ingest_rays(mixed)
    assert cell(grid, 2) == 100


def test_high_and_low_beams_do_not_erase_an_obstacle_at_another_height():
    grid = RayOccupancy2D(min_height=0.0, max_height=2.0)
    for _ in range(2):
        grid.ingest_rays(ray(2, 0.5))
    for _ in range(15):
        grid.ingest_rays(np.vstack((ray(4, 1.5, (0, 0, 1.5)), ray(4, 0.1, (0, 0, 0.1)))))
    assert cell(grid, 2) == 100
    before = grid.values.copy()
    grid.ingest_rays(np.empty((0, 6)))
    np.testing.assert_array_equal(grid.values, before)


def test_true_rear_sensor_origin_and_range_limit():
    grid = RayOccupancy2D(min_height=0.4, max_height=0.6, max_ray_range=3.0)
    for _ in range(2):
        grid.ingest_rays(ray(-2.0, origin=(-0.5, 0, 0.5)))
    for _ in range(5):
        grid.ingest_rays(ray(-10.0, origin=(-0.5, 0, 0.5)))
    assert cell(grid, -2) == 0
    assert cell(grid, -0.2) == -1  # not on the rear sensor's ray
    assert grid.spec.origin_x > -5  # unobserved far endpoint never expands map to -10


def test_growth_preserves_evidence_and_clear_discards_it():
    grid = RayOccupancy2D(min_height=0.0, max_height=2.0)
    grid.ingest_rays(ray(2))
    grid.ingest_rays(ray(-10))
    grid.ingest_rays(ray(2))
    assert cell(grid, 2) == 100
    grid.clear()
    grid.ingest_rays(ray(2))
    assert cell(grid, 2) == -1


def test_invalid_rays_rejected_before_any_map_mutation():
    grid = RayOccupancy2D()
    with pytest.raises(ValueError):
        grid.ingest_rays([[0, 0, 0, np.nan, 0, 0]])
    assert not grid.ready


def test_incremental_refresh_preserves_distant_wall_and_resets_previous_footprint():
    grid = RayOccupancy2D(min_height=0.0, max_height=2.0)
    for _ in range(2):
        grid.ingest_rays(ray(2), robot_xy=(0.0, 1.0))
    assert cell(grid, 0, 1) == 0
    distant = np.array([[10, 10, 0.5, 14, 10, 0.5]])
    grid.ingest_rays(distant, robot_xy=(10.0, 11.0))
    assert cell(grid, 2) == 100
    assert cell(grid, 0, 1) == -1
    assert cell(grid, 10, 11) == 0


def test_single_height_clear_leaves_unobserved_column_unknown():
    grid = RayOccupancy2D(min_height=0.0, max_height=2.0)
    for _ in range(2):
        grid.ingest_rays(ray(2))
    for _ in range(12):
        grid.ingest_rays(ray(4))
    assert cell(grid, 1) == -1  # no hit, but other heights have never been seen
    assert cell(grid, 2) == -1  # old obstacle removed, not declared traversable
    assert cell(grid, 4) == 100


def test_all_height_bins_are_required_including_partial_top_bin():
    grid = RayOccupancy2D(min_height=0.0, max_height=0.45, height_resolution=0.2)
    grid.ingest_rays(ray(4, 0.1, (0, 0, 0.1)))
    grid.ingest_rays(ray(4, 0.3, (0, 0, 0.3)))
    assert cell(grid, 2) == -1
    grid.ingest_rays(ray(4, 0.425, (0, 0, 0.425)))
    assert cell(grid, 2) == 0


@pytest.mark.parametrize("height", [0.1, 1.1])
def test_low_obstacle_and_tabletop_survive_other_height_clearing(height):
    grid = RayOccupancy2D(min_height=0.0, max_height=1.2)
    obstacle = ray(2, height, (0, 0, height))
    for _ in range(2):
        grid.ingest_rays(obstacle)
    other_layers = np.vstack(
        [ray(4, z, (0, 0, z)) for z in np.arange(0.1, 1.2, 0.2) if not np.isclose(z, height)]
    )
    for _ in range(15):
        grid.ingest_rays(other_layers)
    assert cell(grid, 2) == 100


def test_foreground_return_occludes_obstacle_behind_it():
    grid = RayOccupancy2D(min_height=0.4, max_height=0.6)
    for _ in range(2):
        grid.ingest_rays(ray(3))
    for _ in range(15):
        grid.ingest_rays(ray(1))
    assert cell(grid, 1) == 100
    assert cell(grid, 2) == 0  # retains the earlier observation, no new clearing
    assert cell(grid, 3) == 100


def test_footprint_does_not_hide_single_hit_or_confirmed_obstacle():
    grid = RayOccupancy2D(min_height=0.0, max_height=2.0)
    grid.ingest_rays(ray(0.225), robot_xy=(0.0, 0.0))
    assert cell(grid, 0.225) == -1
    grid.ingest_rays(ray(0.225), robot_xy=(0.0, 0.0))
    assert cell(grid, 0.225) == 100
    grid.ingest_rays(ray(-10), robot_xy=(-10.0, 0.0))
    assert cell(grid, 0.225) == 100  # growth and footprint refresh retain the hit


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "rules, hits, misses_to_unblock",
    [
        # Defaults: four hits saturate at 8; six misses drop below occupied=3.
        ({}, 4, 6),
        # A lower cap clears a saturated trail after two misses.
        ({"score_max": 4}, 4, 2),
        # Stronger misses clear faster even with the default cap.
        ({"score_miss": 3}, 4, 2),
        # Confirm on the first hit; a single miss clears it again.
        ({"score_occupied": 2}, 1, 1),
    ],
)
def test_score_rules_control_trail_clearing(monkeypatch, native, rules, hits, misses_to_unblock):
    from robot_mapping.core import ray_grid

    if native and ray_grid.kernels is None:
        pytest.skip("native ray kernels unavailable")
    if not native:
        monkeypatch.setattr(ray_grid, "kernels", None)
    grid = RayOccupancy2D(min_height=0.4, max_height=0.6, **rules)
    for _ in range(hits):
        grid.ingest_rays(ray(2))
    assert cell(grid, 2) == 100
    for _ in range(misses_to_unblock - 1):
        grid.ingest_rays(ray(4))
    assert cell(grid, 2) == 100
    grid.ingest_rays(ray(4))
    assert cell(grid, 2) != 100
    assert grid._scores.min() >= grid.score_min and grid._scores.max() <= grid.score_max


@pytest.mark.parametrize(
    "rules, message",
    [
        ({"score_hit": 0}, "score_hit"),
        ({"score_miss": 128}, "score_miss"),
        ({"score_min": 0}, "score_min"),
        ({"score_min": -129}, "score_min"),
        ({"score_occupied": 9}, "score_occupied"),
        ({"score_max": 128}, "score_max"),
        ({"score_occupied": 0}, "score_occupied"),
        ({"score_hit": 2.0}, "score_hit"),
        ({"score_miss": True}, "score_miss"),
    ],
)
def test_invalid_score_rules_rejected(rules, message):
    with pytest.raises(ValueError, match=message):
        RayOccupancy2D(**rules)
