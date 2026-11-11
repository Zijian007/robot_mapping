"""Deterministic map geometry and grid encoding for synthetic scenes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from robot_mapping.core import spatial
from robot_mapping.core.map_types import SE3Pose
from robot_mapping.core.online_map import OnlineMap
from robot_mapping.core.pose_math import gravity_aligned_frame


@pytest.mark.parametrize("tilted", [False, True])
def test_map_matches_fixed_geometry_and_encoding(tilted, monkeypatch):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/map_snapshot.json").read_text()
    )
    expected = fixture["tilted" if tilted else "level"]
    mapping = OnlineMap(voxel_resolution=0.15, grid_resolution=0.05, keyframe_dist_th=0.1)
    with monkeypatch.context() as context:
        context.setattr(spatial, "kernels", None)
        reference = OnlineMap(voxel_resolution=0.15, grid_resolution=0.05, keyframe_dist_th=0.1)
    scene = np.random.default_rng(19).normal(size=(600, 3)) + [3.0, 1.0, 0.3]
    for current in (mapping, reference):
        if tilted:
            current.set_map_alignment(gravity_aligned_frame([-0.15, 0.01, 1.0], [1, 0, 0]))
        for x in (0.0, 1.0, 2.0, 1.0, 0.0):
            current.ingest_local_points(
                SE3Pose((x, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)), scene - [x, 0, 0]
            )
    grid, spec = mapping.snapshot_grid()
    points = mapping.voxels.points_array()
    assert len(points) == expected["point_count"]
    assert mapping.keyframe_count() == expected["keyframes"]
    assert vars(spec) == expected["grid_spec"]
    assert hashlib.sha256(grid.astype("<i2").tobytes()).hexdigest() == expected["grid_sha256"]
    # Check fixed grid/point encoding, then compare native geometry
    # independent of hash iteration order.
    reference_points = reference.voxels.points_array()
    assert (
        hashlib.sha256(reference_points.astype("<f8").tobytes()).hexdigest()
        == expected["points_sha256"]
    )
    np.testing.assert_array_equal(
        points[np.lexsort(points.T)], reference_points[np.lexsort(reference_points.T)]
    )
    np.testing.assert_allclose(mapping.freeze_for_save().map_from_lio, expected["map_from_lio"])
