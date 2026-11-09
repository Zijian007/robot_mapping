"""Reproduce exact no-loop maps captured before extracting the package."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from robot_mapping.core.online_map import OnlineMap, SE3Pose
from robot_mapping.core.pose_math import gravity_aligned_frame
from robot_mapping.core import spatial


@pytest.mark.parametrize("tilted", [False, True])
def test_map_matches_original_algorithm_and_encoding(tilted, monkeypatch):
    fixture = json.loads((Path(__file__).parents[1]/"docs/baseline-map-fixtures.json").read_text())
    expected = fixture["tilted" if tilted else "level"]
    mapping = OnlineMap(voxel_resolution=.15, grid_resolution=.05, keyframe_dist_th=.1)
    with monkeypatch.context() as context:
        context.setattr(spatial, 'kernels', None)
        reference = OnlineMap(voxel_resolution=.15, grid_resolution=.05, keyframe_dist_th=.1)
    scene = np.random.default_rng(19).normal(size=(600, 3))+[3., 1., .3]
    for current in (mapping, reference):
        if tilted:
            current.set_map_alignment(gravity_aligned_frame([-.15,.01,1.], [1, 0, 0]))
        for x in (0., 1., 2., 1., 0.):
            current.ingest_local_points(SE3Pose((x, 0., 0.), (0., 0., 0., 1.)), scene-[x, 0, 0])
    grid, spec = mapping.snapshot_grid()
    points = mapping.voxels.points_array()
    assert len(points) == expected["point_count"]
    assert mapping.keyframe_count() == expected["keyframes"]
    assert vars(spec) == expected["grid_spec"]
    assert hashlib.sha256(grid.astype('<i2').tobytes()).hexdigest() == expected["grid_sha256"]
    # Keep checking the original fixture byte-for-byte with its Python set
    # ordering, then check native geometry bit-for-bit independent of hash order.
    reference_points = reference.voxels.points_array()
    assert hashlib.sha256(reference_points.astype('<f8').tobytes()).hexdigest() == expected["points_sha256"]
    np.testing.assert_array_equal(points[np.lexsort(points.T)],
                                  reference_points[np.lexsort(reference_points.T)])
    np.testing.assert_allclose(mapping.freeze_for_save().map_from_lio, expected["map_from_lio"])
