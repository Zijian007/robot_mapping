"""Point fields, row order, timestamp windows and native/reference parity."""
from dataclasses import replace

import numpy as np
import pytest

from robot_mapping.backends import fastlio
from robot_mapping.backends.ray_native import kernels
from robot_mapping.filters.body_filter import BodyBox
from robot_mapping.runtime import frontend
from robot_mapping.types import LidarScan

pytestmark = pytest.mark.skipif(kernels is None, reason="optional native extension not built")


@pytest.mark.parametrize('origins', [False, True])
@pytest.mark.parametrize('blind', [0., .5, 1.25])
def test_preparation_matches_reference_at_blind_boundaries(monkeypatch, origins, blind):
    rng = np.random.default_rng(15)
    xyz = rng.uniform(-2, 2, (1000, 3)).astype(np.float32)
    xyz[:4] = 0.
    xyz[:3, 0] = [np.nextafter(np.float32(blind), -np.inf, dtype=np.float32), blind,
                   np.nextafter(np.float32(blind), np.inf, dtype=np.float32)]
    offsets = np.linspace(0, 100_000_000, len(xyz), dtype=np.int64)
    scan = LidarScan('front', 'lidar', 'sensor', 1_788_000_000_000_000_000,
                     100_000_000, 1, xyz, offsets, rng.uniform(0, 255, len(xyz)))
    outputs = []
    for backend in (None, kernels):
        with monkeypatch.context() as context:
            context.setattr(fastlio, 'kernels', backend)
            outputs.append(fastlio.prepare_scan(scan, blind, include_origins=origins))
    a, b = outputs
    assert (a.frame_stamp, a.scan_duration_s, a.raw_points, a.dropped_points) == (
        b.frame_stamp, b.scan_duration_s, b.raw_points, b.dropped_points)
    assert a.points_array.tobytes() == b.points_array.tobytes()
    assert b.points_array.shape[1] == (8 if origins else 5)
    assert not scan.xyz.flags.writeable
    empty = replace(scan, xyz=np.empty((0, 3)), intensity=np.empty(0), point_offsets_ns=np.empty(0, dtype=np.int64))
    assert fastlio.prepare_scan(empty, blind, include_origins=origins).points_array.shape == (0, 8 if origins else 5)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_aabb_compaction_keeps_fields_and_scalar_boundary_semantics(dtype):
    rng = np.random.default_rng(77)
    padded = rng.uniform(-2, 2, (1000, 22)).astype(dtype)
    points = padded[::-1, ::2]  # negative row strides and noncontiguous columns
    centers = rng.uniform(-1, 1, (35, 3))
    boxes = np.stack((centers-.2, centers+.2), axis=1)
    def compare(points, boxes):
        keep = np.ones(len(points), dtype=bool)
        x, y, z = np.ascontiguousarray(points[:, :3].T)
        for lower, upper in boxes:
            keep &= ~((x>=lower[0]) & (x<=upper[0]) & (y>=lower[1]) & (y<=upper[1])
                       & (z>=lower[2]) & (z<=upper[2]))
        actual, removed = kernels.filter_aabbs(points, boxes, np.__version__.startswith('1.'))
        assert removed == int((~keep).sum())
        assert actual.dtype == points.dtype
        assert actual.tobytes() == points[keep].tobytes()
        if not removed:
            assert actual is points
    compare(points, boxes)
    compare(points, np.empty((0, 2, 3)))
    for bound in (.1, .2, -.2, 1e-50, np.nextafter(3.4e38, 0), 3.4e38, -3.4e38):
        # Compare using scalar bounds, as the original loop does. Broadcasted
        # vector comparisons would have different NumPy 1.x promotion rules.
        special = np.array([[bound, 0., 0., 7., -0.]], dtype=dtype)
        box = np.array([[[bound, -1., -1.], [bound, 1., 1.]]])
        compare(special, box)
    points.flags.writeable = False
    compare(points, boxes)  # inputs are read-only; outputs own their storage


@pytest.mark.parametrize('columns', [5, 8, 11])
@pytest.mark.parametrize('dt', [-.005, 0., .005, .2])
@pytest.mark.parametrize('empty_front', [False, True])
def test_fused_merge_matches_reference_bytes_and_accounting(monkeypatch, columns, dt, empty_front):
    rng = np.random.default_rng(812)
    padded = rng.uniform(-2, 2, (1000, columns*2)).astype(np.float32)
    back_array = padded[::-1, ::2]
    back_array[:, 4] = np.linspace(-5, 105, len(back_array))
    back_array[:5, 4] = [0., 100., np.nextafter(np.float32(100.), np.float32(np.inf)), np.nan, -0.]
    front_array = back_array[:0] if empty_front else back_array[:3].copy()
    front = fastlio.BackendScan(front_array, 10., .1000000001, len(front_array)+3, 3)
    back = fastlio.BackendScan(back_array, 10.+dt, .1, len(back_array)+5, 5)
    before_back, before_front = back_array.copy(), front_array.copy()
    angle = .4321
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    outputs = []
    for backend in (None, kernels):
        with monkeypatch.context() as context:
            context.setattr(frontend, 'kernels', backend)
            outputs.append(frontend.merge_back_into_front(front, back,
                rotation_front=rotation, translation_front=np.array([.03, .01, .5]),
                rotation_back=rotation.T, translation_back=np.array([-.6, .1, .02]),
                body_box=BodyBox(-.5, .5, -.5, .5, -.5, .5), max_scan_duration_s=.12))
    (a, removed_a, kept_a), (b, removed_b, kept_b) = outputs
    assert (removed_a, kept_a, a.raw_points, a.dropped_points, a.frame_stamp, a.scan_duration_s) == (
        removed_b, kept_b, b.raw_points, b.dropped_points, b.frame_stamp, b.scan_duration_s)
    assert a.points_array.tobytes() == b.points_array.tobytes()
    if kept_b == 0:
        assert a is b is front
    assert back_array.tobytes() == before_back.tobytes()
    assert front_array.tobytes() == before_front.tobytes()


def test_preprocessing_rejects_invalid_shapes_without_modifying_inputs():
    xyz = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        kernels.prepare_points(xyz, np.zeros(1, dtype=np.float32), np.zeros(2, dtype=np.int64), .25, True)
    with pytest.raises((ValueError, TypeError)):
        kernels.prepare_points(xyz.astype(np.float64), np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.int64), .25, True)
    with pytest.raises(ValueError):
        kernels.filter_aabbs(xyz, np.zeros((1, 3)))
    with pytest.raises(ValueError):
        kernels.filter_aabbs(xyz, np.array([[[1., 0., 0.], [0., 1., 1.]]]))
    front = np.zeros((1, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        kernels.merge_prepared(front, front, np.zeros((1, 3)), np.empty((0, 3)), 0., 100.)
    np.testing.assert_array_equal(xyz, 0.)
    np.testing.assert_array_equal(front, 0.)
