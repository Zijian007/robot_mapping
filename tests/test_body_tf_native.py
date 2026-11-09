"""Native TF predicates must preserve the original filter's tolerances."""
from unittest.mock import Mock

import numpy as np
import pytest

from robot_mapping.backends.ray_native import kernels
from robot_mapping.filters import urdf_filter

pytestmark = pytest.mark.skipif(kernels is None, reason="optional native extension not built")


def original_valid(matrix):
    return (matrix.shape == (4, 4) and np.isfinite(matrix).all()
            and np.allclose(matrix[3], [0, 0, 0, 1])
            and np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            and np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5))


def test_transform_validation_matches_oracle_near_acceptance_boundaries(monkeypatch):
    rng = np.random.default_rng(741)
    matrices = [np.eye(4), np.diag([-1., 1., 1., 1.]), np.eye(3), np.empty((0, 4)), np.zeros(4)]
    # Adjacent floats on both sides of affine-row, orthogonality and determinant
    # thresholds. The determinant is always computed by the same NumPy routine.
    for value in (1e-8, 1e-5, 2e-5, 1e-5+1e-8):
        for sign in (-1, 1):
            for delta in (np.nextafter(value, 0.), value, np.nextafter(value, np.inf)):
                for row, col in ((3, 0), (3, 3), (0, 0), (0, 1)):
                    matrix = np.eye(4)
                    matrix[row, col] += sign*delta
                    matrices.append(matrix)
    for _ in range(500):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        matrix = np.eye(4)
        matrix[:3, :3] = rotation + rng.normal(scale=10.**rng.uniform(-9, -3), size=(3, 3))
        matrix[:3, 3] = rng.normal(size=3)*1e6
        matrices.append(matrix)
    for bad in (np.nan, np.inf, -np.inf):
        for i in range(4):
            for j in range(4):
                matrix = np.eye(4)
                matrix[i, j] = bad
                matrices.append(matrix)
    for matrix in matrices:
        # The production caller promotes SDK matrices to float64 first.
        matrix = np.asarray(matrix, dtype=np.float64)
        expected = bool(original_valid(matrix))
        for backend in (None, kernels):
            with monkeypatch.context() as context:
                context.setattr(urdf_filter, 'kernels', backend)
                assert urdf_filter._valid_transform(matrix) == expected
    matrix = np.eye(4, order='F')
    matrix.flags.writeable = False
    assert urdf_filter._valid_transform(matrix)


def test_native_predicates_match_numpy_component_thresholds():
    for target, tolerance in ((0., 1e-8), (1., 1e-5+1e-8)):
        for side in (-1, 1):
            threshold = target+side*tolerance
            for value in (np.nextafter(threshold, -np.inf), threshold, np.nextafter(threshold, np.inf)):
                matrix = np.eye(4)
                matrix[3, 0 if target == 0 else 3] = value
                assert kernels.rigid_structure_valid(matrix) == bool(np.allclose(matrix[3], [0, 0, 0, 1]))
    for i, j in ((0, 0), (0, 1)):
        target = float(i == j)
        for side in (-1, 1):
            threshold = target+side*(1e-5+1e-5*target)
            for value in (np.nextafter(threshold, -np.inf), threshold, np.nextafter(threshold, np.inf)):
                gram = np.eye(3)
                gram[i, j] = value
                assert kernels.rigid_gram_valid(gram) == bool(np.allclose(gram, np.eye(3), atol=1e-5))
    for threshold in (1.-2e-5, 1.+2e-5, np.nan, np.inf, -np.inf):
        for value in (np.nextafter(threshold, -np.inf), threshold, np.nextafter(threshold, np.inf)):
            assert kernels.rigid_determinant_valid(value) == bool(np.isclose(value, 1, atol=1e-5))
    assert not kernels.rigid_structure_valid(np.zeros((4, 3)))
    assert not kernels.rigid_gram_valid(np.zeros((3, 4)))


@pytest.mark.parametrize('failure', ['missing', 'affine', 'reflection', 'scale', 'nonfinite', 'shape'])
def test_late_invalid_tf_rejects_entire_scan_and_names_failed_link(tmp_path, monkeypatch, failure):
    path = tmp_path/'robot.urdf'
    links = ''.join(f'<link name="{name}"><collision><geometry><box size="1 1 1"/>'
                    f'</geometry></collision></link><joint name="{name}_joint" type="revolute">'
                    f'<parent link="base_link"/><child link="{name}"/></joint>' for name in ('first', 'second'))
    path.write_text(f'<robot name="test"><link name="base_link"/>{links}</robot>')
    body = urdf_filter.UrdfBodyFilter(path, tmp_path)
    points = np.array([[0., 0., 0., 7., 1.], [3., 0., 0., 8., 2.]], dtype=np.float32)
    before = points.tobytes()
    invalid = np.eye(4)
    if failure == 'affine': invalid[3, 0] = 1e-4
    if failure == 'reflection': invalid[0, 0] = -1.
    if failure == 'scale': invalid[0, 0] = 1.01
    if failure == 'nonfinite': invalid[0, 3] = np.nan
    if failure == 'shape': invalid = np.eye(3)
    for backend in (None, kernels):
        calls = []
        def lookup(target, source, stamp):
            calls.append((target, source, stamp))
            if source == 'second':
                if failure == 'missing': raise RuntimeError('missing historical TF')
                return invalid
            return np.eye(4)
        with monkeypatch.context() as context:
            context.setattr(urdf_filter, 'kernels', backend)
            compact = Mock(wraps=kernels.filter_aabbs)
            context.setattr(kernels, 'filter_aabbs', compact)
            with pytest.raises(urdf_filter.MissingBodyTransform, match='second.*123') as exc:
                body.filter_points(points, np.eye(3), np.zeros(3), 123, lookup)
            assert exc.value.__cause__ is not None
            compact.assert_not_called()
        assert calls == [('base_link', 'first', 123), ('base_link', 'second', 123)]
        assert points.tobytes() == before


def test_lookup_is_once_per_link_per_scan_and_empty_scan_skips_lookup(tmp_path):
    path = tmp_path/'robot.urdf'
    path.write_text('<robot name="test"><link name="base_link"/><link name="arm">'
        '<collision><geometry><box size="1 1 1"/></geometry></collision>'
        '<collision><geometry><sphere radius="0.2"/></geometry></collision></link>'
        '<joint name="joint" type="revolute"><parent link="base_link"/><child link="arm"/></joint></robot>')
    body = urdf_filter.UrdfBodyFilter(path, tmp_path)
    lookup = Mock(return_value=np.eye(4))
    for stamp in (123, 124):
        body.filter_points(np.array([[2., 0., 0.]]), np.eye(3), np.zeros(3), stamp, lookup)
    assert lookup.call_count == 2
    body.filter_points(np.empty((0, 3)), np.eye(3), np.zeros(3), 125, lookup)
    assert lookup.call_count == 2
