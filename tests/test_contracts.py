"""Reject ambiguous timing/ownership at the application boundary."""
from dataclasses import replace
import numpy as np
import pytest

from robot_mapping.types import LidarScan, ImuBatch


def scan(**overrides):
    values = dict(sensor_id="front", frame_id="lidar_front", clock_id="sensor",
        begin_ns=1_000_000_000, duration_ns=100_000_000, recv_monotonic_ns=5_000_000_000,
        xyz=np.array([[1., 2., 3.], [4., 5., 6.]]),
        point_offsets_ns=np.array([0, 100_000_000], dtype=np.int64))
    values.update(overrides)
    return LidarScan(**values)


def test_scan_owns_buffers_and_preserves_scan_end_when_points_removed():
    xyz = np.array([[1., 2., 3.], [4., 5., 6.]])
    item = scan(xyz=xyz)
    xyz[:] = 42
    np.testing.assert_array_equal(item.xyz[0], [1, 2, 3])
    with pytest.raises(ValueError):
        item.xyz[0, 0] = 42
    filtered = replace(item, xyz=item.xyz[:1], point_offsets_ns=item.point_offsets_ns[:1], intensity=item.intensity[:1])
    assert filtered.end_ns == 1_100_000_000
    assert filtered.begin_ns != filtered.recv_monotonic_ns


@pytest.mark.parametrize("overrides", [
    {"begin_ns": 1.0}, {"begin_ns": True}, {"begin_ns": 0},
    {"clock_id": ""}, {"duration_ns": 0}, {"duration_ns": 2**63-1},
    {"point_offsets_ns": [0, -1]}, {"point_offsets_ns": [0., 1.]},
    {"point_offsets_ns": [0]}, {"point_offsets_ns": [0, 100_000_001]},
    {"xyz": [[float("nan"), 0, 0], [1, 2, 3]]},
])
def test_ambiguous_or_invalid_scan_is_rejected(overrides):
    with pytest.raises(ValueError):
        scan(**overrides)


@pytest.mark.parametrize("stamps", [[2, 1], [1, 1], [0, 1], [1., 2.]])
def test_imu_requires_ordered_source_nanoseconds(stamps):
    with pytest.raises(ValueError):
        ImuBatch("imu", "sensor", stamps, np.zeros((2, 3)), np.zeros((2, 3)))


def test_imu_arrays_are_owned_and_si_fields_remain_distinct():
    acceleration = np.array([[0., 0., 9.81]])
    batch = ImuBatch("imu", "sensor", [1], np.zeros((1, 3)), acceleration)
    acceleration[:] = 0
    assert batch.linear_acceleration[0, 2] == 9.81
    assert not batch.timestamps_ns.flags.writeable
    assert not batch.angular_velocity.flags.writeable

