"""FAST-LIO2 scan preparation from owned DTOs; message decoding belongs to the adapter."""
import numpy as np
import pytest

from robot_mapping.backends.fastlio import prepare_scan
from robot_mapping.types import LidarScan


def test_prepare_scan_drops_blind_zone_without_rewriting_scan_duration():
    scan = LidarScan(
        "front", "lidar_front", "sensor", 1_000_000_000, 100_000_000, 99,
        np.array([[2., 2., .5], [0., 0., 0.]]),
        np.array([0, 100_000_000]),
        np.array([1., 1.1]),
    )
    native = prepare_scan(scan, .5)
    assert len(native.points_array) == 1
    assert scan.end_ns == 1_100_000_000
    assert native.scan_duration_s == pytest.approx(.1)
    assert native.frame_stamp == 1.
