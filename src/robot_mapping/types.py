"""Owned sensor data, SI units, source-clock integer nanoseconds.

Timestamps are never rewritten to host receive time. Arrays are copied on
construction so SDK buffers may be reused without changing archived data.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import numpy as np


def _ns(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be integer nanoseconds")
    if value < (1 if positive else 0) or value > np.iinfo(np.int64).max:
        raise ValueError(f"{name} out of int64 nanosecond range")
    return int(value)


def _owned(value, dtype, shape, name):
    array = np.array(value, dtype=dtype, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape {shape} and finite values")
    array.flags.writeable = False
    return array


def _integer_times(value, name):
    raw = np.asarray(value)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if raw.size and (np.any(raw < 0) or np.any(raw > np.iinfo(np.int64).max)):
        raise ValueError(f"{name} outside int64 range")
    return _owned(raw, np.int64, raw.shape, name)


@dataclass(frozen=True)
class LidarScan:
    sensor_id: str
    frame_id: str
    clock_id: str
    begin_ns: int
    duration_ns: int
    recv_monotonic_ns: int
    xyz: np.ndarray
    point_offsets_ns: np.ndarray
    intensity: np.ndarray | None = None

    def __post_init__(self):
        if not all(isinstance(v, str) and v for v in (self.sensor_id, self.frame_id, self.clock_id)):
            raise ValueError("sensor_id, frame_id and clock_id are required")
        for name in ("begin_ns", "duration_ns", "recv_monotonic_ns"):
            object.__setattr__(self, name, _ns(getattr(self, name), name, positive=name != "recv_monotonic_ns"))
        _ns(self.begin_ns + self.duration_ns, "end_ns", positive=True)
        points = np.asarray(self.xyz)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")
        count = len(points)
        object.__setattr__(self, "xyz", _owned(points, np.float32, (count, 3), "xyz"))
        offsets = _integer_times(self.point_offsets_ns, "point_offsets_ns")
        if len(offsets) != count or np.any(offsets > self.duration_ns):
            raise ValueError("point offsets must match XYZ and stay inside the scan interval")
        object.__setattr__(self, "point_offsets_ns", offsets)
        values = np.zeros(count) if self.intensity is None else self.intensity
        object.__setattr__(self, "intensity", _owned(values, np.float32, (count,), "intensity"))

    @property
    def end_ns(self):
        return self.begin_ns + self.duration_ns


@dataclass(frozen=True)
class ImuBatch:
    frame_id: str
    clock_id: str
    timestamps_ns: np.ndarray
    angular_velocity: np.ndarray  # rad/s
    linear_acceleration: np.ndarray  # m/s^2

    def __post_init__(self):
        if not self.frame_id or not self.clock_id:
            raise ValueError("frame_id and clock_id are required")
        stamps = _integer_times(self.timestamps_ns, "timestamps_ns")
        if np.any(stamps <= 0) or np.any(np.diff(stamps) <= 0):
            raise ValueError("IMU timestamps must be positive and strictly increasing")
        object.__setattr__(self, "timestamps_ns", stamps)
        for name in ("angular_velocity", "linear_acceleration"):
            object.__setattr__(self, name, _owned(getattr(self, name), np.float64, (len(stamps), 3), name))

    def __len__(self):
        return len(self.timestamps_ns)

