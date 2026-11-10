"""SDK-free sensor contracts for mapping inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypedDict

import numpy as np

from .types import ImuBatch, LidarScan

if TYPE_CHECKING:
    from .backends.fastlio import BackendScan


class _LioState(TypedDict):
    status: str
    stamp: float


class LioResult(_LioState, total=False):
    """Only scan_to_map with lidar_update_valid=True may contribute to a map.

    prediction_only carries IMU-predicted state after a failed lidar update.
    """

    lidar_update_valid: bool
    position: Sequence[float] | np.ndarray
    orientation_xyzw: Sequence[float] | np.ndarray
    scan_body_points: np.ndarray
    scan_body_rays: np.ndarray  # (N,6), deskewed origin and measured endpoint in IMU axes
    matched_points: int
    map_points: int
    mean_abs_residual: float


class SensorSource(Protocol):
    def open(self) -> None: ...
    def next_lidar(self, timeout_s: float) -> LidarScan | None: ...
    def imu_between(self, begin_ns: int, end_ns: int, *, clock_id: str) -> ImuBatch: ...
    def close(self) -> None: ...


class TransformProvider(Protocol):
    def lookup(self, target: str, source: str, timestamp_ns: int) -> np.ndarray:
        """Return historical T_target_source; raise when unavailable."""
        ...


class LioBackend(Protocol):
    """Algorithm backend operating on normalized scans and an (N, 7) IMU array."""

    def process(self, scan: BackendScan, imu: np.ndarray) -> LioResult: ...
    def reset(self) -> None: ...
