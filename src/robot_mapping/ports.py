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
    """Pose/cloud fields are present for scan_to_map; metrics are optional."""
    position: Sequence[float] | np.ndarray
    orientation_xyzw: Sequence[float] | np.ndarray
    scan_body_points: np.ndarray
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

