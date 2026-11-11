"""Map poses, immutable observations, snapshots and keyframe selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from robot_mapping.ports import LioResult

from .grid import OccupancyGridSpec


class MappingPhase(str, Enum):
    IDLE = "idle"
    MAPPING = "mapping"


@dataclass(frozen=True)
class SE3Pose:
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    @property
    def yaw(self) -> float:
        x, y, z, w = self.orientation_xyzw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)


def pose_from_odom(odom: LioResult) -> SE3Pose:
    position = odom["position"]
    orientation = odom["orientation_xyzw"]
    return SE3Pose(
        position=(float(position[0]), float(position[1]), float(position[2])),
        orientation_xyzw=(
            float(orientation[0]),
            float(orientation[1]),
            float(orientation[2]),
            float(orientation[3]),
        ),
    )


def yaw_delta(first_yaw: float, second_yaw: float) -> float:
    wrapped_delta = (first_yaw - second_yaw + math.pi) % (2.0 * math.pi) - math.pi
    return abs(wrapped_delta)


class KeyframeGate:
    def __init__(self, dist_th: float = 0.5, angle_th: float = 0.2) -> None:
        self.dist_th = float(dist_th)
        self.angle_th = float(angle_th)
        self._last: SE3Pose | None = None

    def reset(self) -> None:
        self._last = None

    def accept(self, pose: SE3Pose) -> bool:
        if self._last is None:
            self._last = pose
            return True
        dx = pose.position[0] - self._last.position[0]
        dy = pose.position[1] - self._last.position[1]
        distance = math.hypot(dx, dy)
        if distance >= self.dist_th or yaw_delta(pose.yaw, self._last.yaw) >= self.angle_th:
            self._last = pose
            return True
        return False


@dataclass
class OnlineMapStats:
    keyframes: int = 0
    voxels: int = 0
    occupied_cells: int = 0
    free_cells: int = 0
    last_ingest_ms: float = 0.0
    loops: int = 0
    grid_observations: int = 0
    removed_voxels: int = 0


@dataclass(frozen=True)
class KeyframeRecord:
    lio_pose: SE3Pose
    local_points: np.ndarray
    local_rays: np.ndarray | None = None


@dataclass(frozen=True)
class GridObservation:
    anchor_id: int
    anchor_from_tracking: np.ndarray
    local_rays: np.ndarray
    keyframe: bool = False


@dataclass(frozen=True)
class MapSnapshot:
    generation: int
    version: int
    points: np.ndarray
    grid: tuple[np.ndarray, OccupancyGridSpec] | None
    map_from_lio: np.ndarray
    tracking_from_footprint: np.ndarray
    grid_min_height: float
    grid_max_height: float
