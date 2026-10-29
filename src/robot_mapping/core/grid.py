"""Navigation-compatible occupancy encoding and geometry, independent of HAL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

GDK_UNKNOWN = -1
GDK_FREE = 0
GDK_OCCUPIED = 100


@dataclass(frozen=True)
class OccupancyGridSpec:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float

    @classmethod
    def from_map_payload(cls, payload: dict[str, Any]) -> OccupancyGridSpec:
        origin = ((payload.get("origin") or {}).get("position") or {})
        return cls(
            width=int(payload["width"]),
            height=int(payload["height"]),
            resolution=float(payload["resolution"]),
            origin_x=float(origin.get("x", 0.0)),
            origin_y=float(origin.get("y", 0.0)),
        )

    @property
    def max_x(self) -> float:
        return self.origin_x + self.width * self.resolution

    @property
    def max_y(self) -> float:
        return self.origin_y + self.height * self.resolution

    def world_to_grid(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.floor((xy[:, 0] - self.origin_x) / self.resolution).astype(np.int32)
        row = np.floor((xy[:, 1] - self.origin_y) / self.resolution).astype(np.int32)
        return row, col

    def grid_to_world(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        x = self.origin_x + (col.astype(np.float32) + 0.5) * self.resolution
        y = self.origin_y + (row.astype(np.float32) + 0.5) * self.resolution
        return np.column_stack((x, y)).astype(np.float32)
