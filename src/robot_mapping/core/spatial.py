"""Spatial storage: voxel deduplication and an expanding occupancy grid."""
from __future__ import annotations

import math
from typing import Sequence
import numpy as np

from .grid import GDK_FREE, GDK_OCCUPIED, GDK_UNKNOWN, OccupancyGridSpec


class WorldVoxelHash:
    """Axis-aligned voxel set in world frame (dedupe by cell index)."""

    def __init__(self, resolution: float = 0.15) -> None:
        if resolution <= 0:
            raise ValueError("voxel resolution must be > 0")
        self.resolution = float(resolution)
        self._inv = 1.0 / self.resolution
        self._cells: set[tuple[int, int, int]] = set()

    def clear(self) -> None:
        self._cells.clear()

    def __len__(self) -> int:
        return len(self._cells)

    def _key(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        return (
            int(math.floor(x * self._inv)),
            int(math.floor(y * self._inv)),
            int(math.floor(z * self._inv)),
        )

    def add_points(self, points: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Insert points; return centers of newly created voxels as float32 (M,3)."""
        arr = np.asarray(points, dtype=np.float64)
        if arr.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("points must be (N, >=3)")
        keys = np.floor(arr[:, :3] * self._inv).astype(np.int64)
        new_centers: list[tuple[float, float, float]] = []
        half = 0.5 * self.resolution
        for i in range(keys.shape[0]):
            key = (int(keys[i, 0]), int(keys[i, 1]), int(keys[i, 2]))
            if key in self._cells:
                continue
            self._cells.add(key)
            new_centers.append(
                (
                    key[0] * self.resolution + half,
                    key[1] * self.resolution + half,
                    key[2] * self.resolution + half,
                )
            )
        if not new_centers:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(new_centers, dtype=np.float32)

    def points_array(self) -> np.ndarray:
        if not self._cells:
            return np.empty((0, 3), dtype=np.float32)
        keys = np.fromiter(
            (c for cell in self._cells for c in cell),
            dtype=np.int64,
            count=len(self._cells) * 3,
        ).reshape((-1, 3))
        centers = (keys.astype(np.float64) + 0.5) * self.resolution
        return centers.astype(np.float32)


class GrowingOccupancy2D:
    """Expandable occupancy grid: occupied from hits, free from robot footprint."""

    def __init__(
        self,
        *,
        resolution: float = 0.05,
        min_height: float = -0.25,
        max_height: float = 1.8,
        margin_cells: int = 5,
        expand_chunk_cells: int = 40,
        free_radius_m: float = 0.35,
    ) -> None:
        if resolution <= 0:
            raise ValueError("grid resolution must be > 0")
        self.resolution = float(resolution)
        self.min_height = float(min_height)
        self.max_height = float(max_height)
        self.margin_cells = int(margin_cells)
        self.expand_chunk_cells = max(8, int(expand_chunk_cells))
        self.free_radius_m = float(free_radius_m)
        self._values: np.ndarray | None = None
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._width = 0
        self._height = 0

    def clear(self) -> None:
        self._values = None
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._width = 0
        self._height = 0

    @property
    def ready(self) -> bool:
        return self._values is not None

    @property
    def spec(self) -> OccupancyGridSpec | None:
        if self._values is None:
            return None
        return OccupancyGridSpec(
            width=self._width,
            height=self._height,
            resolution=self.resolution,
            origin_x=self._origin_x,
            origin_y=self._origin_y,
        )

    @property
    def values(self) -> np.ndarray | None:
        return self._values

    def occupied_count(self) -> int:
        if self._values is None:
            return 0
        return int(np.sum(self._values == GDK_OCCUPIED))

    def free_count(self) -> int:
        if self._values is None:
            return 0
        return int(np.sum(self._values == GDK_FREE))

    def _ensure_seed(self, x: float, y: float) -> None:
        if self._values is not None:
            return
        m = self.margin_cells
        side = 2 * m + 1
        self._width = side
        self._height = side
        self._origin_x = x - m * self.resolution
        self._origin_y = y - m * self.resolution
        self._values = np.full((side, side), GDK_UNKNOWN, dtype=np.int16)

    def _expand_to_include(self, xs: np.ndarray, ys: np.ndarray) -> None:
        assert self._values is not None
        if xs.size == 0:
            return
        min_x = float(xs.min())
        max_x = float(xs.max())
        min_y = float(ys.min())
        max_y = float(ys.max())
        pad = self.margin_cells * self.resolution

        need_min_x = min_x - pad
        need_max_x = max_x + pad
        need_min_y = min_y - pad
        need_max_y = max_y + pad

        cur_max_x = self._origin_x + self._width * self.resolution
        cur_max_y = self._origin_y + self._height * self.resolution

        left = 0
        right = 0
        bottom = 0
        top = 0

        while need_min_x < self._origin_x - 1e-9:
            add = max(self.expand_chunk_cells, int(math.ceil((self._origin_x - need_min_x) / self.resolution)))
            left += add
            self._origin_x -= add * self.resolution
        while need_max_x > cur_max_x + 1e-9:
            add = max(self.expand_chunk_cells, int(math.ceil((need_max_x - cur_max_x) / self.resolution)))
            right += add
            cur_max_x += add * self.resolution
        while need_min_y < self._origin_y - 1e-9:
            add = max(self.expand_chunk_cells, int(math.ceil((self._origin_y - need_min_y) / self.resolution)))
            bottom += add
            self._origin_y -= add * self.resolution
        while need_max_y > cur_max_y + 1e-9:
            add = max(self.expand_chunk_cells, int(math.ceil((need_max_y - cur_max_y) / self.resolution)))
            top += add
            cur_max_y += add * self.resolution

        if left == 0 and right == 0 and bottom == 0 and top == 0:
            return

        new_w = self._width + left + right
        new_h = self._height + bottom + top
        new_values = np.full((new_h, new_w), GDK_UNKNOWN, dtype=np.int16)
        new_values[bottom : bottom + self._height, left : left + self._width] = self._values
        self._values = new_values
        self._width = new_w
        self._height = new_h

    def _world_to_rc(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        col = np.floor((x - self._origin_x) / self.resolution).astype(np.int32)
        row = np.floor((y - self._origin_y) / self.resolution).astype(np.int32)
        return row, col

    def ingest_points(self, xyz: np.ndarray, robot_xy: tuple[float, float] | None = None) -> None:
        arr = np.asarray(xyz, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 3:
            if robot_xy is not None:
                self._ensure_seed(robot_xy[0], robot_xy[1])
                self._mark_free_disk(robot_xy[0], robot_xy[1])
            return

        z = arr[:, 2]
        mask = (z >= self.min_height) & (z <= self.max_height)
        xy = arr[mask, :2]
        if robot_xy is not None:
            self._ensure_seed(robot_xy[0], robot_xy[1])
        elif len(xy) > 0:
            self._ensure_seed(float(xy[0, 0]), float(xy[0, 1]))
        else:
            return

        assert self._values is not None
        if len(xy) > 0:
            self._expand_to_include(xy[:, 0], xy[:, 1])
            row, col = self._world_to_rc(xy[:, 0], xy[:, 1])
            in_bounds = (
                (row >= 0)
                & (row < self._height)
                & (col >= 0)
                & (col < self._width)
            )
            self._values[row[in_bounds], col[in_bounds]] = GDK_OCCUPIED

        if robot_xy is not None:
            self._expand_to_include(
                np.array([robot_xy[0]], dtype=np.float64),
                np.array([robot_xy[1]], dtype=np.float64),
            )
            self._mark_free_disk(robot_xy[0], robot_xy[1])

    def _mark_free_disk(self, x: float, y: float) -> None:
        assert self._values is not None
        r_cells = max(1, int(math.ceil(self.free_radius_m / self.resolution)))
        crow = int(math.floor((y - self._origin_y) / self.resolution))
        ccol = int(math.floor((x - self._origin_x) / self.resolution))
        r0 = max(0, crow - r_cells)
        r1 = min(self._height, crow + r_cells + 1)
        c0 = max(0, ccol - r_cells)
        c1 = min(self._width, ccol + r_cells + 1)
        rr = np.arange(r0, r1, dtype=np.int32)[:, None]
        cc = np.arange(c0, c1, dtype=np.int32)[None, :]
        dist2 = (rr - crow) ** 2 + (cc - ccol) ** 2
        disk = dist2 <= r_cells * r_cells
        patch = self._values[r0:r1, c0:c1]
        # Only paint unknown → free; do not erase occupied hits under the robot.
        unknown = patch == GDK_UNKNOWN
        patch[disk & unknown] = GDK_FREE


