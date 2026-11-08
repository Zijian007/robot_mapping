"""Height-binned occupancy evidence from measured, deskewed lidar rays.

The exported map stays 2D. Independent vertical evidence prevents a ray above
an obstacle from erasing it. Evidence is updated once per cell per observation;
current hits win over free rays in that observation. Ray-derived 2D free space
requires negative evidence in every configured height bin; unseen bins stay
unknown. This discrete coverage is not a continuous swept-volume guarantee.
"""
from __future__ import annotations

import numpy as np

from .grid import GDK_FREE, GDK_OCCUPIED, GDK_UNKNOWN
from .spatial import GrowingOccupancy2D
from robot_mapping.backends.ray_native import kernels


class RayOccupancy2D(GrowingOccupancy2D):
    def __init__(self, *, height_resolution=.2, max_ray_range=30., max_rays=2048, **kwargs):
        super().__init__(**kwargs)
        if not np.isfinite([height_resolution, max_ray_range]).all() or min(height_resolution, max_ray_range) <= 0:
            raise ValueError("ray height resolution and range must be finite and positive")
        if max_rays < 1:
            raise ValueError("max_rays must be positive")
        self.height_resolution = float(height_resolution)
        self.max_ray_range = float(max_ray_range)
        self.max_rays = int(max_rays)
        self._scores = None
        self._last_footprint_xy = None

    def clear(self):
        super().clear()
        self._scores = None
        self._last_footprint_xy = None

    def _ensure_seed(self, x, y):
        super()._ensure_seed(x, y)
        if self._scores is None:
            layers = max(1, int(np.ceil((self.max_height-self.min_height)/self.height_resolution)))
            self._scores = np.zeros((self._height, self._width, layers), dtype=np.int8)

    def _expand_to_include(self, xs, ys):
        old_x, old_y = self._origin_x, self._origin_y
        old_h, old_w = self._height, self._width
        super()._expand_to_include(xs, ys)
        if (old_h, old_w) != (self._height, self._width):
            left = round((old_x-self._origin_x)/self.resolution)
            bottom = round((old_y-self._origin_y)/self.resolution)
            self._scores = np.pad(self._scores, ((bottom, self._height-old_h-bottom),
                (left, self._width-old_w-left), (0, 0)))

    def ingest_points(self, xyz, robot_xy=None):
        raise ValueError("ray occupancy requires measured origins and endpoints; use ingest_rays")

    def _mark_free_disk(self, x, y):
        # Retain the existing current-footprint prior, but even a single-hit
        # candidate takes precedence over it. Work on a local patch only.
        row, col = self._world_to_rc(np.asarray(x), np.asarray(y))
        radius = max(1, int(np.ceil(self.free_radius_m/self.resolution)))
        r0, r1 = max(0, row-radius), min(self._height, row+radius+1)
        c0, c1 = max(0, col-radius), min(self._width, col+radius+1)
        patch = self._values[r0:r1, c0:c1]
        positive = (self._scores[r0:r1, c0:c1] > 0).any(axis=2)
        rr, cc = np.ogrid[r0:r1, c0:c1]
        disk = (rr-row)**2+(cc-col)**2 <= radius**2
        patch[disk & (patch == GDK_UNKNOWN) & ~positive] = GDK_FREE

    def _keys(self, xyz):
        rows, cols = self._world_to_rc(xyz[:, 0], xyz[:, 1])
        layers = np.floor((xyz[:, 2]-self.min_height)/self.height_resolution).astype(np.int64)
        layers = np.minimum(layers, self._scores.shape[2]-1)  # inclusive upper boundary
        valid = ((xyz[:, 2] >= self.min_height) & (xyz[:, 2] <= self.max_height)
                 & (rows >= 0) & (rows < self._height) & (cols >= 0) & (cols < self._width))
        return ((rows[valid].astype(np.int64)*self._width+cols[valid])*self._scores.shape[2]+layers[valid])

    def _select_rays(self, rays):
        delta = rays[:, 3:]-rays[:, :3]
        distance = np.linalg.norm(delta, axis=1)
        azimuth = np.floor((np.arctan2(delta[:, 1], delta[:, 0])+np.pi)*180/np.pi).astype(int) % 360
        layer = np.clip(np.floor((rays[:, 5]-self.min_height)/self.height_resolution),
                        -1, self._scores.shape[2]).astype(int)
        # Sampling groups only: each retained ray still has its exact origin.
        origins = np.round(rays[:, :3]/.25)
        if kernels is not None:
            chosen = kernels.select_grouped(np.column_stack((origins, azimuth, layer)).astype(np.int64), distance)
            if len(chosen) > self.max_rays:
                chosen = chosen[np.linspace(0, len(chosen)-1, self.max_rays, dtype=int)]
            return rays[chosen]
        origin_order = np.lexsort((origins[:, 2], origins[:, 1], origins[:, 0]))
        ordered_origins = origins[origin_order]
        boundaries = np.r_[True, (ordered_origins[1:] != ordered_origins[:-1]).any(axis=1)]
        sensors = np.empty(len(rays), dtype=np.int64)
        sensors[origin_order] = np.cumsum(boundaries)-1
        key = (sensors*360+azimuth)*(self._scores.shape[2]+2)+layer+1
        order = np.lexsort((distance, key))
        chosen = order[np.r_[True, np.diff(key[order]) != 0]]
        if len(chosen) > self.max_rays:
            chosen = chosen[np.linspace(0, len(chosen)-1, self.max_rays, dtype=int)]
        return rays[chosen]

    def _free_keys(self, rays):
        if kernels is not None:
            origin, endpoint = rays[:, :3], rays[:, 3:]
            delta = endpoint-origin
            scale = np.minimum(1., self.max_ray_range/np.maximum(np.linalg.norm(delta, axis=1), 1e-9))
            endpoint = origin+delta*scale[:, None]
            r0, c0 = self._world_to_rc(origin[:, 0], origin[:, 1])
            r1, c1 = self._world_to_rc(endpoint[:, 0], endpoint[:, 1])
            return kernels.grid_free_keys(np.column_stack((r0, c0, r1, c1)),
                np.column_stack((origin[:, 2], endpoint[:, 2])), self.min_height,
                self.max_height, self.height_resolution, self._scores.shape[2], self._width, self._height)
        result = []
        depth = self._scores.shape[2]
        # Bound temporary matrices; vectorize the ray traversal within each batch.
        for start in range(0, len(rays), 128):
            batch = rays[start:start+128]
            origin, endpoint = batch[:, :3], batch[:, 3:]
            delta = endpoint-origin
            scale = np.minimum(1., self.max_ray_range/np.maximum(np.linalg.norm(delta, axis=1), 1e-9))
            endpoint = origin+delta*scale[:, None]
            r0, c0 = self._world_to_rc(origin[:, 0], origin[:, 1])
            r1, c1 = self._world_to_rc(endpoint[:, 0], endpoint[:, 1])
            dr, dc = r1-r0, c1-c0
            steps = np.maximum(abs(dr), abs(dc)).astype(np.int64)
            if not steps.max():
                continue
            k = np.arange(1, steps.max(), dtype=np.int64)[None, :]
            n = np.maximum(steps[:, None], 1)
            # Integer rasterization (Bresenham-equivalent rounding), excluding returns.
            rows = r0[:, None]+(k*dr[:, None]+n//2)//n
            cols = c0[:, None]+(k*dc[:, None]+n//2)//n
            z = origin[:, 2, None]+k/n*(endpoint[:, 2]-origin[:, 2])[:, None]
            layers = np.minimum(np.floor((z-self.min_height)/self.height_resolution).astype(int), depth-1)
            valid = ((k < steps[:, None]) & (z >= self.min_height) & (z <= self.max_height)
                     & (rows >= 0) & (rows < self._height) & (cols >= 0) & (cols < self._width))
            result.append(((rows*self._width+cols)*depth+layers)[valid])
        return np.unique(np.concatenate(result)) if result else np.empty(0, dtype=np.int64)

    def ingest_rays(self, rays, robot_xy=None):
        rays = np.asarray(rays, dtype=np.float64)
        if rays.ndim != 2 or rays.shape[1] != 6 or not np.isfinite(rays).all():
            raise ValueError("rays must be finite (N,6): origin xyz, endpoint xyz")
        if not len(rays):
            return  # no return is not evidence of free space
        origin, endpoint = rays[:, :3], rays[:, 3:]
        distances = np.linalg.norm(endpoint-origin, axis=1)
        keep = distances > 1e-6
        rays, distances = rays[keep], distances[keep]
        if not len(rays):
            return
        self._ensure_seed(*(robot_xy if robot_xy is not None else rays[0, :2]))
        clipped = rays[:, :3]+(rays[:, 3:]-rays[:, :3])*np.minimum(1., self.max_ray_range/distances)[:, None]
        extent = np.concatenate((rays[:, :3], clipped), axis=0)
        if robot_xy is not None:
            extent = np.vstack((extent, [*robot_xy, 0.]))
        self._expand_to_include(extent[:, 0], extent[:, 1])
        hits = np.unique(self._keys(rays[distances <= self.max_ray_range, 3:]))
        misses = np.setdiff1d(self._free_keys(self._select_rays(rays)), hits, assume_unique=True)
        scores = self._scores.reshape(-1)
        scores[misses] = np.maximum(scores[misses]-1, -4)
        scores[hits] = np.minimum(scores[hits]+2, 8)
        # Two distinct observations confirm occupation. Single-hit candidates
        # stay unknown, even if another height in the same column is free.
        # Recompute only columns touched by this observation. Include the prior
        # footprint so its temporary unknown->free paint keeps the old semantics.
        depth = self._scores.shape[2]
        columns = [hits//depth, misses//depth]
        if self._last_footprint_xy is not None:
            row, col = self._world_to_rc(*np.asarray(self._last_footprint_xy))
            radius = max(1, int(np.ceil(self.free_radius_m/self.resolution)))
            rr = np.arange(max(0, row-radius), min(self._height, row+radius+1))
            cc = np.arange(max(0, col-radius), min(self._width, col+radius+1))
            columns.append((rr[:, None]*self._width+cc).ravel())
        columns = np.unique(np.concatenate(columns))
        changed = self._scores.reshape(-1, depth)[columns]
        occupied = (changed >= 3).any(axis=1)
        free = (changed < 0).all(axis=1)
        values = self._values.reshape(-1)
        values[columns] = GDK_UNKNOWN
        values[columns[free]] = GDK_FREE
        values[columns[occupied]] = GDK_OCCUPIED
        if robot_xy is not None:
            self._mark_free_disk(*robot_xy)
        self._last_footprint_xy = None if robot_xy is None else tuple(robot_xy)
