"""Conservative removal of persistent 3D voxels using measured free segments.

Only stored voxels carry miss counters. A sampled beam must cross the inner
60% of a voxel, away from its return; grazing a cell corner is insufficient.
This is geometric evidence, not person recognition or a full-volume guarantee.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from robot_mapping.backends.ray_native import kernels

from .spatial import WorldVoxelHash


class RayVoxelHash(WorldVoxelHash):
    def __init__(self, resolution=0.15, *, clear_observations=3, max_ray_range=30.0, max_rays=2048):
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("voxel resolution must be finite and positive")
        super().__init__(resolution)
        if int(clear_observations) != clear_observations or clear_observations < 2:
            raise ValueError("3D clearing requires at least two observations")
        if not np.isfinite(max_ray_range) or max_ray_range <= 0 or max_rays < 1:
            raise ValueError("3D ray range and limit must be positive")
        self.clear_observations = int(clear_observations)
        self.max_ray_range = float(max_ray_range)
        self.max_rays = int(max_rays)
        self._python_misses = {}
        self._python_removed = 0  # removal events, not unique objects

    @property
    def _misses(self):
        if self._native_store is not None:
            return self._native_store.misses()
        return self._python_misses

    @property
    def removed_voxels(self):
        if self._native_store is not None:
            return self._native_store.removed_voxels
        return self._python_removed

    def clear(self):
        super().clear()
        self._python_misses.clear()
        self._python_removed = 0

    def add_points(self, points):
        added = super().add_points(points)
        if self._native_store is not None:
            return added
        # Keyframe geometry is positive evidence, even when voxel filtering
        # moved its centroid into a neighbouring cell of the raw return.
        arr = np.asarray(points, dtype=np.float64)
        if arr.size:
            keys = np.floor(arr[:, :3] * self._inv).astype(np.int64)
            for key in keys:
                self._misses.pop(tuple(key), None)
        return added

    def _select_rays(self, rays, distance):
        delta = rays[:, 3:] - rays[:, :3]
        az = np.floor(np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))).astype(int)
        el = np.floor(
            np.degrees(np.arctan2(delta[:, 2], np.linalg.norm(delta[:, :2], axis=1)))
        ).astype(int)
        origins = np.round(rays[:, :3] / 0.25).astype(np.int64)
        groups = np.column_stack((origins, az, el))
        if kernels is not None and self._native_store is not None:
            chosen = kernels.select_grouped(groups, distance)
        else:
            order = np.lexsort((distance, *groups[:, ::-1].T))
            chosen = order[np.r_[True, (groups[order][1:] != groups[order][:-1]).any(axis=1)]]
        if len(chosen) > self.max_rays:
            chosen = chosen[np.linspace(0, len(chosen) - 1, self.max_rays, dtype=int)]
        return rays[chosen], distance[chosen]

    def ingest_rays(self, rays):
        rays = np.asarray(rays, dtype=np.float64)
        if rays.ndim != 2 or rays.shape[1] != 6 or not np.isfinite(rays).all():
            raise ValueError("3D rays must be finite (N,6)")
        if not len(rays) or not len(self):
            return
        distance = np.linalg.norm(rays[:, 3:] - rays[:, :3], axis=1)
        keep = distance > 1e-6
        rays, distance = rays[keep], distance[keep]
        if not len(rays):
            return  # missing/zero-length returns never clear anything
        if self._native_store is not None:
            selected, lengths = self._select_rays(rays, distance)
            self._native_store.ingest_rays(
                rays, selected, lengths, self.max_ray_range, self.clear_observations
            )
            return
        # Every return protects its voxel and a one-cell neighbourhood, even
        # when its ray is excluded from the bounded clearing subset.
        hits = set(map(tuple, np.floor(rays[:, 3:] * self._inv).astype(np.int64)))
        neighbours = tuple(product((-1, 0, 1), repeat=3))

        def protected(key):
            x, y, z = key
            return any((x + dx, y + dy, z + dz) in hits for dx, dy, dz in neighbours)

        for key in list(self._misses):
            if protected(key):
                del self._misses[key]
        selected, lengths = self._select_rays(rays, distance)
        crossed = set()
        step = self.resolution * 0.5
        for start in range(0, len(selected), 128):
            batch, length = selected[start : start + 128], lengths[start : start + 128]
            # Stop two voxels before the measured surface, and never exceed
            # the configured range. No extrapolation beyond a return.
            limit = np.minimum(length - 2 * self.resolution, self.max_ray_range)
            if limit.max() <= self.resolution:
                continue
            travel = np.arange(self.resolution, limit.max(), step)
            xyz = (
                batch[:, None, :3]
                + (batch[:, 3:] - batch[:, :3])[:, None, :]
                * (travel[None, :] / length[:, None])[:, :, None]
            )
            scaled = xyz * self._inv
            keys = np.floor(scaled).astype(np.int64)
            fraction = scaled - keys
            interior = ((fraction > 0.2) & (fraction < 0.8)).all(axis=2)
            valid = interior & (travel[None, :] < limit[:, None])
            # Deduplicate once per observation, regardless of beam density.
            crossed.update(map(tuple, keys[valid]))
        self._apply_misses(key for key in self._cells.intersection(crossed) if not protected(key))

    def _apply_misses(self, keys):
        for key in keys:
            count = self._misses.get(key, 0) + 1
            if count >= self.clear_observations:
                self._cells.remove(key)
                self._misses.pop(key, None)
                self._python_sums.pop(key, None)  # A cleared voxel forgets its old surface.
                self._python_removed += 1
            else:
                self._misses[key] = count
