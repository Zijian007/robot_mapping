"""Height-binned occupancy evidence from measured, deskewed lidar rays.

The exported map stays 2D. Independent vertical evidence prevents a ray above
an obstacle from erasing it. Evidence is updated once per cell per observation;
current hits win over free rays in that observation. The default height_complete
mode requires negative evidence in every height bin. Projected mode maintains
separate repeated planar visibility evidence with all-height obstacle vetoes.
Polar mode keeps the same planar evidence but, per observation and per angular
sector around each sensor origin, clears only toward the nearest in-band return
(vendor Mapping2D style): what lies behind the first obstacle at any height in
that sector is not cleared by the same observation.
No mode establishes a continuous swept-volume or floor-support guarantee.
"""

from __future__ import annotations

import numpy as np

from robot_mapping.backends.ray_native import kernels

from .grid import GDK_FREE, GDK_OCCUPIED, GDK_UNKNOWN
from .spatial import GrowingOccupancy2D

FREE_SPACE_MODES = ("height_complete", "projected", "polar")
PLANAR_MODES = ("projected", "polar")
SCORE_RULES = ("score_hit", "score_miss", "score_max", "score_min", "score_occupied")


def validate_score_rules(score_hit, score_miss, score_max, score_min, score_occupied):
    """Check per-layer evidence rules; scores are stored as int8."""
    values = dict(
        score_hit=score_hit,
        score_miss=score_miss,
        score_max=score_max,
        score_min=score_min,
        score_occupied=score_occupied,
    )
    for name, value in values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if not 1 <= score_hit <= 127:
        raise ValueError("score_hit must be in [1, 127]")
    if not 1 <= score_miss <= 127:
        raise ValueError("score_miss must be in [1, 127]")
    if not -128 <= score_min <= -1:
        raise ValueError("score_min must be in [-128, -1]")
    if not 1 <= score_occupied <= score_max <= 127:
        raise ValueError(
            "score_occupied/score_max must satisfy 1 <= score_occupied <= score_max <= 127"
        )
    return tuple(int(value) for value in values.values())


class RayOccupancy2D(GrowingOccupancy2D):
    def __init__(
        self,
        *,
        height_resolution=0.2,
        max_ray_range=30.0,
        max_rays=2048,
        free_space_mode="height_complete",
        projected_min_observations=2,
        projected_max_range=None,
        polar_bins=180,
        polar_closest_margin=0.2,
        score_hit=2,
        score_miss=1,
        score_max=8,
        score_min=-4,
        score_occupied=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        (
            self.score_hit,
            self.score_miss,
            self.score_max,
            self.score_min,
            self.score_occupied,
        ) = validate_score_rules(score_hit, score_miss, score_max, score_min, score_occupied)
        if (
            not np.isfinite([height_resolution, max_ray_range]).all()
            or min(height_resolution, max_ray_range) <= 0
        ):
            raise ValueError("ray height resolution and range must be finite and positive")
        if max_rays < 1:
            raise ValueError("max_rays must be positive")
        self.height_resolution = float(height_resolution)
        self.max_ray_range = float(max_ray_range)
        self.max_rays = int(max_rays)
        if free_space_mode not in FREE_SPACE_MODES:
            raise ValueError("unknown free_space_mode")
        self.free_space_mode = free_space_mode
        if (
            isinstance(projected_min_observations, (bool, np.bool_))
            or not isinstance(projected_min_observations, (int, np.integer))
            or not 1 <= projected_min_observations <= 127
        ):
            raise ValueError("projected_min_observations must be an integer in [1, 127]")
        self.projected_min_observations = int(projected_min_observations)
        self.projected_max_range = float(
            max_ray_range if projected_max_range is None else projected_max_range
        )
        if not np.isfinite(self.projected_max_range) or self.projected_max_range <= 0:
            raise ValueError("projected_max_range must be finite and positive")
        if (
            isinstance(polar_bins, (bool, np.bool_))
            or not isinstance(polar_bins, (int, np.integer))
            or not 8 <= polar_bins <= 3600
        ):
            raise ValueError("polar_bins must be an integer in [8, 3600]")
        self.polar_bins = int(polar_bins)
        if not np.isfinite(polar_closest_margin) or polar_closest_margin < 0:
            raise ValueError("polar_closest_margin must be finite and nonnegative")
        self.polar_closest_margin = float(polar_closest_margin)
        self._projected_counts = None
        self._scores = None
        self._last_footprint_xy = None

    def clear(self):
        super().clear()
        self._scores = None
        self._last_footprint_xy = None
        self._projected_counts = None

    def _ensure_seed(self, x, y):
        super()._ensure_seed(x, y)
        if self._scores is None:
            layers = max(
                1, int(np.ceil((self.max_height - self.min_height) / self.height_resolution))
            )
            self._scores = np.zeros((self._height, self._width, layers), dtype=np.int8)
            if self.free_space_mode in PLANAR_MODES:
                self._projected_counts = np.zeros((self._height, self._width), dtype=np.int8)

    def _expand_to_include(self, xs, ys):
        old_x, old_y = self._origin_x, self._origin_y
        old_h, old_w = self._height, self._width
        super()._expand_to_include(xs, ys)
        if (old_h, old_w) != (self._height, self._width):
            left = round((old_x - self._origin_x) / self.resolution)
            bottom = round((old_y - self._origin_y) / self.resolution)
            self._scores = np.pad(
                self._scores,
                (
                    (bottom, self._height - old_h - bottom),
                    (left, self._width - old_w - left),
                    (0, 0),
                ),
            )
            if self._projected_counts is not None:
                self._projected_counts = np.pad(
                    self._projected_counts,
                    ((bottom, self._height - old_h - bottom), (left, self._width - old_w - left)),
                )

    def _project_free(self, rays, hits):
        """Planar visibility model: repeated observations, all-height obstacle veto.

        Height scores still clear obstacles only with same-height measurements.
        Planar free space assumes a static observed scene; it does not assert
        that unseen heights or the supporting floor have been measured.
        """
        counts = self._projected_counts.reshape(-1)
        positive = (self._scores > 0).any(axis=2).ravel()
        hit_columns = np.unique(hits // self._scores.shape[2])
        positive[hit_columns] = True  # even a hit into previously negative evidence
        counts[positive] = 0
        seen = np.zeros(counts.shape, dtype=bool)
        for beam in self._select_rays(rays):
            # Out-of-band returns do not justify planar free space.
            if not (
                self.min_height <= beam[2] <= self.max_height
                and self.min_height <= beam[5] <= self.max_height
            ):
                continue
            delta = beam[3:] - beam[:3]
            distance = np.linalg.norm(delta)
            length = min(distance, self.max_ray_range)
            endpoint = beam[:3] + delta * min(1.0, self.max_ray_range / distance)
            row0, col0 = self._world_to_rc(beam[0], beam[1])
            row1, col1 = self._world_to_rc(endpoint[0], endpoint[1])
            dr, dc = int(row1 - row0), int(col1 - col0)
            steps = max(abs(dr), abs(dc))
            for k in range(1, steps):
                if (k / steps) * length >= self.projected_max_range:
                    break
                row = int(row0) + (k * dr + steps // 2) // steps
                col = int(col0) + (k * dc + steps // 2) // steps
                if not (0 <= row < self._height and 0 <= col < self._width):
                    continue
                column = row * self._width + col
                if positive[column]:
                    break  # known obstacle at ANY height occludes the planar ray
                seen[column] = True
        counts[seen] = np.minimum(
            counts[seen].astype(np.int16) + 1, self.projected_min_observations
        )
        return np.flatnonzero(seen)

    def _polar_rays(self, rays, distances):
        """In-band rays cut at the nearest in-band return of their sensor sector.

        Sectors are `polar_bins` equal bearings around each sensor origin
        (origins grouped like `_ray_groups`). A return within
        `polar_closest_margin` of the sector's nearest keeps its full length;
        a farther one is cut to the nearest planar range, because what lies
        beyond the sector's first obstacle is not observed in the plane.
        Returned endpoints are never cleared themselves.
        """
        band = (
            (rays[:, 2] >= self.min_height)
            & (rays[:, 2] <= self.max_height)
            & (rays[:, 5] >= self.min_height)
            & (rays[:, 5] <= self.max_height)
            & (distances <= self.max_ray_range)
        )
        planar = np.hypot(rays[:, 3] - rays[:, 0], rays[:, 4] - rays[:, 1])
        band &= planar > 1e-6
        rays, planar = rays[band], planar[band]
        if not len(rays):
            return rays
        bearing = np.arctan2(rays[:, 4] - rays[:, 1], rays[:, 3] - rays[:, 0])
        bins = np.floor((bearing + np.pi) / (2 * np.pi) * self.polar_bins).astype(np.int64)
        bins %= self.polar_bins
        # One int64 key per (origin group, sector); 15 bits per origin axis covers +-4 km.
        origin = np.round(rays[:, :3] / 0.25).astype(np.int64) + (1 << 14)
        if (origin < 0).any() or (origin >= 1 << 15).any():
            raise ValueError("ray origin outside the supported +-4 km range")
        keys = ((origin[:, 0] << 30) | (origin[:, 1] << 15) | origin[:, 2]) * self.polar_bins + bins
        _, sector = np.unique(keys, return_inverse=True)
        sector = sector.reshape(-1)
        nearest = np.full(sector.max() + 1, np.inf)
        np.minimum.at(nearest, sector, planar)
        limit = nearest[sector]
        cut = planar > limit + self.polar_closest_margin
        if cut.any():
            rays = rays.copy()
            rays[cut, 3:] = (
                rays[cut, :3]
                + (rays[cut, 3:] - rays[cut, :3]) * (limit[cut] / planar[cut])[:, None]
            )
        # Rays sharing an origin cell and an endpoint cell rasterize identically.
        r0, c0 = self._world_to_rc(rays[:, 0], rays[:, 1])
        r1, c1 = self._world_to_rc(rays[:, 3], rays[:, 4])
        cell_key = (np.asarray(r0, np.int64) * self._width + c0) * (self._height * self._width) + (
            np.asarray(r1, np.int64) * self._width + c1
        )
        _, first = np.unique(cell_key, return_index=True)
        return rays[np.sort(first)]

    def _polar_update(self, rays, distances):
        """Add planar evidence along polar rays and mark newly qualified columns free."""
        depth = self._scores.shape[2]
        counts = self._projected_counts.reshape(-1)
        scores = self._scores.reshape(-1, depth)
        hit_columns = np.unique(self._keys(rays[distances <= self.max_ray_range, 3:]) // depth)
        counts[hit_columns] = 0
        selected = self._polar_rays(rays, distances)
        if len(selected):
            origin, endpoint = selected[:, :2], selected[:, 3:5]
            planar = np.linalg.norm(endpoint - origin, axis=1)
            # Rasterize only up to the planar range limit; the endpoint cell is never cleared.
            scale = np.minimum(1.0, self.projected_max_range / planar)
            endpoint = origin + (endpoint - origin) * scale[:, None]
            r0, c0 = (
                np.asarray(v, np.int64) for v in self._world_to_rc(origin[:, 0], origin[:, 1])
            )
            r1, c1 = (
                np.asarray(v, np.int64) for v in self._world_to_rc(endpoint[:, 0], endpoint[:, 1])
            )
            # Obstacle test at any height, evaluated once inside the window the rays cover.
            top, bottom = (
                max(0, int(min(r0.min(), r1.min()))),
                min(self._height, int(max(r0.max(), r1.max())) + 1),
            )
            left, right = (
                max(0, int(min(c0.min(), c1.min()))),
                min(self._width, int(max(c0.max(), c1.max())) + 1),
            )
            window_w = right - left
            stop_mask = (self._scores[top:bottom, left:right] > 0).any(axis=2)
            hr, hc = np.divmod(hit_columns, self._width)
            inside = (hr >= top) & (hr < bottom) & (hc >= left) & (hc < right)
            # A current hit vetoes even before its score becomes positive.
            stop_mask[hr[inside] - top, hc[inside] - left] = True
            stop_mask = stop_mask.reshape(-1)
            seen = []
            for begin in range(0, len(selected), 512):
                part = slice(begin, begin + 512)
                dr, dc = r1[part] - r0[part], c1[part] - c0[part]
                steps = np.maximum(np.abs(dr), np.abs(dc))
                if steps.max() < 2:
                    continue
                k = np.arange(1, steps.max(), dtype=np.int64)[None, :]
                n = np.maximum(steps[:, None], 1)
                rows = r0[part, None] + (k * dr[:, None] + n // 2) // n
                cols = c0[part, None] + (k * dc[:, None] + n // 2) // n
                valid = (
                    (k < steps[:, None])
                    & (rows >= top)
                    & (rows < bottom)
                    & (cols >= left)
                    & (cols < right)
                )
                local = np.where(valid, (rows - top) * window_w + (cols - left), 0)
                # Everything from the first known obstacle (any height) onward is occluded.
                keep = valid & (np.cumsum(valid & stop_mask[local], axis=1) == 0)
                seen.append(rows[keep] * self._width + cols[keep])
            if seen:
                seen = np.unique(np.concatenate(seen))
                counts[seen] = np.minimum(
                    counts[seen].astype(np.int16) + 1, self.projected_min_observations
                )
        # The height update may have reset earlier planar-free columns anywhere in range.
        values = self._values.reshape(-1)
        candidates = np.flatnonzero(
            (counts >= self.projected_min_observations)
            & (values != GDK_FREE)
            & (values != GDK_OCCUPIED)
        )
        free = candidates[~(scores[candidates] > 0).any(axis=1)]
        values[free] = GDK_FREE

    def ingest_points(self, xyz, robot_xy=None):
        raise ValueError("ray occupancy requires measured origins and endpoints; use ingest_rays")

    def _mark_free_disk(self, x, y, hit_columns=None):
        # Retain the existing current-footprint prior, but even a single-hit
        # candidate takes precedence over it. Work on a local patch only.
        row, col = self._world_to_rc(np.asarray(x), np.asarray(y))
        radius = max(1, int(np.ceil(self.free_radius_m / self.resolution)))
        r0, r1 = max(0, row - radius), min(self._height, row + radius + 1)
        c0, c1 = max(0, col - radius), min(self._width, col + radius + 1)
        patch = self._values[r0:r1, c0:c1]
        positive = (self._scores[r0:r1, c0:c1] > 0).any(axis=2)
        if hit_columns is not None:
            columns = np.arange(r0, r1)[:, None] * self._width + np.arange(c0, c1)
            positive |= np.isin(columns, hit_columns)
        rr, cc = np.ogrid[r0:r1, c0:c1]
        disk = (rr - row) ** 2 + (cc - col) ** 2 <= radius**2
        patch[disk & (patch == GDK_UNKNOWN) & ~positive] = GDK_FREE

    def _keys(self, xyz):
        rows, cols = self._world_to_rc(xyz[:, 0], xyz[:, 1])
        layers = np.floor((xyz[:, 2] - self.min_height) / self.height_resolution).astype(np.int64)
        layers = np.minimum(layers, self._scores.shape[2] - 1)  # inclusive upper boundary
        valid = (
            (xyz[:, 2] >= self.min_height)
            & (xyz[:, 2] <= self.max_height)
            & (rows >= 0)
            & (rows < self._height)
            & (cols >= 0)
            & (cols < self._width)
        )
        return (rows[valid].astype(np.int64) * self._width + cols[valid]) * self._scores.shape[
            2
        ] + layers[valid]

    def _select_rays(self, rays):
        groups = self._ray_groups(rays)
        distances = np.linalg.norm(rays[:, 3:] - rays[:, :3], axis=1)
        if kernels is not None:
            chosen = kernels.select_grouped(groups, distances)
        else:
            origins, azimuth, layer = groups[:, :3], groups[:, 3], groups[:, 4]
            origin_order = np.lexsort((origins[:, 2], origins[:, 1], origins[:, 0]))
            ordered_origins = origins[origin_order]
            boundaries = np.r_[True, (ordered_origins[1:] != ordered_origins[:-1]).any(axis=1)]
            sensors = np.empty(len(rays), dtype=np.int64)
            sensors[origin_order] = np.cumsum(boundaries) - 1
            key = (sensors * 360 + azimuth) * (self._scores.shape[2] + 2) + layer + 1
            order = np.lexsort((distances, key))
            chosen = order[np.r_[True, np.diff(key[order]) != 0]]
        if len(chosen) > self.max_rays:
            chosen = chosen[np.linspace(0, len(chosen) - 1, self.max_rays, dtype=int)]
        return rays[chosen]

    def _free_keys(self, rays):
        if kernels is not None:
            origin, endpoint = rays[:, :3], rays[:, 3:]
            delta = endpoint - origin
            scale = np.minimum(
                1.0, self.max_ray_range / np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
            )
            endpoint = origin + delta * scale[:, None]
            r0, c0 = self._world_to_rc(origin[:, 0], origin[:, 1])
            r1, c1 = self._world_to_rc(endpoint[:, 0], endpoint[:, 1])
            return kernels.grid_free_keys(
                np.column_stack((r0, c0, r1, c1)),
                np.column_stack((origin[:, 2], endpoint[:, 2])),
                self.min_height,
                self.max_height,
                self.height_resolution,
                self._scores.shape[2],
                self._width,
                self._height,
            )
        result = []
        depth = self._scores.shape[2]
        # Bound temporary matrices; vectorize the ray traversal within each batch.
        for start in range(0, len(rays), 128):
            batch = rays[start : start + 128]
            origin, endpoint = batch[:, :3], batch[:, 3:]
            delta = endpoint - origin
            scale = np.minimum(
                1.0, self.max_ray_range / np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
            )
            endpoint = origin + delta * scale[:, None]
            r0, c0 = self._world_to_rc(origin[:, 0], origin[:, 1])
            r1, c1 = self._world_to_rc(endpoint[:, 0], endpoint[:, 1])
            dr, dc = r1 - r0, c1 - c0
            steps = np.maximum(abs(dr), abs(dc)).astype(np.int64)
            if not steps.max():
                continue
            k = np.arange(1, steps.max(), dtype=np.int64)[None, :]
            n = np.maximum(steps[:, None], 1)
            # Integer rasterization (Bresenham-equivalent rounding), excluding returns.
            rows = r0[:, None] + (k * dr[:, None] + n // 2) // n
            cols = c0[:, None] + (k * dc[:, None] + n // 2) // n
            z = origin[:, 2, None] + k / n * (endpoint[:, 2] - origin[:, 2])[:, None]
            layers = np.minimum(
                np.floor((z - self.min_height) / self.height_resolution).astype(int), depth - 1
            )
            valid = (
                (k < steps[:, None])
                & (z >= self.min_height)
                & (z <= self.max_height)
                & (rows >= 0)
                & (rows < self._height)
                & (cols >= 0)
                & (cols < self._width)
            )
            result.append(((rows * self._width + cols) * depth + layers)[valid])
        return np.unique(np.concatenate(result)) if result else np.empty(0, dtype=np.int64)

    def _ray_groups(self, rays):
        # Keep NumPy's exact angular bins and round-to-even origin grouping.
        # Both backends select nearest rays and apply the same bounded subset.
        delta = rays[:, 3:] - rays[:, :3]
        azimuth = (
            np.floor((np.arctan2(delta[:, 1], delta[:, 0]) + np.pi) * 180 / np.pi).astype(int) % 360
        )
        layer = np.clip(
            np.floor((rays[:, 5] - self.min_height) / self.height_resolution),
            -1,
            self._scores.shape[2],
        ).astype(int)
        origins = np.round(rays[:, :3] / 0.25)
        return np.column_stack((origins, azimuth, layer)).astype(np.int64)

    def ingest_rays(self, rays, robot_xy=None):
        if self.free_space_mode != "polar":
            return self._ingest_rays(rays, robot_xy)
        # Polar: height evidence and height-complete free space as usual, then
        # planar evidence on top. A column is free when either rule holds; both
        # require that no height layer has positive evidence.
        prepared = self._ingest_rays(rays, robot_xy)
        if prepared is not None:
            self._polar_update(*prepared)

    def _ingest_rays(self, rays, robot_xy=None):
        """Height evidence and the projected/height-complete value rule.

        Returns the filtered rays and their lengths, or None when nothing was ingested.
        """
        rays = np.asarray(rays, dtype=np.float64)
        if rays.ndim != 2 or rays.shape[1] != 6 or not np.isfinite(rays).all():
            raise ValueError("rays must be finite (N,6): origin xyz, endpoint xyz")
        if not len(rays):
            return  # no return is not evidence of free space
        origin, endpoint = rays[:, :3], rays[:, 3:]
        distances = np.linalg.norm(endpoint - origin, axis=1)
        keep = distances > 1e-6
        rays, distances = rays[keep], distances[keep]
        if not len(rays):
            return
        self._ensure_seed(*(robot_xy if robot_xy is not None else rays[0, :2]))
        clipped = (
            rays[:, :3]
            + (rays[:, 3:] - rays[:, :3]) * np.minimum(1.0, self.max_ray_range / distances)[:, None]
        )
        extent = np.concatenate((rays[:, :3], clipped), axis=0)
        if robot_xy is not None:
            extent = np.vstack((extent, [*robot_xy, 0.0]))
        self._expand_to_include(extent[:, 0], extent[:, 1])
        # After any growth (padding reallocates). Polar evidence is added by ingest_rays.
        projected = self._projected_counts if self.free_space_mode == "projected" else None
        if kernels is not None:
            kernels.grid_update(
                rays,
                distances,
                self._ray_groups(rays),
                self._scores,
                self._values,
                self._origin_x,
                self._origin_y,
                self.resolution,
                self.min_height,
                self.max_height,
                self.height_resolution,
                self.max_ray_range,
                self.max_rays,
                self.free_radius_m,
                self._last_footprint_xy,
                None if robot_xy is None else tuple(robot_xy),
                projected,
                self.projected_min_observations,
                self.projected_max_range,
                self.score_hit,
                self.score_miss,
                self.score_min,
                self.score_max,
                self.score_occupied,
            )
            self._last_footprint_xy = None if robot_xy is None else tuple(robot_xy)
            return rays, distances
        hits = np.unique(self._keys(rays[distances <= self.max_ray_range, 3:]))
        misses = np.setdiff1d(self._free_keys(self._select_rays(rays)), hits, assume_unique=True)
        scores = self._scores.reshape(-1)
        # Widen before arithmetic: configured steps may exceed the int8 range.
        scores[misses] = np.maximum(
            scores[misses].astype(np.int16) - self.score_miss, self.score_min
        )
        scores[hits] = np.minimum(scores[hits].astype(np.int16) + self.score_hit, self.score_max)
        # A layer is occupied at score_occupied (default: two distinct hits). Lower
        # candidates stay unknown, even if another height in the same column is free.
        # Recompute only columns touched by this observation. Include the prior
        # footprint so its temporary unknown->free paint keeps the old semantics.
        depth = self._scores.shape[2]
        columns = [hits // depth, misses // depth]
        if projected is not None:
            columns.append(self._project_free(rays, hits))
        if self._last_footprint_xy is not None:
            row, col = self._world_to_rc(*np.asarray(self._last_footprint_xy))
            radius = max(1, int(np.ceil(self.free_radius_m / self.resolution)))
            rr = np.arange(max(0, row - radius), min(self._height, row + radius + 1))
            cc = np.arange(max(0, col - radius), min(self._width, col + radius + 1))
            columns.append((rr[:, None] * self._width + cc).ravel())
        columns = np.unique(np.concatenate(columns))
        changed = self._scores.reshape(-1, depth)[columns]
        occupied = (changed >= self.score_occupied).any(axis=1)
        free = (changed < 0).all(axis=1)
        if projected is not None:
            free = (projected.reshape(-1)[columns] >= self.projected_min_observations) & ~(
                changed > 0
            ).any(axis=1)
        values = self._values.reshape(-1)
        values[columns] = GDK_UNKNOWN
        values[columns[free]] = GDK_FREE
        values[columns[occupied]] = GDK_OCCUPIED
        if robot_xy is not None:
            self._mark_free_disk(
                *robot_xy, hit_columns=hits // depth if projected is not None else None
            )
        self._last_footprint_xy = None if robot_xy is None else tuple(robot_xy)
        return rays, distances
