"""Timestamp-preserving preparation, body filtering and dual-lidar merging."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from robot_mapping.backends.fastlio import BackendScan
from robot_mapping.backends.ray_native import kernels
from robot_mapping.filters.body_filter import (
    BodyBox,
    filter_points_array,
    lidar_b_to_lidar_a_rt,
    transform_points_array,
    transform_xyz,
)


@dataclass(frozen=True)
class FrontendScan:
    """Native scan plus the source time for TF and integration time for LIO."""

    scan: BackendScan
    timestamp_ns: int
    integration_stamp: float


def apply_body_filter(
    scan: BackendScan, rotation: np.ndarray, translation: np.ndarray, box: BodyBox
) -> tuple[BackendScan, int]:
    scan_points = scan.points_array
    filtered_points, removed_count = filter_points_array(scan_points, rotation, translation, box)
    if removed_count == 0:
        return scan, 0
    return replace(
        scan, points_array=filtered_points, dropped_points=int(scan.dropped_points) + removed_count
    ), removed_count


def merge_back_into_front(
    front_scan: BackendScan,
    back_scan: BackendScan,
    *,
    rotation_front: np.ndarray,
    translation_front: np.ndarray,
    rotation_back: np.ndarray,
    translation_back: np.ndarray,
    body_box: BodyBox | None,
    max_scan_duration_s: float,
) -> tuple[BackendScan, int, int]:
    """Put back-lidar points into the front lidar frame and concatenate.

    Returns (merged_scan, body_removed_on_back, n_back_kept).
    """
    back_points = back_scan.points_array
    if len(back_points) == 0:
        return front_scan, 0, 0
    body_removed = 0
    if body_box is not None:
        back_points, body_removed = filter_points_array(
            back_points, rotation_back, translation_back, body_box
        )
        if len(back_points) == 0:
            return front_scan, body_removed, 0
    front_from_back_rotation, front_from_back_translation = lidar_b_to_lidar_a_rt(
        rotation_front, translation_front, rotation_back, translation_back
    )
    back_time_offset_ms = (float(back_scan.frame_stamp) - float(front_scan.frame_stamp)) * 1000.0
    # Keep actual capture times inside the primary scan's IMU interval. Clipping
    # times to the boundary gives out-of-window returns a fictitious pose.
    scan_end_ms = float(front_scan.scan_duration_s) * 1000.0
    if max_scan_duration_s > 0.0:
        scan_end_ms = min(scan_end_ms, max_scan_duration_s * 1000.0)
    front_points = front_scan.points_array
    if (
        kernels is not None
        and front_points.dtype == back_points.dtype == np.float32
        and front_points.shape[1] == back_points.shape[1]
        and np.isfinite(scan_end_ms)
        and abs(scan_end_ms) < 3.4e38
    ):
        # Preserve NumPy's matrix arithmetic, but compact time-valid back points
        # directly into the final buffer instead of copying entire scans twice.
        transformed_back_xyz = transform_xyz(
            back_points[:, :3], front_from_back_rotation, front_from_back_translation
        )
        origins = (
            back_points[:, 5:8] @ front_from_back_rotation.T + front_from_back_translation
            if back_points.shape[1] >= 8
            else np.empty((0, 3))
        )
        merged, time_removed, back_point_count = kernels.merge_prepared(
            front_points,
            back_points,
            transformed_back_xyz,
            origins,
            np.float32(back_time_offset_ms),
            np.float32(scan_end_ms),
        )
        if back_point_count == 0:
            return front_scan, body_removed, 0
    else:
        back_in_front = transform_points_array(
            back_points, front_from_back_rotation, front_from_back_translation
        )
        if back_points.shape[1] >= 8:
            back_in_front[:, 5:8] = (
                back_points[:, 5:8] @ front_from_back_rotation.T + front_from_back_translation
            )
        back_in_front[:, 4] += np.float32(back_time_offset_ms)
        in_window = (back_in_front[:, 4] >= 0.0) & (back_in_front[:, 4] <= scan_end_ms)
        time_removed = int(np.count_nonzero(~in_window))
        back_in_front = back_in_front[in_window]
        back_point_count = len(back_in_front)
        if back_point_count == 0:
            return front_scan, body_removed, 0
        merged = (
            back_in_front
            if len(front_points) == 0
            else np.concatenate([front_points, back_in_front], axis=0)
        )
    return (
        replace(
            front_scan,
            points_array=merged,
            raw_points=int(front_scan.raw_points) + int(back_scan.raw_points),
            dropped_points=(
                int(front_scan.dropped_points)
                + int(back_scan.dropped_points)
                + body_removed
                + time_removed
            ),
        ),
        body_removed,
        back_point_count,
    )
