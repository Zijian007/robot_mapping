"""Projection of accepted observations shared by live ingestion and loop rebuild."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from .map_types import GridObservation, KeyframeRecord
from .pose_math import transform_points
from .ray_grid import RayOccupancy2D
from .ray_voxels import RayVoxelHash
from .spatial import GrowingOccupancy2D, WorldVoxelHash


def project_observation(
    voxels: WorldVoxelHash | RayVoxelHash,
    grid: GrowingOccupancy2D | RayOccupancy2D,
    keyframe_points: np.ndarray | None,
    map_from_tracking: np.ndarray,
    tracking_from_footprint: np.ndarray,
    local_rays: np.ndarray | None = None,
    *,
    clear_voxels: bool = False,
) -> dict[str, float]:
    """Project one accepted observation, for both ingress and loop replay.

    Only keyframes supply geometry. Transform measured rays once and share
    them between the independent 2D and 3D evidence accumulators.
    """
    stage_started_at = time.perf_counter()
    if keyframe_points is not None:
        map_points = transform_points(map_from_tracking, keyframe_points)
        voxels.add_points(map_points)
    voxel_ms = (time.perf_counter() - stage_started_at) * 1000.0
    stage_started_at = time.perf_counter()
    # 2D evidence must retain original XY and Z. Projecting 15 cm voxel
    # centers onto 5 cm cells creates a 3-cell lattice and shifts heights.
    if local_rays is None:
        grid.ingest_points(
            map_points, robot_xy=tuple((map_from_tracking @ tracking_from_footprint)[:2, 3])
        )
        grid_ms = (time.perf_counter() - stage_started_at) * 1000.0
    else:
        map_rays = project_grid_rays(grid, local_rays, map_from_tracking, tracking_from_footprint)
        grid_ms = (time.perf_counter() - stage_started_at) * 1000.0
        if clear_voxels:
            stage_started_at = time.perf_counter()
            voxels.ingest_rays(map_rays)
            voxel_ms += (time.perf_counter() - stage_started_at) * 1000.0
    return {"grid_ms": grid_ms, "voxel_ms": voxel_ms}


def replay_observation(
    voxels: WorldVoxelHash | RayVoxelHash,
    grid: GrowingOccupancy2D | RayOccupancy2D,
    observation: GridObservation,
    keyframes: Sequence[KeyframeRecord],
    optimized_poses: Sequence[np.ndarray],
    tracking_from_footprint: np.ndarray,
    *,
    clear_voxels: bool = False,
) -> None:
    map_from_tracking = optimized_poses[observation.anchor_id] @ observation.anchor_from_tracking
    keyframe_points = (
        keyframes[observation.anchor_id].local_points if observation.keyframe else None
    )
    project_observation(
        voxels,
        grid,
        keyframe_points,
        map_from_tracking,
        tracking_from_footprint,
        observation.local_rays,
        clear_voxels=clear_voxels,
    )


def project_grid_rays(
    grid: RayOccupancy2D,
    local_rays: np.ndarray,
    map_from_tracking: np.ndarray,
    tracking_from_footprint: np.ndarray,
) -> np.ndarray:
    map_rays = np.column_stack(
        (
            transform_points(map_from_tracking, local_rays[:, :3]),
            transform_points(map_from_tracking, local_rays[:, 3:]),
        )
    )
    grid.ingest_rays(map_rays, robot_xy=tuple((map_from_tracking @ tracking_from_footprint)[:2, 3]))
    return map_rays
