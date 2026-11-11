"""Typed configuration domains shared by mapping algorithms and acquisition apps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .filters.body_filter import BodyBox


@dataclass(frozen=True)
class LioSettings:
    lidar: str
    imu: str
    lidar_ready: float
    cube_side_length: float
    det_range: float
    filter_size_map: float
    filter_size_surf: float
    blind: float
    print_interval: float
    max_back_dt: float
    wait_imu_s: float
    min_imu_samples: int
    max_scan_duration: float
    max_lidar_frame_gap: float
    warmup_frames: int
    match_distance: float
    match_iterations: int
    plane_fit_mode: str
    time_source: str
    require_imu: bool
    max_imu_gap_s: float
    max_imu_boundary_gap_s: float
    imu_startup_timeout_s: float
    max_match_failures: int

    @property
    def needs_back(self) -> bool:
        return self.lidar == "both"

    @property
    def primary_lidar(self) -> str:
        return "front" if self.lidar == "both" else self.lidar


@dataclass(frozen=True)
class BodyFilterSettings:
    enabled: bool
    mode: str
    urdf_file: Path
    package_path: Path
    collision_threshold: float
    tf_timeout_s: float
    front: BodyBox
    back: BodyBox


@dataclass(frozen=True)
class GridSettings:
    resolution: float
    min_height: float
    max_height: float
    margin_cells: int
    ray_clearing: bool
    update_interval_s: float
    height_resolution: float
    max_ray_range: float
    max_rays: int
    free_space_mode: str
    projected_min_observations: int
    projected_max_range: float


@dataclass(frozen=True)
class LoopSettings:
    enabled: bool
    search_radius_m: float
    min_keyframe_gap: int
    neighbor_keyframes: int
    max_correction_m: float
    max_correction_rad: float
    min_inlier_ratio: float
    max_rmse_m: float
    vgicp_resolution: float
    vgicp_max_correspondence_distance: float
    vgicp_num_threads: int
    odom_sigmas: tuple[float, float, float, float, float, float]
    loop_sigmas: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class OnlineSettings:
    keyframe_dist_th: float
    keyframe_angle_th: float
    voxel_resolution: float
    voxel_clearing: bool
    voxel_clear_observations: int
    voxel_max_rays: int
    free_radius_m: float
    map_point_stride: int
    live_png: bool
    live_http_port: int
    live_refresh_hz: float
    loop: LoopSettings


@dataclass(frozen=True)
class MappingSettings:
    config_path: Path
    maps_root: Path
    lio: LioSettings
    body_filter: BodyFilterSettings
    grid: GridSettings
    online: OnlineSettings
