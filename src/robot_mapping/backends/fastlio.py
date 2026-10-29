"""Optional FAST-LIO2 construction; no device SDK or robot assets."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from robot_mapping.types import LidarScan


@dataclass(frozen=True)
class BackendScan:
    points_array: np.ndarray
    frame_stamp: float  # BEGIN here; the frontend normalizes to END exactly once.
    scan_duration_s: float
    raw_points: int
    dropped_points: int


def prepare_scan(scan: LidarScan, blind: float):
    keep = np.sum(scan.xyz.astype(np.float64)**2, axis=1) > blind**2
    array = np.empty((int(keep.sum()),5), dtype=np.float32)
    array[:,:3] = scan.xyz[keep]
    array[:,3] = scan.intensity[keep]
    array[:,4] = scan.point_offsets_ns[keep]*1e-6
    return BackendScan(array, scan.begin_ns*1e-9, scan.duration_ns*1e-9,
                       len(scan.xyz), int((~keep).sum()))


def create_backend(settings, imu_from_lidar):
    from .capabilities import load_fastlio2
    fastlio2 = load_fastlio2()

    mapping = {
        "scan_match_stride": 1,
        "map_add_stride": 1,
        "match_distance": settings.match_distance,
        "match_iterations": settings.match_iterations,
        "cube_side_length": settings.cube_side_length,
        "det_range": settings.det_range,
        "filter_size_surf": settings.filter_size_surf,
        "filter_size_map": settings.filter_size_map,
        "plane_fit_mode": settings.plane_fit_mode,
        "extrinsic_est_en": False,
        "export_scan_points": True,
    }
    config = fastlio2.FastLIO2Config(
        blind=settings.blind,
        point_filter_num=1,
        build_point_list=False,
        max_scan_duration_s=settings.max_scan_duration,
        extrinsic_T=tuple(imu_from_lidar[:3, 3]),
        extrinsic_R=tuple(imu_from_lidar[:3, :3].ravel()),
        mapping=mapping,
    )
    return fastlio2.FastLIO2(config, require_native=True)
