"""Algorithm defaults in generic.json; user files overlay robot, source and switches."""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from .filters.body_filter import BodyBox

DEFAULT_CONFIG = Path(__file__).parent / "resources" / "generic.json"


@dataclass(frozen=True)
class MappingSettings:
    config_path: Path
    maps_root: Path
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
    body_filter_enabled: bool
    body_filter_mode: str
    body_urdf_file: Path
    body_package_path: Path
    body_collision_threshold: float
    body_tf_timeout_s: float
    front_box: BodyBox
    back_box: BodyBox
    grid_resolution: float
    grid_min_height: float
    grid_max_height: float
    grid_margin_cells: int
    keyframe_dist_th: float
    keyframe_angle_th: float
    voxel_resolution: float
    free_radius_m: float
    map_point_stride: int
    live_png: bool
    live_http_port: int
    live_refresh_hz: float
    loop_enabled: bool
    loop_search_radius_m: float
    loop_min_keyframe_gap: int
    loop_neighbor_keyframes: int
    loop_max_correction_m: float
    loop_max_correction_rad: float
    loop_min_inlier_ratio: float
    loop_max_rmse_m: float
    loop_vgicp_resolution: float
    loop_vgicp_max_correspondence_distance: float
    loop_vgicp_num_threads: int
    loop_odom_sigmas: tuple[float, float, float, float, float, float]
    loop_loop_sigmas: tuple[float, float, float, float, float, float]

    @property
    def needs_back(self) -> bool:
        return self.lidar == "both"

    @property
    def primary_lidar(self) -> str:
        return "front" if self.lidar == "both" else self.lidar


def _as_six(value: Any, name: str) -> tuple[float, float, float, float, float, float]:
    seq = list(value)
    if len(seq) != 6:
        raise ValueError(f"{name} 需要 6 个数，实际 {len(seq)}")
    return tuple(float(v) for v in seq)  # type: ignore[return-value]


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def algorithm_profile() -> dict[str, Any]:
    return json.loads(files("robot_mapping.resources").joinpath("generic.json").read_text())


def resolve_profile(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Algorithm defaults, then the user overlay. Saved full snapshots merge idempotently."""
    return _deep_merge(algorithm_profile(), profile or {})


def load_mapping_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        if str(path).endswith(".json"):
            data = json.load(f)
        else:
            import yaml
            data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置不是 mapping: {path}")
    return data


def _read_body_box(body: dict[str, Any], name: str) -> BodyBox:
    nested = body.get(name)
    if not isinstance(nested, dict):
        raise ValueError(f"body_filter.{name} 缺少盒子边界")
    return BodyBox(**{key: float(nested[key]) for key in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")})


def mapping_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = resolve_profile(cfg)
    lio = cfg["lio"]
    body = cfg["body_filter"]
    grid = cfg["grid"]
    online = cfg["online"]
    loop = online["loop"]
    session = cfg.get("session") or {}
    maps_root = session.get("maps_root", "maps")
    return {
        "maps_root": str(Path.cwd() / maps_root),
        "lidar": str(lio["lidar"]),
        "imu": str(lio["imu"]),
        "lidar_ready": float(lio["lidar_ready"]),
        "cube_side_length": float(lio["cube_side_length"]),
        "det_range": float(lio["det_range"]),
        "filter_size_map": float(lio["filter_size_map"]),
        "filter_size_surf": float(lio["filter_size_surf"]),
        "blind": float(lio["blind"]),
        "print_interval": float(lio["print_interval"]),
        "max_back_dt": float(lio["max_back_dt"]),
        "wait_imu_s": float(lio["wait_imu_s"]),
        "min_imu_samples": int(lio["min_imu_samples"]),
        "max_scan_duration": float(lio["max_scan_duration"]),
        "max_lidar_frame_gap": float(lio["max_lidar_frame_gap"]),
        "warmup_frames": int(lio["warmup_frames"]),
        "match_distance": float(lio["match_distance"]),
        "match_iterations": int(lio["match_iterations"]),
        "plane_fit_mode": str(lio["plane_fit_mode"]),
        "time_source": str(lio["time_source"]),
        "require_imu": bool(lio["require_imu"]),
        "body_filter_enabled": bool(body["enabled"]),
        "body_filter_mode": str(body["mode"]),
        "body_urdf_file": str(body.get("urdf_file", "")),
        "body_package_path": str(body.get("package_path", "")),
        "body_collision_threshold": float(body["collision_threshold"]),
        "body_tf_timeout_s": float(body["tf_timeout_s"]),
        "front_box": _read_body_box(body, "front"),
        "back_box": _read_body_box(body, "back"),
        "grid_resolution": float(grid["resolution"]),
        "grid_min_height": float(grid["min_height"]),
        "grid_max_height": float(grid["max_height"]),
        "grid_margin_cells": int(grid["margin_cells"]),
        "keyframe_dist_th": float(online["keyframe_dist_th"]),
        "keyframe_angle_th": float(online["keyframe_angle_th"]),
        "voxel_resolution": float(online["voxel_resolution"]),
        "free_radius_m": float(online["free_radius_m"]),
        "map_point_stride": int(online["map_point_stride"]),
        "live_png": bool(online["live_png"]),
        "live_http_port": int(online["live_http_port"]),
        "live_refresh_hz": float(online["live_refresh_hz"]),
        "loop_enabled": bool(loop["enabled"]),
        "loop_search_radius_m": float(loop["search_radius_m"]),
        "loop_min_keyframe_gap": int(loop["min_keyframe_gap"]),
        "loop_neighbor_keyframes": int(loop["neighbor_keyframes"]),
        "loop_max_correction_m": float(loop["max_correction_m"]),
        "loop_max_correction_rad": float(loop["max_correction_rad"]),
        "loop_min_inlier_ratio": float(loop["min_inlier_ratio"]),
        "loop_max_rmse_m": float(loop["max_rmse_m"]),
        "loop_vgicp_resolution": float(loop["vgicp_resolution"]),
        "loop_vgicp_max_correspondence_distance": float(loop["vgicp_max_correspondence_distance"]),
        "loop_vgicp_num_threads": int(loop["vgicp_num_threads"]),
        "loop_odom_sigmas": _as_six(loop["odom_sigmas"], "online.loop.odom_sigmas"),
        "loop_loop_sigmas": _as_six(loop["loop_sigmas"], "online.loop.loop_sigmas"),
    }


def settings_from_profile(profile: dict[str, Any], config_path: str | Path, **overrides) -> MappingSettings:
    """Build settings without command-line parsing; validate once at session entry."""
    import os

    path = Path(config_path).resolve()
    values = mapping_defaults(profile)
    values.update(overrides)
    for name in ("maps_root", "body_urdf_file", "body_package_path"):
        value = Path(os.path.expandvars(str(values[name]))).expanduser()
        values[name] = value if value.is_absolute() or name == "maps_root" else path.parent / value
    return MappingSettings(config_path=path, **values)


def load_settings(path: str | Path = DEFAULT_CONFIG, **overrides) -> MappingSettings:
    """Read a user overlay (or a full saved snapshot) on top of algorithm defaults."""
    return settings_from_profile(load_mapping_config(path), path, **overrides)


def validate_settings(settings: MappingSettings) -> None:
    for name,value in vars(settings).items():
        if isinstance(value,float) and not math.isfinite(value):
            raise ValueError(f"{name} 必须为有限数值")
    if settings.lidar not in {"front", "back", "both"}:
        raise ValueError("lio.lidar must be front, back or both")
    if settings.imu not in {"front", "back", "chassis"}:
        raise ValueError("lio.imu must be front, back or chassis")
    if settings.time_source != "header":
        raise ValueError("standalone mapping requires header/source time")
    if settings.grid_min_height > settings.grid_max_height:
        raise ValueError("grid min_height must not exceed max_height")
    if settings.body_filter_mode not in {"urdf", "box"}:
        raise ValueError("body_filter.mode 必须为 urdf 或 box")
    if settings.body_collision_threshold < 0:
        raise ValueError("body collision_threshold 必须为有限非负数")
    if settings.body_tf_timeout_s <= 0:
        raise ValueError("body tf_timeout_s 必须为有限正数")
    if settings.lidar_ready < 0:
        raise ValueError("--lidar-ready 不能小于 0")
    if settings.cube_side_length <= 0 or settings.det_range <= 0:
        raise ValueError("--cube-side-length / --det-range 必须大于 0")
    if settings.filter_size_map < 0 or settings.filter_size_surf < 0:
        raise ValueError("filter-size 不能小于 0")
    if settings.blind < 0:
        raise ValueError("--blind 不能小于 0")
    if settings.grid_resolution <= 0:
        raise ValueError("--grid-resolution 必须大于 0")
    if settings.grid_margin_cells < 0:
        raise ValueError("--grid-margin-cells 不能小于 0")
    if settings.keyframe_dist_th < 0 or settings.keyframe_angle_th < 0:
        raise ValueError("keyframe 阈值不能小于 0")
    if settings.voxel_resolution <= 0:
        raise ValueError("--voxel-resolution 必须大于 0")
    if settings.live_http_port < 0:
        raise ValueError("--live-http-port 不能小于 0")
    for prefix in ("front", "back"):
        box = getattr(settings, f"{prefix}_box")
        if box.min_x >= box.max_x or box.min_y >= box.max_y or box.min_z >= box.max_z:
            raise ValueError(f"{prefix} body box min 必须小于 max")
