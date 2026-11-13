"""Load user profiles, resolve defaults and serialize effective typed settings."""

from __future__ import annotations

import copy
import json
import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import asdict, fields
from importlib.resources import files
from numbers import Integral
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

from .filters.body_filter import BodyBox
from .settings import (
    BodyFilterSettings,
    GridSettings,
    LioSettings,
    LoopSettings,
    MappingSettings,
    OnlineSettings,
)

DEFAULT_CONFIG = Path(__file__).parent / "resources" / "generic.json"
# Counts and evidence scores must stay integers: truncating 2.5 would change the rule.
STRICT_INTEGERS = frozenset(
    {
        "projected_min_observations",
        "score_hit",
        "score_miss",
        "score_max",
        "score_min",
        "score_occupied",
    }
)
Section = TypeVar("Section")


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
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
    """User files and saved snapshots overlay the same algorithm defaults."""
    return _deep_merge(algorithm_profile(), profile or {})


def load_mapping_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        if str(path).endswith(".json"):
            profile = json.load(stream)
        else:
            import yaml

            profile = yaml.safe_load(stream) or {}
    if not isinstance(profile, dict):
        raise ValueError(f"配置不是 mapping: {path}")
    return profile


def _read_section(section_type: type[Section], profile: Mapping[str, Any]) -> Section:
    """Coerce the existing scalar profile syntax at the configuration boundary."""
    annotations = get_type_hints(section_type)
    constructor_fields = {}
    for field in fields(section_type):
        value = profile[field.name]
        if value is None:
            raise ValueError(f"{field.name} must not be null")
        scalar_type = annotations[field.name]
        if scalar_type in (int, float, str, bool) and field.name not in STRICT_INTEGERS:
            value = scalar_type(value)
        constructor_fields[field.name] = value
    return section_type(**constructor_fields)


def _resource_path(value: str | Path, profile_directory: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else profile_directory / path


def _read_body_box(profile: Mapping[str, Any], sensor: str) -> BodyBox:
    bounds = profile.get(sensor)
    if not isinstance(bounds, dict):
        raise ValueError(f"body_filter.{sensor} 缺少盒子边界")
    return BodyBox(**{field.name: float(bounds[field.name]) for field in fields(BodyBox)})


def _as_six(value: Any, name: str) -> tuple[float, float, float, float, float, float]:
    sigmas = tuple(float(sigma) for sigma in value)
    if len(sigmas) != 6:
        raise ValueError(f"{name} 需要 6 个数，实际 {len(sigmas)}")
    return sigmas


def settings_from_profile(profile: dict[str, Any], config_path: str | Path) -> MappingSettings:
    """Resolve domains once; validation runs at the session boundary."""
    resolved = resolve_profile(profile)
    config_path = Path(config_path).resolve()
    grid = resolved["grid"]
    online = resolved["online"]
    body_filter = resolved["body_filter"]
    for key in ("urdf_file", "package_path"):
        body_filter[key] = _resource_path(body_filter.get(key, ""), config_path.parent)
    for sensor in ("front", "back"):
        body_filter[sensor] = _read_body_box(body_filter, sensor)
    loop = online["loop"]
    for key in ("odom_sigmas", "loop_sigmas"):
        loop[key] = _as_six(loop[key], f"online.loop.{key}")
    online["loop"] = _read_section(LoopSettings, loop)
    maps_root = (resolved.get("session") or {}).get("maps_root", "maps")
    maps_root = Path(os.path.expandvars(str(Path.cwd() / maps_root))).expanduser()
    return MappingSettings(
        config_path=config_path,
        maps_root=maps_root,
        lio=_read_section(LioSettings, resolved["lio"]),
        body_filter=_read_section(BodyFilterSettings, body_filter),
        grid=_read_section(GridSettings, grid),
        online=_read_section(OnlineSettings, online),
    )


def load_settings(path: str | Path = DEFAULT_CONFIG) -> MappingSettings:
    return settings_from_profile(load_mapping_config(path), path)


def profile_from_settings(settings: MappingSettings, profile: dict[str, Any]) -> dict[str, Any]:
    """Save effective domain settings while retaining caller-owned calibration/source keys."""
    effective = asdict(settings)
    effective.pop("config_path")
    effective["session"] = {"maps_root": str(effective.pop("maps_root"))}
    for key in ("urdf_file", "package_path"):
        effective["body_filter"][key] = str(effective["body_filter"][key])
    snapshot = _deep_merge(resolve_profile(profile), effective)
    snapshot.pop("effective_settings", None)
    return snapshot


def _setting_values(sections: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    for name, value in sections.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            yield from _setting_values(value, path)
        else:
            yield path, value


def validate_settings(settings: MappingSettings) -> None:
    for name, value in _setting_values(asdict(settings)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} 必须为有限数值")
    if settings.lio.lidar not in {"front", "back", "both"}:
        raise ValueError("lio.lidar must be front, back or both")
    if settings.lio.imu not in {"front", "back", "chassis"}:
        raise ValueError("lio.imu must be front, back or chassis")
    if settings.lio.time_source != "header":
        raise ValueError("standalone mapping requires header/source time")
    if settings.grid.min_height > settings.grid.max_height:
        raise ValueError("grid min_height must not exceed max_height")
    if (
        min(settings.grid.height_resolution, settings.grid.max_ray_range, settings.grid.max_rays)
        <= 0
    ):
        raise ValueError("grid ray parameters must be positive")
    if settings.grid.update_interval_s < 0:
        raise ValueError("grid update_interval_s must be nonnegative")
    if settings.grid.free_space_mode not in {"height_complete", "projected", "polar"}:
        raise ValueError("grid.free_space_mode must be height_complete, projected or polar")
    count = settings.grid.projected_min_observations
    if isinstance(count, bool) or not isinstance(count, Integral) or not 1 <= count <= 127:
        raise ValueError("grid.projected_min_observations must be an integer in [1, 127]")
    if settings.grid.projected_max_range <= 0:
        raise ValueError("grid.projected_max_range must be positive")
    if not 0 <= settings.grid.free_end_margin <= 1.0:
        raise ValueError("grid.free_end_margin must be in [0, 1] m")
    from .core.ray_grid import validate_score_rules

    grid = settings.grid
    try:
        validate_score_rules(
            grid.score_hit, grid.score_miss, grid.score_max, grid.score_min, grid.score_occupied
        )
    except ValueError as error:
        raise ValueError(f"grid.{error}") from None
    if settings.grid.free_space_mode in {"projected", "polar"} and not settings.grid.ray_clearing:
        raise ValueError(f"{settings.grid.free_space_mode} free space requires grid.ray_clearing")
    if settings.online.voxel_clearing and not settings.grid.ray_clearing:
        raise ValueError("online.voxel_clearing requires grid.ray_clearing")
    if settings.online.voxel_clear_observations < 2:
        raise ValueError("online.voxel_clear_observations must be at least 2")
    if settings.online.voxel_max_rays <= 0:
        raise ValueError("online.voxel_max_rays must be positive")
    if settings.body_filter.mode not in {"urdf", "box"}:
        raise ValueError("body_filter.mode 必须为 urdf 或 box")
    if settings.body_filter.collision_threshold < 0:
        raise ValueError("body collision_threshold 必须为有限非负数")
    if settings.body_filter.tf_timeout_s <= 0:
        raise ValueError("body tf_timeout_s 必须为有限正数")
    if settings.lio.lidar_ready < 0:
        raise ValueError("--lidar-ready 不能小于 0")
    for name in (
        "max_imu_gap_s",
        "max_imu_boundary_gap_s",
        "imu_startup_timeout_s",
        "min_imu_samples",
        "max_match_failures",
    ):
        if getattr(settings.lio, name) <= 0:
            raise ValueError(f"{name} 必须大于 0")
    if settings.lio.cube_side_length <= 0 or settings.lio.det_range <= 0:
        raise ValueError("--cube-side-length / --det-range 必须大于 0")
    if settings.lio.filter_size_map < 0 or settings.lio.filter_size_surf < 0:
        raise ValueError("filter-size 不能小于 0")
    if settings.lio.blind < 0:
        raise ValueError("--blind 不能小于 0")
    if settings.grid.resolution <= 0:
        raise ValueError("--grid-resolution 必须大于 0")
    if settings.grid.margin_cells < 0:
        raise ValueError("--grid-margin-cells 不能小于 0")
    if settings.online.keyframe_dist_th < 0 or settings.online.keyframe_angle_th < 0:
        raise ValueError("keyframe 阈值不能小于 0")
    if settings.online.voxel_resolution <= 0:
        raise ValueError("--voxel-resolution 必须大于 0")
    if settings.online.live_http_port < 0:
        raise ValueError("--live-http-port 不能小于 0")
    for prefix in ("front", "back"):
        box = getattr(settings.body_filter, prefix)
        if box.min_x >= box.max_x or box.min_y >= box.max_y or box.min_z >= box.max_z:
            raise ValueError(f"{prefix} body box min 必须小于 max")
