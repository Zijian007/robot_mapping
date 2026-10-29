"""Backend-neutral loop configuration and immutable detection snapshots."""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LoopConfig:
    enabled: bool = False
    search_radius_m: float = 8.0
    min_keyframe_gap: int = 30
    neighbor_keyframes: int = 2
    max_correction_m: float = 2.0
    max_correction_rad: float = 0.6
    min_inlier_ratio: float = 0.4
    max_rmse_m: float = 0.3
    vgicp_resolution: float = 0.5
    vgicp_max_correspondence_distance: float = 1.0
    vgicp_num_threads: int = 2
    min_points: int = 50

    def __post_init__(self):
        for name in ("search_radius_m", "max_correction_m", "max_correction_rad",
                     "max_rmse_m", "vgicp_resolution", "vgicp_max_correspondence_distance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"loop.{name} must be finite and > 0")
        if not 0 < self.min_inlier_ratio <= 1:
            raise ValueError("loop.min_inlier_ratio must be in (0, 1]")
        if self.min_keyframe_gap < 1 or self.neighbor_keyframes < 0 or self.min_points < 3:
            raise ValueError("invalid loop keyframe gap / neighbors / min_points")
        if self.vgicp_num_threads < 1:
            raise ValueError("loop.vgicp_num_threads must be >= 1")




@dataclass(frozen=True)
class LoopSnapshot:
    generation: int
    pose_revision: int
    clouds: tuple[np.ndarray, ...]
    poses: tuple[np.ndarray, ...]




@dataclass(frozen=True)
class LoopMatch:
    query_id: int
    match_id: int
    T_match_query: np.ndarray
    correction_m: float
    correction_rad: float
    fitness: float  # Actual fraction of ALL query points with a valid correspondence.
    rmse_m: float = 0.0
    generation: int = -1
    pose_revision: int = -1

