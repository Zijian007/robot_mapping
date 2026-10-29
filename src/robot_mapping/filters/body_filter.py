"""Remove LiDAR points that fall inside a robot AABB in base_link."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class MissingBodyTransform(RuntimeError):
    """A complete body mask cannot be constructed for this scan timestamp."""



@dataclass(frozen=True)
class BodyBox:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    def contains_xyzw(self, xyz_base: np.ndarray) -> np.ndarray:
        return (
            (xyz_base[:, 0] >= self.min_x)
            & (xyz_base[:, 0] <= self.max_x)
            & (xyz_base[:, 1] >= self.min_y)
            & (xyz_base[:, 1] <= self.max_y)
            & (xyz_base[:, 2] >= self.min_z)
            & (xyz_base[:, 2] <= self.max_z)
        )




def transform_xyz(xyz: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    return xyz @ rotation.T + translation


def lidar_b_to_lidar_a_rt(
    rotation_a: np.ndarray,
    translation_a: np.ndarray,
    rotation_b: np.ndarray,
    translation_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """p_a = R @ p_b + t, using each lidar's transform to the same parent (base_link)."""
    rotation = rotation_a.T @ rotation_b
    translation = rotation_a.T @ (translation_b - translation_a)
    return rotation, translation


def transform_points_array(
    points_array: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    out = np.array(points_array, dtype=np.float32, copy=True)
    if len(out) == 0:
        return out
    out[:, :3] = transform_xyz(out[:, :3], rotation, translation).astype(np.float32)
    return out


def keep_outside_body_mask(
    xyz_lidar: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    box: BodyBox,
) -> np.ndarray:
    if len(xyz_lidar) == 0:
        return np.zeros(0, dtype=bool)
    xyz_base = transform_xyz(xyz_lidar, rotation, translation)
    return ~box.contains_xyzw(xyz_base)


def filter_points_array(
    points_array: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    box: BodyBox,
) -> tuple[np.ndarray, int]:
    """Keep rows whose xyz is outside the robot box. Returns (kept, n_removed)."""
    if points_array is None or len(points_array) == 0:
        return points_array, 0
    keep = keep_outside_body_mask(points_array[:, :3], rotation, translation, box)
    n_removed = int(np.count_nonzero(~keep))
    if n_removed == 0:
        return points_array, 0
    return points_array[keep], n_removed
