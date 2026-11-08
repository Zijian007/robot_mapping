"""SE3 helpers shared by online map / pose graph / loop closure."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def gravity_aligned_frame(up, forward, origin=(0., 0., 0.)) -> np.ndarray:
    """Return output_from_input from three vectors expressed in input axes.

    +Z follows up; +X follows the horizontal projection of forward. The caller
    supplies the origin and reference vectors; no robot frame is assumed.
    """
    up = np.array(up, dtype=np.float64, copy=True)
    forward = np.array(forward, dtype=np.float64, copy=True)
    origin = np.asarray(origin, dtype=np.float64)
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (up, forward, origin)):
        raise ValueError("frame vectors must be finite 3-vectors")
    if np.linalg.norm(up) < 1e-6:
        raise ValueError("cannot determine up direction")
    up /= np.linalg.norm(up)
    forward -= up * np.dot(forward, up)
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError("cannot determine horizontal heading")
    forward /= np.linalg.norm(forward)
    mat = np.eye(4)
    mat[:3, :3] = np.stack([forward, np.cross(up, forward), up])
    mat[:3, 3] = -mat[:3, :3] @ origin
    return mat


def is_se3(mat: np.ndarray) -> bool:
    m = np.asarray(mat)
    return (m.shape == (4, 4) and bool(np.isfinite(m).all()) and
            bool(np.allclose(m[3], [0, 0, 0, 1], atol=1e-5)) and
            bool(np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-4)) and
            bool(np.isclose(np.linalg.det(m[:3, :3]), 1., atol=1e-4)))


def quat_xyzw_to_rot(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_parts_to_matrix(
    position: Sequence[float],
    orientation_xyzw: Sequence[float],
) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = quat_xyzw_to_rot(orientation_xyzw)
    t[:3, 3] = np.asarray(position, dtype=np.float64)
    return t


def invert_se3(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64)
    r = m[:3, :3]
    t = m[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ t
    return out


def relative_se3(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    """T_parent_child such that child = parent @ relative."""
    return invert_se3(parent) @ np.asarray(child, dtype=np.float64)


def transform_points(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("points must be (N, >=3)")
    r = mat[:3, :3]
    t = mat[:3, 3]
    return (arr[:, :3] @ r.T) + t


def se3_translation_norm(mat: np.ndarray) -> float:
    return float(np.linalg.norm(mat[:3, 3]))


def se3_rotation_angle(mat: np.ndarray) -> float:
    r = mat[:3, :3]
    cos_theta = float((np.trace(r) - 1.0) * 0.5)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return float(math.acos(cos_theta))
