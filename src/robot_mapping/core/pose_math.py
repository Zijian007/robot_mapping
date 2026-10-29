"""SE3 helpers shared by online map / pose graph / loop closure."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def gravity_aligned_map_from_lio(mean_acc: np.ndarray, base_from_imu: np.ndarray) -> np.ndarray:
    """Map +Z follows initial accelerometer up; +X follows projected base heading.

    FAST-LIO2 keeps its initial IMU axes and estimates a gravity vector in that
    frame. Rotate map output, without changing its EKF state or observations.
    The map origin remains the initial IMU origin.
    """
    up = np.asarray(mean_acc, dtype=np.float64)
    rotation = np.asarray(base_from_imu, dtype=np.float64)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-6:
        raise ValueError("无法从初始化 IMU 数据确定重力方向")
    up = up / np.linalg.norm(up)
    forward = rotation.T @ np.array([1., 0., 0.])
    forward -= up * np.dot(forward, up)
    if not np.isfinite(forward).all() or np.linalg.norm(forward) < 1e-6:
        raise ValueError("无法确定初始底盘水平朝向")
    forward /= np.linalg.norm(forward)
    mat = np.eye(4)
    mat[:3, :3] = np.stack([forward, np.cross(up, forward), up])
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


def rot_to_quat_xyzw(r: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(r, dtype=np.float64)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def pose_parts_to_matrix(
    position: Sequence[float],
    orientation_xyzw: Sequence[float],
) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = quat_xyzw_to_rot(orientation_xyzw)
    t[:3, 3] = np.asarray(position, dtype=np.float64)
    return t


def matrix_to_pose_parts(mat: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    m = np.asarray(mat, dtype=np.float64)
    pos = (float(m[0, 3]), float(m[1, 3]), float(m[2, 3]))
    return pos, rot_to_quat_xyzw(m[:3, :3])


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
