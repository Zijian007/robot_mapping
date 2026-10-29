"""URDF collision bounds + timestamped TF, following the installed SLAM filter.

Bounds are inflated in collision coordinates, then their eight corners are
transformed into the input lidar frame and bounded again (conservative AABB).
No SDK imports or hardware access are needed to load/test the geometry.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET

import numpy as np

from robot_mapping.core.pose_math import invert_se3
from robot_mapping.filters.body_filter import MissingBodyTransform


def _vector(text: str | None, default: str = "0 0 0") -> np.ndarray:
    value = np.array([float(v) for v in (text or default).split()])
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"无效 URDF 三维向量: {text!r}")
    return value


def _origin(element: ET.Element | None) -> np.ndarray:
    pose = np.eye(4)
    if element is None:
        return pose
    r, p, y = _vector(element.get("rpy"))
    cr, cp, cy = math.cos(r), math.cos(p), math.cos(y)
    sr, sp, sy = math.sin(r), math.sin(p), math.sin(y)
    pose[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                    [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                    [-sp, cp*sr, cp*cr]]
    pose[:3, 3] = _vector(element.get("xyz"))
    return pose


def _corners(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.array(list(itertools.product(*zip(lower, upper))))


def _stl_vertices(path: Path) -> np.ndarray:
    data = path.read_bytes()
    # Binary STL headers may also start with 'solid'; determine format by length.
    count = int.from_bytes(data[80:84], "little") if len(data) >= 84 else 0
    if count and len(data) == 84 + 50 * count:
        dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)),
                          ("attribute", "<u2")])
        vertices = np.frombuffer(data, dtype=dtype, offset=84)["vertices"].reshape(-1, 3)
    else:
        vertices = np.array([_vector(line.strip()[7:]) for line in data.decode("ascii").splitlines()
                             if line.strip().startswith("vertex ")])
    if vertices.size == 0 or not np.isfinite(vertices).all():
        raise ValueError(f"STL 无有效顶点: {path}")
    return vertices


def _geometry_bounds(geometry: ET.Element, urdf: Path, package_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = geometry.find("mesh")
    if mesh is not None:
        filename = mesh.attrib["filename"]
        if filename.startswith("package://"):
            path = package_path / filename[len("package://"):]
        elif filename.startswith("file://"):
            path = Path(filename[len("file://"):])
        else:
            path = urdf.parent / filename
        if path.suffix.lower() != ".stl":
            raise ValueError(f"本体过滤仅支持 STL 网格: {path}")
        vertices = _stl_vertices(path) * _vector(mesh.get("scale"), "1 1 1")
        return vertices.min(axis=0), vertices.max(axis=0)
    box, cylinder, sphere = geometry.find("box"), geometry.find("cylinder"), geometry.find("sphere")
    if box is not None:
        half = _vector(box.get("size")) / 2
    elif cylinder is not None:
        half = np.array([float(cylinder.attrib["radius"])] * 2
                        + [float(cylinder.attrib["length"]) / 2])
    elif sphere is not None:
        half = np.full(3, float(sphere.attrib["radius"]))
    else:
        raise ValueError("本体过滤不支持此 URDF collision 几何")
    if not np.isfinite(half).all() or np.any(half <= 0):
        raise ValueError("URDF collision 尺寸必须为有限正数")
    return -half, half


@dataclass(frozen=True)
class CollisionBounds:
    frame: str
    corners: np.ndarray  # Eight corners in the nearest movable ancestor frame.


class UrdfBodyFilter:
    def __init__(self, urdf_file: str | Path, package_path: str | Path,
                 collision_threshold: float = 0.1, base_frame: str = "base_link") -> None:
        if not math.isfinite(collision_threshold) or collision_threshold < 0:
            raise ValueError("collision_threshold 必须为有限非负数")
        self.base_frame = base_frame
        self.urdf_file = Path(urdf_file)
        root = ET.parse(self.urdf_file).getroot()
        if base_frame not in {link.attrib["name"] for link in root.findall("link")}:
            raise ValueError(f"URDF 缺少 base_frame={base_frame}")
        fixed = {}
        for joint in root.findall("joint"):
            if joint.get("type") == "fixed":
                fixed[joint.find("child").attrib["link"]] = (
                    joint.find("parent").attrib["link"], _origin(joint.find("origin")))
        self.bounds: list[CollisionBounds] = []
        for link in root.findall("link"):
            frame = link.attrib["name"]
            parent_pose = np.eye(4)
            seen = set()
            while frame != base_frame and frame in fixed:
                if frame in seen:
                    raise ValueError("URDF fixed joint 存在环")
                seen.add(frame)
                frame, pose = fixed[frame]
                parent_pose = pose @ parent_pose
            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                if geometry is None:
                    raise ValueError(f"{link.attrib['name']} collision 缺少 geometry")
                lower, upper = _geometry_bounds(geometry, self.urdf_file, Path(package_path))
                corners = _corners(lower - collision_threshold, upper + collision_threshold)
                pose = parent_pose @ _origin(collision.find("origin"))
                self.bounds.append(CollisionBounds(frame, corners @ pose[:3, :3].T + pose[:3, 3]))
        if not self.bounds:
            raise ValueError(f"URDF 没有 collision 几何: {self.urdf_file}")

    def filter_points(self, points: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
                      timestamp_ns: int, lookup: Callable[[str, str, int], np.ndarray]) -> tuple[np.ndarray, int]:
        """Remove the union of link AABBs; preserve all remaining point fields.

        ``lookup(target, source, timestamp_ns)`` returns T_target_source.
        Missing/invalid TF rejects the whole scan, never leaves a partial mask.
        """
        if timestamp_ns <= 0:
            raise MissingBodyTransform("点云缺少有效源时间戳，无法查询本体 TF")
        if len(points) == 0:
            return points, 0
        lidar_pose = np.eye(4)
        lidar_pose[:3, :3], lidar_pose[:3, 3] = rotation, translation
        lidar_from_base = invert_se3(lidar_pose)
        transforms = {self.base_frame: np.eye(4)}
        boxes = []
        for bound in self.bounds:
            if bound.frame not in transforms:
                try:
                    mat = np.asarray(lookup(self.base_frame, bound.frame, timestamp_ns), dtype=np.float64)
                    if (mat.shape != (4, 4) or not np.isfinite(mat).all()
                            or not np.allclose(mat[3], [0, 0, 0, 1])
                            or not np.allclose(mat[:3, :3].T @ mat[:3, :3], np.eye(3), atol=1e-5)
                            or not np.isclose(np.linalg.det(mat[:3, :3]), 1, atol=1e-5)):
                        raise ValueError("invalid SE3")
                    transforms[bound.frame] = mat
                except Exception as exc:
                    raise MissingBodyTransform(f"本体 TF 缺失: {self.base_frame} <- {bound.frame} @ {timestamp_ns}") from exc
            pose = lidar_from_base @ transforms[bound.frame]
            corners = bound.corners @ pose[:3, :3].T + pose[:3, 3]
            boxes.append((corners.min(axis=0), corners.max(axis=0)))
        # Repeated per-link comparisons should traverse contiguous coordinate
        # arrays rather than stride through intensity/time fields each time.
        x, y, z = np.ascontiguousarray(points[:, :3].T)
        keep = np.ones(len(points), dtype=bool)
        for lower, upper in boxes:
            # Avoid allocating/reducing an N x 3 temporary for every link.
            inside = ((x >= lower[0]) & (x <= upper[0])
                      & (y >= lower[1]) & (y <= upper[1])
                      & (z >= lower[2]) & (z <= upper[2]))
            keep &= ~inside
        removed = int(np.count_nonzero(~keep))
        return (points[keep] if removed else points), removed
