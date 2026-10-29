"""Height slicing without voxel-center quantization."""
from __future__ import annotations
import numpy as np

def slice_pointcloud_by_height(
    parsed: dict[str, np.ndarray],
    min_height: float | None,
    max_height: float | None,
) -> dict[str, np.ndarray]:
    if min_height is None and max_height is None:
        return parsed
    xyz = parsed["xyz"]
    mask = np.ones(len(xyz), dtype=bool)
    if min_height is not None:
        mask &= xyz[:, 2] >= min_height
    if max_height is not None:
        mask &= xyz[:, 2] <= max_height
    out: dict[str, np.ndarray] = {"xyz": xyz[mask].astype(np.float32, copy=False)}
    if "reflectivity" in parsed:
        out["reflectivity"] = parsed["reflectivity"][mask]
    return out
