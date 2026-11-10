"""Profile fixed 2D observations and compare exact grid evidence, without hardware.

Execute each backend in a fresh process. Input decoding, coordinate conversion
and diagnostic hashing are excluded from timings. Method timings are disjoint;
unattributed time includes validation and any unwrapped reference-array work.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_backend(args):
    if args.backend == "legacy":
        if args.legacy_root is None:
            raise ValueError("legacy backend requires --legacy-root")
        import robot_mapping

        (extension,) = args.legacy_root.glob("_ray_native*.so")
        native = load_module("robot_mapping._ray_native", extension)
        robot_mapping._ray_native = native
        from robot_mapping.backends import ray_native

        ray_native.kernels = native
        return load_module(
            "robot_mapping.core.ray_grid", args.legacy_root / "ray_grid.py"
        ).RayOccupancy2D
    from robot_mapping.core import ray_grid

    if args.backend == "python":
        ray_grid.kernels = None
    elif ray_grid.kernels is None:
        raise RuntimeError("build the native extension before benchmarking")
    return ray_grid.RayOccupancy2D


def fingerprint(grid):
    digest = hashlib.sha256()
    digest.update(json.dumps(vars(grid.spec), sort_keys=True).encode())
    digest.update(grid._scores.tobytes())
    digest.update(grid.values.astype("<i2").tobytes())
    return digest.hexdigest()


def profile_methods(grid, times):
    for name in (
        "_ensure_seed",
        "_expand_to_include",
        "_keys",
        "_select_rays",
        "_free_keys",
        "_mark_free_disk",
        "_native_groups",
    ):
        if not hasattr(grid, name):
            continue
        original = getattr(grid, name)

        def wrapped(*args, _call=original, _name=name, **kwargs):
            start = time.perf_counter()
            result = _call(*args, **kwargs)
            times[_name] = times.get(_name, 0.0) + (time.perf_counter() - start) * 1000.0
            return result

        setattr(grid, name, wrapped)


def summary(values):
    return dict(zip(("p50", "p95", "p99"), map(float, np.percentile(values, [50, 95, 99]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "python", "legacy"), required=True)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    factory = configure_backend(args)
    from robot_mapping.core.pose_math import transform_points

    with np.load(args.observations, allow_pickle=False) as archive:
        indices = sorted(int(key[5:]) for key in archive.files if key.startswith("rays_"))
        if not indices:
            raise ValueError("archive contains no ray observations")
        observations = []
        footprint = archive["footprint"] if "footprint" in archive else np.eye(4)
        for i in indices:
            rays, matrix = archive[f"rays_{i}"], archive[f"mat_{i}"]
            world = np.column_stack(
                (transform_points(matrix, rays[:, :3]), transform_points(matrix, rays[:, 3:]))
            )
            observations.append((world, tuple((matrix @ footprint)[:2, 3])))
    rows, evidence, times = [], [], {}
    if args.backend == "native":
        native_module = sys.modules[factory.__module__].kernels
        original = native_module.grid_update

        def timed_update(*args, **kwargs):
            start = time.perf_counter()
            result = original(*args, **kwargs)
            times["native_update_ms"] = (time.perf_counter() - start) * 1000.0
            return result

        native_module.grid_update = timed_update
    for rep in range(args.repeats):
        grid = factory()
        profile_methods(grid, times)
        for rays, xy in observations:
            times.clear()
            start = time.perf_counter()
            grid.ingest_rays(rays, robot_xy=xy)
            total = (time.perf_counter() - start) * 1000.0
            remainder = total - sum(times.values())
            rows.append(dict(total_ms=total, unattributed_ms=remainder, **times))
            if rep == 0:
                evidence.append(fingerprint(grid))
    result = {
        "backend": args.backend,
        "repeats": args.repeats,
        "observations": len(observations),
        "hardware_accessed": False,
        "input_sha256": hashlib.sha256(args.observations.read_bytes()).hexdigest(),
        "timings_ms": {key: summary([row.get(key, 0.0) for row in rows]) for key in rows[0]},
        "evidence_after_each_observation": evidence,
        "grid_spec": vars(grid.spec),
        "score_shape": list(grid._scores.shape),
        "occupied": grid.occupied_count(),
        "free": grid.free_count(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "evidence_after_each_observation"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
