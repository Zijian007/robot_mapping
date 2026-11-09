"""Compare voxel storage on fixed observations without opening hardware.

Run each backend in a fresh process. Timings exclude input decoding, coordinate
transforms and diagnostic snapshots. Legacy input is an explicitly saved copy
of spatial.py, ray_voxels.py and the matching _ray_native extension.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_backend(args):
    from robot_mapping.backends import ray_native
    if args.backend == "legacy":
        if args.legacy_root is None:
            raise ValueError("legacy backend requires --legacy-root")
        extension, = args.legacy_root.glob("_ray_native*.so")
        ray_native.kernels = load_module("robot_mapping._ray_native", extension)
        load_module("robot_mapping.core.spatial", args.legacy_root / "spatial.py")
        module = load_module("robot_mapping.core.ray_voxels", args.legacy_root / "ray_voxels.py")
    else:
        from robot_mapping.core import spatial, ray_voxels
        if args.backend == "python":
            spatial.kernels = ray_voxels.kernels = None
        elif ray_native.kernels is None:
            raise RuntimeError("build the native extension before benchmarking")
        module = ray_voxels
    return module.RayVoxelHash


def fingerprint(voxels):
    payload = {"cells": sorted(tuple(map(int, key)) for key in voxels._cells),
               "misses": sorted((*map(int, key), int(count)) for key, count in voxels._misses.items()),
               "removed": voxels.removed_voxels}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def measure(call):
    start = time.perf_counter()
    result = call()
    return result, (time.perf_counter() - start) * 1000.


def summary(values):
    return dict(zip(("p50", "p95", "p99"), map(float, np.percentile(values, [50, 95, 99]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "python", "legacy"), required=True)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    factory = configure_backend(args)
    from robot_mapping.core.pose_math import transform_points
    with np.load(args.observations, allow_pickle=False) as archive:
        indices = sorted(int(key[5:]) for key in archive.files if key.startswith("rays_"))
        observations = []
        for i in indices:
            rays, matrix = archive[f"rays_{i}"], archive[f"mat_{i}"]
            world = np.column_stack((transform_points(matrix, rays[:, :3]),
                                     transform_points(matrix, rays[:, 3:])))
            points = (transform_points(matrix, archive[f"points_{i}"])
                      if f"points_{i}" in archive else None)
            observations.append((points, world))
    samples = {name: [] for name in ("insert_ms", "rays_ms", "observation_ms", "export_ms")}
    evidence = []
    for rep in range(args.repeats):
        voxels = factory()
        for points, rays in observations:
            insert_ms = 0.
            if points is not None:
                _, insert_ms = measure(lambda: voxels.add_points(points))
                samples["insert_ms"].append(insert_ms)
            _, rays_ms = measure(lambda: voxels.ingest_rays(rays))
            samples["rays_ms"].append(rays_ms)
            samples["observation_ms"].append(insert_ms + rays_ms)
            if rep == 0:
                evidence.append(fingerprint(voxels))
        _, elapsed = measure(voxels.points_array)
        samples["export_ms"].append(elapsed)
    # A separate synthetic workload covers insertion/export at a larger map size.
    # It is not an end-to-end LiDAR throughput measurement.
    points = np.random.default_rng(812).uniform([-30., -30., -1.], [30., 30., 3.], (100_000, 3))
    synthetic = {name: [] for name in ("insert_ms", "duplicate_insert_ms", "export_ms")}
    for _ in range(args.repeats):
        large = factory()
        for name, call in (("insert_ms", lambda: large.add_points(points)),
                           ("duplicate_insert_ms", lambda: large.add_points(points)),
                           ("export_ms", large.points_array)):
            _, elapsed = measure(call)
            synthetic[name].append(elapsed)
    result = {"backend": args.backend, "repeats": args.repeats,
              "observations": len(observations), "hardware_accessed": False,
              "input_sha256": hashlib.sha256(args.observations.read_bytes()).hexdigest(),
              "recorded_ms": {key: summary(value) for key, value in samples.items() if value},
              "evidence_after_each_observation": evidence,
              "final_voxels": len(voxels), "removed_voxels": voxels.removed_voxels,
              "synthetic_input_points": len(points), "synthetic_voxels": len(large),
              "synthetic_ms": {key: summary(value) for key, value in synthetic.items()},
              "synthetic_evidence": fingerprint(large)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "evidence_after_each_observation"}, indent=2))


if __name__ == "__main__":
    main()
