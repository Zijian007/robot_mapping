"""Time point preparation, timestamped URDF filtering and dual-LiDAR merging.

Only recorded scans and historical TF are used. Decoding, model loading and
result hashing are outside measured regions. Run each backend in a new process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure(args):
    if args.backend == "legacy":
        if args.legacy_root is None:
            raise ValueError("legacy backend requires --legacy-root")
        import robot_mapping

        (extension,) = args.legacy_root.glob("_ray_native*.so")
        native = load_module("robot_mapping._ray_native", extension)
        robot_mapping._ray_native = native
        from robot_mapping.backends import ray_native

        ray_native.kernels = native
        modules = [
            load_module(name, args.legacy_root / file)
            for name, file in (
                ("robot_mapping.filters.body_filter", "body_filter.py"),
                ("robot_mapping.filters.urdf_filter", "urdf_filter.py"),
                ("robot_mapping.backends.fastlio", "fastlio.py"),
                ("robot_mapping.runtime.frontend", "frontend.py"),
            )
        ]
    else:
        from robot_mapping.backends import fastlio
        from robot_mapping.filters import body_filter, urdf_filter
        from robot_mapping.runtime import frontend

        modules = [body_filter, urdf_filter, fastlio, frontend]
        if args.backend == "python":
            for module in modules:
                module.kernels = None
        else:
            from robot_mapping.backends.ray_native import kernels

            if kernels is None:
                raise RuntimeError("build the native extension before benchmarking")
    return modules[1:]


def summary(values):
    return dict(zip(("p50", "p95", "p99"), map(float, np.percentile(values, [50, 95, 99]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("legacy", "native", "python"), required=True)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.frames, args.repeats) < 1:
        parser.error("frames and repeats must be positive")
    urdf, fastlio, frontend = configure(args)
    from robot_mapping_gdk.recording import ReplaySource

    from robot_mapping.config import settings_from_profile
    from robot_mapping.runtime.calibration import calibration

    source = ReplaySource(args.recording)
    source.open()
    profile = source.manifest["configuration"]
    settings = settings_from_profile(profile, args.recording / "configuration.json")
    if not settings.needs_back or settings.body_filter_mode != "urdf":
        raise ValueError("benchmark requires a dual-LiDAR recording with URDF filtering")
    extrinsics, _ = calibration(profile, settings)
    body = urdf.UrdfBodyFilter(
        settings.body_urdf_file, settings.body_package_path, settings.body_collision_threshold
    )
    pipeline = frontend.MappingFrontend(
        source, settings, lidar_extrinsics=extrinsics, body_filter=body, body_lookup=source.lookup
    )
    all_frames = source._frames
    indices = np.linspace(0, len(all_frames) - 1, min(args.frames, len(all_frames)), dtype=int)
    source._frames = [all_frames[i] for i in indices]
    frames = []
    for _ in indices:
        primary = source.next_lidar(0.0)
        frames.append((primary, source.aux_lidar(), source._current))
    direct_origins = "include_origins" in inspect.signature(fastlio.prepare_scan).parameters

    def prepare(scan):
        if direct_origins:
            return fastlio.prepare_scan(
                scan, settings.blind, include_origins=settings.grid_ray_clearing
            )
        native = fastlio.prepare_scan(scan, settings.blind)
        if settings.grid_ray_clearing:
            native = replace(
                native,
                points_array=np.column_stack(
                    (
                        native.points_array,
                        np.zeros((len(native.points_array), 3), dtype=native.points_array.dtype),
                    )
                ),
            )
        return native

    rows, evidence, counts = [], [], []
    for rep in range(args.repeats):
        for primary, auxiliary, current in frames:
            source._current = current
            start = time.perf_counter()
            front = prepare(primary)
            back = prepare(auxiliary) if auxiliary is not None else None
            prepared = time.perf_counter()
            front, removed = pipeline._filter_scan(
                front,
                settings.primary_lidar,
                primary.begin_ns,
                extrinsics[settings.primary_lidar][:3, :3],
                extrinsics[settings.primary_lidar][:3, 3],
            )
            if (
                back is not None
                and abs(back.frame_stamp - front.frame_stamp) <= settings.max_back_dt
            ):
                back, back_removed = pipeline._filter_scan(
                    back,
                    "back",
                    auxiliary.begin_ns,
                    extrinsics["back"][:3, :3],
                    extrinsics["back"][:3, 3],
                )
                removed += back_removed
            else:
                back = None
            filtered = time.perf_counter()
            kept = 0
            if back is not None:
                front, _, kept = frontend.merge_back_into_front(
                    front,
                    back,
                    rotation_front=extrinsics["front"][:3, :3],
                    translation_front=extrinsics["front"][:3, 3],
                    rotation_back=extrinsics["back"][:3, :3],
                    translation_back=extrinsics["back"][:3, 3],
                    body_box=None,
                    max_scan_duration_s=settings.max_scan_duration,
                )
            end = time.perf_counter()
            rows.append(
                dict(
                    prepare_ms=(prepared - start) * 1000.0,
                    filter_ms=(filtered - prepared) * 1000.0,
                    merge_ms=(end - filtered) * 1000.0,
                    total_ms=(end - start) * 1000.0,
                )
            )
            if rep == 0:
                metadata = dict(
                    stamp=front.frame_stamp,
                    duration=front.scan_duration_s,
                    raw=front.raw_points,
                    dropped=front.dropped_points,
                    body_removed=removed,
                    back_kept=kept,
                )
                digest = hashlib.sha256(front.points_array.tobytes())
                digest.update(json.dumps(metadata, sort_keys=True).encode())
                evidence.append(digest.hexdigest())
                counts.append(metadata)
    result = dict(
        backend=args.backend,
        hardware_accessed=False,
        frames=len(frames),
        repeats=args.repeats,
        indices=indices.tolist(),
        input_frames_sha256=[all_frames[i]["sha256"] for i in indices],
        timings_ms={key: summary([row[key] for row in rows]) for key in rows[0]},
        collision_bounds=len(body.bounds),
        evidence_after_each_frame=evidence,
        counts=counts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key
                not in ("indices", "input_frames_sha256", "evidence_after_each_frame", "counts")
            },
            indent=2,
        )
    )
    source.close()


if __name__ == "__main__":
    main()
