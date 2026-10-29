"""Algorithm diagnostics. Live sensors, replay and map export belong to robot_mapping_gdk."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform

from . import __version__


def entrypoint():
    """Run one algorithm diagnostic command."""
    import traceback
    try:
        return main()
    except KeyboardInterrupt:
        return 130
    except Exception:
        traceback.print_exc()
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="LiDAR mapping algorithm diagnostics. Live sessions are started by an adapter.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Inspect this interpreter without hardware access")
    doctor.add_argument("--native", action="store_true", help="Actually load/check native FAST-LIO2")
    doctor.add_argument("--loop", action="store_true", help="Actually exercise installed GTSAM and patched VGICP")
    args = parser.parse_args(argv)
    report = {"robot_mapping": __version__, "python": platform.python_version(),
              "machine": platform.machine(), "hardware_accessed": False}
    report["installed"] = {}
    for name in ("numpy", "scipy", "gtsam", "Pillow", "matplotlib", "fastlio2"):
        try:
            report["installed"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["installed"][name] = None
    ok = True
    if args.native:
        from .backends.capabilities import fastlio2_available
        report["fastlio2_scan_export"] = fastlio2_available()
        ok &= report["fastlio2_scan_export"]
    if args.loop:
        from .backends.capabilities import gtsam_available, pygicp_available
        report["gtsam"] = gtsam_available()
        report["patched_vgicp"] = pygicp_available()
        ok &= report["gtsam"] and report["patched_vgicp"]
        if ok:
            import numpy as np
            from .backends.pose_graph import PoseGraphBackend
            from .backends.loop_closure import run_fast_vgicp
            graph = PoseGraphBackend()
            graph.add_keyframe(np.eye(4), relative_to_prev=None)
            points = np.random.default_rng(7).normal(size=(300, 3))
            transform, converged, mse = run_fast_vgicp(points, points, np.eye(4),
                resolution=.5, max_correspondence_distance=1., num_threads=1)
            # Voxel Gaussian approximations need not produce exact identity
            # even for identical clouds. Check a small geometric error.
            report["loop_smoke_mse_m2"] = mse
            report["loop_smoke"] = bool(converged and mse < .02**2
                and np.allclose(transform, np.eye(4), atol=.02)
                and len(graph.optimized_matrices()) == 1)
            ok &= report["loop_smoke"]
    print(json.dumps(report, indent=2))
    return 0 if ok else 1
