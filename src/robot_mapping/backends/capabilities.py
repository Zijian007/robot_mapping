"""Load native backends at construction; boolean probes are for diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType


def load_gtsam() -> ModuleType:
    import gtsam

    if not hasattr(gtsam, "Pose3") or not hasattr(gtsam, "ISAM2"):
        raise RuntimeError("GTSAM requires Pose3 and ISAM2 bindings")
    return gtsam


def load_pygicp() -> ModuleType:
    import pygicp

    if not getattr(pygicp, "mapping_gil_released", False) or not hasattr(
        getattr(pygicp, "FastVGICP", None), "has_converged"
    ):
        raise RuntimeError("VGICP requires the convergence/GIL patched binding")
    return pygicp


def load_fastlio2(*, require_rays: bool = False) -> ModuleType:
    import fastlio2

    if not fastlio2.native_available():
        raise RuntimeError("FAST-LIO2 native backend is unavailable")
    from fastlio2 import _fastlio2

    if not getattr(_fastlio2, "SCAN_BODY_POINTS_VERSION", 0):
        raise RuntimeError("FAST-LIO2 requires the frame point-cloud export interface")
    if not getattr(_fastlio2, "LIDAR_UPDATE_VALIDITY_VERSION", 0):
        raise RuntimeError(
            "FAST-LIO2 requires the lidar update validity/map insertion guard; rebuild the native backend"
        )
    if require_rays and not getattr(_fastlio2, "SCAN_BODY_RAYS_VERSION", 0):
        raise RuntimeError("ray clearing requires deskewed ray export; rebuild FAST-LIO2")
    return fastlio2


def _available(loader: Callable[[], ModuleType]) -> bool:
    try:
        loader()
    except (ImportError, OSError, RuntimeError):
        return False
    return True


def gtsam_available() -> bool:
    return _available(load_gtsam)


def pygicp_available() -> bool:
    return _available(load_pygicp)


def fastlio2_available() -> bool:
    return _available(load_fastlio2)
