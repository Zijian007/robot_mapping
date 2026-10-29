"""Load native backends at construction; boolean probes are for diagnostics."""
from __future__ import annotations

from types import ModuleType
from typing import Callable
import numpy as np


def _require_numpy1() -> None:
    # These native wheels were built against NumPy 1.26.4; NumPy 2 can crash Python.
    if np.__version__ != "1.26.4":
        raise RuntimeError("GTSAM/VGICP require this deployment's NumPy 1.26.4 runtime")


def load_gtsam() -> ModuleType:
    _require_numpy1()
    import gtsam
    if not hasattr(gtsam, "Pose3") or not hasattr(gtsam, "ISAM2"):
        raise RuntimeError("GTSAM requires Pose3 and ISAM2 bindings")
    return gtsam


def load_pygicp() -> ModuleType:
    _require_numpy1()
    import pygicp
    if not getattr(pygicp, "mapping_gil_released", False) or not hasattr(pygicp.FastVGICP, "has_converged"):
        raise RuntimeError("VGICP requires the convergence/GIL patched binding")
    return pygicp


def load_fastlio2() -> ModuleType:
    import fastlio2
    if not fastlio2.native_available():
        raise RuntimeError("FAST-LIO2 native backend is unavailable")
    from fastlio2 import _fastlio2
    if not getattr(_fastlio2, "SCAN_BODY_POINTS_VERSION", 0):
        raise RuntimeError("FAST-LIO2 requires the frame point-cloud export interface")
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
