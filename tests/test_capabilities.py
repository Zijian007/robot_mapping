import sys
from types import ModuleType, SimpleNamespace

import pytest

from robot_mapping.backends import capabilities


def test_fastlio_ray_requirement_is_checked_only_when_requested(monkeypatch):
    module = ModuleType("fastlio2")
    module.native_available = lambda: True
    module._fastlio2 = SimpleNamespace(SCAN_BODY_POINTS_VERSION=1, LIDAR_UPDATE_VALIDITY_VERSION=1)
    monkeypatch.setitem(sys.modules, "fastlio2", module)
    assert capabilities.load_fastlio2() is module
    with pytest.raises(RuntimeError, match="deskewed ray"):
        capabilities.load_fastlio2(require_rays=True)
    module._fastlio2.SCAN_BODY_RAYS_VERSION = 1
    assert capabilities.load_fastlio2(require_rays=True) is module


def test_incomplete_vgicp_binding_reports_unavailable(monkeypatch):
    monkeypatch.setattr(capabilities, "_require_numpy1", lambda: None)
    monkeypatch.setitem(sys.modules, "pygicp", SimpleNamespace(mapping_gil_released=True))
    assert capabilities.pygicp_available() is False
    with pytest.raises(RuntimeError, match="convergence/GIL"):
        capabilities.load_pygicp()
