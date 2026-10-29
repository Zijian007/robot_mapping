"""Install a wheel in a temporary target and exercise it outside the source tree.

Uses the caller interpreter's existing dependencies, never installs dependencies
or changes its environment. Native loop regression runs separately.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def main():
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "--native"):
        raise SystemExit("Usage: python scripts/verify_wheel.py dist/robot_mapping-*.whl [--native]")
    wheel = Path(sys.argv[1]).resolve(strict=True)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert not any("g2_omnipicker" in name or name.endswith("adapters/gdk.py")
                       for name in names)
        for forbidden in ("robot_mapping/api.py", "robot_mapping/session_run.py",
                          "robot_mapping/adapters/", "robot_mapping/storage/",
                          "robot_mapping/view/"):
            assert not any(name.startswith(forbidden) or name.endswith(forbidden) for name in names), forbidden
        assert "robot_mapping/resources/generic.json" in names
        assert "robot_mapping/resources/user.json" in names
        generic = json.loads(archive.read("robot_mapping/resources/generic.json"))
        assert "robot" not in generic and "source" not in generic
        assert generic["lio"]["filter_size_map"] == 0.15
    with tempfile.TemporaryDirectory(prefix="robot-mapping-wheel-") as scratch:
        root = Path(scratch)
        target = root / "site"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                        "--target", str(target), str(wheel)], check=True, cwd=root)
        script = r'''
import importlib, importlib.abc, json, sys
from pathlib import Path
target = Path(sys.argv[1])
sys.path.insert(0, str(target))
class NoApplication(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'move_odom','wholebody','agibot_gdk','robot_mapping_gdk'}:
            raise AssertionError('Application dependency: '+fullname)
sys.meta_path.insert(0, NoApplication())
import robot_mapping
assert Path(robot_mapping.__file__).is_relative_to(target)
from robot_mapping.core.online_map import OnlineMap, SE3Pose
from robot_mapping.filters.body_filter import BodyBox
from robot_mapping.filters.urdf_filter import UrdfBodyFilter
from robot_mapping.types import LidarScan, ImuBatch
from robot_mapping.cli import main
import numpy as np
mapping = OnlineMap()
assert mapping.ingest_local_points(SE3Pose((0,0,0),(0,0,0,1)), [[2.,2.,.5]])
assert mapping.grid.occupied_count() == 1
assert not ({'gtsam','pygicp','fastlio2','scipy','PIL','matplotlib'} & set(sys.modules))
model = Path('robot.urdf')
model.write_text('<robot name="test"><link name="base_link"><collision><geometry><box size="1 1 1"/></geometry></collision></link></robot>')
body = UrdfBodyFilter(model, Path('.'), .1)
kept, removed = body.filter_points(np.array([[0.,0.,0.],[2.,2.,2.]]), np.eye(3), np.zeros(3), 1,
                                 lambda *_: np.eye(4))
assert removed == 1 and len(kept) == 1
assert main(['doctor']) == 0
for name in ('robot_mapping.api','robot_mapping.adapters','robot_mapping.storage','robot_mapping.view'):
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError('application module still present: '+name)
values, spec = mapping.snapshot_grid()
assert values.shape == (spec.height, spec.width)
assert not ({'move_odom','wholebody','agibot_gdk','robot_mapping_gdk'} & set(sys.modules))
if sys.argv[2] == '--native':
    assert main(['doctor','--native','--loop']) == 0
print(json.dumps({'wheel_import': str(robot_mapping.__file__),
                  'isolated_core': 'PASS', 'cli': 'PASS',
                  'native_doctor': 'PASS' if sys.argv[2]=='--native' else 'NOT_REQUESTED'}))
'''
        native = "--native" if len(sys.argv) == 3 else ""
        result = subprocess.run([sys.executable, "-I", "-c", script, str(target), native], cwd=root,
                                capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        print(result.stdout)
    print(json.dumps({"wheel": str(wheel), "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
                      "relocation": "PASS", "dependencies_installed": False}, indent=2))


if __name__ == "__main__":
    main()
