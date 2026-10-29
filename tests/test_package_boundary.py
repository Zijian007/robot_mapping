"""A core-only process must not load an SDK, app, or optional solver."""
import importlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_core_import_without_applications_or_optional_dependencies(tmp_path):
    script = r'''
import importlib.abc, sys
blocked = {'move_odom', 'wholebody', 'agibot_gdk', 'robot_mapping_gdk', 'gtsam', 'pygicp',
           'fastlio2', 'scipy', 'PIL', 'matplotlib', 'yaml'}
class BlockApplications(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('unexpected import: '+fullname)
sys.meta_path.insert(0, BlockApplications())
before_path = list(sys.path)
from robot_mapping.core.online_map import OnlineMap, SE3Pose
from robot_mapping.filters.body_filter import BodyBox
from robot_mapping.filters.urdf_filter import UrdfBodyFilter
from robot_mapping.ports import SensorSource, TransformProvider
from robot_mapping.types import LidarScan, ImuBatch
mapping = OnlineMap()
assert mapping.ingest_local_points(SE3Pose((0,0,0),(0,0,0,1)), [[2,2,.5]])
assert mapping.grid.occupied_count() == 1
assert before_path == sys.path
assert not (blocked & set(sys.modules))
print('isolated core OK')
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_default_doctor_does_not_import_numerical_or_native_code(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    script = '''
import sys
from robot_mapping.cli import main
assert main(['doctor']) == 0
assert not ({'numpy', 'gtsam', 'pygicp', 'fastlio2', 'agibot_gdk'} & set(sys.modules))
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_algorithm_cli_has_no_application_commands():
    from robot_mapping.cli import main
    for command in ("run", "replay", "project", "init-config"):
        with pytest.raises(SystemExit) as exc:
            main([command])
        assert exc.value.code == 2


def test_application_modules_are_not_in_the_algorithm_package():
    for name in (
        "robot_mapping.api",
        "robot_mapping.session_run",
        "robot_mapping.adapters",
        "robot_mapping.adapters.recording",
        "robot_mapping.adapters.conversion",
        "robot_mapping.storage",
        "robot_mapping.view",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)
