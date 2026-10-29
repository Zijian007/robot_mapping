"""Verify isolated imports, native execution and actual loaded library paths."""
from pathlib import Path
import importlib.abc
import json
import subprocess
import sys


def main():
    root=Path(sys.argv[1] if len(sys.argv)>1 else sys.prefix).resolve()
    assert Path(sys.prefix).resolve()==root
    assert "include-system-site-packages = false" in (root/"pyvenv.cfg").read_text().lower()
    class BlockApplications(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split(".")[0] in {"move_odom","wholebody","agibot_gdk"}:
                raise AssertionError("unexpected application/SDK import: "+fullname)
    sys.meta_path.insert(0,BlockApplications())
    import robot_mapping,fastlio2,pygicp,gtsam,numpy,scipy,matplotlib,PIL,yaml
    modules=[robot_mapping,fastlio2,pygicp,gtsam,numpy,scipy,matplotlib,PIL,yaml]
    for module in modules:
        assert Path(module.__file__).resolve().is_relative_to(root),(module.__name__,module.__file__)
    assert numpy.__version__=="1.26.4"
    assert getattr(pygicp,"mapping_gil_released",False)
    from robot_mapping.cli import main as command
    assert command(["doctor","--native","--loop"])==0
    # Exercise the optional companion module and its transitive library lookup.
    import gtsam_unstable
    assert Path(gtsam_unstable.__file__).resolve().is_relative_to(root)
    maps=Path("/proc/self/maps").read_text()
    for line in maps.splitlines():
        if any(name in line for name in ("libgtsam", "libmetis-gtsam", "libfast_gicp")):
            path=Path(line.split()[-1]).resolve()
            assert path.is_relative_to(root),line
    print(json.dumps({"runtime":str(root),"imports":{m.__name__:m.__file__ for m in modules},
        "native_execution":"PASS","loaded_libraries":"inside runtime","hardware_accessed":False},indent=2))


if __name__=="__main__":
    main()
