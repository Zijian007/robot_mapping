"""Verify isolated imports, native execution and actual loaded library paths."""

import importlib.abc
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else sys.prefix).resolve()
    assert Path(sys.prefix).resolve() == root
    assert "include-system-site-packages = false" in (root / "pyvenv.cfg").read_text().lower()

    class BlockApplications(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"move_odom", "wholebody", "agibot_gdk"}:
                raise AssertionError("unexpected application/SDK import: " + fullname)

    sys.meta_path.insert(0, BlockApplications())
    import fastlio2
    import gtsam
    import matplotlib
    import numpy
    import PIL
    import pygicp
    import scipy
    import yaml

    import robot_mapping

    modules = [robot_mapping, fastlio2, pygicp, gtsam, numpy, scipy, matplotlib, PIL, yaml]
    for module in modules:
        assert Path(module.__file__).resolve().is_relative_to(root), (
            module.__name__,
            module.__file__,
        )
    assert numpy.__version__ == "1.26.4"
    assert getattr(pygicp, "mapping_gil_released", False)
    from fastlio2 import _fastlio2

    assert _fastlio2.PROFILE_TIMINGS_VERSION >= 1
    assert _fastlio2.WORKSPACE_REUSE_VERSION >= 1
    from types import SimpleNamespace

    lio = fastlio2.FastLIO2(
        fastlio2.FastLIO2Config(
            mapping={
                "filter_size_surf": 0.15,
                "filter_size_map": 0.15,
                "omp_num_threads": 1,
                "profile_timings": True,
                "cube_side_length": 1000.0,
                "det_range": 70.0,
            }
        ),
        require_native=True,
    )
    x, y = numpy.meshgrid(numpy.linspace(1.0, 5.0, 20), numpy.linspace(-2.0, 2.0, 20))
    scan_points = numpy.zeros((x.size, 5))
    scan_points[:, :3] = numpy.column_stack((x.ravel(), y.ravel(), numpy.full(x.size, 0.5)))
    scan_points[:, 4] = numpy.linspace(0.0, 100.0, len(scan_points))
    for i in range(12):
        end = 100.0 + i * 0.1
        imu_samples = numpy.zeros((20, 7))
        imu_samples[:, 0] = numpy.linspace(end - 0.095, end, 20)
        imu_samples[:, 6] = 9.81
        odom = lio.process(
            SimpleNamespace(frame_stamp=end, scan_duration_s=0.1, points_array=scan_points),
            imu_samples,
        )
    assert odom["lidar_update_valid"] and odom["timings_ms"]["match_ms"] > 0
    from robot_mapping.cli import main as command

    assert command(["doctor", "--native", "--loop"]) == 0
    from robot_mapping.backends.ray_native import kernels

    assert kernels is not None, "managed runtime requires compiled ray kernels"
    assert kernels.API_VERSION >= 5, "rebuild native kernels for body TF validation"
    assert Path(kernels.__file__).resolve().is_relative_to(root)
    from robot_mapping.core.ray_voxels import RayVoxelHash

    voxels = RayVoxelHash()
    assert voxels._native_store is not None, "managed runtime requires native voxel storage"
    voxels.add_points([[2.025, 0.075, 0.525]])
    for _ in range(3):
        voxels.ingest_rays([[0.075, 0.075, 0.525, 4.125, 0.075, 0.525]])
    assert len(voxels) == 0 and voxels.removed_voxels == 1
    from robot_mapping.core.ray_grid import RayOccupancy2D

    grid = RayOccupancy2D(min_height=0.4, max_height=0.6)
    for _ in range(2):
        grid.ingest_rays([[0.0, 0.0, 0.5, 2.0, 0.0, 0.5]])
    row, col = grid._world_to_rc(numpy.array([2.0]), numpy.array([0.0]))
    assert grid.values[row[0], col[0]] == 100
    for _ in range(5):
        grid.ingest_rays([[0.0, 0.0, 0.5, 4.0, 0.0, 0.5]])
    row, col = grid._world_to_rc(numpy.array([2.0]), numpy.array([0.0]))
    assert grid.values[row[0], col[0]] == 0
    from robot_mapping.backends.fastlio import prepare_scan
    from robot_mapping.types import LidarScan

    scan = LidarScan(
        "front",
        "lidar",
        "sensor",
        1_000_000_000,
        100_000_000,
        1,
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.5]],
        numpy.array([0, 10_000_000]),
        [1.0, 2.0],
    )
    prepared = prepare_scan(scan, 0.5, include_origins=True)
    assert prepared.points_array.shape == (1, 8) and prepared.dropped_points == 1
    points, removed = kernels.filter_aabbs(
        prepared.points_array, numpy.array([[[1.0, -1.0, 0.0], [3.0, 1.0, 1.0]]])
    )
    assert len(points) == 0 and removed == 1
    merged, removed, kept = kernels.merge_prepared(
        prepared.points_array,
        prepared.points_array,
        prepared.points_array[:, :3],
        numpy.zeros((1, 3)),
        5.0,
        100.0,
    )
    assert kept == 1 and removed == 0 and merged.shape == (2, 8) and merged[1, 4] == 15.0
    from robot_mapping.filters.urdf_filter import _valid_transform

    assert _valid_transform(numpy.eye(4))
    assert not _valid_transform(numpy.diag([-1.0, 1.0, 1.0, 1.0]))
    # Exercise the optional companion module and its transitive library lookup.
    import gtsam_unstable

    assert Path(gtsam_unstable.__file__).resolve().is_relative_to(root)
    maps = Path("/proc/self/maps").read_text()
    for line in maps.splitlines():
        if any(name in line for name in ("libgtsam", "libmetis-gtsam", "libfast_gicp")):
            path = Path(line.split()[-1]).resolve()
            assert path.is_relative_to(root), line
    print(
        json.dumps(
            {
                "runtime": str(root),
                "imports": {m.__name__: m.__file__ for m in modules},
                "native_execution": "PASS",
                "loaded_libraries": "inside runtime",
                "hardware_accessed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
