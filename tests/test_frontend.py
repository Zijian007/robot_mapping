"""Mapping pipeline over injected DTOs; no session, files or HTTP."""
from dataclasses import replace
import json
import threading

import numpy as np
import pytest

from robot_mapping.config import load_settings
from robot_mapping.runtime.frontend import MappingFrontend
from robot_mapping.types import LidarScan, ImuBatch


def config_file(tmp_path):
    profile = {
        "session": {"maps_root": str(tmp_path / "maps")},
        "lio": {"lidar": "front", "lidar_ready": 0., "min_imu_samples": 1, "print_interval": 999.},
        "body_filter": {"enabled": False},
        "online": {"live_png": False, "keyframe_dist_th": .1},
        "robot": {"lidar_to_base": {"front": np.eye(4).tolist()},
                  "imu_from_lidar": np.eye(4).tolist(), "imu_frame_id": "imu_front"},
        "source": {"clock_id": "gdk_sensor"},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(profile))
    return path


class Frames:
    def __init__(self, count=1):
        self.index, self.count, self.exhausted = 0, count, False
        self.closed = False

    def open(self):
        pass

    def next_lidar(self, timeout_s):
        if self.index == self.count:
            self.exhausted = True
            return None
        stamp = 100_000_000_000 + self.index * 100_000_000
        self.index += 1
        return LidarScan("front", "lidar_front", "gdk_sensor", stamp, 100_000_000, 1,
            np.array([[2., 2., .5], [3., 2., .5], [2., 3., .5]]), np.array([0, 50_000_000, 100_000_000]))

    def imu_between(self, begin_ns, end_ns, *, clock_id):
        return ImuBatch("imu_front", clock_id, np.array([end_ns]), np.zeros((1, 3)), np.array([[0., 0., 9.81]]))

    def close(self):
        self.closed = True


class Lio:
    def __init__(self):
        self.index = 0
        self.stamps = []

    def process(self, scan, imu):
        self.index += 1
        self.stamps.append(scan.frame_stamp)
        if self.index <= 10:
            return {"status": "initializing", "stamp": scan.frame_stamp}
        return {"status": "scan_to_map", "stamp": scan.frame_stamp,
            "position": [.2 * (self.index - 11), 0, 0], "orientation_xyzw": [0, 0, 0, 1],
            "scan_body_points": scan.points_array[:, :3]}

    def reset(self):
        self.index = 0


def test_core_frontend_filters_dual_dtos_at_source_times(tmp_path):
    settings = replace(load_settings(config_file(tmp_path)), lidar="both",
        body_filter_enabled=True, body_filter_mode="urdf")
    source = Frames(1)
    original = source.next_lidar
    source.aux_lidar = lambda: replace(source.current, sensor_id="back", begin_ns=source.current.begin_ns + 5_000_000)

    def next_lidar(timeout):
        source.current = original(timeout)
        return source.current

    source.next_lidar = next_lidar
    stamps = []

    class BodyFilter:
        def filter_points(self, points, rotation, translation, stamp, lookup):
            stamps.append(stamp)
            return points[1:], 1

    lio = Lio()
    frontend = MappingFrontend(source, settings, lidar_extrinsics={"front": np.eye(4), "back": np.eye(4)},
        imu_from_lidar=np.eye(4), body_filter=BodyFilter(), body_lookup=lambda *args: np.eye(4),
        clock_id="gdk_sensor", imu_frame_id="imu_front", lio=lio)
    frontend.setup_lio()
    frontend.run(threading.Event())
    assert stamps == [100_000_000_000, 100_005_000_000]
    assert frontend.body_removed_total == 2
    assert frontend.back_merged_frames == 1
    assert lio.stamps == pytest.approx([100.1])


def test_invalid_settings_raise_value_error(tmp_path):
    from robot_mapping.config import validate_settings
    settings = replace(load_settings(config_file(tmp_path)), grid_resolution=0.)
    with pytest.raises(ValueError, match="grid-resolution"):
        validate_settings(settings)
