"""Mapping pipeline over injected DTOs; no session, files or HTTP."""

import json
import threading
from dataclasses import replace

import numpy as np
import pytest

from robot_mapping.config import load_settings
from robot_mapping.runtime.frontend import MappingFrontend
from robot_mapping.types import ImuBatch, LidarScan


def config_file(tmp_path):
    profile = {
        "session": {"maps_root": str(tmp_path / "maps")},
        "lio": {
            "lidar": "front",
            "lidar_ready": 0.0,
            "min_imu_samples": 1,
            "print_interval": 999.0,
        },
        "body_filter": {"enabled": False},
        "online": {"live_png": False, "keyframe_dist_th": 0.1},
        "robot": {
            "lidar_to_base": {"front": np.eye(4).tolist()},
            "imu_from_lidar": np.eye(4).tolist(),
            "imu_frame_id": "imu_front",
        },
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
        return LidarScan(
            "front",
            "lidar_front",
            "gdk_sensor",
            stamp,
            100_000_000,
            1,
            np.array([[2.0, 2.0, 0.5], [3.0, 2.0, 0.5], [2.0, 3.0, 0.5]]),
            np.array([0, 50_000_000, 100_000_000]),
        )

    def imu_between(self, begin_ns, end_ns, *, clock_id):
        stamps = np.arange(begin_ns + 5_000_000, end_ns + 1, 5_000_000, dtype=np.int64)
        return ImuBatch(
            "imu_front",
            clock_id,
            stamps,
            np.zeros((len(stamps), 3)),
            np.tile([0.0, 0.0, 9.81], (len(stamps), 1)),
        )

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
        return {
            "status": "scan_to_map",
            "stamp": scan.frame_stamp,
            "lidar_update_valid": True,
            "position": [0.2 * (self.index - 11), 0, 0],
            "orientation_xyzw": [0, 0, 0, 1],
            "scan_body_points": scan.points_array[:, :3],
        }

    def reset(self):
        self.index = 0


def test_core_frontend_filters_dual_dtos_at_source_times(tmp_path):
    base_settings = load_settings(config_file(tmp_path))
    settings = replace(
        base_settings,
        lio=replace(base_settings.lio, lidar="both"),
        body_filter=replace(base_settings.body_filter, enabled=True, mode="urdf"),
    )
    source = Frames(1)
    original = source.next_lidar
    source.aux_lidar = lambda: replace(
        source.current, sensor_id="back", begin_ns=source.current.begin_ns + 5_000_000
    )

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
    frontend = MappingFrontend(
        source,
        settings,
        lidar_extrinsics={"front": np.eye(4), "back": np.eye(4)},
        imu_from_lidar=np.eye(4),
        body_filter=BodyFilter(),
        body_lookup=lambda *args: np.eye(4),
        clock_id="gdk_sensor",
        imu_frame_id="imu_front",
        lio=lio,
    )
    frontend.setup_lio()
    frontend.run(threading.Event())
    assert stamps == [100_000_000_000, 100_005_000_000]
    assert frontend.body_removed_total == 2
    assert frontend.back_merged_frames == 1
    assert lio.stamps == pytest.approx([100.1])


def test_invalid_settings_raise_value_error(tmp_path):
    from robot_mapping.config import validate_settings

    base_settings = load_settings(config_file(tmp_path))
    settings = replace(base_settings, grid=replace(base_settings.grid, resolution=0.0))
    with pytest.raises(ValueError, match="grid-resolution"):
        validate_settings(settings)


@pytest.mark.parametrize("validity", [False, None])
def test_map_refuses_invalid_or_unreported_lidar_update(validity):
    from robot_mapping.core.online_map import OnlineMap

    mapping = OnlineMap()
    odom = {
        "status": "scan_to_map",
        "position": [0, 0, 0],
        "orientation_xyzw": [0, 0, 0, 1],
        "scan_body_points": [[1, 2, 0.5]],
    }
    if validity is not None:
        odom["lidar_update_valid"] = validity
    assert not mapping.ingest_odom(odom)
    assert mapping.keyframe_count() == 0
    assert len(mapping.voxels) == 0


@pytest.mark.parametrize(
    "field",
    [
        "max_imu_gap_s",
        "max_imu_boundary_gap_s",
        "imu_startup_timeout_s",
        "max_match_failures",
        "min_imu_samples",
    ],
)
def test_tracking_guard_settings_must_be_positive(tmp_path, field):
    from robot_mapping.config import validate_settings

    base_settings = load_settings(config_file(tmp_path))
    settings = replace(base_settings, lio=replace(base_settings.lio, **{field: 0}))
    with pytest.raises(ValueError, match=field):
        validate_settings(settings)


@pytest.mark.parametrize("inject_policy", [False, True])
def test_frontend_delegates_frame_choice_and_defaults_to_backend_coordinates(
    tmp_path, inject_policy
):
    from robot_mapping.core.online_map import OnlineMap

    settings = load_settings(config_file(tmp_path))
    mapping = OnlineMap()
    calls = []
    custom = np.eye(4)
    custom[:3, 3] = [10, 20, 30]

    def initialize(mean_acc, generation):
        calls.append((mean_acc.copy(), generation))
        return mapping.set_map_alignment(custom, generation=generation)

    mounting = np.eye(4)
    mounting[:3, 3] = [0.2, 0.3, 0.4]
    frontend = MappingFrontend(
        Frames(14),
        settings,
        online_map=mapping,
        lidar_extrinsics={"front": mounting},
        clock_id="gdk_sensor",
        imu_frame_id="imu_front",
        lio=Lio(),
        initialize_map=initialize if inject_policy else None,
    )
    frontend.setup_lio()
    frontend.run(threading.Event())
    np.testing.assert_array_equal(
        mapping.keyframe_opt_matrix(0), custom if inject_policy else np.eye(4)
    )
    if inject_policy:
        assert len(calls) == 1
        np.testing.assert_allclose(calls[0][0], [0, 0, 9.81])
        assert calls[0][1] == mapping.generation


def test_dual_lidar_merge_transforms_ray_origins_with_endpoints():
    from robot_mapping.backends.fastlio import BackendScan
    from robot_mapping.runtime.scan_preprocessing import merge_back_into_front

    front = BackendScan(np.array([[2, 0, 0.5, 1, 20, 0, 0, 0]], dtype=np.float32), 10.0, 0.1, 1, 0)
    back = replace(front, points_array=front.points_array.copy())
    rotation = np.diag([-1.0, -1.0, 1.0])
    translation = np.array([-0.6, 0.1, 0.02])
    merged, _, count = merge_back_into_front(
        front,
        back,
        rotation_front=np.eye(3),
        translation_front=np.zeros(3),
        rotation_back=rotation,
        translation_back=translation,
        body_box=None,
        max_scan_duration_s=0.12,
    )
    assert count == 1
    np.testing.assert_allclose(merged.points_array[0, 5:8], 0.0)
    np.testing.assert_allclose(merged.points_array[1, 5:8], translation)
    np.testing.assert_allclose(merged.points_array[1, :3], rotation @ [2, 0, 0.5] + translation)
