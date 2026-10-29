"""Algorithm defaults live in generic.json; user files overlay robot and switches."""
import json
from importlib.resources import files

from robot_mapping.config import algorithm_profile, load_settings, resolve_profile


def test_algorithm_defaults_exclude_robot_and_source():
    profile = algorithm_profile()
    assert "robot" not in profile
    assert "source" not in profile
    assert "session" not in profile
    assert profile["lio"]["filter_size_map"] == 0.15
    assert profile["body_filter"]["enabled"] is False


def test_empty_user_overlay_merges_algorithm_knobs():
    overlay = json.loads(files("robot_mapping.resources").joinpath("user.json").read_text())
    assert "lio" not in overlay
    assert overlay["robot"]["imu_from_lidar"] is None
    merged = resolve_profile(overlay)
    assert merged["lio"]["filter_size_map"] == 0.15
    assert merged["body_filter"]["enabled"] is False
    assert merged["robot"]["imu_frame_id"] == ""


def test_partial_overlay_keeps_algorithm_defaults():
    overlay = {"lio": {"lidar": "both", "lidar_ready": 2.0}, "body_filter": {"enabled": True}}
    merged = resolve_profile(overlay)
    assert merged["lio"]["lidar"] == "both"
    assert merged["lio"]["lidar_ready"] == 2.0
    assert merged["lio"]["filter_size_map"] == 0.15
    assert merged["lio"]["cube_side_length"] == 1000.0
    assert merged["body_filter"]["enabled"] is True
    assert merged["body_filter"]["mode"] == "urdf"
    assert resolve_profile(merged) == merged


def test_load_settings_reads_overlay_on_defaults(tmp_path):
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps({
        "lio": {"lidar": "both", "lidar_ready": 2.0},
        "body_filter": {"enabled": True},
    }))
    settings = load_settings(path)
    assert settings.filter_size_map == 0.15
    assert settings.lidar == "both"
    assert settings.lidar_ready == 2.0
    assert settings.body_filter_enabled is True
