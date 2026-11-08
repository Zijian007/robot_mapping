"""Algorithm defaults live in generic.json; user files overlay robot and switches."""
import json
from dataclasses import replace
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


def test_effective_settings_snapshot_roundtrips_all_runtime_overrides(tmp_path):
    from robot_mapping.config import settings_from_profile, profile_from_settings
    from robot_mapping.filters.body_filter import BodyBox
    path = tmp_path / "config.json"
    user = {"robot": {"imu_frame_id": "robot_imu"}, "source": {"clock_id": "sensor"},
            "effective_settings": {"grid_resolution": 999.}}
    original = json.dumps(user, sort_keys=True)
    base = settings_from_profile(user, path)
    # Exercise every numeric/bool setting, including fields previously omitted
    # by MappingSession's handwritten partial snapshot.
    overrides = {}
    for name, value in vars(base).items():
        if isinstance(value, bool):
            overrides[name] = not value
        elif isinstance(value, int):
            overrides[name] = value + 1
        elif isinstance(value, float):
            overrides[name] = value + .001
        elif isinstance(value, tuple):
            overrides[name] = tuple(v + .001 for v in value)
    overrides.update(maps_root=tmp_path / "maps", body_urdf_file=tmp_path / "model.urdf",
        body_package_path=tmp_path / "models", front_box=BodyBox(-1, 1, -2, 2, -3, 3),
        back_box=BodyBox(-2, 2, -3, 3, -4, 4), lidar="back", imu="back", body_filter_mode="box")
    settings = replace(base, **overrides)
    snapshot = profile_from_settings(settings, user)
    path.write_text(json.dumps(snapshot))
    assert load_settings(path) == settings
    assert snapshot["robot"] == user["robot"] and snapshot["source"] == user["source"]
    assert "effective_settings" not in snapshot
    assert json.dumps(user, sort_keys=True) == original
