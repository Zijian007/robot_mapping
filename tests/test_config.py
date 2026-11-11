"""Algorithm defaults live in generic.json; user files overlay robot and switches."""

import json
from dataclasses import replace
from importlib.resources import files

import pytest

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
    path.write_text(
        json.dumps(
            {
                "lio": {"lidar": "both", "lidar_ready": 2.0},
                "body_filter": {"enabled": True},
            }
        )
    )
    settings = load_settings(path)
    assert settings.lio.filter_size_map == 0.15
    assert settings.lio.lidar == "both"
    assert settings.lio.lidar_ready == 2.0
    assert settings.body_filter.enabled is True


def test_effective_settings_snapshot_roundtrips_all_runtime_overrides(tmp_path):
    from robot_mapping.config import profile_from_settings, settings_from_profile
    from robot_mapping.filters.body_filter import BodyBox

    path = tmp_path / "config.json"
    user = {
        "robot": {"imu_frame_id": "robot_imu"},
        "source": {"clock_id": "sensor"},
        "effective_settings": {"grid_resolution": 999.0},
    }
    original = json.dumps(user, sort_keys=True)
    base = settings_from_profile(user, path)
    # Exercise every numeric/bool setting, including fields previously omitted
    # by MappingSession's handwritten partial snapshot.
    from dataclasses import is_dataclass

    def perturb_fields(section):
        changed_fields = {}
        for name, value in vars(section).items():
            if is_dataclass(value):
                changed_fields[name] = perturb_fields(value)
            elif isinstance(value, bool):
                changed_fields[name] = not value
            elif isinstance(value, int):
                changed_fields[name] = value + 1
            elif isinstance(value, float):
                changed_fields[name] = value + 0.001
            elif isinstance(value, tuple):
                changed_fields[name] = tuple(sigma + 0.001 for sigma in value)
        return replace(section, **changed_fields)

    changed = perturb_fields(base)
    settings = replace(
        changed,
        maps_root=tmp_path / "maps",
        lio=replace(changed.lio, lidar="back", imu="back"),
        body_filter=replace(
            changed.body_filter,
            urdf_file=tmp_path / "model.urdf",
            package_path=tmp_path / "models",
            front=BodyBox(-1, 1, -2, 2, -3, 3),
            back=BodyBox(-2, 2, -3, 3, -4, 4),
            mode="box",
        ),
    )
    snapshot = profile_from_settings(settings, user)
    path.write_text(json.dumps(snapshot))
    assert load_settings(path) == settings
    assert snapshot["robot"] == user["robot"] and snapshot["source"] == user["source"]
    assert "effective_settings" not in snapshot
    assert json.dumps(user, sort_keys=True) == original


@pytest.mark.parametrize("budget", [None, 2048])
def test_voxel_budget_legacy_fallback_and_explicit_override_roundtrip(tmp_path, budget):
    from robot_mapping.config import profile_from_settings, settings_from_profile, validate_settings

    profile = {"grid": {"max_rays": 8192}}
    if budget is not None:
        profile["online"] = {"voxel_max_rays": budget}
    path = tmp_path / "config.json"
    settings = settings_from_profile(profile, path)
    validate_settings(settings)
    assert settings.grid.max_rays == 8192
    assert settings.online.voxel_max_rays == (8192 if budget is None else budget)
    snapshot = profile_from_settings(settings, profile)
    assert snapshot["online"]["voxel_max_rays"] == settings.online.voxel_max_rays
    assert settings_from_profile(snapshot, path) == settings


@pytest.mark.parametrize("budget", [0, -1])
def test_voxel_budget_must_be_positive(budget):
    from robot_mapping.config import settings_from_profile, validate_settings

    with pytest.raises(ValueError, match="voxel_max_rays"):
        validate_settings(settings_from_profile({"online": {"voxel_max_rays": budget}}, "unused"))


@pytest.mark.parametrize(
    "profile",
    [
        {"grid": {"free_space_mode": "any"}},
        {
            "grid": {"free_space_mode": "projected", "ray_clearing": False},
            "online": {"voxel_clearing": False},
        },
    ],
)
def test_invalid_free_space_projection_rejected(profile):
    from robot_mapping.config import settings_from_profile, validate_settings

    with pytest.raises(ValueError):
        validate_settings(settings_from_profile(profile, "unused"))


@pytest.mark.parametrize("value", [0, 128, 2.5, True, "4", None])
def test_invalid_projected_observation_count_rejected(value):
    from robot_mapping.config import settings_from_profile, validate_settings

    with pytest.raises(ValueError, match="projected_min_observations"):
        validate_settings(
            settings_from_profile({"grid": {"projected_min_observations": value}}, "unused")
        )


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_projected_range_rejected(value):
    from robot_mapping.config import settings_from_profile, validate_settings

    with pytest.raises(ValueError, match="projected_max_range"):
        validate_settings(settings_from_profile({"grid": {"projected_max_range": value}}, "unused"))


def test_old_profile_keeps_two_observations_and_original_range():
    from robot_mapping.config import settings_from_profile

    settings = settings_from_profile({"grid": {"max_ray_range": 12.0}}, "unused")
    assert settings.grid.projected_min_observations == 2
    assert settings.grid.projected_max_range == 12.0
