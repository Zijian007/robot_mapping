"""Timestamped robot geometry without GDK or application dependencies."""
import struct
from unittest.mock import Mock
import numpy as np
import pytest
from robot_mapping.filters.urdf_filter import UrdfBodyFilter, MissingBodyTransform


def model(tmp_path, geometry='<box size="0.2 0.2 0.2"/>', origin="", extra="", padding=0.1):
    path = tmp_path / "robot.urdf"
    path.write_text(f'''<robot name="test"><link name="base_link"/>
      <link name="arm"><collision>{origin}<geometry>{geometry}</geometry></collision></link>
      <joint name="arm_joint" type="revolute"><parent link="base_link"/><child link="arm"/></joint>
      {extra}</robot>''')
    return UrdfBodyFilter(path, tmp_path, padding)


def pose(x=0, angle=0):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0, x], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.]])


def test_moving_link_replaces_old_mask_and_keeps_point_fields(tmp_path):
    body = model(tmp_path)
    points = np.array([[1, 0, 0, 41, 8], [2, 0, 0, 42, 9], [4, 0, 0, 43, 10]], dtype=np.float32)
    lookup = Mock(side_effect=lambda target, source, stamp: pose(stamp))
    first, removed = body.filter_points(points, np.eye(3), np.zeros(3), 1, lookup)
    second, _ = body.filter_points(points, np.eye(3), np.zeros(3), 2, lookup)
    assert removed == 1
    np.testing.assert_array_equal(first, points[[1, 2]])
    np.testing.assert_array_equal(second, points[[0, 2]])
    np.testing.assert_array_equal(points[:, 3], [41, 42, 43])
    assert lookup.call_args_list[0].args == ("base_link", "arm", 1)


def test_rotated_bounds_include_all_eight_corners_in_lidar_frame(tmp_path):
    body = model(tmp_path, '<box size="2 0.2 0.2"/>', padding=0)
    rotation = pose(angle=np.pi / 2)[:3, :3]
    # Link rotated 45 degrees; AABB deliberately also removes its outer corner.
    base_points = np.array([[1, .65, 0], [1.7, .65, 0], [3, 0, 0]])
    translation = np.array([.4, -.3, .2])
    points = (base_points - translation) @ rotation
    kept, removed = body.filter_points(points, rotation, translation, 123,
                                      lambda *_: pose(1, np.pi / 4))
    assert removed == 2
    np.testing.assert_allclose(kept, points[2:])


@pytest.mark.parametrize("geometry", ['<box size="1 1 1"/>',
                                     '<cylinder radius="0.5" length="1"/>',
                                     '<sphere radius="0.5"/>'])
def test_primitive_padding_and_collision_origin_match_reference_bounds(tmp_path, geometry):
    body = model(tmp_path, geometry, origin='<origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>')
    points = np.random.default_rng(5).uniform(0, 4, size=(1000, 5))
    # All three primitive AABBs are cubes here; padding extends each side to 0.6.
    expected_inside = np.all(np.abs(points[:, :3] - [1, 2, 3]) <= .6, axis=1)
    kept, removed = body.filter_points(points, np.eye(3), np.zeros(3), 123, lambda *_: pose())
    assert removed == expected_inside.sum()
    np.testing.assert_array_equal(kept, points[~expected_inside])


@pytest.mark.parametrize("binary", [False, True])
def test_stl_scale_collision_origin_padding_and_fixed_child(tmp_path, binary):
    mesh_path = tmp_path / "robot_description" / "part.stl"
    mesh_path.parent.mkdir()
    vertices = [[0, 0, 0], [1, 1, 1], [0, 1, 0]]
    if binary:
        mesh_path.write_bytes(b"solid binary".ljust(80, b" ") + struct.pack("<I", 1)
                             + struct.pack("<12fH", 0, 0, 1, *np.ravel(vertices), 0))
    else:
        mesh_path.write_text("solid test\n" + "\n".join("vertex " + " ".join(map(str, v)) for v in vertices)
                             + "\nendsolid test\n")
    body = model(tmp_path, extra='''<link name="tool"><collision><origin xyz="0 2 0"/>
      <geometry><mesh filename="package://robot_description/part.stl" scale="2 1 1"/></geometry>
      </collision></link><joint name="tool_mount" type="fixed"><parent link="arm"/>
      <child link="tool"/><origin xyz="0 0 3"/></joint>''')
    lookup = Mock(return_value=pose(1))
    points = np.array([[3.05, 2.5, 3.5], [3.2, 2.5, 3.5], [5, 0, 0]])
    kept, removed = body.filter_points(points, np.eye(3), np.zeros(3), 5, lookup)
    assert removed == 1
    np.testing.assert_array_equal(kept, points[1:])
    assert lookup.call_count == 1  # Fixed tool composes from the arm; no tool TF required.


def test_missing_or_invalid_tf_rejects_whole_scan(tmp_path):
    body = model(tmp_path)
    points = np.array([[0, 0, 0, 1, 7], [2, 0, 0, 2, 8]])
    for lookup in [Mock(side_effect=RuntimeError("no tf")), Mock(return_value=np.full((4, 4), np.nan)),
                   Mock(return_value=np.diag([2, 1, 1, 1]))]:
        with pytest.raises(MissingBodyTransform, match="arm"):
            body.filter_points(points, np.eye(3), np.zeros(3), 123, lookup)
    with pytest.raises(MissingBodyTransform, match="时间戳"):
        body.filter_points(points, np.eye(3), np.zeros(3), 0, Mock())
    np.testing.assert_array_equal(points[:, 3:], [[1, 7], [2, 8]])


def test_invalid_assets_fail_loading_instead_of_silently_disabling_filter(tmp_path):
    with pytest.raises(FileNotFoundError):
        model(tmp_path, '<mesh filename="missing.stl"/>')
    with pytest.raises(ValueError, match="collision"):
        model(tmp_path, '<unsupported/>')
