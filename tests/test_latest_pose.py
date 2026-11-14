"""The newest footprint pose is available for every accepted frame, for live display."""
import numpy as np

from robot_mapping.core.map_types import SE3Pose
from robot_mapping.core.online_map import OnlineMap


def _yaw(angle):
    m = np.eye(4)
    m[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return m


def test_latest_footprint_pose_follows_every_frame_and_resets():
    mapping = OnlineMap(voxel_resolution=0.15, grid_resolution=0.05, keyframe_dist_th=1.0)
    assert mapping.latest_footprint_pose() is None
    # Tracking frame (IMU) sits 0.2 m ahead of the footprint; map rotated 90 deg from LIO.
    tracking_from_footprint = np.eye(4)
    tracking_from_footprint[0, 3] = -0.2
    mapping.set_map_alignment(_yaw(np.pi / 2), tracking_from_footprint=tracking_from_footprint)
    scene = np.random.default_rng(3).normal(size=(300, 3)) + [3.0, 0.0, 0.3]
    for stamp, x in enumerate((0.0, 0.3)):
        mapping.ingest_local_points(SE3Pose((x, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                                    scene - [x, 0, 0], stamp=float(stamp))
    # 0.3 m is below the keyframe distance, so only the first frame became a keyframe,
    # yet the pose still tracks the second frame.
    assert mapping.keyframe_count() == 1
    pose = mapping.latest_footprint_pose()
    expected = _yaw(np.pi / 2)
    expected[:3, 3] = expected[:3, :3] @ [0.3 - 0.2, 0.0, 0.0]
    np.testing.assert_allclose(pose, expected, atol=1e-12)
    pose[0, 3] = 99.0  # a copy, never the internal state
    assert mapping.latest_footprint_pose()[0, 3] != 99.0
    mapping.cancel_mapping()
    assert mapping.latest_footprint_pose() is None
