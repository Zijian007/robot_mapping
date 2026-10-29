"""Explicit robot configuration and resource path resolution."""
from pathlib import Path
import numpy as np

from robot_mapping.core.pose_math import is_se3


def calibration(profile, settings):
    robot = profile.get("robot",{})
    if not robot.get("imu_frame_id"):
        raise ValueError("robot.imu_frame_id must explicitly identify the calibrated IMU")
    matrices = {name:np.asarray(value,dtype=np.float64) for name,value in robot.get("lidar_to_base",{}).items()}
    required = {"front","back"} if settings.needs_back else {settings.primary_lidar}
    if not required <= matrices.keys():
        raise ValueError(f"robot.lidar_to_base requires {sorted(required)}")
    imu = np.asarray(robot.get("imu_from_lidar"),dtype=np.float64)
    if not all(is_se3(v) for v in (*matrices.values(),imu)):
        raise ValueError("robot calibration must contain finite rigid 4x4 transforms")
    if settings.imu != settings.primary_lidar and not robot.get("cross_sensor_imu_calibrated",False):
        raise ValueError("mixed LiDAR/IMU selection requires explicit calibrated imu_from_lidar and cross_sensor_imu_calibrated")
    return matrices,imu

