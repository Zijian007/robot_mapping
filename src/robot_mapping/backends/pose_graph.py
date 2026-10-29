"""GTSAM Pose3 / iSAM2 pose-graph backend for online mapping loop closure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_mapping.core.pose_math import matrix_to_pose_parts, is_se3
from .capabilities import load_gtsam


@dataclass(frozen=True)
class PoseGraphNoise:
    odom_sigmas: tuple[float, float, float, float, float, float] = (
        0.05,
        0.05,
        0.05,
        0.02,
        0.02,
        0.02,
    )
    loop_sigmas: tuple[float, float, float, float, float, float] = (
        0.1,
        0.1,
        0.1,
        0.05,
        0.05,
        0.05,
    )
    prior_sigmas: tuple[float, float, float, float, float, float] = (
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
    )

    def __post_init__(self):
        for values in (self.odom_sigmas, self.loop_sigmas, self.prior_sigmas):
            array = np.asarray(values)
            if array.shape != (6,) or not np.isfinite(array).all() or np.any(array <= 0):
                raise ValueError("Pose3 noise needs six finite positive sigmas (rotation, translation)")


class PoseGraphBackend:
    """Thin wrapper: nodes = Pose3, odom/loop = BetweenFactorPose3, solver = iSAM2."""

    def __init__(self, noise: PoseGraphNoise | None = None) -> None:
        gtsam = load_gtsam()

        self._gtsam = gtsam
        self.noise = noise or PoseGraphNoise()
        self._params = gtsam.ISAM2Params()
        self._params.setRelinearizeThreshold(0.1)
        self._isam = gtsam.ISAM2(self._params)
        self._odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(self.noise.odom_sigmas, dtype=np.float64)
        )
        self._loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(self.noise.loop_sigmas, dtype=np.float64)
        )
        self._prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.asarray(self.noise.prior_sigmas, dtype=np.float64)
        )
        self._n_nodes = 0
        self._loop_count = 0
        self._factor_id = 0

    def reset(self) -> None:
        self._isam = self._gtsam.ISAM2(self._params)
        self._n_nodes = 0
        self._loop_count = 0
        self._factor_id = 0

    @property
    def size(self) -> int:
        return self._n_nodes

    @property
    def loop_count(self) -> int:
        return self._loop_count

    def _pose3_from_matrix(self, mat: np.ndarray) -> Any:
        if not is_se3(mat):
            raise ValueError("invalid SE3 factor measurement")
        gtsam = self._gtsam
        r = mat[:3, :3]
        t = mat[:3, 3]
        return gtsam.Pose3(gtsam.Rot3(r), gtsam.Point3(t[0], t[1], t[2]))

    def _matrix_from_pose3(self, pose3: Any) -> np.ndarray:
        t = np.eye(4, dtype=np.float64)
        t[:3, :3] = np.asarray(pose3.rotation().matrix(), dtype=np.float64)
        p = pose3.translation()
        t[0, 3] = float(p[0])
        t[1, 3] = float(p[1])
        t[2, 3] = float(p[2])
        return t

    def add_keyframe(self, world_T_body: np.ndarray, *, relative_to_prev: np.ndarray | None) -> int:
        """Insert node. relative_to_prev is T_{k-1}_k when k>0."""
        gtsam = self._gtsam
        idx = self._n_nodes
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        pose = self._pose3_from_matrix(world_T_body)
        values.insert(idx, pose)
        if idx == 0:
            graph.add(gtsam.PriorFactorPose3(0, pose, self._prior_noise))
        else:
            if relative_to_prev is None:
                raise ValueError("relative_to_prev required for keyframe > 0")
            between = self._pose3_from_matrix(relative_to_prev)
            graph.add(
                gtsam.BetweenFactorPose3(idx - 1, idx, between, self._odom_noise)
            )
        self._isam.update(graph, values)
        self._n_nodes = idx + 1
        return idx

    def add_loop(
        self,
        i: int,
        j: int,
        T_i_j: np.ndarray,
        *,
        noise: np.ndarray | None = None,
    ) -> None:
        if i < 0 or j < 0 or i >= self._n_nodes or j >= self._n_nodes:
            raise IndexError(f"loop indices out of range: {i}, {j}, n={self._n_nodes}")
        gtsam = self._gtsam
        graph = gtsam.NonlinearFactorGraph()
        model = self._loop_noise
        if noise is not None:
            model = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(noise, dtype=np.float64))
        graph.add(gtsam.BetweenFactorPose3(i, j, self._pose3_from_matrix(T_i_j), model))
        self._isam.update(graph, gtsam.Values())
        self._isam.update()
        self._loop_count += 1

    def optimized_matrices(self) -> list[np.ndarray]:
        estimate = self._isam.calculateEstimate()
        out: list[np.ndarray] = []
        for i in range(self._n_nodes):
            out.append(self._matrix_from_pose3(estimate.atPose3(i)))
        return out

    def optimized_pose(self, idx: int) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        mats = self.optimized_matrices()
        return matrix_to_pose_parts(mats[idx])

    def predicted_matrix(self, idx: int) -> np.ndarray:
        return self.optimized_matrices()[idx]
