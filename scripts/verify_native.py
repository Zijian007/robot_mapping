"""Exercise the native interfaces that previously imported but crashed at runtime."""
import numpy as np
import gtsam
import pygicp
import fastlio2

if np.__version__ != "1.26.4":
    raise SystemExit(f"Expected private NumPy 1.26.4, got {np.__version__}: {np.__file__}")
if not getattr(pygicp, "mapping_gil_released", False):
    raise SystemExit("pygicp binding needs the convergence/GIL patch")
pose = gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1., 2., 3.))
graph = gtsam.NonlinearFactorGraph()
values = gtsam.Values()
values.insert(0, pose)
graph.add(gtsam.PriorFactorPose3(0, pose, gtsam.noiseModel.Diagonal.Sigmas(np.ones(6)*.01)))
isam = gtsam.ISAM2()
isam.update(graph, values)
np.testing.assert_allclose(isam.calculateEstimate().atPose3(0).translation(), [1, 2, 3])
points = np.random.default_rng(9).normal(size=(500, 3))
reg = pygicp.FastVGICP()
reg.set_num_threads(1)
reg.set_resolution(.5)
reg.set_input_source(points)
reg.set_input_target(points + [.1, -.1, 0])
transform = reg.align(np.eye(4))
assert reg.has_converged()
np.testing.assert_allclose(transform[:3, 3], [.1, -.1, 0], atol=.07)
assert fastlio2.native_available()
from fastlio2 import _fastlio2
assert getattr(_fastlio2, "SCAN_BODY_POINTS_VERSION", 0) == 1
print("[third_party] NumPy / GTSAM Pose3+iSAM2 / FastVGICP / FAST-LIO2 scan export: OK")
