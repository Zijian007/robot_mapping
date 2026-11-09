"""Optional standalone ray kernels; no PCL, Eigen, SDK or LIO dependency."""
from setuptools import Extension, setup
import pybind11

setup(ext_modules=[Extension(
    "robot_mapping._ray_native", ["csrc/ray_native.cpp"],
    depends=["csrc/grid_update.hpp", "csrc/point_preprocess.hpp"],
    include_dirs=[pybind11.get_include()], language="c++", optional=True,
    extra_compile_args=["-O3", "-std=c++17", "-ffp-contract=off"],
)])
