# 算法开发工具

从 `robot_mapping/` 执行，使用独立开发环境或调用方提供的 Python 环境，
无需 GDK 应用。环境安装与 wheel 构建见 [开发指南](../docs/development.md)。

| 用途 | 入口 |
|---|---|
| Python/C++ 规范检查 | `check_quality.py` |
| 执行源码测试 | `run_tests.sh`，使用 `PYTHON` 指定解释器 |
| 验证算法 wheel | `verify_wheel.py` |
| 构建回环后端 wheel（GTSAM、FastVGICP） | `build_loop_backends.sh`，调用 `patch_gtsam.py`、`patch_fast_gicp.py`、`package_loop_backends.py` |
| 固定观测的体素存储对比 | `benchmark_voxels.py` |
| 固定观测的栅格对比与阶段测量 | `benchmark_grid.py` |
| 录包预处理及双雷达合并对比 | `benchmark_preprocess.py` |

G2 应用的安装、离线部署和启动工具位于
[robot_mapping_gdk](../../robot_mapping_gdk/scripts/README.md)，不参与本算法包的独立安装和构建。
性能工具在定位具体瓶颈时按需使用，输入和输出目录由调用方提供。
