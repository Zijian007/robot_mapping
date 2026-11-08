# 工具索引

以下命令从 `robot_mapping/` 执行。部署脚本保持现有路径，离线恢复包也使用这些路径。

| 用途 | 入口 |
|---|---|
| 安装锁定运行环境 | `install_runtime.sh` / `install_runtime.py` |
| 诊断当前安装 | 根目录 `run.sh doctor --native --loop` |
| 执行源码测试 | `run_tests.sh`，可用 `PYTHON=.venv/bin/python` 选择解释器 |
| 构建原生依赖 | `build_native.sh`，内部使用 `patch_fast_gicp.py` |
| 导出锁定 wheelhouse | `runtime_bundle.py` |
| 打包离线恢复包 | `package_runtime.py` |
| 验证原生依赖构建 | `verify_native.py` |
| 验证单个算法 wheel | `verify_wheel.py` |
| 验证已安装运行环境 | `verify_runtime.py` |
| 验证完整离线恢复包 | `verify_restore.py` |

安装流程见 [运行环境说明](../docs/runtime.md)，源码与 wheel 验证见
[开发指南](../docs/development.md)。一次性实验脚本保存在本机
`artifacts/history/20261108/build/`，其中硬编码的历史路径不是受支持的工具接口。
