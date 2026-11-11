# 算法开发与构建

`robot_mapping` 是独立的算法包，依赖和构建入口由 `pyproject.toml` 声明。
安装、开发、构建和算法测试均不要求 `robot_mapping_gdk`、`robot_hal` 或设备 SDK。

## 独立安装

使用已有 Python 3.10+ 环境，或在仓库外创建开发环境。从算法仓库根目录执行：

```bash
python3 -m venv ~/.venvs/robot-dev
source ~/.venvs/robot-dev/bin/activate
python -m pip install .
python -m robot_mapping doctor
```

虚拟环境由开发者或调用方选择，不是算法包的一部分，也不绑定到任何应用目录。
基础安装仅依赖 NumPy；`.[runtime]` 增加 YAML 支持。

使用默认 LIO 时，另行安装符合本包接口要求的 [FAST-LIO2](../../fastlio2/README.md)，
或通过 `LioBackend` 注入其他实现。启用回环时安装 `.[loop]` 和带收敛状态、GIL 补丁的
FastVGICP。回环使用支持 NumPy 2 的 `gtsam==4.2.2+numpy2.1` 构建，无需降级 NumPy。
这些依赖均为算法后端，不需要通过 GDK 应用安装。

### 构建回环后端 wheel

只有升级 GTSAM、FastVGICP 或 pybind11 时才需要重新编译，已有 wheel 可直接安装：

```bash
PYTHON=/path/to/venv/bin/python bash scripts/build_loop_backends.sh [all|gtsam|fast-gicp]
```

- 源码和中间文件放在 `${XDG_CACHE_HOME:-~/.cache}/robot_mapping/loop-backends`（`WORK_DIR` 可覆盖），
  可随时删除；wheel 输出到 `dist/`（`OUTPUT` 可覆盖）。
- GTSAM 4.2.2 由 `patch_gtsam.py` 改用解释器中的 pybind11 2.12+，版本标为 `4.2.2+numpy2.1`；
  FastVGICP 固定修订，由 `patch_fast_gicp.py` 增加收敛状态和 GIL 释放。
- `package_loop_backends.py` 把依赖的动态库打进 wheel，RPATH 改为 `$ORIGIN`，安装后不依赖构建目录。
- 构建结束时从临时目录安装新 wheel，执行 `robot_mapping doctor --loop`；不改动所选解释器的环境。
- 并行数默认取 CPU 亲和度允许的核数（`JOBS=N` 可覆盖）。GTSAM 绑定单个编译进程峰值约 6GB 内存，
  在 Orin 上全量编译需要数十分钟。

```bash
# 仅检查已选用的后端，不连接硬件
python -m robot_mapping doctor --native
python -m robot_mapping doctor --loop
```

## 源码开发

在独立开发环境中安装可编辑源码和开发工具：

```bash
python -m pip install -e '.[dev]'
PYTHON=python bash scripts/run_tests.sh
```

`run_tests.sh` 从本仓库读取源码，使用 `PYTHON` 指定的解释器，不安装依赖。
需要 YAML 时改用 `python -m pip install -e '.[dev,runtime]'`。
没有可选后端时，相应原生测试会跳过；验证相关改动时应安装对应后端。
格式及类型约定见 [代码规范](code-style.md)，按改动影响范围选择必要检查。
修改公共接口时，再检查实际调用方；GDK 应用的集成验证属于应用侧工作。

## 构建 wheel

```bash
python -m pip wheel --no-deps --wheel-dir dist .
```

wheel 可独立分发，在匹配 Python ABI 和平台的调用方环境中安装。
普通安装、可编辑安装、wheel 安装、uv 环境替代命令和卸载统一见 [README](../README.md#安装)。
默认构建隔离会安装 `pyproject.toml` 声明的构建依赖。
离线构建需提前提供 setuptools、wheel、pybind11 和所需依赖，使用
`--no-index --no-build-isolation`，或通过 `--find-links` 指定离线依赖目录。

射线扩展需要 C++17 编译器和 pybind11，安装已编译 wheel 无需编译工具。
开发中验证 C++ 改动时可原地构建：

```bash
python -m pip install 'setuptools>=68' wheel 'pybind11>=2.11'
python setup.py build_ext --inplace
```

没有扩展时仍可使用 Python/NumPy 参考实现，原生专用测试会跳过。
本包 wheel 不包含 FAST-LIO2、FastVGICP 或 GTSAM，使用方按需安装。

## 应用集成

任意 Python 应用都可以安装本包并实现采集端口。依赖方向为
`应用 → robot_mapping → 所选算法后端`，算法包不反向依赖应用。

`robot_mapping_gdk` 是面向 G2 的源码应用，负责采集、会话和设备部署，
该应用统一使用 `/home/agi/.venv`，离线 wheelhouse 和安装脚本用于补齐共享环境的依赖，详见
[应用运行环境](../../robot_mapping_gdk/docs/runtime.md)。算法包仍可安装到任意调用方环境。
应用使用安装的 wheel 时，算法源码修改后需重新构建并更新应用内的依赖。

## 目录维护

| 路径 | 内容 |
|---|---|
| `src/`、`csrc/` | Python 算法和 C++ 扩展 |
| `tests/`、`tests/fixtures/` | 功能测试及合成场景固定数据 |
| `scripts/` | 算法构建、检查及按需性能诊断工具 |
| `docs/` | 当前算法、参数和接口说明 |
| `build/`、`dist/`、`*.egg-info` | 构建时生成，使用后可清理 |

Python 参考实现、共享空间索引及投影模块都有实际用途，不能仅凭内部引用次数删除。
实验报告、备份和历史流水账不作为项目交付物保存。
