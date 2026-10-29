# 独立运行环境与离线恢复

`robot_mapping` 自己管理算法运行环境。move_odom 是独立的地图消费者，不使用这个解释器；算法包及其原生依赖均不从导航目录加载。SDK、模型和标定继续由设备适配层提供。

## 布局与启动

```text
robot_mapping/
  .venv/                 独立 Python 环境，包含全部算法依赖和自有动态库
  wheelhouse/            28 个锁定 wheel、requirements.lock、manifest.json
  run.sh                 离线入口：doctor（本包 .venv）
  scripts/               构建、导出、安装、验证与恢复包工具
```

在线 G2 启动在 `robot_mapping_gdk/scripts/`（`run_g2.sh`、`drive_g2.sh`、`install_map.sh`），用同一个 `.venv`。

```bash
bash scripts/install_runtime.sh
bash run.sh doctor --native --loop
bash ../robot_mapping_gdk/scripts/run_g2.sh
python -m robot_mapping_gdk replay /path/to/recording --maps-root ./maps
```

无需为 GTSAM/VGICP 设置 PYTHONPATH 或 LD_LIBRARY_PATH。安装器清除继承的这些变量，创建不继承系统 site-packages 的 venv，使用 `--no-index --require-hashes` 安装。校验 Python ABI、CPU 架构、libc 和每个 wheel 摘要；遇到已有非托管目录拒绝修改。相同依赖锁重复安装只做验证，失败环境不会被启动入口当作就绪环境。

默认路径为本项目 `.venv`。可用 `--venv /path/to/env` 安装到其他位置，运行时设 `MAPPING_VENV=/path/to/env`。脚本不修改共用 `/home/agi/.venv`。旧 `/home/agi/move_odom/tools/mapping/.third_party` 已删除，当前启动和安装均不依赖它。历史恢复包及源码快照位于 `/data/robot_mapping-backups/`；该目录只用于回退，不参与运行。

GDK SDK 未包含在依赖集中。需要在线使用时，先配置 SDK 的动态库环境；可用 `MAPPING_SDK_PYTHONPATH=/path/to/sdk/python` 显式提供 SDK 模块路径。新入口不继承普通 PYTHONPATH，以免误加载其他工程的 NumPy 或原生扩展。

## 备份与恢复

生成不含 SDK、模型、地图、日志或凭证的恢复包：

```bash
.venv/bin/python scripts/package_runtime.py \
  --output dist/robot-mapping-runtime-cp310-linux_aarch64.tar.gz
```

备份生成的 tar.gz 和对应 sha256 文件到重刷系统时会保留的位置。恢复到相同 ABI/操作系统环境：

```bash
sha256sum -c robot-mapping-runtime-cp310-linux_aarch64.tar.gz.sha256
tar -xzf robot-mapping-runtime-cp310-linux_aarch64.tar.gz
cd robot-mapping-runtime
bash scripts/install_runtime.sh
bash run.sh doctor --native --loop
```

恢复器重新创建 venv，不复制不可迁移的旧 venv 或绝对路径 shebang。不需要原 move_odom、fastlio2 源码或联网下载 Python 包。恢复包中的 wheels 已包含本次验证过的 Python 模块及原生扩展。

**适用范围：Ubuntu 22.04、aarch64、CPython 3.10、glibc 2.35。** 这是本机 ABI 的运行环境备份，不是跨平台 manylinux 通用发行版，也不包含操作系统本身。宿主需有 Python venv 支持及匹配的 PCL 1.12 / Boost 1.74 系统库；原生验证会实际加载和执行算法。缺失时先安装对应系统包，例如 Ubuntu 22.04 的 `python3.10-venv`、`libpcl-filters1.12`、`libpcl-search1.12`、`libpcl-common1.12`、`libboost-serialization1.74.0`、`libboost-filesystem1.74.0`、`libboost-timer1.74.0`，及其包管理器自动安装的依赖。脚本不自动运行 sudo/apt。

这还不等于重刷任意操作系统后直接在线建图：实机 SDK、模型、标定与设备通信环境必须另行恢复，新采集链和 SDK 退出策略仍待实机验收。

## 原生包的来源

当前 wheelhouse 对应 `robot_mapping 0.4.0` 和 `robot_mapping_gdk 0.2.0`，包含独立的 FAST-LIO2、pygicp 和 GTSAM 原生 wheel，不包含 grid_icp。

- FAST-LIO2：从独立 `/home/agi/fastlio2` 项目编译 Python 绑定及 C++ 后端。
- FastVGICP：从固定 commit 的 fast_gicp 源码编译，包含释放 GIL 和暴露收敛状态的绑定补丁；Python 导入名为 `pygicp`。
- GTSAM：从 4.2.2 源码编译 C++ 库和 Python 绑定。

流程是源码编译后导出 wheel，再由安装器离线安装到 `.venv`。日常部署只使用 wheelhouse；修改 C++ 或切换目标平台时才重新构建。

## 重新构建和导出

本次首次迁移显式读取已验证的本地安装来导出二进制，输入目录只作为一次性 bootstrap 和审计来源，后续安装及运行不需要它。

常规依赖更新可在本包内构建，构建目录默认 `build/native`：

```bash
bash scripts/build_native.sh                 # GTSAM 4.2.2 和补丁 VGICP
bash scripts/build_native.sh --fast-gicp-only
```

需要系统编译工具、CMake、Eigen/PCL/Boost 开发包，以及联网获取固定版本源码。构建工作目录不属于最终运行环境。FAST-LIO2 由其独立项目构建；导出时显式提供已编译的源码根目录：

```bash
.venv/bin/python scripts/runtime_bundle.py \
  --native-prefix ./build/native \
  --fastlio2-root /path/to/fastlio2 \
  --gdk-root /path/to/robot_mapping_gdk \
  --output ./wheelhouse-new
```

`--gdk-root` 可省略。工具复制所需模块和库、移除调试段、改成基于 `$ORIGIN` 的原生库搜索路径，并生成锁定 wheel。VGICP 版本包含补丁标识；GTSAM 原版本保留，但 wheel SHA-256 与实际来源写入 manifest，不能把本地重新打包的文件冒充上游原始 wheel。

普通 Python 依赖按当前已安装分发的 RECORD 导出，排除解释器外部文件、editable 路径钩子与缓存；console scripts 由 pip 重建。这是当前验证环境的恢复快照，不声称所有上游依赖均已做可重复源码构建。

新 bundle 生成到新目录，验证通过后再选择它安装：

```bash
bash scripts/install_runtime.sh --wheelhouse ./wheelhouse-new
```

## 验证

```bash
.venv/bin/python -I scripts/verify_runtime.py
.venv/bin/python -m pytest -q tests -p no:cacheprovider
```

验证器阻止 SDK/上层应用导入，检查模块路径、NumPy 版本、FAST-LIO2 帧点云接口、实际 VGICP 配准和 GTSAM 优化，并检查 `/proc/self/maps` 中实际加载的自有动态库都属于该 venv。单看 import 成功不足以确认迁移完成。
