# robot_mapping

可独立安装和调用的雷达建图算法包，当前版本 **0.4.0**。C++ FAST-LIO2 负责里程计和去畸变，可选 FastVGICP / GTSAM 负责回环；Python 管理过滤、关键帧和栅格。不依赖 `robot_mapping_gdk`、`robot_hal`、`move_odom`、`wholebody` 或设备 SDK。

开发约定与统一检查入口见 [算法包代码规范](docs/code-style.md)。
模块职责及 Python 配置迁移见 [架构与开发入口](docs/architecture.md)。

## 依赖边界

| 部分 | 职责 | 是否需要设备 SDK |
|---|---|---|
| `robot_mapping` | 数据类型、LIO/回环、过滤、关键帧、栅格 | 否 |
| FAST-LIO2 / VGICP / GTSAM | 可选 C++ 算法后端，选择相应能力时加载 | 否 |
| `robot_mapping_gdk` | 会话、录包/回放、预览、地图导出、GDK 采集、G2 启动 | 仅打开实际设备时需要 |
| 机器人数据 | URDF、碰撞网格、雷达/IMU 外参、时间和单位约定 | 使用方从外部提供 |

依赖方向是 **调用方 → robot_mapping → 所选算法后端**。`robot_mapping_gdk` 是其中一个调用方，安装和使用本算法包不需要它的源码、运行环境或启动脚本。原生后端不承担传感器读取。模型和标定是输入数据，核心模板不预填机器人外参。

## 安装

要求 Python 3.10+。以下命令均从本仓库根目录执行，`python` 指向实际使用本包的环境。
已有应用环境时直接使用它；独立使用时可在仓库外创建环境：

```bash
python3 -m venv ~/.venvs/robot-dev
source ~/.venvs/robot-dev/bin/activate
```

### 普通安装

```bash
python -m pip install .
python -c "import robot_mapping; print(robot_mapping.__version__)"
python -m robot_mapping doctor
```

| 能力 | 依赖 |
|---|---|
| 基础算法、JSON 配置、栅格和地图管理 | NumPy，由基础安装自动安装 |
| YAML 配置 | `python -m pip install '.[runtime]'` |
| 默认 LIO 与去畸变 | 另行安装符合接口要求的 FAST-LIO2；也可注入自定义 `LioBackend` |
| 回环 | `python -m pip install --find-links /path/to/wheels '.[loop]' pygicp` 安装 SciPy、支持 NumPy 2 的 `gtsam==4.2.2+numpy2.1` 和带收敛状态与 GIL 补丁的 FastVGICP；这两个 wheel 由 `scripts/build_loop_backends.sh` 构建 |

基础安装不自动安装 FAST-LIO2 或 FastVGICP。安装相应后端后，可通过
`python -m robot_mapping doctor --native` 或 `python -m robot_mapping doctor --loop`
检查对应能力，不连接机器人。源码构建的可选射线扩展需要 C++17 编译器和 pybind11；
没有该扩展时使用 Python/NumPy 参考实现。

### 源码开发

```bash
python -m pip install -e '.[dev]'
```

需要 YAML 时使用 `python -m pip install -e '.[dev,runtime]'`。
可编辑安装直接读取 Python 源码，修改后重启调用进程即可。
修改 C++、头文件或构建配置后，重新执行安装命令并重启进程；依赖或包元数据变化也需重装。

### wheel 构建与安装

```bash
python -m pip wheel --no-deps --wheel-dir dist .
python -m pip install /path/to/wheels/robot_mapping-*.whl
```

安装目录只保留待安装的一个目标 wheel。包含原生扩展的 wheel 需匹配 Python ABI、CPU 架构及系统平台。
接收方无需编译器或本仓库源码，所选 LIO/回环后端仍需单独提供。
离线安装需准备全部依赖 wheel，使用
`python -m pip install --no-index --find-links /path/to/wheels robot-mapping`。
同版本 wheel 更新时，在已满足依赖的环境中，加上 `--force-reinstall --no-deps` 安装目标 wheel。

### 指定环境与卸载

未激活环境时，用 `/path/to/env/bin/python -m pip ...` 指定目标解释器。
uv 管理且未提供 pip 的环境，可用 `uv pip install --python /path/to/env/bin/python .`
或 `uv pip install --python /path/to/env/bin/python -e '.[dev]'`。
默认保留构建隔离；仅在已安装构建依赖的环境中使用 `--no-build-isolation`。

```bash
python -m pip uninstall robot-mapping
```

卸载只移除当前环境中的包或可编辑链接，不删除仓库和本地构建产物。
普通安装或 wheel 安装不会随源码更新；需要重新构建并安装。
更详细的开发和构建说明见 [开发指南](docs/development.md)。

## 项目边界

`robot_mapping` 是算法库：`LidarScan` / `ImuBatch` 进，位姿、去畸变点和栅格出。会话、录包、HTTP 预览、Nav2 导出和 GDK 采集都在 [robot_mapping_gdk](../robot_mapping_gdk/README.md)。move_odom 只消费已生成的地图文件。

数据定义见 `types.py`，接口见 `ports.py`。PointCloud2 解码在适配包。

## 配置分层

算法固有超参数只写在 `src/robot_mapping/resources/generic.json`（LIO、栅格、回环、盒子过滤几何）。用户文件只覆盖本机项：雷达选择、标定、URDF 路径、采集源。`load_settings` / `resolve_profile` 在加载时把用户文件叠到算法默认上。完整配置（如地图保存的快照）也可直接加载，因为再叠一次是幂等的。存盘地图带合并后的快照，默认值以后改了也不影响回放。

空标定模板是 `resources/user.json`，不含算法旋钮。
保存快照通过 `profile_from_settings` 将实际运行参数完整写回标准配置字段，包括调用方的参数覆盖，
同时保留机器人标定和采集源配置。录包回放直接读取这些字段。

## 地图坐标的职责边界

核心不选择地图原点或初始航向。`MappingFrontend` 默认保留 LIO 坐标，应用通过
`initialize_map(mean_acc, generation)` 注入初始化策略。`OnlineMap.set_map_alignment`
只接受固定的 `map_from_lio` 和 `tracking_from_footprint`，负责一致地应用到点云、
自由区域和回环重建，并随保存快照冻结。`grid.min_height/max_height` 始终是地图 Z 坐标。
`tracking_from_footprint` 将自由区域中心坐标系变换到 LIO 跟踪的传感器坐标系。

G2 的初始 `base_link` 原点、水平航向和重力向上由
`robot_mapping_gdk/map_frame.py` 实现。核心中的 `gravity_aligned_frame` 仅按调用方
给定的 up、forward、origin 三个向量计算坐标变换，不选择机器人坐标系。

## 2D 射线清空

默认按高度层积累真实射线的命中与空闲证据；任意未观测高度层会使整列保持未知。
关键帧和定时观测都可更新栅格，回环后按修正位姿重建。
参数、脚下空闲先验及适用边界见 [2D 射线清空说明](docs/mapping.md#2d-射线清空)。

## 3D 点云残影清理

默认以多次真实射线穿透证据清理持久体素图；FAST-LIO 内部定位图保持独立。
参数、回波邻域保护及适用边界见 [3D 点云清理说明](docs/mapping.md#3d-点云残影清理)。

## 原生后端与代码入口

可选 C++ 扩展 `_ray_native` 负责射线分组、2D 栅格遍历、3D 体素证据更新、
点云预处理和本体 TF 校验，计算时释放 GIL。Python 管理栅格扩容、观测顺序、
地图生命周期及回环；原生后端不读取设备。

`VoxelStore` 持有体素和清理计数；融合栅格内核直接更新栅格数组。
点云准备保留逐点时间、强度及实际射线起点，双雷达合并只接受主扫描积分窗口内的点。
本体过滤先完成历史 TF 验证，再按碰撞包围盒筛选点。
算法规则和参数见 [栅格与点云清理](docs/mapping.md)，模块入口见 [架构说明](docs/architecture.md)。

已安装的扩展要求 API v8 或更新；版本不兼容时明确报错，未安装时使用 Python/NumPy 参考实现。
`doctor` 显示各算法后端及 `ray_native_api`。点云导出不保证点的排列顺序。
源码构建需要 C++17 编译器和 pybind11；安装已编译 wheel 不需要编译工具。

G2 启动脚本将 OpenBLAS 限为一个线程，FAST-LIO 的 OpenMP 并行设置保持独立。
帧日志增加 `lio_ms`、`grid_ms`、`voxel_ms`、`total_ms`；`ingest_ms` 表示当前帧
地图摄入的实际耗时。会话状态 `timings` 给出最近 256 个已处理帧的各阶段
P50/P95/P99/最大耗时。`read_ms` 包含等待、解码以及录包包装器可能的等待，
`prepare_ms` 包含本体过滤和双雷达合并；`grid_ms`/`voxel_ms` 是 `map_ms` 的子项，不能重复相加。
没有地图更新的帧记零；提前拒绝的帧不进入该统计窗口。历史录包中的时间缺口不会因优化而消失。

- `backends/capabilities.py`：导入并检查 `fastlio2`、`pygicp`、`gtsam`
- `backends/fastlio.py`：创建 FAST-LIO2 并准备点云数组
- `backends/loop_closure.py` / `pose_graph.py`：回环配准与 GTSAM
- `runtime/frontend.py` → `core/online_map.py`：一帧扫描到关键帧/栅格

## 建图质量保护

原生 FAST-LIO2 必须提供 `LIDAR_UPDATE_VALIDITY_VERSION=1`。激光观测更新无效时，后端返回
`status: prediction_only` / `lidar_update_valid: false`，保留 IMU 预测，禁止局部地图插点；
Python 同时禁止该帧进入关键帧和导出地图。注入自定义 `LioBackend` 时，成功结果必须显式带
`status: scan_to_map` 和 `lidar_update_valid: true`。初始化期间仍允许为局部地图建立种子。

`lio` 下的保护参数（用户配置可以覆盖，时间单位为秒）：

| 参数 | 默认值 | 行为 |
|---|---|---|
| `max_match_failures` | 5 帧 | 连续激光更新无效达到阈值时结束会话，保存已有地图；成功帧清零连续计数 |
| `max_imu_boundary_gap_s` | 0.02 s | IMU 请求窗口首尾允许的最大观测缺口 |
| `max_imu_gap_s` | 0.05 s | 窗口内相邻 IMU 样本的最大时间间隔 |
| `wait_imu_s` | 0.35 s | 适配层等待及补取当前 IMU 窗口的预算 |
| `max_lidar_frame_gap` | 1.0 s | 相邻处理帧允许的最大时间跨度 |
| `imu_startup_timeout_s` | 1.0 s | 传感器预热结束后，等待第一组完整 IMU 的最长时间 |

默认 `require_imu: true` 时，IMU 同时检查样本数、首尾覆盖和内部断点。开始积分后发现不完整
窗口即停止，不跨缺口继续预测；启动阶段允许在超时前跳帧等待。会话的自动保存由适配包负责。
这些检查不等同于退化检测或定位精度保证，重复场景、外参和初始化静止条件仍需验证。

修改原生源码后需重建 FAST-LIO2，再与本包一起安装；旧扩展会在打开传感器前被拒绝。

本包命令行只提供 `doctor`，算法通过 Python API 调用。
G2 实机启动、录包回放、配置生成和地图导出属于可选的
[robot_mapping_gdk 应用](../robot_mapping_gdk/README.md)，其环境部署见
[应用运行环境说明](../robot_mapping_gdk/docs/runtime.md)。

## Python API

调用方实现 `ports.py` 中的 `SensorSource`，并选择内置或自定义的 `LioBackend`，
通过 `MappingFrontend` 组织建图。配置和前端均直接从算法包导入：

```python
from robot_mapping.config import load_settings
from robot_mapping.runtime.frontend import MappingFrontend
```

## 目录与开发

| 目录 | 用途 |
|---|---|
| `src/`、`csrc/` | Python 算法与 C++ 射线内核源码 |
| `tests/` | 数据契约、建图、原生内核和算法包边界测试 |
| `scripts/` | 算法构建、检查和性能诊断工具，见 [工具索引](scripts/README.md) |
| `docs/` | 当前设计、参数和运行说明 |
| `build/`、`dist/` | 按需生成的构建与发布产物，不作为运行依赖 |

开发、打包及清理约定见 [开发指南](docs/development.md)。
合成场景的固定测试数据放在 `tests/fixtures/`，不依赖本机地图或录包。
