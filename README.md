# robot_mapping

与设备 SDK 解耦的雷达建图算法包，当前版本 **0.4.0**。C++ FAST-LIO2 负责里程计和去畸变，可选 FastVGICP / GTSAM 负责回环；Python 管理过滤、关键帧和栅格。不导入 `move_odom`、`wholebody` 或设备适配包。

开发约定与统一检查入口见 [算法包代码规范](docs/code-style.md)。

## 依赖边界

| 部分 | 职责 | 是否需要设备 SDK |
|---|---|---|
| `robot_mapping` | 数据类型、LIO/回环、过滤、关键帧、栅格 | 否 |
| FAST-LIO2 / VGICP / GTSAM | 可选 C++ 算法后端，选择相应能力时加载 | 否 |
| `robot_mapping_gdk` | 会话、录包/回放、预览、地图导出、GDK 采集、G2 启动 | 仅打开实际设备时需要 |
| 机器人数据 | URDF、碰撞网格、雷达/IMU 外参、时间和单位约定 | 使用方从外部提供 |

依赖方向是 **适配包 → 算法包**。原生后端是算法实现，不承担传感器读取。模型和标定是输入数据。核心模板不预填机器人外参。

## 安装

推荐使用本包自己管理的独立环境：

```bash
bash scripts/install_runtime.sh
bash run.sh doctor --native --loop
```

安装器创建 `robot_mapping/.venv`，离线安装带 SHA-256 锁定的依赖。完整步骤见 [独立运行环境](docs/runtime.md)。

如果只把它作为库使用，也可以在自行管理的 Python 3.10+ 环境中安装本包，按需要提供原生算法后端。
基础依赖仅 NumPy；`.[runtime]` 补充 YAML 配置支持，`.[loop]` 补充回环依赖。
地图导出、预览及其依赖由 `robot_mapping_gdk[runtime]` 提供。

默认 doctor 只报告安装信息；`--native --loop` 实际执行原生算法但不连接机器人。

## 项目边界

`robot_mapping` 是算法库：`LidarScan` / `ImuBatch` 进，位姿、去畸变点和栅格出。会话、录包、HTTP 预览、Nav2 导出和 GDK 采集都在 [robot_mapping_gdk](../robot_mapping_gdk/README.md)。move_odom 只消费已生成的地图文件。

数据定义见 `types.py`，接口见 `ports.py`。PointCloud2 解码在适配包。

## 配置分层

算法固有超参数只写在 `src/robot_mapping/resources/generic.json`（LIO、栅格、回环、盒子过滤几何）。用户文件只覆盖本机项：雷达选择、标定、URDF 路径、采集源。`load_settings` / `resolve_profile` 在加载时把用户文件叠到算法默认上。完整旧配置仍然可用，因为再叠一次是幂等的。存盘地图带合并后的快照，默认值以后改了也不影响回放。

空标定模板是 `resources/user.json`，不含算法旋钮。
保存快照通过 `profile_from_settings` 将实际运行参数完整写回标准配置字段，包括调用方的参数覆盖，
同时保留机器人标定和采集源配置。录包回放直接读取这些字段。

## 地图坐标的职责边界

核心不选择地图原点或初始航向。`MappingFrontend` 默认保留 LIO 坐标，应用通过
`initialize_map(mean_acc, generation)` 注入初始化策略。`OnlineMap.set_map_alignment`
只接受固定的 `map_from_lio`、`tracking_from_footprint` 和地图 Z 坐标下的
`height_bounds`，负责一致地应用到点云、自由区域和回环重建，并随保存快照冻结。
`tracking_from_footprint` 将自由区域中心坐标系变换到 LIO 跟踪的传感器坐标系。

G2 的初始 `base_link` 原点、水平航向、重力向上和高度参数换算由
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

射线热点由本包的可选 C++ 扩展 `_ray_native` 执行：射线分组、2D 栅格遍历、3D
采样/去重及回波邻域保护。扩展独立于 FAST-LIO、PCL 和设备 SDK，计算时释放 GIL。
API v2 的 `VoxelStore` 直接拥有体素集合、清理计数和累计删除数，批量完成插点、
正观测重置、射线证据更新和点云导出，不再同步 Python 集合与 C++ 索引。
API v3 的融合栅格内核批量完成近处回波筛选、命中/空闲去重、分值更新和占据状态刷新，
直接写入现有栅格数组。Python 管理栅格分配/扩容、角度分组、观测顺序和回环；
API v4 增加盲区过滤、点字段准备、URDF 包围盒点筛选及双雷达合并的行压缩内核，
减少预处理中间数组。API v5 将本体 TF 的有限性、齐次末行及容差比较改为小型 C++ 谓词；
Gram 矩阵乘法和行列式仍由 NumPy 计算，查询顺序、容差与错误归属保持不变。
没有扩展时使用 Python/NumPy 参考实现。
`doctor` 显示 `ray_backend`、`voxel_backend`、`grid_backend`、`preprocess_backend`、
`body_tf_validation` 和 `ray_native_api`；受管理的运行环境要求 C++ API v5。
旧扩展会回退到参考实现，需要重建才能启用原生路径。
点云导出不保证点的排列顺序；几何坐标、栅格编码及证据规则保持一致。
迁移范围、等价性验证和性能测量见 [原生体素存储](docs/native-voxels.md)。
2D 更新的迁移和逐观测测量见 [融合栅格更新](docs/native-grid.md)。
点云准备与双雷达预处理的迁移见 [原生点云预处理](docs/native-preprocess.md)。
本体 TF 校验的后续优化见 [原生 TF 校验](docs/native-body-tf.md)。
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

本包命令行只有 `doctor`。域控启动、回放和导出：

```bash
bash /home/agi/robot_mapping_gdk/scripts/run_g2.sh
bash /home/agi/robot_mapping_gdk/scripts/drive_g2.sh
robot-mapping-gdk replay ./recordings/session_001 --maps-root ./replayed
robot-mapping-gdk project --pcd map.pcd --output-dir exported
```

通用空标定模板：

```bash
robot-mapping-gdk init-config --generic --output mapping.json
```

## Python API

算法侧注入 `LioBackend` 和 `SensorSource` 后跑 `MappingFrontend`。进程内会话在适配包：

```python
from robot_mapping.config import load_settings
from robot_mapping.runtime.frontend import MappingFrontend
from robot_mapping_gdk.session import MappingSession
from robot_mapping_gdk import create_source
```

## 目录与开发

| 目录 | 用途 |
|---|---|
| `src/`、`csrc/` | Python 算法与 C++ 射线内核源码 |
| `tests/` | 数据契约、建图、原生内核和部署回归测试 |
| `scripts/` | 安装、构建、发布及验证工具，见 [工具索引](scripts/README.md) |
| `docs/` | 当前设计和运行说明；历史验收记录在 `docs/history/` |
| `.venv/`、`wheelhouse/` | 当前运行环境及锁定离线安装包 |
| `artifacts/` | 本机评估基准、历史诊断附件和日志，不进入 Git |
| `build/`、`dist/` | 按需生成的构建与发布产物，不作为运行依赖 |

开发、打包及清理约定见 [开发指南](docs/development.md)。
地图回归测试依赖 [baseline-map-fixtures.json](docs/baseline-map-fixtures.json)，这是测试数据，需保留。
历史实验结论见 [2026-11-08 归档](docs/history/20261108/README.md)，旧迁移记录在
[2026-10-29 归档](docs/history/20261029/)。历史记录中的路径和版本仅反映当时环境。
