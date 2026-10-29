# robot_mapping

与设备 SDK 解耦的雷达建图算法包，当前版本 **0.4.0**。C++ FAST-LIO2 负责里程计和去畸变，可选 FastVGICP / GTSAM 负责回环；Python 管理过滤、关键帧和栅格。不导入 `move_odom`、`wholebody` 或设备适配包。

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

如果只把它作为库使用，也可以在自行管理的 Python 3.10+ 环境中安装 `.[runtime]`，按需要提供原生算法后端。基础依赖仅 NumPy；不开回环时无需 VGICP/GTSAM。

默认 doctor 只报告安装信息；`--native --loop` 实际执行原生算法但不连接机器人。

## 项目边界

`robot_mapping` 是算法库：`LidarScan` / `ImuBatch` 进，位姿、去畸变点和栅格出。会话、录包、HTTP 预览、Nav2 导出和 GDK 采集都在 [robot_mapping_gdk](../robot_mapping_gdk/README.md)。move_odom 只消费已生成的地图文件。

数据定义见 `types.py`，接口见 `ports.py`。PointCloud2 解码在适配包。

## 配置分层

算法固有超参数只写在 `src/robot_mapping/resources/generic.json`（LIO、栅格、回环、盒子过滤几何）。用户文件只覆盖本机项：雷达选择、标定、URDF 路径、采集源。`load_settings` / `resolve_profile` 在加载时把用户文件叠到算法默认上。完整旧配置仍然可用，因为再叠一次是幂等的。存盘地图带合并后的快照，默认值以后改了也不影响回放。

空标定模板是 `resources/user.json`，不含算法旋钮。

## 原生后端与代码入口

- `backends/capabilities.py`：导入并检查 `fastlio2`、`pygicp`、`gtsam`
- `backends/fastlio.py`：创建 FAST-LIO2 并准备点云数组
- `backends/loop_closure.py` / `pose_graph.py`：回环配准与 GTSAM
- `runtime/frontend.py` → `core/online_map.py`：一帧扫描到关键帧/栅格

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

## 开发

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q tests -p no:cacheprovider
.venv/bin/python -m pip wheel --no-index --no-deps --no-build-isolation -w dist .
.venv/bin/python scripts/verify_wheel.py dist/robot_mapping-0.4.0-py3-none-any.whl --native
```

### 从 0.3 迁移

- `MappingSession`、录包、回放、HTTP 预览、地图导出和 PointCloud2 解码移到 `robot_mapping_gdk`
- `robot-mapping` 命令行只保留 `doctor`
- `robot-mapping-gdk` 提供 `run` / `replay` / `project` / `init-config`
