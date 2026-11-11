# 建图代码的职责与开发入口

`robot_mapping` 是可独立安装、开发和调用的算法包。
`robot_mapping_gdk` 是可选的 G2 源码应用，负责设备、会话和自身的部署运行环境。
依赖方向为 `robot_mapping_gdk → robot_mapping → 所选算法后端`；
其他应用也可以直接使用算法包，无需 GDK 应用、HAL 或设备 SDK。
算法包不负责选择 G2 地图原点。独立安装与构建见 [开发指南](development.md)。

## 算法包

| 模块 | 职责 |
|---|---|
| `settings.py` | 冻结的配置对象，按 LIO、本体过滤、栅格、在线地图及回环分组 |
| `config.py` | 合并 JSON/YAML 默认配置、解析路径和类型、验证配置、生成保存快照 |
| `runtime/frontend.py` | 获取扫描与 IMU、检查时间覆盖、调用 LIO、管理初始化和失败状态 |
| `runtime/scan_preprocessing.py` | 本体盒过滤、双雷达坐标和时间对齐、扫描合并 |
| `core/map_types.py` | 位姿、关键帧、栅格观测、快照及关键帧选取规则 |
| `core/online_map.py` | 地图状态、观测归档、锁、回环重建及地图切换 |
| `core/map_projection.py` | 在线更新和回环重建共用的观测投影，调用栅格与体素更新 |
| `core/ray_grid.py`、`core/ray_voxels.py` | 2D/3D 射线证据更新；原生与参考路径使用一致的射线分组 |

地图状态和投影分开后，坐标变换及证据更新只维护一份。
回环重建仍使用版本和代次检查，在锁外计算，在短临界区切换地图；不将耗时重建移入采集锁。

## GDK 应用

| 模块 | 职责 |
|---|---|
| `source.py` | GDK 采集、立即复制 SDK 消息、IMU 历史补取及历史 TF |
| `cpu_affinity.py` | 在计算库加载和线程创建前设置 CPU 亲和度 |
| `session.py` | 组装前端与地图、运行会话、预览、关闭及保存 |
| `recording.py` | 有界录制队列、后台写入、完成标记与关闭错误传播 |
| `recording_format.py` | `RecordedFrame`、帧 NPZ 编解码及写入校验和 |
| `replay.py` | 按录制顺序回放，校验完成标记、路径、校验和、IMU 窗口及 TF |
| `map_frame.py`、`storage/` | G2 地图坐标策略、快照和地图文件导出 |
| `cli.py` | 命令行参数及配置覆盖；实时运行在导入计算模块前配置 CPU |
| `main.py`、`run.sh` | 应用源码启动入口 |
| `wheelhouse/`、`scripts/` | 离线依赖与安装启动；实际解释器统一为 `/home/agi/.venv/bin/python` |

录制帧使用 `RecordedFrame` 保存主雷达、辅雷达、IMU 窗口和 TF，避免在采集、写盘、
回放之间传递结构不明的字典。磁盘仍使用版本 1 的 manifest 和 NPZ，原有录包可以直接回放。

## Python 接口

调用方提供自己的 JSON/YAML 配置，未覆盖字段使用算法包内置默认值。
Python 中使用分组配置：
`settings.lio`、`settings.body_filter`、`settings.grid`、`settings.online` 及
`settings.online.loop`。配置对象冻结，运行前覆盖参数使用 `dataclasses.replace`：

```python
from dataclasses import replace

from robot_mapping.config import load_settings

settings = load_settings()  # 读取算法包内置默认配置，也可传入调用方的配置路径
settings = replace(settings, grid=replace(settings.grid, projected_max_range=5.0))
```

算法类型从本包导入：

```python
from robot_mapping.core.map_types import MapSnapshot, SE3Pose
from robot_mapping.settings import MappingSettings
```

使用 GDK 应用时，本机调参编辑 `robot_mapping_gdk/g2_mapping.json`。
录制和回放是应用内部模块，在应用源码环境中导入：

```python
from robot_mapping_gdk.recording import RecordingSource
from robot_mapping_gdk.replay import ReplaySource
```

`benchmark_preprocess.py --backend native|python` 可按需比较预处理后端。
Python/NumPy 路径用于无扩展环境和原生算法的等价性验证。

## 验证

算法包可在自己的开发环境中独立验证，从算法仓库根目录执行：

```bash
PYTHON=python bash scripts/run_tests.sh
```

算法测试覆盖配置往返、原生/参考计算、回环重建并发、时间完整性及包依赖边界。
GDK 应用的录制回放、会话和保存测试在应用侧运行，见
[应用开发入口](../../robot_mapping_gdk/README.md#目录与验证)。
涉及算法行为变化时，按需对比相同输入的处理结果；一般整理无需反复运行全量回放。

应用源码修改后重新启动即可生效。算法库或 HAL 修改后，需要更新应用环境中的依赖。
