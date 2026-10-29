# M0 / M1 / M2 实施记录

日期：2026-10-29。首批基线冻结和核心提取已完成；M0/M1 未连接机器人、未启动遥操或建图会话。

## 当前状态：0.3.5

当前算法包为 robot_mapping 0.4.0，GDK 适配包为 0.2.0。算法包保留 LIO/回环/本体过滤/关键帧/栅格和 `doctor`。会话、录包/回放、HTTP 预览、地图导出、PointCloud2 解码和在线 `run` 都在 `robot_mapping_gdk`。

move_odom 独立负责里程计、定位和控制，只读取已生成的地图。其传感器适配由导航项目维护；既无建图库依赖，也无共享契约包依赖。旧建图整合代码仅存历史归档，原地图数据保留。

两个项目通过地图文件格式、坐标系和分辨率约定交付，不共用采集生命周期。新采集链实机质量、SDK 退出策略和全新系统含 SDK/模型/标定的部署仍待验收。

当前验证记录见 independent-packages-0.3.5-validation.json 和 runtime-restore-0.3.5-20261029.json。

```bash
cd /home/agi/robot_mapping
bash run.sh doctor --native --loop
.venv/bin/python -m pytest -q tests /home/agi/robot_mapping_gdk/tests -p no:cacheprovider
```

下文为历史审计记录，其中的旧依赖、版本与路径不代表当前架构。

## 历史记录阅读说明

下面的 M0/M1/M2 和 0.3.0 节保留当时的版本、测试数量与命令作为审计记录，不代表当前状态。历史 `.third_party` 路径和旧安装命令已经失效，不再作为操作步骤；历史“未完成”项以本页当前状态为准。

## 已交付

- 独立 `src` 布局的 Python 包 `robot-mapping==0.1.0`、wheel 构建元数据和离线 CLI。
- 地图、SE3、栅格、URDF/固定盒过滤、回环、导出与预览共 8 个原模块实现已迁出；旧模块仅保留兼容转发或机器人标定适配。新核心没有 `move_odom`、`wholebody` 或 GDK 导入。
- 通用 `LidarScan`、`ImuBatch` 与传感器/历史 TF Protocol。使用整数纳秒、明确时钟标识、SI 单位与独立只读数组；这些新 DTO 尚未替换旧 GDK 前端，接入在 M2。
- 回环数据类型与算法分离。仅使用核心时只需 NumPy，不加载 SciPy、GTSAM、VGICP、FAST-LIO2 或显示库。后端检测不修改 Python/动态库路径。
- `robot-mapping doctor` 默认不加载数值/原生算法；`--native --loop` 检查帧点云能力并实际执行图优化与配准。`project` 支持独立 PCD 投影导出。
- 旧 `move_odom` 包声明对新核心的依赖，运行入口在核心缺失时给出安装提示。构建产物已无依赖安装到旧建图私有目录，共享环境的依赖版本保持不变。

## 验证结果

| 检查 | 结果 |
|---|---|
| 提取前完整建图测试 | 71 passed，5.64 s |
| 新核心测试，NumPy 1.26.4 + 实际 GTSAM/VGICP | 56 passed，2.57 s |
| 新核心测试，共享 NumPy 2.2.6 | 49 passed，7 skipped，2.08 s；跳过依赖旧原生 ABI 的回环测试 |
| 旧入口调用新包的完整回归 | 71 passed，5.51 s |
| 水平/倾斜地图固定样本 | 栅格、体素结果的 SHA-256 与提取前一致，关键帧数/栅格元信息一致 |
| wheel 临时独立安装 | PASS；在临时目录及 `python -I` 中运行，阻止导入 move_odom/wholebody/GDK，验证核心、URDF、导航编码、PGM、CLI |
| 原生 doctor | PASS；FAST-LIO2 帧点云接口、GTSAM Pose3/iSAM2、补丁 VGICP 配准 |
| shell 语法与 move_odom diff 检查 | PASS |
| 共享依赖 | NumPy 保持 2.2.6，基线记录的其他依赖版本也未变化 |

新包测试有一部分由旧回归迁入，两个套件存在覆盖重叠，不将数量相加当作独立测试总数。wheel 独立安装使用了当前解释器中已有的第三方依赖，没有验证全新机器的完整依赖安装；这属于 M3 部署验收。

基线与样本：`baseline-20261029.json`、`baseline-map-fixtures.json`。wheel 为 `dist/robot_mapping-0.1.0-py3-none-any.whl`，当前 SHA-256 为 `31fae2023943a3c58ab673cc7fc32ed9a1cb7e8d6489820087954e7f6fdc986f`。它仅包含 Python 实现和元数据，不包含 C++ 后端二进制、SDK、机器人模型或历史地图。

## 复现

基础环境测试：

```bash
cd /home/agi/robot_mapping
PYTHON=/home/agi/.venv/bin/python bash scripts/run_tests.sh
/home/agi/.venv/bin/python scripts/verify_wheel.py dist/robot_mapping-0.1.0-py3-none-any.whl
```

本机既有回环依赖环境测试：

```bash
cd /home/agi/robot_mapping
PYTHONPATH=/home/agi/move_odom/tools/mapping/.third_party/lib/python3.10/site-packages \
LD_LIBRARY_PATH=/home/agi/move_odom/tools/mapping/.third_party/lib \
PYTHON=/home/agi/.venv/bin/python bash scripts/run_tests.sh

cd /home/agi/move_odom
bash tools/mapping/scripts/run_tests.sh
PYTHONPATH=tools/mapping/.third_party/lib/python3.10/site-packages \
LD_LIBRARY_PATH=tools/mapping/.third_party/lib \
  /home/agi/.venv/bin/python -m robot_mapping doctor --native --loop
```

独立包本身没有任何固定的 `/home/agi` 源码路径，以上只是本机验证环境的具体路径。迁移后更新核心需重新构建 wheel 并安装至旧建图环境；旧入口不直接读取新包源码目录。

## 后续阶段

范围更新：按用户要求，wholebody 接入延期，当前不修改其配置或代码，也不以其集成或并行遥操验收作为独立包首版完成条件。

M2：迁出 GDK 只读采集，连接新 DTO/Protocol，加入原始录包与回放。

M3：完整会话 API、Unix socket 服务/轻量客户端、manifest 与一致发布、专用环境及原生 wheel 部署。当前核心 `OnlineMap` 的生命周期操作不能冒充这一完整服务 API。

M4：move_odom 旧键盘转客户端，完善命令行与退出管理，保持底盘控制职责。wholebody 配置模型、启动/状态/退出接入延期。

M5：独立建图实机验收，衡量地图质量、实时性与传感器读取稳定性；wholebody 并行运行验收延期。已有旧 PCD 无法替代这一阶段的原始录包。

## M2：独立采集、会话与回放（2026-10-29）

已交付 `robot-mapping==0.2.0`：

- `adapters/gdk.py` 直接使用 SDK 的 Lidar/Imu/TF，只读传感器；不经过 move_odom 工厂，不导入 wholebody，不创建 Robot/Pnc。IMU 采集线程有界缓存、丢间隔补取、时间倒退和错误传播；部分初始化失败也清理设备。
- `adapters/conversion.py` 解析字段、步长和大小端，按显式时间格式生成只读 DTO。扫描结束时间由几何过滤前的有效捕获时间确定；前端只将开始时间转成结束时间一次。历史 TF 使用原始源时间。
- `runtime/frontend.py` 迁出双雷达合并、过滤、FAST-LIO2 接入与重力对齐；完整外参由配置注入。旧 live_mapper 已改成标定兼容适配，两边共享实现。重置后的 source flush 在采集线程执行；启动等待可中断，断流超时保留已有地图。
- `MappingSession` 提供 start/status/snapshot/save/reset/cancel/close；保存后进入 idle，未保存时重置需明确 discard。退出先停工作线程和采集，再保存未保存图。无论成功、传感器异常还是导出失败，状态与错误可观察，保存失败可重试。
- `RecordingSource` 记录处理读取到的主/辅雷达、IMU 请求窗口和实际历史 TF，单帧压缩数值文件，不使用 pickle。队列有界；溢出或写盘失败保持不完整状态。`ReplaySource` 校验格式、完成标记和每帧 SHA-256，按原始时间回放，拒绝不一致的 IMU 窗口和缺失 TF。
- `storage/snapshot.py` 使用 staging 加同文件系统目录重命名，整组发布 PCD、栅格产物、坐标变换、有效配置及 manifest。新 CLI 提供 init-config/run/replay；离线 doctor/project 继续保留。
- 模板配置是包资源 JSON，路径和矩阵显式可编辑。移除了新配置中的底盘控制选项；URDF 路径不再依赖某个工作区。旧 move_odom 的依赖范围已更新到 `>=0.2.0,<0.3`，wheel 已无依赖更新至原有建图私有目录。

验证结果：

| 检查 | 结果 |
|---|---|
| 新包完整测试，私有 NumPy 1.26.4 + 实际原生后端 | 73 passed |
| 新包完整测试，共享 NumPy 2.2.6 | 66 passed，7 skipped；回环旧 ABI 的测试跳过 |
| 旧 move_odom 建图回归，使用已安装 0.2.0 wheel | 71 passed |
| 安装包隔离验收 | PASS；临时目录安装，`python -I`，阻止 move_odom/wholebody/GDK 导入，完成模板生成、会话、录制、保存及原生 CLI 回放 |
| 实际 FAST-LIO2 回放 | 14 帧，无错误，1 个关键帧，3 个体素，生成完整地图 manifest；属于合成输入的接线验证 |
| 原生 doctor 与共享环境保护 | PASS；保留原共享依赖，未安装/升级其 NumPy |

测试覆盖保存失败重试、传感器断流与错误收尾、录包损坏/溢出、历史 TF、IMU 坐标系不匹配、保存后重置/废弃、预热中退出，以及已有重力对齐、栅格、过滤与实际回环回归。新旧套件仍有重叠，数量不相加。

本轮 wheel：`dist/robot_mapping-0.2.0-py3-none-any.whl`；SHA-256：`e41993a85e4264f01d518c80b2f639d6849ec0ef19330eefce2e29e0406c9999`。只分发本包 Python 代码与模板，不包含 SDK、模型或原生后端。

验收命令沿用上面的测试脚本，wheel 校验改为：

```bash
/home/agi/.venv/bin/python scripts/verify_wheel.py dist/robot_mapping-0.2.0-py3-none-any.whl --native
```

限制与后续工作：

- 在线 SDK 通过假设备验证调用契约与清理，实际 native 通过离线合成输入验证。本轮未连接机器人、未采集现场数据，不能据此声称地图噪声或运动质量已经改善。
- 录包记录本前端实际读取到的输入，不是所有硬件消息的完整归档；不包含在线超时/调度和会话控制事件。录制中禁止 reset；要重置原点，关闭后使用新目录开始新会话。回放恢复输入，不恢复历史优化器状态。
- SDK/TF 保留进程级引用，暂不卸载 GDK；console 在会话清理/保存后直接退出以避开原生全局析构阻塞，Python API 不调用进程退出。该策略仍需实机退出验收。
- M3 的 Unix socket 服务、轻量客户端、原生 wheel/SDK/模型完整分发、磁盘配额、依赖锁定与全新 OS 一键安装仍待实施。已有会话 API 和 manifest 是其基础，不代表完整服务已完成。
- M4 还需将旧键盘入口转为服务客户端。wholebody 接入继续延期。M5 仍需静止、直行、转向、闭环和长时实机验收。

## SDK 解耦：0.3.0（2026-10-29）

根据用户要求，补齐默认装配的依赖边界。0.2.0 的核心不导入 SDK，但 MappingSession 未提供 source 时仍自动创建 GDK，配置生成器默认 G2，因此会话装配和配置分发尚未完全分离。

本轮改为：

- `robot_mapping` 只包含通用算法、会话和录包/回放；`MappingSession` 必须显式提供 source，不再创建 GDK。
- `robot_mapping_gdk` 单独包提供 GdkSource、G2 示例及 SDK 特殊退出策略，依赖方向为适配包 → 核心。SDK 只在 open 设备时导入。
- CLI 通过 `robot_mapping.sources` entry point 加载明确选择的插件。指定 `--source gdk` 或配置 `source.plugin: gdk` 才加载适配包；缺失/重名插件明确报错，回放不加载硬件插件。
- 核心 init-config 生成通用空标定模板；G2 模板移到 `robot-mapping-gdk init-config`。模型和实机标定从外部配置输入，核心 wheel 没有 G2 模板和 GDK 采集代码。
- FAST-LIO2 构造移入 `backends/fastlio.py`，由 `LioBackend.process/reset` 表达算法契约。选择 LIO 时仍需相应原生算法实现；它不是设备 SDK。注入自定义 lio 不加载默认后端。
- 旧 move_odom 前端转发保持，更新版本范围为 `>=0.3.0,<0.4`。两份 wheel 无依赖更新至旧私有环境；共享环境 NumPy 保持 2.2.6。

验证：核心私有环境 **75 passed**，共享环境 **68 passed / 7 skipped**，适配包 **3 passed**，旧 move_odom 回归 **71 passed**，实际 native doctor 通过。核心 wheel 在隔离目录、阻止 `robot_mapping_gdk` / GDK / 上层项目导入时完成完整会话、录制保存及真实 native 回放；两 wheel 单独安装后验证 entry point 发现、G2 配置和延迟构造，仍阻止 SDK 导入。新旧套件仍有覆盖重叠，数量不相加。

产物摘要见 `sdk-decoupling-20261029.json`。复现：

```bash
cd /home/agi/robot_mapping
/home/agi/.venv/bin/python scripts/verify_wheel.py dist/robot_mapping-0.3.0-py3-none-any.whl --native
cd /home/agi/robot_mapping_gdk
/home/agi/.venv/bin/python scripts/verify_wheels.py \
  ../robot_mapping/dist/robot_mapping-0.3.0-py3-none-any.whl \
  dist/robot_mapping_gdk-0.1.0-py3-none-any.whl
```

这次改变的是软件依赖与配置所有权，没有连接实机，没有改变标定数值或建图算法。实机质量、一键部署和 SDK 进程退出验证仍待完成；wholebody 未接入。
