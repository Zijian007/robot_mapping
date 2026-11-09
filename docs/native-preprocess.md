# 原生点云预处理

`_ray_native` API v4 增加三个不依赖设备 SDK 的内核：

- `prepare_points`：按盲区距离筛选，直接生成 `float32` 的 XYZ、强度、毫秒偏移和可选起点列。
- `filter_aabbs`：在所有碰撞包围盒的并集中筛点，按原输入顺序一次压缩完整点行。
- `merge_prepared`：调整后雷达的逐点时间，筛选主扫描积分窗口内的点，直接写入最终合并数组。

上述逐点遍历和复制期间释放 GIL。内核不修改输入，支持只读及非连续视图；输入没有需要
剔除的点时，本体过滤仍返回原数组。新增头文件包含在源码分发包中。

## 精度与边界

URDF 资产解析、历史 TF 查询与刚体校验、碰撞角点变换和包围盒构造继续使用原 Python/NumPy
逻辑，全部必要 TF 验证通过后才调用点筛选内核。缺失或无效 TF 仍拒绝整帧。
后雷达的 XYZ 和射线起点矩阵运算也保留 NumPy，C++ 只负责窗口筛选和最终行复制，
以保留现有矩阵运算的舍入结果。

NumPy 1.x 对 float32 数组和 float64 标量边界比较时的弱标量提升规则在内核中显式保留；
NumPy 2.x 调用方关闭该规则。常规 float32 和 float64 的 URDF 点数组使用原生筛选，
其他类型保留参考路径。双雷达融合只对标准 float32 点数组使用原生压缩。

盲区仍使用严格的距离平方比较。`prepare_scan` 新增可选参数 `include_origins=False`，
默认仍返回五列；前端开启射线建图时直接准备八列，不再在主循环中补零列并复制一次。
扫描起始时间和持续时间的纳秒到秒换算不变；后雷达的偏移仍以 float32 毫秒累加，
只保留主扫描实际积分窗口内的点，不把窗口外时间裁剪到边界。
强度、其他字段、前后点序，以及 raw/dropped/body/back 计数与原实现一致。

Python/NumPy 参考实现继续用于没有兼容扩展的环境。扫描准备现在直接填起点列，
因此部分工作从前端主循环移到读帧准备阶段；对比性能应使用整个预处理阶段，
不要只比较在线日志中 `prepare_ms` 的单项。

## 2026-11-09 离线测量

从录包 `20261108-200217` 的 3,130 帧中均匀抽取 64 帧，使用实际标定和 35 个 URDF
碰撞包围盒。旧 API v3 与新 API v4 路径各运行 5 轮；Python 参考路径运行 3 轮。
在同一主机串行执行，OpenBLAS 固定一个线程。录包解码、DTO 创建、模型读取和结果摘要
均在计时区间之外；过滤计时包含历史 TF 查询、校验及包围盒构造。

| 操作，中位耗时 | 原实现 | 新实现 |
|---|---:|---:|
| 双雷达点字段与起点列准备 | 2.65 ms | 0.78 ms |
| 双雷达本体过滤（含 TF/包围盒） | 14.78 ms | 12.67 ms |
| 后雷达变换、时间筛选及合并 | 2.17 ms | 1.83 ms |
| 整体预处理 | 19.78 ms | 15.29 ms |

整体 P95 从 20.46 ms 降到 15.68 ms，P99 从 20.80 ms 降到 15.96 ms。
各阶段分位数不能相加。该测量没有包含 FAST-LIO2、地图更新、设备采集或 SDK 调用延迟，
不能解释为整套建图帧率提升；本轮没有连接机器人。

三个后端在全部 64 个抽样帧上的合并点云字节、时间元数据、保留与剔除计数完全一致。
完整原生回归 233 项通过；屏蔽扩展后的参考路径 179 项通过、54 项原生专用测试跳过。
新增测试覆盖盲区边界、浮点比较边界、5/8/11 列、非连续及只读输入、空前雷达、
前后时间偏移、窗口外点、计数和输入不被修改。

## 复现

从核心包目录运行：

```bash
.venv/bin/python setup.py build_ext --inplace
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:../robot_mapping_gdk/src \
  .venv/bin/python scripts/benchmark_preprocess.py --backend native \
  --recording ../robot_mapping_gdk/recordings/20261108-200217 \
  --output ../robot_mapping_gdk/artifacts/evaluations/preprocess-native-20261109/native.json
```

`--backend python` 运行参考实现。`--backend legacy --legacy-root DIR` 在独立进程中加载
旧 `fastlio.py`、`frontend.py`、`body_filter.py`、`urdf_filter.py` 和匹配的扩展。
本次旧源码、三个后端结果及每个抽样帧的输入摘要保存在
`robot_mapping_gdk/artifacts/evaluations/preprocess-native-20261109/`，不作为运行依赖。

安装与锁定运行环境更新遵循 [开发指南](development.md)。预处理内核需要
`preprocess_backend: cpp`、API v4 及以上；管理环境验证会实际执行三个预处理内核。
后续 [本体 TF 校验](native-body-tf.md) 将当前运行环境升级至 API v5。

本次独立核心 wheel 的原生诊断和双包录制/回放/保存验证通过。已更新本机核心 wheel、
锁文件与校验清单，并使用锁定安装器同步 `.venv`，原生执行与库隔离检查通过。
旧核心 wheel、锁文件、清单和运行环境标记保存在评估目录的 `runtime-before/`，
安装日志为 `runtime-install.log`。
