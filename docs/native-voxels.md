# 原生体素存储

`_ray_native` API v2 的 `VoxelStore` 拥有持久化体素集合、每体素清理计数和累计删除数。
`WorldVoxelHash` / `RayVoxelHash` 保留 Python 接口，但原生路径批量执行插点、正观测重置、
射线证据更新、删除、清空和点云导出。计算期间释放 GIL，互斥锁保护原生状态。
不会在每次观测时维护或重建 Python 体素集合，也不会把计数逐项往返传递给 Python。

2D 栅格分值、关键帧与观测顺序、回环事务仍由 Python 管理；FAST-LIO2 的局部定位地图独立。
射线选择与几何证据规则不变。没有兼容扩展时保留 Python/NumPy 参考实现。
对象在构造时选择存储后端；需要更换后端时创建新对象。

## 数据契约

- `add_points` 返回按输入首次出现顺序排列的新增体素中心，类型保持 `float32`。
- `points_array` 返回全部体素中心，不保证点的排列顺序。C++ 和 Python 的哈希迭代顺序不同，
  因此 PCD 文件或无序点云的字节摘要可能不同；几何等价性应先规范排序。
- 插点和正观测计数重置统一以 `float64` 量化。修复了旧实现中 `float32` 边界点的计数重置
  可能与实际存储体素不一致的问题；参考实现同步使用相同规则。
- `_cells`、`_misses` 是私有诊断接口；原生路径返回独立快照，修改快照不会改变地图。
  正常摄入路径不读取这些快照。
- 保留原先的低层 `VoxelIndex` / `voxel_evidence` 接口；新摄入路径不再调用它们。

## 2026-11-09 离线测量

旧实现已经启用了 C++ 射线内核；这里比较的是继续下沉体素存储的收益。
同一主机上串行执行旧实现、新实现各 9 轮，OpenBLAS 固定 1 线程。
实际录包输入含 12 次观测、1 次几何插点，最终 7,101 个体素。
计时不包含文件读取、坐标转换、诊断摘要或 FAST-LIO2。

| 操作 | 旧实现 | 新实现 |
|---|---:|---:|
| 录包插点 P50 | 26.89 ms | 1.07 ms |
| 录包射线更新 P50 | 19.83 ms | 19.84 ms |
| 录包体素更新（插点 + 射线）P95 | 46.75 ms | 21.29 ms |
| 录包点云导出 P50 | 4.82 ms | 0.22 ms |
| 合成 100,000 点首次插入 P50 | 358.57 ms | 22.88 ms |
| 同一批合成点重复插入 P50 | 246.04 ms | 9.70 ms |
| 合成图 98,786 个体素导出 P50 | 73.06 ms | 8.58 ms |

录包逐次观测后的体素集合、计数和删除数，以及合成图的最终几何，在旧实现、新实现和
Python 参考实现之间一致。该录包没有发生体素删除；删除、回波邻域保护、清空、重插入、
浮点边界和并发插入由测试另行验证。历史地图基准保留原始 Python 点序摘要，
再对规范排序后的原生点云做逐元素精确比较，原基准数据不变。

这些数字是局部离线测量，受主机负载影响，不能当作整套建图吞吐量或实机采集稳定性保证。
此次没有连接机器人。

## 复现

先从核心包目录构建扩展：

```bash
.venv/bin/python setup.py build_ext --inplace
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python scripts/benchmark_voxels.py \
  --backend native \
  --observations artifacts/baselines/native-rays-20261108/observations.npz \
  --output ../robot_mapping_gdk/artifacts/evaluations/voxel-native-20261109/native.json \
  --repeats 9
```

使用 `--backend python` 验证参考路径。`--backend legacy --legacy-root DIR` 在独立进程中
加载明确保存的旧 `spatial.py`、`ray_voxels.py` 及匹配的 `_ray_native*.so`。
本次旧代码、原生扩展备份与三个后端的 JSON 结果保存在
`robot_mapping_gdk/artifacts/evaluations/voxel-native-20261109/`；它们是本机评估附件，不是运行依赖。
录包输入 SHA-256：`24d8a633720bdc138339a0cee513e17a9265b2b492567c1db73c28f2ef9e0356`。

安装包验证与运行环境更新遵循 [开发指南](development.md)。体素原生存储需要
`voxel_backend: cpp` 和 API v2 及以上；管理环境的验证还实际执行三次观测删除。
后续 [融合栅格更新](native-grid.md) 将当前运行环境升级至 API v3。

本次原生路径完整回归 195 项通过；屏蔽扩展后的参考路径 179 项通过、16 项原生专用
测试跳过。独立 wheel 的原生诊断、双包录制/回放/保存验证通过。本机 `wheelhouse/`
已更新核心 wheel 并重建摘要清单，使用锁定安装器同步到 `.venv`，原生执行及库隔离检查通过。
旧核心 wheel、锁文件、清单和运行环境标记保存在评估目录的 `runtime-before/`，
安装记录为 `runtime-install.log`。GDK 启动入口继续使用该独立运行环境。
