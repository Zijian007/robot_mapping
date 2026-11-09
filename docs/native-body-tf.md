# 原生本体 TF 校验

`_ray_native` API v5 将 URDF 本体过滤中的小矩阵比较改为 C++ 谓词，减少每次历史 TF
校验反复构造广播数组和调用 `allclose` / `isclose` 的开销。Gram 矩阵乘法和行列式仍使用
相同的 NumPy 实现；角点变换、包围盒构造、历史 TF 查询和点筛选保持原路径。

## 校验契约

`rigid_structure_valid` 检查 `(4,4)` 形状、全部元素有限性和齐次末行；
`rigid_gram_valid` 检查调用方计算的 `R.T @ R`；`rigid_determinant_valid` 检查 NumPy
计算的行列式。几个谓词只有最多 16 个元素，不为它们单独释放 GIL。
原生点筛选和复制仍按原设计释放 GIL。

保留原 URDF 过滤器的容差和依次拒绝顺序：

| 检查 | 绝对容差 | 相对容差 |
|---|---:|---:|
| 齐次末行与 `[0,0,0,1]` | `1e-8` | `1e-5` |
| Gram 矩阵与单位阵 | `1e-5` | `1e-5` |
| 行列式与 1 | `1e-5` | `1e-5` |

历史 TF 仍每帧、每个独立运动链查询一次；同一帧的重复碰撞体复用已校验矩阵，
不跨帧缓存结果。不改变查询时间戳、调用顺序或具体失败连杆的错误信息。
所有必需 TF 通过后才筛点；后续连杆缺失、反射、缩放、非有限值或畸形矩阵仍拒绝整帧。
没有兼容扩展时使用保留的 NumPy 校验路径。

## 2026-11-09 测量与验证

先对 32 个录包抽样帧做 cProfile 诊断：59 次本体过滤中出现 2,366 次 `allclose`、
3,549 次 `isclose`（包含嵌套调用）；`isclose` 累计约占本体过滤时间的一半。
该诊断带有 profiler 开销，嵌套累计时间不能相加，也不用于下表性能对比。

性能比较使用录包 `20261108-200217` 的 3,130 帧中均匀抽取的 64 帧，35 个碰撞包围盒，
旧 API v4 与新 API v5 各执行 5 轮，参考路径执行 3 轮。在同一主机上串行测量，
OpenBLAS 固定 1 线程，不包含录包解码、模型加载和结果摘要。

| 预处理阶段，中位耗时 | 原实现（API v4） | 新实现（API v5） |
|---|---:|---:|
| 双雷达点字段准备 | 0.79 ms | 0.79 ms |
| 双雷达本体过滤（含 TF/包围盒） | 12.69 ms | 4.48 ms |
| 后雷达变换和合并 | 1.86 ms | 1.83 ms |
| 整体预处理 | 15.33 ms | 7.12 ms |

整体 P95 从 15.85 ms 降至 7.39 ms，P99 从 16.03 ms 降至 7.45 ms。
上述时间不含 FAST-LIO2、地图更新和设备采集，受共享主机负载影响，不能当作整套
建图帧率或实机稳定性保证。本轮未连接机器人。

三条路径在全部抽样帧上的合并点云、时间和剔除计数逐字节一致。
完整原生回归 242 项通过；屏蔽扩展后 179 项通过、63 项原生专用测试跳过。
新增测试与保留的 NumPy 判定对照容差阈值两侧相邻浮点数、500 组旋转扰动、
非有限值、只读和非连续矩阵；另验证后续连杆无效时整帧拒绝、输入不变及错误归属。

## 复现

从核心包目录执行：

```bash
.venv/bin/python setup.py build_ext --inplace
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src:../robot_mapping_gdk/src \
  .venv/bin/python scripts/benchmark_preprocess.py --backend native \
  --recording ../robot_mapping_gdk/recordings/20261108-200217 \
  --output ../robot_mapping_gdk/artifacts/evaluations/tf-native-20261109/native.json
```

`--backend python` 运行参考路径，`--backend legacy --legacy-root DIR` 加载保存的旧实现。
旧代码、原生扩展、逐帧输入/输出摘要、三个后端的结果和 profiler 附件保存在
`robot_mapping_gdk/artifacts/evaluations/tf-native-20261109/`，不作为运行依赖。
当前构建的 `doctor` 应报告 `body_tf_validation: cpp`、`ray_native_api: 5`。

本次独立核心 wheel 的原生诊断与双包录制/回放/保存验证均通过，日志保存在评估目录的
`verify-wheel.log`、`verify-wheels.log`。本机核心 wheel、锁文件及校验清单已更新，
通过锁定安装器同步 `.venv`，原生执行和库隔离检查通过。旧核心 wheel、锁文件、
清单与运行环境标记保存在 `runtime-before/`，安装日志为 `runtime-install.log`。
