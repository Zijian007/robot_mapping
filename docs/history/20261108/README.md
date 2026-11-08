# 2026-11-08 实验与维护记录

此目录保存当日验收结论，内容保留实验时的路径、版本和测量条件；不作为当前启动步骤。
当前使用说明见 [项目 README](../../../README.md) 和 [开发指南](../../development.md)。

| 记录 | 内容 |
|---|---|
| [cleanup-20261108.json](cleanup-20261108.json) | 前一轮文档及实现整理 |
| [grid-coverage-20261108.json](grid-coverage-20261108.json) | 高度层覆盖与空闲区域 |
| [voxel-clearing-20261108.json](voxel-clearing-20261108.json) | 3D 体素清理 |
| [performance-20261108.json](performance-20261108.json) | 阶段性能测量 |
| [native-rays-20261108.json](native-rays-20261108.json) | C++ 射线内核与等价性验证 |
| [tracking-guards-20261108.md](tracking-guards-20261108.md) | 匹配有效性和时间覆盖保护 |
| [thread-runtime-20261108.json](thread-runtime-20261108.json) | 线程调度、预览及 IMU 历史回查 |
| [imu-window-20261108.json](imu-window-20261108.json) | IMU 窗口缺口补取 |
| [repository-cleanup.json](repository-cleanup.json) | 本次清理范围、保留内容及最终验证 |

本机附件位于 `robot_mapping/artifacts/history/20261108/`：

- `build/<原实验目录>/`：历史脚本、JSON、日志、回退 wheel 和校验清单。
- `README-before-cleanup.md`：整理前的说明，包含原始实验操作记录。
- 原 `build/` 下的诊断录包、测试地图、旧构建副本已清理。报告中的这些原路径不再可用。
- 部分更早删除的恢复包及 `tracking-guards` 回退包也不再可用；恢复当前环境使用当前 `wheelhouse/`。

原生射线性能测试的观测输入保留在
`robot_mapping/artifacts/baselines/native-rays-20261108/observations.npz`。
适配包保留的完整评估基准见 [GDK 实验归档](../../../../robot_mapping_gdk/docs/history/20261108/README.md)。
历史脚本中包含当时的绝对路径，重跑前需要改为现存输入；在线诊断脚本会读取设备，不是单元测试。
