# 建图质量保护修复记录（2026-11-08）

## 行为变化

- 原生 FAST-LIO2 输出 `lidar_update_valid`。匹配无效时返回 `prediction_only`，不向局部 ikd-tree 插点；若迭代更新中途失效，恢复更新前的 IMU 预测状态和协方差。
- 算法前端和 `OnlineMap.ingest_odom` 只接受显式有效的激光更新。默认连续失败 5 帧后结束建图，由 GDK 会话保存已有地图；有效帧清零连续失败计数。
- 默认要求 IMU 窗口首尾缺口不超过 20 ms、内部相邻样本间隔不超过 50 ms，且满足原有样本数要求。启动允许等待 1 s；开始积分后不完整的窗口在进入 LIO 前被拒绝。
- 会话状态增加 `lio_status`、`match_failures`、`consecutive_match_failures`，新阈值写入录包和地图配置快照。
- 原生扩展增加 `LIDAR_UPDATE_VALIDITY_VERSION=1`，算法包会在打开传感器前拒绝不具备保护能力的旧扩展。

参数及自定义后端协议见 [README](../README.md#建图质量保护)。二维地图投影与自由空间逻辑本轮没有调整。

## 验证

- 重新编译 ARM64 原生后端，更新两个 Python 包的 wheel，以及带 SHA-256 锁定的 wheelhouse。
- 使用安装后的三个包运行测试：**128 passed**。旧适配器的两份测试额外读取本机 Protobuf 定义，未打开 GDK 设备。
- 故障测试覆盖：匹配丢失时局部地图不变、有效扫描恢复、连续失败停图保存、IMU 首尾缺失与内部断点、启动超时与延迟恢复。
- 已有 `robot_mapping_gdk/recordings/20261029-210845` 录包关闭回环回放：123 帧，0 跳帧、0 匹配失败、3 个关键帧、13756 个体素。二维栅格与历史保存结果逐格相同。
- 独立 wheel 安装、合成数据录包与原生回放、运行环境隔离和 GTSAM/VGICP 检查均通过。

这些检查验证故障处理及正常录包兼容性，不替代实机长时间运行、外参和退化场景验收。

## 本地部署与回退

已通过受管安装器更新 `/home/agi/robot_mapping/.venv`，现有 `run_g2.sh` 使用该环境。
只更新本地构建产物，没有连接或驱动机器人。包版本号沿用原版本，本次构建由 wheel SHA-256 和
`wheelhouse/manifest.json` 的 provenance 区分。

旧完整 wheelhouse 保存在：

```text
/home/agi/robot_mapping/build/tracking-guards-20261108/wheelhouse-before
```

需要回退运行环境时：

```bash
python3 /home/agi/robot_mapping/scripts/install_runtime.py \
  --wheelhouse /home/agi/robot_mapping/build/tracking-guards-20261108/wheelhouse-before
```

默认 `wheelhouse/` 仍保存本次修复；之后不带 `--wheelhouse` 执行安装器会再次安装修复版本。
