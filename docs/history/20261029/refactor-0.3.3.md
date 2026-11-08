# 0.3.3 工作流整理

## 代码阅读顺序

1. api.MappingSession 创建配置、标定、前端及可选回环后端，管理资源生命周期。
2. runtime.frontend.MappingFrontend.run 依次采集、过滤、融合双雷达、读取 IMU、运行 LIO 并写入关键帧。逐帧统计和节流在 _report_frame，主循环不再拼接日志。
3. backends.fastlio.create_backend 使用 capabilities.load_fastlio2 构建原生前端；prepare_scan 只负责数组格式和单位转换。
4. backends.loop_closure.AsyncLoopWorker 创建并持有 FastVGICPBackend。后端在构造时导入并验证绑定，每次调用仍创建独立配准实例，避免沿用上次配准状态。run_fast_vgicp 保留为单次调用入口。
5. backends.pose_graph.PoseGraphBackend 在构造时用 load_gtsam 导入并验证 GTSAM。capabilities 中的布尔探测仅供 doctor/测试诊断，正常配准不再反复探测安装能力。
6. core.online_map.OnlineMap.accept_loop 分三阶段：短锁快照、锁外优化重建、追平并发布。_backend_lock 串行化求解器变更，_lock 保护已发布地图。版本检查仍用于拒绝过期任务。
7. storage.snapshot.save_snapshot 冻结并发布完整地图目录。

## 接口与兼容

ports.LioResult 描述里程计返回字典：status/stamp 必需，位姿、点云与统计字段按状态提供；未引入运行时包装或重复校验。LioBackend.process、前端和地图入口使用该类型。回环线程标注实际 OnlineMap 依赖及配准返回类型。

旧 move_odom 入口保留消息适配和 recv 时间模式，移除重复原生库预检查及空操作 helper 调用。原生检查由真正创建后端的入口负责。缺少后端时独立会话仍在传感器 open 之前失败。

本次没有增加低概率防御、全局模块缓存或新的异常吞并。输入点云检查针对每次新的配准数据；地图代次和位姿版本检查属于并发提交条件，与安装能力检查分开。
