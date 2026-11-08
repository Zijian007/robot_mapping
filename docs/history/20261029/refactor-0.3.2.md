# 0.3.2 运行链路整理

配置文件或配置字典直接生成 MappingSettings，在 MappingSession 入口校验一次。CLI 负责参数解析及覆盖；库不使用 argparse，不以 SystemExit 报配置错误。移除未参与独立会话行为的 save_map_stride、lidar_to_imu_t 和 mapper-only 参数；外参统一来自 robot.imu_from_lidar。

MappingFrontend 直接消费 SensorSource 的 LidarScan/ImuBatch。FAST-LIO2 数组转换归 backends.fastlio.prepare_scan；删除 SourceBridge、空 FrontendSource 协议、旧原始消息分支、转换失败后静默继续和未使用的原始 LIO 地图导出接口。输入数据形状与所有权由 DTO 构造校验，前端入口只核对当前传感器、时钟和 IMU 坐标系。旧 move_odom 消息与 recv 时间模式留在其前端适配器中；三个读取方法用于这里的接口实现差异，不拆出单次算法步骤。

core.spatial 负责体素和动态栅格，core.online_map 负责关键帧和会话/回环状态，storage.pcd 负责文件编码。合成回环辅助移到测试，不再作为运行类方法提供。地图版本、异步任务隔离、保存快照和资源关闭保护保持原逻辑。

构建、安装与恢复工具仍留在 scripts/，不进入算法执行路径。公开会话 API 和 move_odom 的键盘入口保持。GDK 适配包更新至 0.1.1，同步配置示例和依赖版本。

未新增推测性防御、异常吞并或全局路径注入。尚未进行实机采集和地图质量验收。
