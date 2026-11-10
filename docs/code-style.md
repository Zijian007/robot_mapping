# 算法包代码规范

fastlio2、grid_icp、robot_mapping 使用相同的 Ruff、clang-format 和检查脚本。
各包独立构建与发布，检查过程不依赖另外两个仓库。

## 工具与入口

- Python 3.10+；Ruff 固定为 0.17.3，配置在 pyproject.toml。
- C++17；clang-format 固定使用 14 系列，配置在 .clang-format。
- 开发环境安装本包的 dev extra 可获得 Ruff 和测试工具；clang-format 单独安装。
- 检查脚本使用调用它的 Python 环境中的 Ruff；缺少工具时直接失败，不自动安装。

从任一包的根目录执行：

```bash
python scripts/check_quality.py        # 只检查，适用于本地和 CI
python scripts/check_quality.py --fix  # 格式化并应用安全 lint 修复
PYTHON=python bash scripts/run_tests.sh
```

本机已有 Ruff 的解释器是 /home/agi/.venv/bin/python。
robot_mapping 的依赖测试使用 /home/agi/robot_mapping/.venv/bin/python。
检查工具和算法测试可以使用不同的解释器。

通过 CLANG_FORMAT 指定 clang-format-14 的路径。检查脚本拒绝不同主版本，
避免不同机器反复改写格式；三个仓库升级工具版本时需同步配置。

检查范围是 src/、tests/、scripts/、tools/ 和 setup.py；
C++ 检查范围是 csrc/ 和 tests/。
third_party/、build/、dist/、wheelhouse/、artifacts/、运行环境及录包不参与格式整理。
Ruff 额外设置 force-exclude，避免手动传入第三方文件时意外改写。

## 排版与命名

- Python 使用 4 空格缩进、100 列格式化目标和双引号。
- C++ 使用 2 空格缩进、100 列格式化目标和附着式大括号。
- Python 类名用 PascalCase，函数和变量用 snake_case，常量用 UPPER_SNAKE_CASE。
- C++ 公共类名、函数名遵循同一约定；已有外部 API 名称保持兼容。
- 一行一个语句；新增 C++ 条件和循环使用明确代码块。
- C++ 头文件顺序保留：部分现有头文件依赖前置声明或宏，排序不能只当作排版操作。
- Python 导入由 Ruff 排序。需要先设置 SDK 环境再导入的探针或测试，
  仅在对应文件说明并豁免 E402，不全局关闭这项检查。
- 新增代码注释和异常消息使用英文，使用文档使用中文；旧注释按需维护。

## 类型与接口边界

Python 类型标注统一使用 list[T]、dict[K, V]、tuple[...]、T | None，
可迭代接口从 collections.abc 导入。新增或修改的公共 API 应有参数和返回值标注。
NumPy 数组的形状、dtype、坐标系和所有权需要在文档与运行时校验中说明，
类型标注本身不能替代这些约束。

三个包保持各自的调用形态：

| 包 | 接口职责 |
|---|---|
| fastlio2 | 有状态的连续 LIO 估计 |
| grid_icp | 单次二维配准，返回 GridIcpResult |
| robot_mapping | 通过输入端口和算法后端组织建图 |

后续接口重构采用明确的配置、输入和结果类型；旧调用路径需要兼容转换。
时间、单位和坐标的迁移单独进行，不通过批量重命名改变现有协议。
新接口使用整数纳秒描述传感器时间并说明时钟域，积分边界转换为相对秒；
长度用米、角度用弧度、四元数顺序为 xyzw，
T_target_source 表示从 source 变换到 target。
现有接口的实际单位仍以各自文档为准。

Python 对象解析和 pybind11 绑定应逐步集中到绑定层，C++ 算法核心使用明确的
C++ 类型或数组视图。这是后续结构重构的目标；现有代码尚未全部完成分离。

## 修改与验证

排版修改、接口调整和算法行为变化分别组织，便于审查和定位回归。

1. 运行 check_quality.py。
2. 运行当前包的现有测试。
3. 修改 C++ 后，使用相应依赖环境执行 python setup.py build_ext --inplace，
   再运行包含原生路径的测试，确认没有因缺扩展而跳过。
4. 修改公共接口时验证消费者：fastlio2/grid_icp 对应 move_odom，
   robot_mapping 对应 robot_mapping_gdk；接口或算法行为变化还需要固定数据回放比较。

算法库测试不连接机器人、不发送运动命令。
源码检查和测试通过不等于部署完成；robot_mapping 的在线环境使用安装的 wheel，
修改源码和原地编译不会自动更新其已安装 wheel 或锁定 wheelhouse。
