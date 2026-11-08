# 开发、发布和目录维护

## 修改与验证

在线入口运行 `.venv` 中安装的 wheel。修改 `src/` 或 `csrc/` 后，源码测试与已安装程序
是两套验证对象；仅修改源码不会自动更新在线程序。

从 `robot_mapping/` 执行两个包的源码测试：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=src:../robot_mapping_gdk/src \
  .venv/bin/python -m pytest -q --import-mode=importlib -p no:cacheprovider \
  tests ../robot_mapping_gdk/tests
```

射线扩展在源码树中不存在时，Python 参考实现仍可运行，原生专用测试会跳过。
FAST-LIO2 自身的 SDK 适配测试另需其开发环境中的传感器消息和 Protobuf 依赖，
不属于这两个包的基本测试依赖；原生运行检查使用 `verify_runtime.py`。
需要验证源码中的 C++ 修改时先构建扩展：

```bash
.venv/bin/python setup.py build_ext --inplace
```

当前 ARM64 / CPython 3.10 环境的打包和独立安装验证：

```bash
.venv/bin/python -m pip wheel --no-index --no-deps --no-build-isolation \
  --wheel-dir dist . ../robot_mapping_gdk
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/verify_wheel.py \
  dist/robot_mapping-0.4.0-cp310-cp310-linux_aarch64.whl --native
OPENBLAS_NUM_THREADS=1 .venv/bin/python ../robot_mapping_gdk/scripts/verify_wheels.py \
  dist/robot_mapping-0.4.0-cp310-cp310-linux_aarch64.whl \
  dist/robot_mapping_gdk-0.2.0-py3-none-any.whl
```

其他 Python 版本或架构使用实际生成的 wheel 文件名。`verify_wheels.py` 检查核心与适配包
隔离、配置生成、录包、回放和保存，不连接设备。验证通过后，按照
[运行环境说明](runtime.md#重新构建和导出) 生成并验证新的锁定 wheelhouse，再安装到运行环境。
不要只替换 wheelhouse 中的 wheel 而保留旧的校验清单。

## 文件分类与清理

| 内容 | 维护规则 |
|---|---|
| `src/`、`csrc/`、`tests/`、`scripts/` | 源码与可维护工具，纳入 Git |
| `docs/` | 当前操作和设计说明；日期性结论移入 `docs/history/<日期>/` |
| `docs/baseline-map-fixtures.json` | 回归测试输入，不能当日志删除 |
| `.venv/` | 当前运行环境；清理构建目录时保留 |
| `wheelhouse/` | 锁定离线安装包；单独备份并保留 manifest 和 lock |
| `build/`、`dist/`、缓存、`*.egg-info` | 可再生成产物；清理前核对是否混入实验输入或回退备份 |
| `artifacts/baselines/` | 已选择的本机评估基准，有维护价值；按实际需要备份 |
| `artifacts/history/` | 一次性脚本、统计、日志和旧 wheel 备份；不是运行依赖 |
| `artifacts/logs/` | 本机诊断日志，不纳入 Git |

GDK 仓库的 `maps/` 和 `recordings/` 是用户数据，不属于构建缓存。
新评估输出放在 GDK 的 `artifacts/evaluations/<实验名>/`，避免再次散落到项目根目录。
本机 `g2_mapping.json` 也不纳入 Git；机器人默认模板维护在适配包的 `resources/` 中。

## 无用代码审查

先查实际导入、调用、命令行入口和测试，再决定删除。Python 回退实现用于没有扩展的环境及
原生等价性验证；`spatial.py` 是射线实现的共享基础；`projection.py` 被适配包的导出工具使用。
这些都不是历史副本。未被内部调用的公开接口也需要检查外部消费者，不能仅凭文本出现次数删除。

本次整理记录见 [2026-11-08 归档](history/20261108/README.md)。
