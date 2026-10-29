#!/usr/bin/env bash
# SDK paths are explicit; numerical dependencies always come from this venv.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="${MAPPING_VENV:-${ROOT}/.venv}"
if [[ ! -x "${RUNTIME}/bin/python" ]]; then
  echo '[mapping] 请先运行 bash scripts/install_runtime.sh 安装独立运行环境。' >&2
  exit 1
fi
if ! "${RUNTIME}/bin/python" -I -c 'import json,sys; from pathlib import Path; assert json.loads((Path(sys.prefix)/"mapping-runtime.json").read_text())["complete"]' >/dev/null 2>&1; then
  echo '[mapping] 独立运行环境尚未安装完成，请重新运行 scripts/install_runtime.sh。' >&2
  exit 1
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="${MAPPING_SDK_PYTHONPATH:-}"
exec "${RUNTIME}/bin/python" -m robot_mapping "$@"
