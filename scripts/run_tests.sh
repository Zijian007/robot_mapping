#!/usr/bin/env bash
# Run in the caller's selected dependency environment. Never install packages.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export MPLBACKEND=Agg
exec "${PY}" -m pytest -q "${ROOT}/tests" -p no:cacheprovider "$@"
