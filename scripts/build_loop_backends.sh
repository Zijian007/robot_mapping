#!/usr/bin/env bash
# Build the optional loop-closure backends and package them as wheels:
#   GTSAM 4.2.2 with NumPy-2-capable pybind11 bindings (patch_gtsam.py)
#   FastVGICP with the convergence/GIL binding patch (patch_fast_gicp.py)
# Never installs into the selected interpreter; the wheels are verified from a
# temporary target with `robot_mapping doctor --loop`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
WORK="${WORK_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/robot_mapping/loop-backends}"
OUTPUT="${OUTPUT:-${ROOT}/dist}"
JOBS="${JOBS:-$(nproc)}"
MODE="${1:-all}"
case "${MODE}" in
  all|gtsam|fast-gicp) ;;
  *) echo 'Usage: build_loop_backends.sh [all|gtsam|fast-gicp]' >&2; exit 2 ;;
esac
SRC="${WORK}/src"
BUILD="${WORK}/build"
mkdir -p "${SRC}" "${BUILD}" "${OUTPUT}"
OUTPUT="$(cd "${OUTPUT}" && pwd)"

# The interpreter's packages must be identical before and after the build.
fingerprint() {
  PYTHONPATH= "${PY}" -c 'import importlib.metadata as m, json, numpy; print(json.dumps({"packages":sorted((d.metadata["Name"],d.version) for d in m.distributions()),"numpy":(numpy.__version__,numpy.__file__)},sort_keys=True))'
}
BEFORE="$(fingerprint)"
CHECK=""
verify_interpreter() {
  result=$?
  trap - EXIT
  if [[ -n "${CHECK}" ]]; then rm -rf "${CHECK}"; fi
  if [[ "$(fingerprint)" != "${BEFORE}" ]]; then
    echo '[loop-backends] ERROR: interpreter environment changed during the build' >&2
    exit 1
  fi
  exit "${result}"
}
trap verify_interpreter EXIT

"${PY}" -c 'import pybind11, wheel; assert tuple(map(int, pybind11.__version__.split(".")[:2])) >= (2, 12)'
PYBIND_DIR="$(PYTHONPATH= "${PY}" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
# CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF also stops pybind11 from adding -flto.
COMMON=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
  -DPython_EXECUTABLE="${PY}" -DPython3_EXECUTABLE="${PY}" -Dpybind11_DIR="${PYBIND_DIR}")
WHEELS=()

if [[ "${MODE}" == all || "${MODE}" == gtsam ]]; then
  GTSAM_SRC="${SRC}/gtsam"
  GTSAM_BUILD="${BUILD}/gtsam"
  if [[ ! -d "${GTSAM_SRC}/.git" ]]; then
    git clone --depth 1 --branch 4.2.2 https://github.com/borglab/gtsam.git "${GTSAM_SRC}"
  fi
  "${PY}" "${ROOT}/scripts/patch_gtsam.py" "${GTSAM_SRC}"
  cmake -S "${GTSAM_SRC}" -B "${GTSAM_BUILD}" "${COMMON[@]}" -DCMAKE_INSTALL_PREFIX="${WORK}" \
    -DGTSAM_BUILD_PYTHON=ON -DGTSAM_PYTHON_VERSION="$("${PY}" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" \
    -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_TIMING=OFF -DGTSAM_USE_SYSTEM_EIGEN=ON \
    -DGTSAM_WITH_TBB=OFF -DGTSAM_BUILD_UNSTABLE=OFF -DGTSAM_UNSTABLE_BUILD_PYTHON=OFF \
    -DPYTHON_EXECUTABLE="${PY}"
  cmake --build "${GTSAM_BUILD}" --target gtsam_py -j "${JOBS}"
  WHEELS+=("$("${PY}" "${ROOT}/scripts/package_loop_backends.py" gtsam \
    --source "${GTSAM_SRC}" --build "${GTSAM_BUILD}" --output "${OUTPUT}" | tail -n 1)")
fi

if [[ "${MODE}" == all || "${MODE}" == fast-gicp ]]; then
  FGICP_SRC="${SRC}/fast_gicp"
  FGICP_BUILD="${BUILD}/fast_gicp"
  # Pin the revision supported by patch_fast_gicp.py.
  FGICP_REV=0e7ec1441c99f7be453db2ea216d5de029387417
  if [[ ! -d "${FGICP_SRC}/.git" ]]; then
    git clone https://github.com/SMRT-AIST/fast_gicp.git "${FGICP_SRC}"
    git -C "${FGICP_SRC}" checkout "${FGICP_REV}"
    git -C "${FGICP_SRC}" submodule update --init --recursive
  fi
  [[ "$(git -C "${FGICP_SRC}" rev-parse HEAD)" == "${FGICP_REV}" ]] || {
    echo "Unsupported fast_gicp revision; expected ${FGICP_REV}" >&2; exit 1;
  }
  "${PY}" "${ROOT}/scripts/patch_fast_gicp.py" "${FGICP_SRC}"
  cmake -S "${FGICP_SRC}" -B "${FGICP_BUILD}" "${COMMON[@]}" \
    -DBUILD_PYTHON_BINDINGS=ON -DBUILD_VGICP_CUDA=OFF -DBUILD_apps=OFF -DBUILD_test=OFF
  cmake --build "${FGICP_BUILD}" --target pygicp -j "${JOBS}"
  WHEELS+=("$("${PY}" "${ROOT}/scripts/package_loop_backends.py" pygicp \
    --source "${FGICP_SRC}" --build "${FGICP_BUILD}" --output "${OUTPUT}" | tail -n 1)")
fi

# Verify the packaged wheels, not the build tree. A backend not rebuilt in this
# run comes from the interpreter's environment.
CHECK="$(mktemp -d)"
PYTHONPATH= "${PY}" -m pip install --quiet --no-index --no-deps --target "${CHECK}" "${WHEELS[@]}"
PYTHONPATH="${CHECK}:${ROOT}/src" OPENBLAS_NUM_THREADS=1 "${PY}" -m robot_mapping doctor --loop
printf '[loop-backends] verified wheel: %s\n' "${WHEELS[@]}"
