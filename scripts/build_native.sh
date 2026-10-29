#!/usr/bin/env bash
# Private native dependencies. No pip --prefix and no shared-venv package writes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${NATIVE_PREFIX:-${ROOT}/build/native}"
SRC="${PREFIX}/src"
BUILD="${PREFIX}/build"
PY="${PYTHON:-${ROOT}/.venv/bin/python}"
JOBS="${JOBS:-2}"
MODE="${1:-all}"
if [[ "${MODE}" != all && "${MODE}" != --verify-only && "${MODE}" != --fast-gicp-only ]]; then
  echo 'Usage: build_native.sh [--verify-only|--fast-gicp-only]' >&2
  exit 2
fi
[[ -x "${PY}" ]] || { echo "Python not found: ${PY}" >&2; exit 1; }
PY_VER="$("${PY}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
SITE="${PREFIX}/lib/python${PY_VER}/site-packages"
mkdir -p "${SRC}" "${BUILD}" "${SITE}" "${PREFIX}/tmp"
export TMPDIR="${PREFIX}/tmp"

# Compare shared package metadata and the NumPy import before and after, even
# when installation fails. PYTHONPATH is cleared for these shared-env probes.
fingerprint() {
  PYTHONPATH= "${PY}" -c 'import importlib.metadata as m, json, numpy; print(json.dumps({"packages":sorted((d.metadata["Name"],d.version) for d in m.distributions()),"numpy":(numpy.__version__,numpy.__file__)},sort_keys=True))'
}
BEFORE="$(fingerprint)"
verify_shared() {
  result=$?
  trap - EXIT
  if [[ "$(fingerprint)" != "${BEFORE}" ]]; then
    echo '[third_party] ERROR: shared environment changed during installation' >&2
    exit 1
  fi
  exit "${result}"
}
trap verify_shared EXIT

if [[ "${MODE}" != --verify-only ]]; then
  # --target --ignore-installed cannot uninstall NumPy from the interpreter's
  # environment. --no-deps prevents unrelated dependency upgrades.
  if ! PYTHONPATH="${SITE}" "${PY}" -c 'import numpy; assert numpy.__version__ == "1.26.4"' 2>/dev/null; then
    PYTHONPATH= "${PY}" -m pip install numpy==1.26.4 --target "${SITE}" \
      --upgrade --ignore-installed --no-deps
  fi
  PYBIND_DIR="$(PYTHONPATH= "${PY}" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
fi

if [[ "${MODE}" == all ]]; then
  GTSAM_SRC="${SRC}/gtsam"
  if [[ ! -d "${GTSAM_SRC}/.git" ]]; then
    git clone --depth 1 --branch 4.2.2 https://github.com/borglab/gtsam.git "${GTSAM_SRC}"
  fi
  GTSAM_BUILD="${BUILD}/gtsam"
  cmake -S "${GTSAM_SRC}" -B "${GTSAM_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DGTSAM_BUILD_PYTHON=ON -DGTSAM_PYTHON_VERSION="${PY_VER}" \
    -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
    -DGTSAM_BUILD_TIMING=OFF -DGTSAM_USE_SYSTEM_EIGEN=ON \
    -DGTSAM_WITH_TBB=OFF -DGTSAM_BUILD_UNSTABLE=OFF \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -DPython_EXECUTABLE="${PY}" -DPython3_EXECUTABLE="${PY}" -Dpybind11_DIR="${PYBIND_DIR}"
  while IFS= read -r rule; do
    sed -i 's/ -flto[^ ]*//g; s/-fno-fat-lto-objects//g' "${rule}"
  done < <(find "${GTSAM_BUILD}/python" \( -name flags.make -o -name link.txt \))
  cmake --build "${GTSAM_BUILD}" --target gtsam gtsam_py -j "${JOBS}"
  cmake --install "${GTSAM_BUILD}"
  PYTHONPATH= "${PY}" -m pip install "${GTSAM_BUILD}/python" --target "${SITE}" \
    --no-deps --no-build-isolation --upgrade --ignore-installed
fi

if [[ "${MODE}" != --verify-only ]]; then
  FGICP_SRC="${SRC}/fast_gicp"
  # Pin the revision supported by patch_fast_gicp.py.
  FGICP_REV=0e7ec1441c99f7be453db2ea216d5de029387417
  if [[ ! -d "${FGICP_SRC}/.git" ]]; then
    git clone --recursive https://github.com/SMRT-AIST/fast_gicp.git "${FGICP_SRC}"
    git -C "${FGICP_SRC}" checkout "${FGICP_REV}"
    git -C "${FGICP_SRC}" submodule update --init --recursive
  fi
  [[ "$(git -C "${FGICP_SRC}" rev-parse HEAD)" == "${FGICP_REV}" ]] || {
    echo "Unsupported fast_gicp revision; expected ${FGICP_REV}" >&2; exit 1;
  }
  "${PY}" "${ROOT}/scripts/patch_fast_gicp.py" "${FGICP_SRC}"
  FGICP_BUILD="${BUILD}/fast_gicp"
  cmake -S "${FGICP_SRC}" -B "${FGICP_BUILD}" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DBUILD_PYTHON_BINDINGS=ON -DBUILD_VGICP_CUDA=OFF -DBUILD_apps=OFF -DBUILD_test=OFF \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -DPython_EXECUTABLE="${PY}" -DPython3_EXECUTABLE="${PY}" -Dpybind11_DIR="${PYBIND_DIR}"
  cmake --build "${FGICP_BUILD}" --target pygicp -j "${JOBS}"
  cmake --install "${FGICP_BUILD}"
  while IFS= read -r module; do
    cp -v "${module}" "${SITE}/"
  done < <(find "${FGICP_BUILD}" -name 'pygicp*.so')
fi

PYTHONPATH="${SITE}" LD_LIBRARY_PATH="${PREFIX}/lib:${LD_LIBRARY_PATH:-}" \
  "${PY}" "${ROOT}/scripts/verify_native.py"
echo '[third_party] verified; shared environment unchanged'
