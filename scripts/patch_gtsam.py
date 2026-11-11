"""Build GTSAM 4.2.2 bindings with the selected NumPy-2-capable pybind11."""

import sys
from pathlib import Path

source_root = Path(sys.argv[1])
project_configuration = (source_root / "CMakeLists.txt").read_text()
for component, expected in (("MAJOR", 4), ("MINOR", 2), ("PATCH", 2)):
    if f"set (GTSAM_VERSION_{component} {expected})" not in project_configuration:
        raise SystemExit("Only GTSAM 4.2.2 is supported by this binding patch")
cmake_file = source_root / "python/CMakeLists.txt"
cmake_source = cmake_file.read_text()
vendored_binding = "add_subdirectory(${PROJECT_SOURCE_DIR}/wrap/pybind11 pybind11)"
external_binding = "cmake_policy(SET CMP0057 NEW)\nfind_package(pybind11 2.12 CONFIG REQUIRED)"
if vendored_binding in cmake_source:
    cmake_source = cmake_source.replace(vendored_binding, external_binding)
elif "find_package(pybind11 2.12 CONFIG REQUIRED)" in cmake_source:
    if external_binding not in cmake_source:
        cmake_source = cmake_source.replace(
            "find_package(pybind11 2.12 CONFIG REQUIRED)", external_binding
        )
else:
    raise SystemExit("Unsupported GTSAM Python build configuration")
cmake_file.write_text(cmake_source)

setup_file = source_root / "python/setup.py.in"
setup_source = setup_file.read_text()
setup_source = setup_source.replace("version='${GTSAM_VERSION_STRING}'", "version='4.2.2+numpy2.1'")
if "version='4.2.2+numpy2.1'" not in setup_source:
    raise SystemExit("Unsupported GTSAM package version declaration")
setup_file.write_text(setup_source)
print(f"[third_party] GTSAM NumPy 2 binding build ready: {source_root}")
