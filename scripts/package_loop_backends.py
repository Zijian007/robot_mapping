"""Package locally built loop-closure backends as self-contained wheels.

The shared libraries are copied next to each extension and RPATH is rewritten
to $ORIGIN, so installed wheels do not depend on the build tree. Build trees are
read only; all edits happen on staged copies.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

GTSAM_VERSION = "4.2.2+numpy2.1"
PYGICP_VERSION = "0.0.1+mapping.2"
FAST_GICP_REVISION = "0e7ec1441c99f7be453db2ea216d5de029387417"


def patch_rpath(path, replacement):
    """Replace an existing ELF64 LE path in-place, using transitive DT_RPATH.

    Fail closed when the input has no reserved slot or does not match the
    supported layout.
    """
    data = bytearray(Path(path).read_bytes())
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError(f"unsupported ELF format: {path}")
    phoff = struct.unpack_from("<Q", data, 32)[0]
    stride, count = struct.unpack_from("<HH", data, 54)
    segments = [struct.unpack_from("<IIQQQQQQ", data, phoff + i * stride) for i in range(count)]
    dynamic = next((p for p in segments if p[0] == 2), None)
    if dynamic is None:
        raise ValueError(f"ELF has no dynamic section: {path}")
    entries = []
    for offset in range(dynamic[2], dynamic[2] + dynamic[5], 16):
        tag, value = struct.unpack_from("<qQ", data, offset)
        entries.append((offset, tag, value))
        if tag == 0:
            break
    address = next(v for _, t, v in entries if t == 5)
    segment = next(p for p in segments if p[0] == 1 and p[3] <= address < p[3] + p[5])
    strings = segment[2] + address - segment[3]
    paths = [(o, v) for o, t, v in entries if t in (15, 29)]
    if len(paths) != 1:
        raise ValueError(f"expected one existing RPATH/RUNPATH: {path}")
    entry, index = paths[0]
    start = strings + index
    end = data.index(0, start)
    encoded = replacement.encode()
    if len(encoded) > end - start:
        raise ValueError(f"ELF path slot too small: {path}")
    # Do not overwrite a shared suffix referenced by another dynamic string.
    for _, tag, value in entries:
        if tag in (1, 14) and index <= value <= end - strings:
            raise ValueError(f"ELF path overlaps another dynamic string: {path}")
    data[start:end] = encoded + b"\0" * (end - start - len(encoded))
    struct.pack_into("<q", data, entry, 15)
    Path(path).write_bytes(data)


def native_tag():
    python = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return f"{python}-{python}-linux_{platform.machine()}"


def only(paths, description):
    paths = list(paths)
    if len(paths) != 1:
        raise ValueError(f"expected exactly one {description}, found {len(paths)}")
    return paths[0]


def write_wheel_info(info, requires=()):
    """Write native WHEEL metadata; `wheel pack` regenerates RECORD."""
    (info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: robot_mapping.package_loop_backends\n"
        f"Root-Is-Purelib: false\nTag: {native_tag()}\n"
    )
    metadata = info / "METADATA"
    if not metadata.exists():
        name, version = info.name.removesuffix(".dist-info").split("-", 1)
        text = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        metadata.write_text(text + "".join(f"Requires-Dist: {r}\n" for r in requires) + "\n")


def pack(stage, output):
    for path in stage.rglob("*.so*"):
        if path.is_file():
            subprocess.run(["strip", "--strip-debug", str(path)], check=True)
    subprocess.run(
        [sys.executable, "-m", "wheel", "pack", str(stage), "--dest-dir", str(output)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    wheel = output / f"{stage.name}-{native_tag()}.whl"
    if not wheel.is_file():
        raise ValueError(f"wheel pack did not produce {wheel.name}")
    return wheel


def package_gtsam(source, build, output, work):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            "--quiet",
            "--wheel-dir",
            str(work / "raw"),
            str(build / "python"),
        ],
        check=True,
    )
    raw = only((work / "raw").glob("gtsam-*.whl"), "GTSAM binding wheel")
    subprocess.run(
        [sys.executable, "-m", "wheel", "unpack", str(raw), "--dest", str(work)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    stage = work / f"gtsam-{GTSAM_VERSION}"
    if not stage.is_dir():
        raise ValueError(f"GTSAM build must be version {GTSAM_VERSION}; run patch_gtsam.py")
    package = stage / "gtsam"
    shutil.copy2(build / "gtsam/libgtsam.so.4.2.2", package / "libgtsam.so.4")
    shutil.copy2(
        build / "gtsam/3rdparty/metis/libmetis/libmetis-gtsam.so", package / "libmetis-gtsam.so"
    )
    for path in (*package.glob("gtsam*.so"), package / "libgtsam.so.4"):
        patch_rpath(path, "$ORIGIN")
    info = only(stage.glob("*.dist-info"), "GTSAM dist-info")
    shutil.copy2(source / "LICENSE.BSD", info / "LICENSE.BSD")
    shutil.copy2(source / "gtsam/3rdparty/metis/LICENSE.txt", info / "LICENSE.metis.txt")
    write_wheel_info(info)
    return pack(stage, output)


def package_pygicp(source, build, output, work):
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != FAST_GICP_REVISION:
        raise ValueError(f"unverified fast_gicp revision: {revision}")
    stage = work / f"pygicp-{PYGICP_VERSION}"
    stage.mkdir()
    module = only(build.glob("pygicp*.so"), "compiled pygicp extension")
    shutil.copy2(module, stage / module.name)
    patch_rpath(stage / module.name, "$ORIGIN/")
    shutil.copy2(build / "libfast_gicp.so", stage / "libfast_gicp.so")
    info = stage / f"pygicp-{PYGICP_VERSION}.dist-info"
    info.mkdir()
    shutil.copy2(source / "LICENSE", info / "LICENSE")
    (info / "BUILD.json").write_text(
        json.dumps({"revision": revision, "patch": "convergence/GIL"}, indent=2) + "\n"
    )
    write_wheel_info(info, requires=("numpy>=1.26.4,<3",))
    return pack(stage, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("gtsam", "pygicp"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package = package_gtsam if args.backend == "gtsam" else package_pygicp
    with tempfile.TemporaryDirectory(prefix=f"{args.backend}-wheel-") as temporary:
        wheel = package(args.source.resolve(), args.build.resolve(), output, Path(temporary))
    print(wheel)


if __name__ == "__main__":
    main()
