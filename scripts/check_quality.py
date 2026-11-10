#!/usr/bin/env python3
"""Check or format this package's own Python and C++ sources."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(arguments: list[str]) -> None:
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="Apply safe lint fixes and formatting")
    args = parser.parse_args()

    formatter = os.environ.get("CLANG_FORMAT", "clang-format")
    if shutil.which(formatter) is None:
        parser.error("clang-format 14 is required; set CLANG_FORMAT to its executable")
    version = subprocess.check_output([formatter, "--version"], text=True)
    if re.search(r"\bversion 14\.", version) is None:
        parser.error("use clang-format 14 for reproducible formatting")

    # Explicit roots exclude vendored code, wheels, environments and recordings.
    python_paths = [
        name for name in ("src", "tests", "scripts", "tools", "setup.py") if (ROOT / name).exists()
    ]
    cpp_paths = sorted(
        str(path.relative_to(ROOT))
        for directory in ("csrc", "tests")
        for path in (ROOT / directory).rglob("*")
        if path.suffix in {".cpp", ".hpp", ".h"}
    )
    ruff = [sys.executable, "-m", "ruff"]
    if args.fix:
        run([*ruff, "format", *python_paths])
        run([*ruff, "check", "--fix", *python_paths])
        run([*ruff, "format", *python_paths])
    else:
        run([*ruff, "check", *python_paths])
        run([*ruff, "format", "--check", *python_paths])
    if cpp_paths:
        mode = ["-i"] if args.fix else ["--dry-run", "--Werror"]
        run([formatter, "--style=file", *mode, *cpp_paths])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from None
