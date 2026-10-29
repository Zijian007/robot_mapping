"""Restore the locked wheelhouse into a dedicated, managed virtual environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import venv


def clean_environment():
    result=os.environ.copy()
    for name in ("PYTHONPATH","PYTHONHOME","LD_LIBRARY_PATH"):
        result.pop(name,None)
    result.update(PYTHONNOUSERSITE="1",PIP_CONFIG_FILE=os.devnull,PIP_NO_INDEX="1")
    return result


def validate_bundle(directory):
    manifest=json.loads((directory/"manifest.json").read_text())
    if manifest.get("format_version")!=1 or not manifest.get("complete"):
        raise ValueError("wheelhouse is incomplete or has an unsupported format")
    for key,value in (("machine",platform.machine()),("system",platform.system()),
                      ("soabi",sysconfig.get_config_var("SOABI"))):
        if manifest[key]!=value:
            raise ValueError(f"wheelhouse {key} mismatch: {manifest[key]} != {value}")
    if tuple(manifest["libc"])!=platform.libc_ver():
        raise ValueError("wheelhouse requires the recorded libc; rebuild for this operating system")
    def check(path,expected):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f"wheelhouse checksum mismatch: {path.name}")
    check(directory/"requirements.lock",manifest["requirements_sha256"])
    seen=set()
    for wheel in manifest["wheels"]:
        path=directory/wheel["file"]
        if path.name!=wheel["file"] or path.suffix!=".whl" or path.name in seen:
            raise ValueError("invalid or duplicate wheelhouse path")
        seen.add(path.name)
        check(path,wheel["sha256"])
    if {p.name for p in directory.glob("*.whl")}!=seen:
        raise ValueError("wheelhouse has unrecorded or missing wheels")
    return manifest


def main(argv=None):
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse",type=Path,default=root/"wheelhouse")
    parser.add_argument("--venv",type=Path,default=root/".venv")
    args=parser.parse_args(argv)
    bundle=args.wheelhouse.resolve()
    manifest=validate_bundle(bundle)
    destination=args.venv.absolute()
    marker=destination/"mapping-runtime.json"
    if destination.exists() and not marker.is_file():
        raise RuntimeError(f"refusing to modify an unmanaged environment: {destination}")
    env=clean_environment()
    python=destination/"bin/python"
    if marker.is_file():
        current=json.loads(marker.read_text())
        if current.get("complete") and current.get("requirements_sha256")==manifest["requirements_sha256"]:
            subprocess.run([str(python),"-I","-m","pip","check"],env=env,check=True)
            subprocess.run([str(python),"-I",str(root/"scripts/verify_runtime.py"),str(destination)],env=env,check=True)
            print(f"Mapping runtime already installed and verified: {destination}")
            return
    if not destination.exists():
        venv.EnvBuilder(with_pip=True,system_site_packages=False).create(destination)
    marker.write_text(json.dumps({"complete":False,"wheelhouse":str(bundle)},indent=2)+"\n")
    subprocess.run([str(python),"-I","-m","pip","install","--no-index","--find-links",str(bundle),
        "--require-hashes","--force-reinstall","-r",str(bundle/"requirements.lock")],env=env,check=True)
    subprocess.run([str(python),"-I","-m","pip","check"],env=env,check=True)
    subprocess.run([str(python),"-I",str(root/"scripts/verify_runtime.py"),str(destination)],env=env,check=True)
    marker.write_text(json.dumps({"complete":True,"wheelhouse":str(bundle),
        "requirements_sha256":manifest["requirements_sha256"],"packages":manifest["wheels"]},indent=2)+"\n")
    print(f"Mapping runtime ready: {destination}")


if __name__=="__main__":
    main()
