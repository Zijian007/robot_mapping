"""Offline restore must reject corrupt bundles before touching environments."""
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sysconfig

import pytest

spec=importlib.util.spec_from_file_location("mapping_installer",Path(__file__).parents[1]/"scripts/install_runtime.py")
installer=importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def bundle(tmp_path):
    directory=tmp_path/"bundle"
    directory.mkdir()
    wheel=directory/"example-1-py3-none-any.whl"
    wheel.write_bytes(b"test wheel contents")
    lock=directory/"requirements.lock"
    lock.write_text("example==1\n")
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={"format_version":1,"complete":True,"machine":platform.machine(),
        "system":platform.system(),"soabi":sysconfig.get_config_var("SOABI"),
        "libc":platform.libc_ver(),"requirements_sha256":digest(lock),
        "wheels":[{"file":wheel.name,"sha256":digest(wheel)}]}
    (directory/"manifest.json").write_text(json.dumps(manifest))
    return directory


def test_corrupt_wheel_rejected_before_environment_creation(tmp_path):
    directory=bundle(tmp_path)
    next(directory.glob("*.whl")).write_bytes(b"corrupt")
    destination=tmp_path/"runtime"
    with pytest.raises(ValueError,match="checksum"):
        installer.main(["--wheelhouse",str(directory),"--venv",str(destination)])
    assert not destination.exists()


def test_unmanaged_environment_is_preserved(tmp_path):
    directory=bundle(tmp_path)
    destination=tmp_path/"existing"
    destination.mkdir()
    sentinel=destination/"keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(RuntimeError,match="unmanaged"):
        installer.main(["--wheelhouse",str(directory),"--venv",str(destination)])
    assert sentinel.read_text()=="keep"


def test_foreign_python_abi_is_rejected(tmp_path):
    directory=bundle(tmp_path)
    manifest=json.loads((directory/"manifest.json").read_text())
    manifest["soabi"]="incompatible-abi"
    (directory/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="soabi mismatch"):
        installer.validate_bundle(directory)


def test_wheel_lock_uses_top_level_metadata_not_vendored_metadata(tmp_path):
    import zipfile
    spec=importlib.util.spec_from_file_location("mapping_bundle",Path(__file__).parents[1]/"scripts/runtime_bundle.py")
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wheel=tmp_path/"setuptools-80.10.2-py3-none-any.whl"
    with zipfile.ZipFile(wheel,"w") as archive:
        archive.writestr("setuptools/_vendor/other-1.dist-info/METADATA","Name: other\nVersion: 1\n")
        archive.writestr("setuptools-80.10.2.dist-info/METADATA","Name: setuptools\nVersion: 80.10.2\n")
    module.write_manifest(tmp_path,{})
    entries=json.loads((tmp_path/"manifest.json").read_text())["wheels"]
    assert [(e["name"],e["version"]) for e in entries]==[("setuptools","80.10.2")]
