"""Archive installers and locked wheels, excluding environments and user data."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from install_runtime import validate_bundle


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse",type=Path,default=root/"wheelhouse")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    manifest=validate_bundle(args.wheelhouse)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    files=[root/"run.sh",root/"docs/runtime.md",root/"scripts/install_runtime.sh",
        root/"scripts/install_runtime.py",root/"scripts/verify_runtime.py"]
    with tarfile.open(args.output,"w:gz") as archive:
        for path in files:
            archive.add(path,arcname="robot-mapping-runtime/"+path.relative_to(root).as_posix())
        for path in sorted(args.wheelhouse.iterdir()):
            if path.is_file():
                archive.add(path,arcname="robot-mapping-runtime/wheelhouse/"+path.name)
    checksum=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix+".sha256").write_text(f"{checksum}  {args.output.name}\n")
    print(json.dumps({"archive":str(args.output),"sha256":checksum,
        "wheel_count":len(manifest["wheels"]),"contains_sdk_or_models":False},indent=2))


if __name__=="__main__":
    main()
