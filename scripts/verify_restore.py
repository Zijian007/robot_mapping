"""Restore a released archive using base Python and exercise the restored runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from install_runtime import clean_environment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--python",default=sys._base_executable)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args()
    archive=args.archive.resolve(strict=True)
    checksum=hashlib.sha256(archive.read_bytes()).hexdigest()
    expected=archive.with_suffix(archive.suffix+".sha256").read_text().split()[0]
    if checksum!=expected:
        raise ValueError("restore archive checksum mismatch")
    report={"archive":str(archive),"sha256":checksum,"python":args.python,"hardware_accessed":False}
    log=[]
    def run(command,cwd):
        result=subprocess.run(command,cwd=cwd,env=clean_environment(),capture_output=True,text=True,timeout=180)
        log.append(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f"restore command failed: {command}\n{log[-1]}")
        return result.stdout
    try:
        with tempfile.TemporaryDirectory(prefix="mapping-restore-") as temporary:
            destination=Path(temporary)
            with tarfile.open(archive,"r:gz") as tar:
                for member in tar.getmembers():
                    target=(destination/member.name).resolve()
                    if not target.is_relative_to(destination) or member.issym() or member.islnk():
                        raise ValueError("unsafe path in restore archive")
                tar.extractall(destination)
            root=destination/"robot-mapping-runtime"
            command=[args.python,str(root/"scripts/install_runtime.py")]
            run(command,root)
            python=root/".venv/bin/python"
            # Repeating installation must verify rather than replace packages.
            package=root/".venv/lib/python3.10/site-packages/robot_mapping/__init__.py"
            before=package.stat().st_mtime_ns
            repeated=run(command,root)
            assert "already installed and verified" in repeated
            assert package.stat().st_mtime_ns==before
            run(["bash",str(root/"run.sh"),"doctor","--native","--loop"],root)
            smoke=r'''
import json,sys
from pathlib import Path
import numpy as np
from robot_mapping_gdk.session import MappingSession
from robot_mapping_gdk.recording import ReplaySource
from robot_mapping_gdk.cli import main
from robot_mapping.types import LidarScan,ImuBatch
assert main(['init-config','--generic','--output','config.json'])==0
path=Path('config.json')
profile=json.loads(path.read_text())
profile.setdefault('lio',{}).update(lidar='front',lidar_ready=0.,min_imu_samples=1,print_interval=999.)
profile.setdefault('online',{}).update(live_png=False,keyframe_dist_th=.1)
profile.setdefault('robot',{}).update(lidar_to_base={'front':np.eye(4).tolist()},imu_from_lidar=np.eye(4).tolist(),imu_frame_id='imu')
profile.setdefault('source',{})['clock_id']='sensor'
path.write_text(json.dumps(profile))
class Frames:
    index=0
    exhausted=False
    def open(self): pass
    def next_lidar(self,timeout):
        import time
        if self.index==14:
            self.exhausted=True
            return None
        stamp=100_000_000_000+self.index*100_000_000
        self.index+=1
        time.sleep(.01)
        return LidarScan('front','lidar','sensor',stamp,100_000_000,1,
            np.array([[2.,2.,.5],[3.,2.,.5],[2.,3.,.5]]),np.array([0,50_000_000,100_000_000]))
    def imu_between(self,begin,end,*,clock_id):
        return ImuBatch('imu',clock_id,np.array([end]),np.zeros((1,3)),np.array([[0.,0.,9.81]]))
    def close(self): pass
import time
def finish(session):
    deadline=time.monotonic()+15
    while not session.status()['finished']:
        assert time.monotonic()<deadline,session.status()
        time.sleep(.01)
    assert session.status()['error'] is None,session.status()
    assert session.map.keyframe_count()>=1
    destination=session.close()
    assert json.loads((destination/'manifest.json').read_text())['complete']
first=MappingSession(path,source=Frames(),record_dir='recording').start()
finish(first)
second=MappingSession(path,source=ReplaySource('recording')).start()
finish(second)
for forbidden in ('move_odom','wholebody','agibot_gdk'):
    assert forbidden not in sys.modules
print('PASS: native session, recording, replay and complete save from restored archive')
'''
            run([str(python),"-I","-c",smoke],root)
            report.update(restore="PASS",repeated_install="PASS",native_execution="PASS",
                native_record_replay_save="PASS",source_tree_required=False,
                shared_environment_required=False,network_required=False)
    except BaseException:
        report["restore"]="FAIL"
        raise
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+"\n")
        args.report.with_suffix(".log").write_text("\n".join(log))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
