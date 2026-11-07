"""One original stdlib-only constructor-initializer text census; no RNG draws."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT=BASE/'component-census001'
CAP=33554432

def need(value,reason):
    if not value:raise ValueError(reason)
def save(name,value):
    data=(json.dumps(value,sort_keys=True)+'\n').encode()
    need(sum(p.stat().st_size for p in OUT.iterdir())+len(data)<=CAP,'Census whole metadata cap')
    with (OUT/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
class Guard(CpuPrerequisiteGuard):
    def _sample(self):
        sample,reason=super()._sample();sample['output_bytes']=sum(p.lstat().st_size for p in OUT.iterdir())
        if sample['output_bytes']>CAP:reason='census_output_cap'
        return sample,reason

def main(expected):
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need({p.name for p in OUT.iterdir()}=={'plan.json'},'One original census attempt')
        save('attempt.json',dict(pid=os.getpid(),plan_sha256=expected,no_retry=True,no_replay=True))
        try:
            with Guard(OUT/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
                raw=(OUT/'plan.json').read_bytes();need(hashlib.sha256(raw).hexdigest()==expected,'Census plan pin')
                p=json.loads(raw)
                receipt=json.loads((BASE/'component-proof-source001/review.json').read_text())
                need(receipt['accepted'] is True and receipt['census_plan_sha256']==expected,'Independent source gate first')
                need(os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>=10770972672,'32MiB/reserve/extra entry')
                for pin in p['sources']:
                    data=Path(pin['path']).read_bytes();need(len(data)==pin['size'] and hashlib.sha256(data).hexdigest()==pin['sha256'],'Census source pin')
                exclude={str((ROOT/x).absolute()) for x in p['exclude']}
                paths=set()
                def excluded(path):return any(str(path)==x or str(path).startswith(x+'/') for x in exclude)
                for folder in p['history_roots']:
                    root=Path(folder)
                    if not root.is_absolute():root=ROOT/root
                    if not root.exists():continue
                    for parent,dirs,files in os.walk(root,followlinks=False):
                        dirs[:]=sorted(d for d in dirs if d!='__pycache__' and not (Path(parent)/d).is_symlink() and not excluded(Path(parent)/d))
                        for name in files:
                            path=Path(parent)/name
                            if not path.is_symlink() and path.is_file() and not excluded(path):paths.add(str(path))
                for path in ROOT.iterdir():
                    if path.is_file() and not path.is_symlink():paths.add(str(path))
                need(len(paths)<=75000 and sum(len(json.dumps(dict(path=x,size=Path(x).stat().st_size,device=Path(x).stat().st_dev,inode=Path(x).stat().st_ino,kind='NUL_binary_prefix_only',matches=75000,text_sha256='0'*64),sort_keys=True).encode())+1 for x in paths)+65536<CAP,'Complete anticipated inventory fits cap')
                pattern=re.compile(rb'(?<![0-9])'+str(p['initializer']).encode()+rb'(?![0-9])')
                total=0;text_bytes=0;text_files=0;binary_files=0;hits=[]
                inv=OUT/'inventory.jsonl'
                with inv.open('xb') as journal:
                    for name in sorted(paths):
                        path=Path(name);before=path.lstat();need(stat.S_ISREG(before.st_mode),'Regular census file')
                        binary=False;matches=0;tail=b'';pending=0;hashed=hashlib.sha256();count=0
                        with path.open('rb') as stream:
                            while True:
                                block=stream.read(1048576)
                                if not block:break
                                total+=len(block);need(total<=8589934592,'Whole census read cap')
                                if b'\0' in block:binary=True;break
                                combined=tail+block;boundary=max(pending,len(combined)-32)
                                matches+=sum(pending<=m.start()<boundary for m in pattern.finditer(combined))
                                tail=combined[boundary-1:] if boundary else combined
                                pending=1 if boundary else 0
                                hashed.update(block);count+=len(block)
                        if not binary:matches+=sum(m.start()>=pending for m in pattern.finditer(tail))
                        after=path.lstat()
                        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Historical input changed during census')
                        if binary:binary_files+=1
                        else:
                            text_files+=1;text_bytes+=count
                            if matches:hits.append(dict(path=name,matches=matches))
                        record=dict(path=name,size=before.st_size,device=before.st_dev,inode=before.st_ino,kind='NUL_binary_prefix_only' if binary else 'text',matches=None if binary else matches,text_sha256=None if binary else hashed.hexdigest())
                        data=(json.dumps(record,sort_keys=True)+'\n').encode();journal.write(data)
                        need(journal.tell()+65536<CAP,'Inventory upper bound during stream')
                    journal.flush();os.fsync(journal.fileno())
                result=dict(status='accepted_unused_exclusive_original_component_initializer' if not hits else 'closed_prior_role_match',initializer=p['initializer'],role_status='RESERVED_EXCLUSIVE_COMPONENT_MATH001_UNUSED_NOT_DRAWN',candidate_paths=len(paths),text_files=text_files,text_bytes=text_bytes,binary_classifications=binary_files,streamed_bytes=total,prior_matches=hits,inventory_size=inv.stat().st_size,inventory_sha256=hashlib.sha256(inv.read_bytes()).hexdigest(),no_Torch_component_RNG_import_or_calls=True)
                save('result.json',result);need(not hits,'Initializer already present in declared history')
            save('STOP',dict(status='complete_original_exclusive_initializer_census',no_retry=True,no_replay=True))
        except BaseException as exc:
            save('STOP',dict(status='closed_failed_original_initializer_census',error=repr(exc),no_retry=True,no_replay=True));raise
if __name__=='__main__':main(sys.argv[1])
