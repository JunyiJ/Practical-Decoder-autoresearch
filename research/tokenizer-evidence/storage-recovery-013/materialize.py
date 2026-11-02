"""Restore an exact protected raw checkpoint from storage013; never run training."""
import argparse,hashlib,importlib.util,json,os,shutil,tarfile
from pathlib import Path
R=Path(__file__).resolve().parents[3];E=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('primitives',R/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a);a.E=E
from research.gradient_clip_quality_v2.quality_execution import idle

def materialize(member,destination):
 completed=json.loads((E/'raw-completed.json').read_text());plan=a.bound(completed['plan'])
 assert completed['status']=='completed_lossless_raw_storage' and completed['all_raw_bytes_preserved'] and not completed['raw_payloads_retired']
 a.verify(plan['materializer'])
 rows=[r for r in completed['files'] if r['member']==member];assert len(rows)==1
 row=rows[0];expected=row['original'];archive=a.verify(row['archive'])
 p=destination.absolute();assert p.parent.is_dir() and not p.exists() and not p.is_symlink()
 assert all(not q.is_symlink() for q in [p.parent,*p.parents])
 assert p.is_relative_to(R) or p.is_relative_to(Path('/private/tmp'))
 assert shutil.disk_usage(p.parent).free>=8*1024**3+expected['size']+512*1024**2
 tmp=p.with_name(p.name+'.materializing');assert not tmp.exists()
 state=a.run_stream([a.ZSTD,'-d','--long=30','--stdout',str(archive)],archive,'verify');proc=state[0];seen=[]
 try:
  with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
   for m in tar:
    assert m.isfile() and m.name==member and m.size==expected['size'] and not seen
    h=hashlib.sha256();count=0
    with tmp.open('xb') as out:
     stream=tar.extractfile(m)
     while data:=stream.read(4*1024**2):out.write(data);h.update(data);count+=len(data)
     out.flush();os.fsync(out.fileno())
    assert count==expected['size'] and h.hexdigest()==expected['sha256'];seen.append(m.name)
    os.chmod(tmp,m.mode);os.utime(tmp,ns=(expected['mtime_ns'],expected['mtime_ns']))
  while proc.stdout.read(4*1024**2):pass
  proc.stdout.close();a.finish(state)
 except BaseException:
  if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
  state[1].set();state[2].join();raise
 assert seen==[member] and a.file_digest(tmp)==expected['sha256'];a.verify(row['archive'])
 os.link(tmp,p);tmp.unlink();a.syncdir(p.parent)
 assert p.stat().st_size==expected['size'] and a.file_digest(p)==expected['sha256']
 print(json.dumps(dict(status='exact_raw_materialized_archive_preserved',original_member=member,destination=str(p),sha256=expected['sha256'],bytes=expected['size'],training=False)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('member');p.add_argument('destination',type=Path);args=p.parse_args()
 with a.session_lock(R/'research_state/tokenizer-worker-lock'):
  idle()
  with a.CpuPrerequisiteGuard(E/'materialize-check-guard.json',R,timeout_seconds=120):materialize(args.member,args.destination)
