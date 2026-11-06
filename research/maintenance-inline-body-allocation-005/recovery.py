"""Four inactive alternate Metal compiler-cache files, exact new evidence only."""
import fcntl,hashlib,json,os,shutil,subprocess
from pathlib import Path
R=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI');B=Path(__file__).resolve().parent
C=Path('/private/var/folders/kq/88lvc8ps3s5_4jpn83v68f8c0000gn/C/org.python.python/com.apple.metal')
PATHS=[C/'16777235_403/functions1.data',C/'16777235_403/functions1.list',C/'32024/libraries1.data',C/'32024/libraries1.list']
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def row(p):
    s=p.stat();assert p.resolve()==p and p.is_file() and not p.is_symlink() and s.st_nlink==1 and s.st_uid==os.getuid()
    return dict(path=str(p),dev=s.st_dev,ino=s.st_ino,uid=s.st_uid,gid=s.st_gid,nlink=s.st_nlink,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,sha256=sha(p))
def check_consumers():
    ps=subprocess.check_output(['ps','-axo','command='],text=True)
    assert not any(token in line for line in ps.splitlines() for token in ['-m research.body_allocation_worker_v1.execution','-m research.body_allocation_quality_v1.execution run-all','resume001.py run'])
    p=subprocess.run(['lsof','-nP',*map(str,PATHS)],capture_output=True,text=True);assert p.returncode==1 and not p.stdout.strip()
def save(name,value):
    with (B/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
with (R/'research_state/external-cache-maintenance.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (B/'manifest.json').exists();check_consumers()
    observed=json.loads((R/'research/maintenance-inline-body-allocation-004/manifest.json').read_text())['retained_metal'];known={r['path']:r for r in observed}
    entries=[]
    for p in PATHS:
        r=row(p);v=known[str(p)];assert r['sha256']==v['sha256'] and r['ino']==v['inode'] and r['mtime_ns']==v['mtime_ns'];entries.append(r)
    save('manifest.json',dict(entries=entries,reason='Inactive alternate numbered compiler cache in Python-owned per-user Metal cache root, observed after all six workers and parents ended. No open consumer, pending worker or compiler. Exact four data/list cache files only; no SDK/app/framework/metallib runtime or research checkpoint.',
        evidence='Completed worker and parent guards00–05 plus clean capacity-wait controller exit; exact retained004 identity unchanged; all compilation can regenerate from protected installed runtime and original sources.',required=13327286044,target=15474769692,free_before=shutil.disk_usage(R).free))
    check_consumers()
    with (B/'journal.jsonl').open('x') as j:
        for p,r in zip(PATHS,entries):
            check_consumers();assert row(p)==r;j.write(json.dumps(dict(event='intent_remove_verified_inactive_cache',file=r))+'\n');j.flush();os.fsync(j.fileno())
            p.unlink();fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
            j.write(json.dumps(dict(event='removed',path=str(p)))+'\n');j.flush();os.fsync(j.fileno())
    free=shutil.disk_usage(R).free;result=dict(status='completed',files=4,allocated_removed=sum(r['allocated'] for r in entries),free_after=free,required=13327286044,target=15474769692,
        mandatory_deficit=max(0,13327286044-free),target_deficit=max(0,15474769692-free),no_app_or_worker_stopped=True,no_runtime_or_research_weight_modified=True)
    save('completed.json',result);print(json.dumps(result))
