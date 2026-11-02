"""Exact retirement of unused external download caches; no study payload deletion."""
import fcntl, hashlib, json, os, shutil, stat, subprocess, sys, time
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, idle, session_lock

R = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent
HF = Path('/Users/junyi/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct')
UV = Path('/private/tmp/rsi-uv-cache')
TARGETS = (HF, UV)
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''): h.update(b)
    return h.hexdigest()

def save(name, obj):
    data = (json.dumps(obj, sort_keys=True) + '\n').encode()
    assert len(data) < 32*1024**2
    with (E/name).open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())

def record(p):
    s = p.lstat()
    v = dict(size=s.st_size, inode=s.st_ino, device=s.st_dev, mtime=s.st_mtime_ns,
             allocated=s.st_blocks*512, mode=s.st_mode, uid=s.st_uid, links=s.st_nlink)
    if p.is_symlink(): v.update(kind='link', target=os.readlink(p))
    else:
        assert stat.S_ISREG(s.st_mode)
        v.update(kind='file', sha256=sha(p))
    t = p.lstat()
    assert (s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns)
    return v

def inventory():
    files = {}; dirs = []
    for root in TARGETS:
        assert root.is_dir() and not root.is_symlink()
        for parent in root.parents: assert not parent.is_symlink()
        for base, ds, fs in os.walk(root, followlinks=False):
            dirs.append(base)
            for n in list(ds):
                p = Path(base)/n
                if p.is_symlink(): fs.append(n); ds.remove(n)
            for n in fs:
                p=Path(base)/n; v=record(p)
                assert v['uid']==os.getuid() and v['links']==1
                if v['kind']=='link': assert p.resolve().is_relative_to(root)
                files[str(p)]=v
    return dict(files=files, dirs=sorted(dirs))

def survivors():
    out={}
    for root in [R/n for n in ('research','research_runs','research_archives','research_state','src','data','.venv')]+[Path('/private/tmp/rsi-figure-libs')]:
        for base, ds, fs in os.walk(root, followlinks=False):
            ds[:]=[n for n in ds if n!='__pycache__' and Path(base)/n!=E and n!='tokenizer-worker-lock' and n!='maintenance-cache-cleanup-002']
            for n in fs:
                p=Path(base)/n
                if n.endswith('.lock') or n=='tokenizer-quality-progress.json': continue
                out[str(p)]=record(p)
    return out

def unused():
    idle()
    ps=subprocess.check_output(['ps','-axo','pid,comm'],text=True)
    for line in ps.splitlines()[1:]:
        parts=line.strip().split(maxsplit=1)
        if len(parts)!=2 or int(parts[0])==os.getpid(): continue
        name=Path(parts[1]).name.lower()
        assert not (name.startswith('python') or name in ('uv','pip','huggingface-cli','hf','ollama','vllm')),line
    for root in TARGETS:
        result=subprocess.run(['/usr/sbin/lsof','-nP','+D',str(root)],capture_output=True,text=True,timeout=30)
        assert result.returncode==1 and not result.stdout and not result.stderr,(root,result.stdout,result.stderr)

def main(mode):
    with session_lock(GLOBAL_LOCK.parent), CpuPrerequisiteGuard(E/(mode+'-guard.json'),R,timeout_seconds=900):
        unused()
        if mode=='plan':
            assert (HF/'refs/main').read_text().strip()==REVISION
            inv=inventory()
            weight=HF/'blobs/dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'
            assert inv['files'][str(weight)]['sha256']==weight.name
            protected=survivors()
            shards=[]
            entries=list(protected.items())
            for i in range(0,len(entries),1000):
                name=f'survivors-{i//1000:03d}.json';save(name,dict(entries[i:i+1000]));shards.append(dict(name=name,sha256=sha(E/name)))
            save('plan.json',dict(authorization='Human: clean up storage aggressively while not impacting the token model RSI study.',
                source_sha256=sha(Path(__file__)),inventory=inv,survivors=shards,
                model='Qwen/Qwen2.5-1.5B-Instruct',revision=REVISION,
                reason='Unused third-party download cache, not RSI-trained weights; no study source references. UV cache is disposable; installed figure libraries preserved.',
                limitations='Using Qwen again requires downloading this pinned revision. Future UV installs may redownload packages.',
                excluded='All study corpus/tokenizers, selected/fresh weights, all protected archives, installed environments, active app updater and VS Code.',
                free_before=shutil.disk_usage(R).free,required_panel=12381585408))
            print(json.dumps(dict(status='planned',files=len(inv['files']),allocated=sum(v['allocated'] for v in inv['files'].values()),survivors=len(protected))),flush=True)
            return
        m=json.loads((E/'plan.json').read_text()); assert sha(Path(__file__))==m['source_sha256']
        assert inventory()==m['inventory']
        protected={}
        for s in m['survivors']:
            assert sha(E/s['name'])==s['sha256'];protected.update(json.loads((E/s['name']).read_text()))
        assert survivors()==protected
        unused()
        locks=[]
        try:
            paths=[UV/'.lock']+list((HF.parent/'.locks'/HF.name).glob('*.lock'))
            for p in paths:
                assert not p.is_symlink();fd=os.open(p,os.O_RDWR|os.O_NOFOLLOW);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(fd)
            before=shutil.disk_usage(R).free
            with (E/'deletion-journal.jsonl').open('x') as journal:
                def event(v): journal.write(json.dumps(v)+'\n');journal.flush();os.fsync(journal.fileno())
                ordered=sorted(m['inventory']['files'].items(),key=lambda kv:kv[1]['kind']!='link')
                for name,v in ordered:
                    p=Path(name);assert record(p)==v
                    event(dict(event='intent',path=name,expected=v));p.unlink()
                    fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
                    event(dict(event='removed',path=name))
                for name in sorted(m['inventory']['dirs'],key=lambda s:len(Path(s).parts),reverse=True):Path(name).rmdir()
            assert all(not p.exists() for p in TARGETS)
            assert survivors()==protected
            idle()
            save('completed.json',dict(status='completed_verified',files_removed=len(ordered),
                logical_bytes=sum(v['size'] for _,v in ordered if v['kind']=='file'),allocated_bytes=sum(v['allocated'] for _,v in ordered),
                free_before=before,free_after=shutil.disk_usage(R).free,required_panel=12381585408,
                protected_survivors_verified=len(protected),study_payloads_removed=0,model_revision=REVISION,
                completed_unix=time.time(),plan_sha256=sha(E/'plan.json')))
            print((E/'completed.json').read_text(),flush=True)
        finally:
            for fd in locks:os.close(fd)

if __name__=='__main__':
    assert sys.argv[1:] in (['plan'],['apply']);main(sys.argv[1])
