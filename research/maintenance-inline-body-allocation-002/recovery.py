"""Exact closed-package bottle and regenerable signed Homebrew catalog cache."""
import fcntl,hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
R=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI');B=Path(__file__).resolve().parent
C=Path('/Users/junyi/Library/Caches/Homebrew');I=Path('/opt/homebrew/Cellar/ripgrep/15.2.0')
N='600794189233be6914687aca6912157202b4e66fc991d9996ea54c63cbdc60d4--ripgrep--15.2.0.arm64_tahoe.bottle.1.tar.gz'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def row(p):
    s=p.stat();assert p.resolve()==p and not p.is_symlink() and s.st_nlink==1 and s.st_uid==os.getuid()
    return dict(path=str(p),dev=s.st_dev,ino=s.st_ino,uid=s.st_uid,gid=s.st_gid,nlink=s.st_nlink,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,sha256=sha(p))
def opens():
    p=subprocess.run(['lsof','-nP','-Fn'],text=True,capture_output=True);assert p.returncode==0 and not p.stderr
    return {s[1:] for s in p.stdout.splitlines() if s.startswith('n')}
def consumers():
    p=subprocess.check_output(['ps','-axo','command='],text=True)
    assert not any('/opt/homebrew/Library/Homebrew/' in line or 'brew.rb' in line for line in p.splitlines())
def save(n,d):
    with (B/n).open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
with (R/'research_state/external-cache-maintenance.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);consumers();opened=opens();assert not (B/'manifest.json').exists()
    retained={str(p):sha(p) for p in (I/'INSTALL_RECEIPT.json',I/'bin/rg')};receipt=json.loads((I/'INSTALL_RECEIPT.json').read_text());assert receipt['poured_from_bottle'] is True
    candidates=[]
    for n in ('formula.jws.json','cask.jws.json'):
        p=C/'api'/n;value=json.loads(p.read_text());assert set(value)=={'payload','signatures'} and value['signatures'] and isinstance(value['payload'],str)
        # These are signed package catalogs, not installers or pending application updates.
        assert str(p) not in opened;candidates.append(dict(**row(p),reason='Regenerable Homebrew signed API catalog cache; no brew/API consumer or open file, installed software unchanged.'))
    p=C/'downloads'/N;assert str(p) not in opened
    with tarfile.open(p) as tar:
        members=[m for m in tar.getmembers() if m.name.endswith('/bin/rg')];assert len(members)==1 and members[0].isfile()
        assert hashlib.sha256(tar.extractfile(members[0]).read()).hexdigest()==retained[str(I/'bin/rg')]
    candidates.append(dict(**row(p),reason='Cached bottle binary is byte-identical to installed ripgrep15.2.0; installation receipt says poured bottle, no brew process or open cache consumer.'))
    save('manifest.json',dict(entries=candidates,retained_installed_sha256=retained,required=16100421632,target=18247905280,free_before=shutil.disk_usage(R).free))
    consumers();opened=opens()
    with (B/'journal.jsonl').open('x') as j:
        for c in candidates:
            p=Path(c['path']);assert str(p) not in opened and row(p)=={k:v for k,v in c.items() if k!='reason'}
            for v in [dict(event='intent',file=c),dict(event='removed',path=str(p))]:
                if v['event']=='removed':p.unlink()
                j.write(json.dumps(v)+'\n');j.flush();os.fsync(j.fileno())
            fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
    assert all(sha(Path(p))==h for p,h in retained.items());free=shutil.disk_usage(R).free
    result=dict(status='completed',files=3,allocated_removed=sum(c['allocated'] for c in candidates),free_after=free,mandatory_deficit=max(0,16100421632-free),target_deficit=max(0,18247905280-free),installed_rg_and_receipt_unchanged=True)
    save('completed.json',result);print(json.dumps(result))
