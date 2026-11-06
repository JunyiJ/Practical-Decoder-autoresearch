"""New prefix6 capacity blocker: exact regenerated disposable caches only; no closed cleanup rerun."""
import fcntl,hashlib,json,os,re,shutil,stat,subprocess
from pathlib import Path
ROOT=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI');B=Path(__file__).resolve().parent
REQUIRED=13327286044;TARGET=15474769692
METAL=Path('/private/var/folders/kq/88lvc8ps3s5_4jpn83v68f8c0000gn/C/org.python.python/com.apple.metal')
PIP=Path('/Users/junyi/Library/Caches/pip')
APP=[Path('/Users/junyi/Library/Caches/Google/Chrome'),Path('/Users/junyi/Library/Application Support/Code/CachedData')]
INVENTORY=[PIP,METAL,*APP,*map(Path,['/Users/junyi/.cache/pip','/Users/junyi/.cache/uv','/Users/junyi/.npm/_cacache','/Users/junyi/Library/Caches/Homebrew','/Users/junyi/anaconda3/pkgs'])]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def row(p):
    s=p.lstat();assert p.resolve()==p and stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1
    return dict(path=str(p),device=s.st_dev,inode=s.st_ino,uid=s.st_uid,gid=s.st_gid,hardlinks=s.st_nlink,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,sha256=sha(p))
def save(name,value):
    with (B/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def opens():
    p=subprocess.run(['/usr/sbin/lsof','-nP','-Fn'],text=True,capture_output=True);assert p.returncode==0 and not p.stderr,p.stderr
    return {line[1:] for line in p.stdout.splitlines() if line.startswith('n')}
def syncdir(p):
    f=os.open(p,os.O_RDONLY)
    try:os.fsync(f)
    finally:os.close(f)
with (ROOT/'research_state/external-cache-maintenance.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (B/'manifest.json').exists()
    ps=subprocess.check_output(['ps','-axo','pid=,command='],text=True)
    installers=[s for s in ps.splitlines() if re.search(r'\b(pip|uv|conda|mamba|brew) (install|update|upgrade|download|build|sync)',s)]
    active_mps=[s for s in ps.splitlines() if '-m research.body_allocation_worker_v1.execution' in s or 'resume001.py run' in s or '-m research.body_allocation_quality_v1.execution run-all' in s]
    open_paths=opens();candidates=[];inventory=[];retained=[]
    for root in INVENTORY:
        unique={};errors=[]
        if root.exists():
            for parent,dirs,names in os.walk(root,followlinks=False,onerror=lambda e:errors.append(str(e))):
                dirs[:]=[d for d in dirs if not (Path(parent)/d).is_symlink()]
                for name in names:
                    p=Path(parent)/name
                    if p.is_symlink():continue
                    s=p.stat();unique[(s.st_dev,s.st_ino)]=s.st_blocks*512
                    reason=None
                    if root==METAL and not active_mps and str(p) not in open_paths and str(p.relative_to(METAL)) in ('16777235_403/functions.data','16777235_403/functions.list','32024/libraries.data','32024/libraries.list'):
                        reason='New regenerated Python Metal compiler cache after all correctness workers finished; not a runtime/library/source/checkpoint.'
                    elif root==PIP and not installers and str(p) not in open_paths and p.parts[len(PIP.parts)] in ('http-v2','wheels'):
                        reason='Completed unopened pip download/wheel cache, no active package installer; installed environment remains intact.'
                    elif root==Path('/Users/junyi/Library/Caches/Homebrew') and not installers and str(p) not in open_paths and p.parent==root/'api' and p.name in ('formula.jws.json','cask.jws.json'):
                        v=json.loads(p.read_text())
                        if set(v)=={'payload','signatures'} and isinstance(v['payload'],str) and v['signatures']:reason='New regenerable signed Homebrew API catalog cache, no active brew consumer; not an installer or pending app update.'
                    elif root in APP and str(p) not in open_paths and re.fullmatch('[0-9a-f]{16}_0',name):
                        with p.open('rb') as f:header=f.read(8)
                        if header==bytes.fromhex('305c72a71b6dfbfc'):reason='Self-identifying disposable Chromium SimpleCache payload, unopened; no indexes/settings/profile/session/extension data.'
                    if reason and s.st_uid==os.getuid() and s.st_nlink==1 and stat.S_ISREG(s.st_mode):candidates.append(dict(**row(p),reason=reason))
                    elif root==METAL and stat.S_ISREG(s.st_mode):retained.append(row(p))
        inventory.append(dict(root=str(root),unique_allocated=sum(unique.values()),files=len(unique),errors=errors))
    # Exact regeneration evidence is recorded, not a rerun of maintenance015/018.
    old=[]
    for n in ('015','016','018'):
        p=ROOT/f'research/maintenance-cache-cleanup-{n}/manifest.json'
        if p.exists():old.extend(json.loads(p.read_text()).get('entries',[]))
    old_ids={(r.get('path'),r.get('ino',r.get('inode')),r.get('mtime_ns')) for r in old}
    for number in ('001','003'):
        q=ROOT/f'research/maintenance-inline-body-allocation-{number}/manifest.json'
        if q.exists():
            for r in json.loads(q.read_text()).get('entries',[]):old_ids.add((r.get('path'),r.get('inode',r.get('ino')),r.get('mtime_ns')))
    for c in candidates:assert (c['path'],c['inode'],c['mtime_ns']) not in old_ids,'Closed candidate cannot repeat'
    candidates.sort(key=lambda r:r['allocated'],reverse=True);before=shutil.disk_usage(ROOT).free
    save('manifest.json',dict(reason='New prefix6 remaining-capacity blocker after six accepted full screens; controller exited cleanly before unattempted slot06; coordinated inline recovery.',required=REQUIRED,target=TARGET,
        inventory=sorted(inventory,key=lambda r:r['unique_allocated'],reverse=True),entries=candidates,retained_metal=retained,free_before=before,
        other_classes='Research generated-payload closure handled separately under research lock. All selected/fresh archives and original evidence protected. Unused Keynote/admin and root-owned npm access blockers unchanged; active/pending updaters, installed/shared runtimes, active containers, unmatched downloads and personal files preserved. No new app dependency proof or access change.',
        active_installers=installers,active_mps=active_mps))
    removed=[];skipped=[]
    with (B/'journal.jsonl').open('x') as j:
        def log(v):j.write(json.dumps(v,sort_keys=True)+'\n');j.flush();os.fsync(j.fileno())
        for i,c in enumerate(candidates):
            if i%100==0:open_paths=opens()
            p=Path(c['path']);expected={k:v for k,v in c.items() if k!='reason'}
            if str(p) in open_paths or not p.exists() or row(p)!=expected:skipped.append(str(p));continue
            log(dict(event='intent_remove_verified_cache',file=c));p.unlink();syncdir(p.parent);removed.append(c);log(dict(event='removed',path=str(p)))
    # Retained compiler bundles, sources and all research files were outside the allowlist.
    assert all(row(Path(r['path']))==r for r in retained)
    free=shutil.disk_usage(ROOT).free
    result=dict(status='completed',removed=len(removed),allocated_removed=sum(r['allocated'] for r in removed),skipped=skipped,free_before=before,free_after=free,
        required=REQUIRED,target=TARGET,mandatory_deficit=max(0,REQUIRED-free),target_deficit=max(0,TARGET-free),retained_metal_verified=len(retained),
        note='Allocated bytes reclaimed are not the observed free-space delta. No app/worker stopped; no credential/session/personal/selected/fresh/original research bytes changed.')
    save('completed.json',result);print(json.dumps(result))
