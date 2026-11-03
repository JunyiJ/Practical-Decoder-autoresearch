"""One-shot audited Clang/Swift module and Homebrew metadata cache eviction."""
import fcntl, hashlib, json, os, shutil, stat, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
MODULES=Path('/private/var/folders/kq/88lvc8ps3s5_4jpn83v68f8c0000gn/C/clang/ModuleCache')
BREW=Path('/Users/junyi/Library/Caches/Homebrew')
EXACT=[BREW/n for n in ('linkage.json','all_commands_list.txt','external_commands_list.txt')]

def inspect(p):
    assert p.is_relative_to(MODULES) or p in EXACT
    for q in (p,*p.parents): assert not q.is_symlink(),str(q)
    s=p.lstat()
    assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1,str(p)
    assert time.time()-s.st_mtime>1200,str(p)
    return dict(path=str(p),size=s.st_size,allocated_bytes=s.st_blocks*512,uid=s.st_uid,mode=s.st_mode,nlink=s.st_nlink,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,reason='Rebuildable compiled Clang/Swift module cache; no compiler or open consumer' if p.is_relative_to(MODULES) else 'Regenerable Homebrew linkage/command metadata; no package operation')

def save(name,value):
    with (HERE/name).open('x') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def consumers():
    ps=subprocess.check_output(['ps','-axo','pid,comm'],text=True)
    for line in ps.splitlines()[1:]:
        cols=line.strip().split(maxsplit=1)
        if len(cols)!=2:continue
        name=Path(cols[1]).name.lower()
        assert not any(name.startswith(x) for x in ('clang','swift','xcodebuild','cc1','gcc','brew')),line
    result=subprocess.run(['/usr/sbin/lsof','-nP','-Fn'],capture_output=True,text=True,timeout=30)
    assert result.returncode==0 and not result.stderr.strip(),result.stderr
    return {x[1:] for x in result.stdout.splitlines() if x.startswith('n')}

if sys.argv[1:] == ['plan']:
    paths=EXACT+[p for p in MODULES.rglob('*') if p.is_file()]
    rows=sorted([inspect(p) for p in paths],key=lambda r:r['allocated_bytes'],reverse=True)
    save('manifest.json',dict(files=rows,allocated_bytes=sum(r['allocated_bytes'] for r in rows),logical_bytes=sum(r['size'] for r in rows),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),authorization='2026-11-02 standing human authorization for verified unused rebuildable caches'))
    print(json.dumps(dict(files=len(rows),allocated_bytes=sum(r['allocated_bytes'] for r in rows))))
elif sys.argv[1:] == ['apply']:
    m=json.loads((HERE/'manifest.json').read_text())
    assert m['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lock=ROOT/'research_state/external-cache-maintenance.lock';assert not lock.is_symlink()
    with lock.open('a+') as held:
        fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
        opened=consumers()
        for r in m['files']:
            assert r['path'] not in opened
            assert inspect(Path(r['path']))==r
        before=shutil.disk_usage(ROOT).free;removed=[];errors=[]
        with (HERE/'deletion-journal.jsonl').open('x') as journal:
            def event(v):
                journal.write(json.dumps(dict(time=time.time(),**v))+'\n');journal.flush();os.fsync(journal.fileno())
            for i,r in enumerate(m['files']):
                if i%50==0: opened=consumers()
                p=Path(r['path'])
                assert str(p) not in opened and inspect(p)==r
                event(dict(event='intent',file=r))
                try: p.unlink()
                except OSError as exc:
                    errors.append(dict(path=str(p),error=repr(exc)));event(dict(event='failed',**errors[-1]));continue
                assert not p.exists();removed.append(r);event(dict(event='removed',path=str(p),allocated_bytes=r['allocated_bytes']))
        after=shutil.disk_usage(ROOT).free
        save('receipt.json',dict(status='completed' if not errors else 'partial',removed_files=len(removed),removed_allocated_bytes=sum(r['allocated_bytes'] for r in removed),removed_logical_bytes=sum(r['size'] for r in removed),free_before=before,free_after=after,observed_free_delta=after-before,errors=errors,research_payloads_modified=False,limitations='Allocated bytes are unlinked st_blocks; APFS snapshots/clones and concurrent worker/system writes may affect free-space delta.',completed_unix=time.time()))
        print((HERE/'receipt.json').read_text())
else: raise SystemExit('plan or apply')
