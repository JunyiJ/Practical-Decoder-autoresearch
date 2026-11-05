"""One-shot retirement of an installed Chrome component download."""
import fcntl, hashlib, json, os, stat, subprocess, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CACHE = Path('/Users/junyi/Library/Application Support/Google/Chrome/component_crx_cache/50c50dfe552378c67888722e4bac990ba88e8a9567df853e350a22e1e0ae881a')
INSTALLED = Path('/Users/junyi/Library/Application Support/Google/Chrome/Crowd Deny/2026.11.5.146')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def identity(p):
    s = p.lstat()
    assert stat.S_ISREG(s.st_mode) and p.resolve() == p
    return dict(path=str(p), size=s.st_size, allocated_bytes=s.st_blocks*512,
                device=s.st_dev, inode=s.st_ino, uid=s.st_uid, gid=s.st_gid,
                mode=s.st_mode, nlink=s.st_nlink, mtime_ns=s.st_mtime_ns,
                sha256=digest(p))

def save(name, value):
    with (OUT/name).open('x') as f:
        json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())

def journal(value):
    with (OUT/'deletion-journal.jsonl').open('a') as f:
        f.write(json.dumps(value)+'\n'); f.flush(); os.fsync(f.fileno())

def free():
    s = os.statvfs(ROOT)
    return s.f_bavail*s.f_frsize

def consumers():
    r = subprocess.run(['/usr/sbin/lsof','-nP',str(CACHE)], capture_output=True, text=True)
    assert r.returncode == 1 and not r.stdout and not r.stderr, (r.returncode,r.stdout,r.stderr)
    return dict(command=['lsof','-nP',str(CACHE)], returncode=r.returncode, open_handles=0)

with (ROOT/'research_state/external-cache-maintenance.lock').open('r') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'manifest.json').exists()
    candidate = identity(CACHE)
    assert candidate['uid'] == os.getuid() and candidate['nlink'] == 1
    assert candidate['sha256'] == CACHE.name
    preserved = []
    with zipfile.ZipFile(CACHE) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            assert not Path(info.filename).is_absolute() and '..' not in Path(info.filename).parts
            p = INSTALLED/info.filename
            assert p.read_bytes() == z.read(info)
            preserved.append(identity(p))
    assert len(preserved) == 3
    before = free()
    save('manifest.json', dict(candidate=candidate, retained=preserved, consumer_check=consumers(),
         reason='Completed component download; all three archive files exactly match installed Crowd Deny 2026.11.5.146. Cache archive has no open handles; installed component and cache metadata preserved.',
         free_before=before, script_sha256=digest(Path(__file__))))
    assert identity(CACHE) == candidate
    assert all(identity(Path(r['path'])) == r for r in preserved)
    consumers()
    journal(dict(event='intent', candidate=candidate))
    CACHE.unlink()
    fd = os.open(CACHE.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    journal(dict(event='removed', path=str(CACHE), allocated_bytes=candidate['allocated_bytes']))
    assert all(identity(Path(r['path'])) == r for r in preserved)
    after = free()
    save('completed.json', dict(status='completed', allocated_bytes_removed=candidate['allocated_bytes'],
         free_before=before, free_after=after, observed_free_delta=after-before,
         retained_files_verified=len(preserved), cleanup_target_bytes=14529069056,
         remaining_deficit_bytes=max(0,14529069056-after), target_met=False,
         limitation='Concurrent writes and APFS accounting can differ from removed file allocation.'))
    print(json.dumps(dict(removed=candidate['allocated_bytes'], free_after=after, free_delta=after-before)))
