"""One-shot, explicit external cache cleanup; never touches research payloads."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CACHE = Path('/Users/junyi/Library/Caches')
BREW = CACHE / 'Homebrew'
TEMP = CACHE / 'dotslash/bc/.tmpOJv2eW'
EXTRACTED = CACHE / 'dotslash/bc/488faa1fc5542c9fb76e25bdefc075c323ed4a'
DIRECTORIES = [BREW / n for n in ('go_cache', 'bootsnap', 'pip_cache', 'api')]
ARCHIVES = [BREW / 'portable-ruby-4.0.1.arm64_big_sur.bottle.tar.gz',
            BREW / 'downloads/980e2eee25a37ba02331e2d5bdee49e03d1b25dcfca4ddba24fb7dbdb57b4712--supabase_darwin_arm64.tar.gz']

def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()

def inspect(path):
    assert path.is_relative_to(CACHE)
    for parent in (path, *path.parents):
        assert not parent.is_symlink(), parent
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and s.st_nlink == 1, path
    with path.open('rb') as f:
        sha = digest(f)
    after = path.lstat()
    identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns, x.st_mode, x.st_uid, x.st_nlink)
    assert identity(s) == identity(after), path
    return dict(path=str(path), size=s.st_size, sha256=sha, inode=s.st_ino,
                device=s.st_dev, mtime_ns=s.st_mtime_ns, allocated_bytes=s.st_blocks * 512)

def save(name, value):
    with (HERE / name).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())

def candidates():
    files = list(ARCHIVES) + [TEMP]
    for directory in DIRECTORIES:
        for base, dirs, names in os.walk(directory, followlinks=False):
            assert not any((Path(base) / n).is_symlink() for n in dirs)
            files.extend(Path(base) / n for n in names)
    return sorted(files)

if sys.argv[1:] == ['plan']:
    # The archive is redundant only if every file/link is present in the
    # extracted runtime. Read members directly; never extract untrusted paths.
    count = 0
    with tarfile.open(TEMP, 'r:gz') as archive:
        for member in archive:
            relative = Path(member.name)
            assert not relative.is_absolute() and '..' not in relative.parts
            target = EXTRACTED / relative
            if member.isfile():
                assert target.is_file() and target.stat().st_size == member.size
                with archive.extractfile(member) as a, target.open('rb') as b:
                    assert digest(a) == digest(b), target
                count += 1
            elif member.issym():
                assert target.is_symlink() and os.readlink(target) == member.linkname
            else:
                assert member.isdir() and target.is_dir(), member.name
    assert Path('/opt/homebrew/Cellar/supabase/2.75.0/INSTALL_RECEIPT.json').is_file()
    assert Path('/opt/homebrew/Library/Homebrew/vendor/portable-ruby/4.0.1/bin/ruby').is_file()
    rows = [inspect(p) for p in candidates()]
    save('manifest.json', dict(authorization='User explicitly requests proactive removal of unused caches and temporary files without impacting RSI.',
        reason='Regenerable Homebrew API/build/pip caches, downloaded installers with installed runtimes, and a redundant dotslash download verified against the extracted runtime.',
        extracted_files_verified=count, files=rows, source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        logical_bytes=sum(r['size'] for r in rows), allocated_bytes=sum(r['allocated_bytes'] for r in rows)))
    print(json.dumps({'files': len(rows), 'bytes': sum(r['size'] for r in rows), 'extracted_files_verified': count}))
elif sys.argv[1:] == ['apply']:
    m = json.loads((HERE / 'manifest.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == m['source']
    # These external installer/build caches have no research consumers. Use
    # an independent maintenance lock so ongoing training is not interrupted.
    # Research artifact cleanup must still use the shared worker lock.
    lock = ROOT / 'research_state/external-cache-maintenance.lock'
    assert not lock.is_symlink()
    with lock.open('a+') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Only block package/cache users; research never consumes these files.
        ps = subprocess.check_output(['ps', '-axo', 'pid,comm'], text=True)
        for line in ps.splitlines():
            command = line.strip().split(maxsplit=1)
            if len(command) == 2:
                name = Path(command[1]).name.lower()
                assert name not in ('brew', 'go', 'pip', 'pip3', 'dotslash'), line
        opened = subprocess.run(['/usr/sbin/lsof', '-nP', '-Fn'], capture_output=True, text=True, timeout=30)
        assert opened.returncode == 0 and not opened.stderr.strip(), opened.stderr
        paths = {line[1:] for line in opened.stdout.splitlines() if line.startswith('n')}
        for row in m['files']:
            assert row['path'] not in paths, row['path']
            assert inspect(Path(row['path'])) == row
        before = shutil.disk_usage(ROOT).free
        with (HERE / 'deletion-journal.jsonl').open('x') as journal:
            def event(record):
                journal.write(json.dumps(record) + '\n'); journal.flush(); os.fsync(journal.fileno())
            for row in m['files']:
                p = Path(row['path'])
                assert inspect(p) == row
                event(dict(event='intent', file=row))
                p.unlink()
                event(dict(event='removed', path=str(p)))
        assert all(not Path(row['path']).exists() for row in m['files'])
        save('receipt.json', dict(status='completed', removed_files=len(m['files']),
             logical_bytes=m['logical_bytes'], allocated_bytes=m['allocated_bytes'],
             free_before=before, free_after=shutil.disk_usage(ROOT).free,
             research_files_modified=False, installed_runtimes_removed=False,
             limitation='Caches can be regenerated; future package operations may download or rebuild them. Free-space delta can include concurrent system activity.',
             completed_unix=time.time()))
        print((HERE / 'receipt.json').read_text())
else:
    raise SystemExit('Use plan or apply')
