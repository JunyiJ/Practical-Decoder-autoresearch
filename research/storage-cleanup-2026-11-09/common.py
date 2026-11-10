"""Exact-file cleanup helpers; no recursive deletion or model loading."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

ROOT = Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI')
OUT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def row(path):
    p = Path(path)
    s = p.lstat()
    assert p.resolve() == p and stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_uid == os.getuid(), p
    result = dict(path=str(p), dev=s.st_dev, ino=s.st_ino, size=s.st_size,
                  allocated=s.st_blocks*512, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, sha256=sha(p))
    t = p.lstat()
    assert (s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns)
    return result


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def opens():
    r = subprocess.run(['/usr/sbin/lsof','-nP','-Fn'], capture_output=True, text=True, timeout=45)
    assert r.returncode == 0 and not r.stderr.strip(), r.stderr
    return {line[1:] for line in r.stdout.splitlines() if line.startswith('n')}


def capacity():
    s = json.loads((ROOT/'research_state/general-rsi-state.json').read_text())
    p = json.loads((ROOT/s['panel']).read_text())
    directory = ROOT/'research_runs'/Path(s['panel']).parent.name
    retained = sum(f.stat().st_size for f in directory.glob('slot-*/*.pt') if f.name in ('raw.pt','h200.pt'))
    required = p['reserve_bytes'] + max(0, p['incremental_peak_bytes']-retained)
    free = shutil.disk_usage(ROOT).free
    return dict(free_bytes=free, required_free_bytes=required, margin_bytes=free-required,
                target_with_extra_2GiB=required+2*1024**3, panel=s['panel'])


def protected_current():
    paths = [ROOT/'research_runs/byte-input-quality-v1-001/10'/name for name in ('raw.pt','h200.pt')]
    paths += [ROOT/'research/tokenizer-evidence/public-inference-correction-001/adoption/recipe.json']
    for study in ('general-rsi-panel-001','general-rsi-exposure-001'):
        paths += [ROOT/f'research/tokenizer-evidence/{study}/plan.json']
        paths += [p for p in (ROOT/f'research_runs/{study}').glob('slot-*/*.pt') if p.name in ('raw.pt','h200.pt')]
    return {str(p):sha(p) for p in paths if p.is_file()}


def remove_entries(entries, journal_path):
    removed, skipped = [], []
    with Path(journal_path).open('x') as journal:
        def event(value):
            journal.write(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **value))+'\n')
            journal.flush(); os.fsync(journal.fileno())
        opened = opens()
        for i, entry in enumerate(entries):
            if i % 100 == 0:
                opened = opens()
            p = Path(entry['path'])
            if str(p) in opened or not p.exists() or row(p) != entry:
                skipped.append(str(p)); continue
            event(dict(event='intent', entry=entry))
            p.unlink()
            event(dict(event='removed', path=str(p)))
            removed.append(entry)
        for parent in {Path(e['path']).parent for e in removed}:
            fd=os.open(parent,os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
    return removed, skipped
