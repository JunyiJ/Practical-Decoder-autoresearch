"""Independent static/role/source gate. Never imports the candidate or proof."""
import ast
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import stat
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT = BASE/'pilot-admission002'
NUMERICAL = BASE/'pilot-math002'
CAP = 67108864


def save(name, data):
    raw = (json.dumps(data, sort_keys=True, allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in OUT.iterdir())+len(raw) < CAP
    with (OUT/name).open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1048576): h.update(block)
    return h.hexdigest()


def census(plan):
    by_decimal={str(value).encode():name for name,value in plan['roles'].items()}
    pattern=re.compile(rb'(?<![0-9])(?:'+b'|'.join(by_decimal)+rb')(?![0-9])')
    excluded = [OUT, NUMERICAL]
    paths = set()
    for folder in plan['role_history_roots']:
        root = Path(folder)
        if not root.exists(): continue
        for parent,dirs,files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != '__pycache__' and
                             not (Path(parent)/d).is_symlink() and Path(parent)/d not in excluded)
            for name in files:
                path = Path(parent)/name
                if path.is_file() and not path.is_symlink(): paths.add(path)
    paths.update(p for p in ROOT.iterdir() if p.is_file() and not p.is_symlink())
    assert len(paths) <= 90000
    read_bytes = 0; hits = []; counts = {'text':0,'binary_prefix':0}
    with (OUT/'role-inventory.jsonl').open('xb') as journal:
        for path in sorted(paths):
            before = path.stat()
            assert stat.S_ISREG(before.st_mode)
            h = hashlib.sha256(); tail = b''; matches = set(); binary = False
            with path.open('rb') as stream:
                block = stream.read(4096)
                while block:
                    read_bytes += len(block)
                    assert read_bytes <= 8589934592
                    if b'\0' in block:
                        binary = True; break
                    h.update(block)
                    text = tail+block
                    # Retain enough overlap to cover every decimal role value.
                    for match in pattern.finditer(text):matches.add(by_decimal[match.group()])
                    tail = text[-64:]
                    block = stream.read(1048576)
            after = path.stat()
            assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) == (
                after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
            kind = 'binary_prefix' if binary else 'text'; counts[kind] += 1
            if matches and not binary: hits.append(dict(path=str(path), roles=sorted(matches)))
            record = dict(path=str(path), size=before.st_size, kind=kind,
                          sha256=None if binary else h.hexdigest(), matches=[] if binary else sorted(matches))
            journal.write((json.dumps(record,sort_keys=True)+'\n').encode())
            assert journal.tell() < CAP-2097152
        journal.flush();os.fsync(journal.fileno())
    result = dict(roles=plan['roles'], matches=hits, files=len(paths), classes=counts,
                  streamed_bytes=read_bytes, scope='declared project/automation text; binary prefixes only',
                  status='reserved_exclusive_unused' if not hits else 'prior_role_match')
    save('roles.json', result)
    assert not hits, 'Role collision; no draws permitted'


def source_review(plan):
    for pin in plan['sources']+plan['assets']+plan['prerequisites']:
        p=Path(pin['path']);assert p.stat().st_size==pin['size'] and digest(p)==pin['sha256'],str(p)
        if p.suffix=='.py':compile(ast.parse(p.read_text()),str(p),'exec')
    for package,version in plan['runtime']['identity']['packages'].items():assert importlib.metadata.version(package)==version
    for name in ['training-review001','inference-review001','cache-review001']:
        assert json.loads((BASE/name/'review.json').read_text())['accepted']
    assert len(plan['roles'])==len(set(plan['roles'].values()))==5
    assert all(type(n)is int and 0<=n<2**63 for n in plan['roles'].values())
    assert plan['required']==dict(arms=3,updates_per_arm=20,total_updates=60,backend='mps',learning_rate=.00125,
        complete_validation_targets=34278,averaging_snapshots=8,quality_selection=False)
    driver=ast.unparse(ast.parse((ROOT/'research/byte_input_pilot_v2/driver.py').read_text()))
    assert "training.write_new(c, archives / (variant + '-raw.pt'))" in driver
    assert "for step in range(1, 21)" in driver and "c.averages.snapshots == 8" in driver
    assert "('native', 'raw_bytes', 'nonlexical_codes')" in driver and "c.step()" in driver
    assert "training.Context(model, build_optimizer(model, 0.00125), view, 20, 0.00125" in driver
    forecast=plan['forecast']
    assert forecast['retained_upper_bytes']==(512+2*320+128+2*80+64)*1024**2
    assert forecast['retained_upper_bytes']+512*1024**2<forecast['output_cap']==2147483648
    assert forecast['entry_bytes']==2147483648+8589934592+2147483648
    assert forecast['rss_upper_forecast']<4294967296 and forecast['mps_upper_forecast']<6442450944
    for field,value in [('source_sha256',plan['sources']),('runtime_sha256',plan['runtime'])]:
        assert plan['binding'][field]==hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(arms=3,updates=60,scope='Fresh-role original20-update resource pilot; no quality selection')


def main(expected):
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(OUT.iterdir())and digest(NUMERICAL/'plan.json')==expected
        plan=json.loads((NUMERICAL/'plan.json').read_text());save('attempt.json',dict(pid=os.getpid(),numerical_plan_sha256=expected,no_replay=True))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=240,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
                source=source_review(plan);census(plan)
                fs=os.statvfs(ROOT);free=fs.f_bavail*fs.f_frsize;assert free>=plan['forecast']['entry_bytes']+CAP
                save('review.json',dict(accepted=True,numerical_plan_sha256=expected,source_review=source,observed_free_bytes=free))
            save('STOP',dict(status='complete_original_admission',no_replay=True))
        except BaseException as exc:
            save('STOP',dict(status='closed_failed_admission',error=repr(exc),no_replay=True));raise
if __name__=='__main__':main(sys.argv[1])
