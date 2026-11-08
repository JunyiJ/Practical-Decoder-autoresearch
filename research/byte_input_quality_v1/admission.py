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
OUT = BASE/'quality001/admission'
NUMERICAL = BASE/'quality001'
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
    from .contract import DESIGN,slots,select,gates
    for pin in plan['sources']+plan['assets']+plan['prerequisites']:
        p=Path(pin['path']);assert p.stat().st_size==pin['size'] and digest(p)==pin['sha256'],str(p)
        if p.suffix=='.py':compile(ast.parse(p.read_text()),str(p),'exec')
    for package,version in plan['runtime']['identity']['packages'].items():assert importlib.metadata.version(package)==version
    for name in ['training-review001','inference-review001','cache-review001','pilot-review003']:
        assert json.loads((BASE/name/'review.json').read_text())['accepted']
    assert json.loads((BASE/'capacity-recovery001/result.json').read_text())['capacity_satisfied']
    assert plan['design']==DESIGN and plan['slots']==slots()
    assert len(plan['roles'])==len(set(plan['roles'].values()))==21
    assert all(type(n)is int and 0<=n<2**63 for n in plan['roles'].values())
    flat={bundle+'.'+k:v for bundle,roles in plan['bundles'].items() for k,v in roles.items()}
    assert dict(flat,bootstrap=plan['bootstrap_seed'])==plan['roles']
    assert list(plan['bundles'])==['screen','fresh0','fresh1','fresh2'] or set(plan['bundles'])=={'screen','fresh0','fresh1','fresh2'}
    assert all(set(r)=={'native_initializer','composer_initializer','nonlexical_key','train_sampler','validation_sampler'} for r in plan['bundles'].values())
    driver=ast.unparse(ast.parse((ROOT/'research/byte_input_quality_v1/driver.py').read_text()))
    assert "training.write_new(c, archives / 'raw.pt')" in driver
    assert "range(1, 1201)" in driver and "c.averages.snapshots == 41" in driver
    assert "training.Context(model, build_optimizer(model, lr), view, 1200, lr" in driver
    assert "(1, 301, 601, 901, 1200)" in driver and "range(10)" in driver
    writer=next(n for n in ast.parse((ROOT/'research/byte_input_training_v2.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='write_new')
    assert [a.arg for a in writer.args.args]==['context','path']
    assert plan['forecast']['entry_bytes']==20*1024**3 and plan['forecast']['output_cap']==10*1024**3
    assert plan['forecast']['all_final_weights_bytes']==8640*1024**2
    assert plan['forecast']['complete_upper_bytes']==9856*1024**2<plan['forecast']['output_cap']
    assert plan['forecast']['rss_upper_forecast']<4*1024**3 and plan['forecast']['mps_upper_forecast']<6*1024**3
    assert plan['forecast']['wall_budget_seconds']==130800
    for field,value in [('source_sha256',plan['sources']),('runtime_sha256',plan['runtime'])]:
        assert plan['binding'][field]==hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    # Pure synthetic rule tests. No candidate import, model/RNG initializer or
    # scientific data scoring; fixed bootstrap fixture seed is not a study role.
    rows=[dict(s,ce=4.) for s in slots()[:9]]
    assert set(select(rows).values())=={.000625}
    rejected=0
    for bad in [rows[:-1],[dict(r,ce=float('nan')) for r in rows],[dict(r,slot=0) for r in rows]]:
        try:select(bad)
        except AssertionError:rejected+=1
        else:raise AssertionError('Invalid selection accepted')
    assert not all(gates([-.02]*3,[-1.]*3,-.02).values())
    assert not all(gates([-.1,.01,-.1],[-1.]*3,-.02).values())
    assert not all(gates([-.03]*3,[0.,-1.,-1.],-.02).values())
    from .statistics import calculate
    counts=[2048]*16+[1510];pairs=[]
    for seed_index in range(3):
        pair={}
        for arm,offset in [('native',0.),('raw_bytes',-.03),('nonlexical_codes',.01)]:
            values=[n*(4.+i*.001+seed_index*.002+offset) for i,n in enumerate(counts)]
            pair[arm]=dict(target=plan['target'],block_loss_sums=values,total_loss=sum(values),ce=sum(values)/34278)
        pairs.append(pair)
    result=calculate(pairs,7)
    assert result['primary_passes'] and result['lexical_content_supported'] and not result['adopted']
    save('rule-tests.json',dict(accepted=True,complete_screen_required=True,exact_tie_lower_lr=True,invalid_selection_rejections=rejected,
        strict_threshold_equality_rejected=True,unfavorable_seed_rejected=True,zero_upper_bound_rejected=True,independent_scalar_bootstrap=True,
        synthetic_only=True,scientific_roles_consumed=False))
    return dict(trajectories=18,updates=21600,scope='Source/role/resource admission and synthetic selection/statistics checks; actual1200-step qualification occurs only through each real slot and independent saved review.')


def main(expected):
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(OUT.iterdir())and digest(NUMERICAL/'plan.json')==expected
        plan=json.loads((NUMERICAL/'plan.json').read_text());save('attempt.json',dict(pid=os.getpid(),numerical_plan_sha256=expected,no_replay=True))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=240,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
                source=source_review(plan);census(plan)
                fs=os.statvfs(ROOT);free=fs.f_bavail*fs.f_frsize;assert free>=plan['forecast']['entry_bytes']+CAP
                save('review.json',dict(accepted=True,plan_sha256=expected,source_review=source,observed_free_bytes=free))
            save('STOP',dict(status='complete_original_admission',no_replay=True))
        except BaseException as exc:
            save('STOP',dict(status='closed_failed_admission',error=repr(exc),no_replay=True));raise
if __name__=='__main__':main(sys.argv[1])
