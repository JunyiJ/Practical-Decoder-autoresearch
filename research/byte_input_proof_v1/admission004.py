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
OUT = BASE/'model-admission004'
NUMERICAL = BASE/'model-math004'
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
    patterns = {name:re.compile(rb'(?<![0-9])'+str(value).encode()+rb'(?![0-9])')
                for name,value in plan['roles'].items()}
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
    assert len(paths) <= 75000
    read_bytes = 0; hits = []; counts = {'text':0,'binary_prefix':0}
    with (OUT/'role-inventory.jsonl').open('xb') as journal:
        for path in sorted(paths):
            before = path.stat()
            assert stat.S_ISREG(before.st_mode)
            h = hashlib.sha256(); tail = b''; matches = set(); binary = False
            with path.open('rb') as stream:
                while block := stream.read(1048576):
                    read_bytes += len(block)
                    assert read_bytes <= 8589934592
                    if b'\0' in block:
                        binary = True; break
                    h.update(block)
                    text = tail+block
                    # Retain enough overlap to cover every decimal role value.
                    for name,pattern in patterns.items():
                        if pattern.search(text): matches.add(name)
                    tail = text[-64:]
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
    trees = {}
    for pin in plan['sources']+plan['assets']:
        path = Path(pin['path'])
        assert path.stat().st_size == pin['size'] and digest(path) == pin['sha256'], str(path)
        if path.parent == ROOT/'research/byte_input_proof_v1' and path.suffix == '.py':
            tree = ast.parse(path.read_text()); compile(tree,str(path),'exec'); trees[path.name] = tree
    for package,version in plan['runtime']['identity']['packages'].items():
        assert importlib.metadata.version(package) == version
    assert len(set(plan['roles'].values())) == 3
    assert all(type(n) is int and 0 <= n < 2**63 for n in plan['roles'].values())
    case_tree = trees['cases.py']
    literals = {n.targets[0].id:ast.literal_eval(n.value) for n in case_tree.body
                if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
                and n.targets[0].id in ('INPUT_REJECTIONS','OPTIMIZER_REJECTIONS','CPU_ONLY_REJECTIONS','BACKENDS','VARIANTS','SHAPE')}
    names = sum((literals[k] for k in ('INPUT_REJECTIONS','OPTIMIZER_REJECTIONS','CPU_ONLY_REJECTIONS')), ())
    assert len(names) == len(set(names)) == 62
    # Every named mutation must be implemented as a literal branch in the actual
    # transaction source. Runtime receipts must subsequently cover every name.
    transaction_strings = {n.value for n in ast.walk(trees['transactions.py'])
                           if isinstance(n,ast.Constant) and type(n.value) is str}
    assert set(names) <= transaction_strings
    assert literals['SHAPE'] == (8,128,50257)
    assert literals['BACKENDS'] == ('cpu','mps')
    assert literals['VARIANTS'] == ('native','raw_bytes','nonlexical_codes')
    entry = ast.unparse(trees['entry004.py'])
    assert entry.index("pins(plan['sources'])") < entry.index('import torch') < entry.index('from .driver004 import run')
    driver = ast.unparse(trees['driver004.py'])
    assert 'result.loss.backward()' in driver and 'ref_loss.backward()' in driver
    assert 'for step in (1, 2)' in driver
    assert 'for name in names' in driver and 'cases.CPU_ONLY_REJECTIONS' in driver
    assert 'state_dict(' not in driver and 'load_state_dict(' not in driver
    assert driver.index("prefix + '-gradient-' + n") < driver.index('ref_parameters[n].grad.copy_(p.grad)') < driver.index('for step in (1, 2)')
    assert "prefix + '-optimizer-input-' + n" in driver
    failed=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001/model-math003'
    assert json.loads((failed/'STOP').read_text())['status']=='closed_failed_original_actual_model_math'
    old=json.loads((failed/'plan.json').read_text())
    assert plan['tolerances']==old['tolerances'] and plan['required_evidence']==old['required_evidence']
    assert set(plan['roles'].values()).isdisjoint(old['roles'].values())
    assert plan['candidate']==old['candidate'] and plan['full_shape']==old['full_shape']
    reference=ast.unparse(trees['model_reference002.py'])
    assert 'torch.stack' in reference and 'F.linear(windows, weights, bias)' in reference
    assert 'F.conv1d' not in reference and '.unfold' not in reference
    diagnostic=json.loads((BASE/'component-arithmetic-diagnosis001/result.json').read_text())
    assert diagnostic['explicit_window_same_fused_primitive_exact'] is True
    assert diagnostic['matches_failed_bias_unit_382'] is True
    assert diagnostic['changed_relu_decisions']==2 and diagnostic['rng_preserved'] is True
    for pin in plan['recovery_evidence_pins']:
        p=Path(pin['path']);assert p.stat().st_size==pin['size'] and digest(p)==pin['sha256']
    coordinate = ast.unparse(trees['coordinates.py'])
    assert coordinate.index('os.fsync(raw.fileno())') < coordinate.index('assert passed')
    forecast = plan['forecast']
    assert forecast['output_cap'] == 17179869184
    assert forecast['raw_coordinate_upper_bytes']+forecast['codebook_framing_metadata_allowance'] < forecast['output_cap']
    assert forecast['rss_upper_forecast'] < 4294967296
    assert forecast['mps_upper_forecast'] < 6442450944
    assert forecast['entry_bytes'] == 17179869184+8589934592+2147483648
    return dict(positive_model_cases=6, rejections=3*(57*2+5),
                checks='AST/compile, exact source/assets, roles, packages, bounds; no numerical claims')


def main(expected):
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(OUT.iterdir()), 'Original admission only'
        assert digest(NUMERICAL/'plan.json') == expected
        plan = json.loads((NUMERICAL/'plan.json').read_text())
        save('attempt.json',dict(pid=os.getpid(),numerical_plan_sha256=expected,no_replay=True))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=240,
                                     rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
                source = source_review(plan)
                census(plan)
                fs = os.statvfs(ROOT); free=fs.f_bavail*fs.f_frsize
                assert free >= plan['forecast']['entry_bytes']+CAP
                save('review.json',dict(accepted=True,numerical_plan_sha256=expected,
                     source_review=source,observed_free_bytes=free,scope='original finite model proof only'))
            save('STOP',dict(status='complete_original_admission',no_replay=True))
        except BaseException as exc:
            save('STOP',dict(status='closed_failed_admission',error=repr(exc),no_replay=True));raise


if __name__=='__main__': main(sys.argv[1])
