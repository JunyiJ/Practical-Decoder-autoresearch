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
OUT = BASE/'state-admission003'
NUMERICAL = BASE/'state-math003'
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
                    for name,pattern in patterns.items():
                        if pattern.search(text): matches.add(name)
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
        path=Path(pin['path']);assert path.stat().st_size==pin['size'] and digest(path)==pin['sha256'],str(path)
        if path.suffix=='.py': compile(ast.parse(path.read_text()),str(path),'exec')
    for package,version in plan['runtime']['identity']['packages'].items():
        assert importlib.metadata.version(package)==version
    accepted=json.loads((BASE/'model-saved-review004/review.json').read_text())
    assert accepted['accepted'] and accepted['all_coordinates']==1412427220
    state_review=json.loads((BASE/'state-source001/review.json').read_text())
    assert state_review['accepted_source_only'] and digest(ROOT/'research/byte_input_state_v1.py')==state_review['source_sha256']
    assert len(set(plan['roles'].values()))==3 and all(type(x)is int and 0<=x<2**63 for x in plan['roles'].values())
    old=json.loads((BASE/'model-math004/plan.json').read_text());assert set(plan['roles'].values()).isdisjoint(old['roles'].values())
    folder=ROOT/'research/byte_input_state_proof_v3'
    entry=ast.unparse(ast.parse((folder/'entry.py').read_text()))
    assert entry.index("pins(plan['sources'])")<entry.index('import torch')<entry.index('from .driver import run')
    cases=ast.parse((folder/'cases.py').read_text())
    constants={n.targets[0].id:ast.literal_eval(n.value) for n in cases.body if isinstance(n,ast.Assign)}
    assert constants['SHAPE']==(8,128,50257) and constants['BACKENDS']==('cpu','mps') and constants['VARIANTS']==('native','raw_bytes','nonlexical_codes')
    assert len(constants['REJECTIONS'])==33 and len(constants['IO_REJECTIONS'])==5
    driver=ast.unparse(ast.parse((folder/'driver.py').read_text()))
    strings={n.value for n in ast.walk(ast.parse(driver)) if isinstance(n,ast.Constant) and type(n.value)is str}
    assert set(constants['REJECTIONS']+constants['IO_REJECTIONS'])<=strings
    assert 'actual_step == expected_step' in driver and 'actual_end == expected_end' in driver
    assert 'actual_start == expected_start' in driver and 'actual_draw == expected_draw' in driver
    assert 'result.loss.backward()' in driver and 'before == after' in driver and 'payload_before == payload_after' in driver
    assert 'adapter.restore_pristine' in driver and 'adapter.read_bound' in driver and 'adapter.write_new' in driver
    assert plan['forecast']['archives_upper_bytes']==12*536870912
    assert plan['forecast']['archives_upper_bytes']+plan['forecast']['metadata_allowance']<plan['forecast']['output_cap']==8589934592
    assert plan['forecast']['entry_bytes']==8589934592+8589934592+2147483648
    assert plan['forecast']['rss_upper_forecast']<4294967296 and plan['forecast']['mps_upper_forecast']<6442450944
    serialized=json.dumps(plan['sources'],sort_keys=True,separators=(',',':')).encode()
    runtime=json.dumps(plan['runtime'],sort_keys=True,separators=(',',':')).encode()
    assert plan['binding']==dict(source_sha256=hashlib.sha256(serialized).hexdigest(),runtime_sha256=hashlib.sha256(runtime).hexdigest())
    failed=json.loads((BASE/'state-math001/plan.json').read_text())
    assert plan['required_evidence']==failed['required_evidence']
    assert plan['forecast']['output_cap']==failed['forecast']['output_cap']
    assert set(plan['roles'].values()).isdisjoint(failed['roles'].values())
    assert (folder/'cases.py').read_bytes()==(ROOT/'research/byte_input_state_proof_v1/cases.py').read_bytes()
    assert 'params[a].shape == params[b].shape' in driver and 'params[target] = params[source]' in driver
    diagnosis=json.loads((BASE/'state-fixture-diagnosis001/diagnosis.json').read_text())
    assert diagnosis['classification']=='test_harness' and diagnosis['same_shape_peers']==[]
    recovery=BASE/'embedding-recovery-review001'
    reviewed=json.loads((recovery/'review.json').read_text())
    assert reviewed['accepted_primitive_and_source_delta'] and reviewed['verified_primitive_gradient_coordinates']==46620
    assert json.loads((recovery/'guard.json').read_text())['status']=='finished'
    assert json.loads((recovery/'STOP').read_text())['status']=='complete_original_primitive_and_source_review'
    for pin in json.loads((recovery/'plan.json').read_text())['pins']:
        path=Path(pin['path']);assert path.stat().st_size==pin['size'] and digest(path)==pin['sha256']
    expected=(ROOT/'research/byte_input_state_proof_v2/driver.py').read_text().replace('research.byte_input_model_v2','research.byte_input_model_v3').replace('byte_input_state_v1 as adapter','byte_input_state_v2 as adapter')
    needle="    publish(label+'-gradients.json',gradients)"
    expected=expected.replace(needle,needle+"\n    if 'composer.byte_weight' in gradients:\n        byte_gradient=dict(model.ordered_parameters())['composer.byte_weight'].grad.detach().cpu()\n        publish(label+'-byte-gradient.json',dict(shape=list(byte_gradient.shape),values=byte_gradient.tolist()))\n        del byte_gradient")
    assert (folder/'driver.py').read_text()==expected
    prior=json.loads((BASE/'state-math002/plan.json').read_text())
    assert plan['required_evidence']==prior['required_evidence']
    assert set(plan['roles'].values()).isdisjoint(prior['roles'].values())
    assert plan['forecast']['grouped_backward_workspace_allowance']==8388608
    return dict(model_cases=6,invalid_transactions=6*(33+5+1),scope='Static source/role/resource admission only')


def main(expected):
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(OUT.iterdir()),'Original admission only'
        assert digest(NUMERICAL/'plan.json')==expected
        plan=json.loads((NUMERICAL/'plan.json').read_text())
        save('attempt.json',dict(pid=os.getpid(),numerical_plan_sha256=expected,no_replay=True))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=240,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
                source=source_review(plan);census(plan)
                fs=os.statvfs(ROOT);free=fs.f_bavail*fs.f_frsize
                assert free>=plan['forecast']['entry_bytes']+CAP
                save('review.json',dict(accepted=True,numerical_plan_sha256=expected,source_review=source,observed_free_bytes=free))
            save('STOP',dict(status='complete_original_admission',no_replay=True))
        except BaseException as exc:
            save('STOP',dict(status='closed_failed_admission',error=repr(exc),no_replay=True));raise


if __name__=='__main__':main(sys.argv[1])
