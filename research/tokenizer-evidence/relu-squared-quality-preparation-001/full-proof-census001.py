"""New25fixture-role recorded-text census; no Torch/NumPy/model/RNG imports."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from research.recorded_role_scan_v1 import scan
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[3]
R=Path(__file__).with_suffix('')
CAP=32*1024**2

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw)<CAP
    with (R/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    assert sha(R/'plan.json')==arg.plan_sha and set(p.name for p in R.iterdir())=={'plan.json','source-review.json'}
    plan=json.loads((R/'plan.json').read_text());roles=plan['roles'];cohorts=[roles['screen'],*roles['fresh']]
    values=[roles['bootstrap'],*[v for c in cohorts for v in c.values()]]
    assert len(cohorts)==4 and len(values)==len(set(values))==25 and all(type(v) is int and 0<=v<2**63 for v in values)
    for cohort in cohorts:
        assert set(cohort)=={'initializer','private_ungated','train','eval','global_cpu','global_mps'}
        assert int.from_bytes(hashlib.sha256(('relu_squared_model_v1:ungated:'+str(cohort['initializer'])).encode()).digest()[:8],'big')%2**63==cohort['private_ungated']
    for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
    scanner=json.loads(Path(plan['scanner_gate']['path']).read_text())
    assert sha(plan['scanner_gate']['path'])==plan['scanner_gate']['sha256'] and scanner['status']=='passed_new_literal_bounded_role_stream_oracle'
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    assert shutil.disk_usage(ROOT).free>=plan['complete_target_bytes']
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        publish('attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
        try:
            with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
                candidates=[]
                for name in ['research','research_runs','research_state','src','tests','configs','scripts']:
                    folder=ROOT/name
                    if folder.exists():candidates.extend(p for p in folder.rglob('*') if p.is_file() and not p.is_symlink()
                        and '__pycache__' not in p.parts and R not in p.parents and p!=Path(__file__).resolve())
                candidates.extend(p for p in ROOT.iterdir() if p.is_file() and not p.is_symlink() and p.suffix in ('.md','.toml','.txt','.yml','.yaml'))
                files=total=inventory_bytes=binaries=0;matches=[];needles={str(v).encode() for v in values}
                with (R/'inventory.jsonl').open('xb') as out:
                    for p in sorted(set(candidates)):
                        with p.open('rb') as f:
                            def chunks():
                                first=f.read(8192)
                                if first:yield first
                                yield from iter(lambda:f.read(1024**2),b'')
                            got=scan(chunks(),needles)
                        if got['binary']:binaries+=1;continue
                        row=dict(path=str(p.relative_to(ROOT)),size=got['bytes'],sha256=got['sha256'])
                        raw=(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode();inventory_bytes+=len(raw)
                        assert inventory_bytes+2*1024**2<CAP;out.write(raw);files+=1;total+=got['bytes']
                        matches.extend(dict(path=row['path'],**m) for m in got['matches'])
                    out.flush();os.fsync(out.fileno())
                if matches:
                    publish('prior-matches.json',dict(matches=matches));raise ValueError('Proposedfixture role occurs in prior recorded text')
                for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
                assert 'torch' not in sys.modules and 'numpy' not in sys.modules
                publish('result.json',dict(status='new_full_horizon_fixture_roles_unused_before_construction',roles=roles,recorded_text_files=files,
                    recorded_text_bytes=total,prior_matches=matches,binary_skips=binaries,inventory_sha256=sha(R/'inventory.jsonl'),source_pins_unchanged=True,
                    excluded_prospective=[str(R),str(Path(__file__).resolve())],no_torch_numpy_model_optimizer_rng_seed_calls=True,
                    corpus_data_directory_not_scanned=True,binary_payloads_not_loaded_as_tensors=True,scope='25fixture-onlyroles;notscientificscreen/freshroles'))
            publish('STOP',dict(status='complete_new_full_horizon_fixture_role_census',no_replay=True,no_retry=True))
        except BaseException as exc:
            if not (R/'STOP').exists():publish('STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
            raise

if __name__=='__main__':main()
