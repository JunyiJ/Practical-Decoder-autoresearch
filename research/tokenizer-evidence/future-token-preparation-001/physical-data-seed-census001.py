"""NEW physical data-view private-role recorded-text census; no Torch/NumPy/model/RNG calls."""
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


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw)<=CAP
    with (R/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    assert sha(R/'plan.json')==arg.plan_sha and set(p.name for p in R.iterdir())=={'plan.json','source-review.json'}
    plan=json.loads((R/'plan.json').read_text());roles=plan['roles']
    assert type(roles) is dict and set(roles)=={'train','val'}
    assert all(type(seed) is int and 0<=seed<2**63 for seed in roles.values()) and len(set(roles.values()))==2
    for name,h in plan['execution_sources'].items():assert sha(ROOT/name)==h,name
    gate=ROOT/plan['scanner_gate']['path']
    assert sha(gate)==plan['scanner_gate']['sha256']
    assert json.loads(gate.read_text())['status']=='passed_new_literal_bounded_role_stream_oracle'
    assert shutil.disk_usage(ROOT).free>=plan['complete_target_bytes']
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        publish('attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
        try:
            with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
                candidates=[]
                for name in ('research','research_runs','research_state','src','tests','configs','scripts'):
                    folder=ROOT/name
                    if folder.exists():candidates.extend(p for p in folder.rglob('*') if p.is_file() and not p.is_symlink()
                        and '__pycache__' not in p.parts and R not in p.parents)
                candidates.extend(p for p in ROOT.iterdir() if p.is_file() and not p.is_symlink() and p.suffix in ('.md','.toml','.txt','.yml','.yaml'))
                files=total=inventory_bytes=binaries=0;matches=[];needles={str(seed).encode() for seed in roles.values()}
                with (R/'inventory.jsonl').open('xb') as out:
                    for path in sorted(set(candidates)):
                        with path.open('rb') as f:
                            def chunks():
                                yield f.read(8192)
                                yield from iter(lambda:f.read(1024**2),b'')
                            got=scan(chunks(),needles)
                        if got['binary']:binaries+=1;continue
                        row=dict(path=str(path.relative_to(ROOT)),size=got['bytes'],sha256=got['sha256'])
                        raw=(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n').encode();inventory_bytes+=len(raw)
                        assert inventory_bytes+2*1024**2<=CAP
                        out.write(raw);files+=1;total+=got['bytes']
                        matches.extend(dict(path=row['path'],**m) for m in got['matches'])
                    out.flush();os.fsync(out.fileno())
                if matches:
                    publish('prior-matches.json',dict(matches=matches));raise ValueError('Private sampler role occurs in prior recorded text')
                for name,h in plan['execution_sources'].items():assert sha(ROOT/name)==h,name
                assert 'torch' not in sys.modules and 'numpy' not in sys.modules
                publish('result.json',dict(status='new_physical_data_private_roles_unused_before_construction',roles=roles,
                    prior_matches=[],recorded_text_files=files,recorded_text_bytes=total,binary_skips=binaries,
                    inventory_sha256=sha(R/'inventory.jsonl'),excluded_current_proposal_and_outputs=str(R.relative_to(ROOT)),
                    no_torch_numpy_model_optimizer_rng_seed_calls=True,no_corpus_loader=True,
                    scope='two_private_roles_exclusive_to_one_original_physical_data_view_proof;no_model_or_scientific_roles'))
            publish('STOP',dict(status='complete_original_physical_data_role_census',no_retry=True,no_replay=True))
        except BaseException as error:
            if not (R/'STOP').exists():publish('STOP',dict(status='closed_failed',error=repr(error),no_retry=True,no_replay=True))
            raise


if __name__=='__main__':main()
