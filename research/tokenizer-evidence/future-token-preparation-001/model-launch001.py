"""One-shot generated proof admission and620second owned-child watchdog; no Torch."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[3]
R=Path(__file__).with_suffix('')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(folder,name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in folder.iterdir() if p.is_file())+len(raw)<=2*1024**2
    with (folder/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    assert sha(R/'plan.json')==arg.plan_sha and set(p.name for p in R.iterdir())=={'plan.json'}
    plan=json.loads((R/'plan.json').read_text());proof=ROOT/plan['proof_directory'];census=ROOT/plan['census_directory']
    publish(R,'attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
    child=None
    try:
        with CpuPrerequisiteGuard(R/'admission-cpu-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
            assert 'torch' not in sys.modules and 'numpy' not in sys.modules
            for ref in plan['required_metadata']:
                assert sha(ROOT/ref['path'])==ref['sha256']
            for path in plan['syntax_only_sources']:compile(ast.parse((ROOT/path).read_text()),path,'exec')
            assert sha(Path(__file__))==plan['launcher_sha256']
            pp=json.loads((proof/'plan.json').read_text())
            assert len(pp['execution_sources'])==1236
            assert json.loads((census/'guard.json').read_text())['status']=='finished'
            assert json.loads((census/'STOP').read_text())['status']=='complete_original_future_token_initializer_census'
            cr=json.loads((census/'result.json').read_text())
            assert cr['status']=='new_future_token_initializer_unused_before_construction' and cr['prior_matches']==[]
            assert json.loads((ROOT/plan['source_review_guard']).read_text())['status']=='finished'
            assert json.loads((ROOT/plan['source_review_STOP']).read_text())['status']=='complete_original_generated_source_review'
            assert set(p.name for p in proof.iterdir())=={'plan.json','source-review.json'}
            assert shutil.disk_usage(ROOT).free>=plan['complete_target_bytes']
            assert plan['complete_target_bytes']==pp['complete_target_bytes']+2*1024**2
            publish(proof,'admission.json',dict(status='accepted_new_census_and_source_resource_gate_only_generated_proof',
                census_result_sha256=sha(census/'result.json'),census_STOP_sha256=sha(census/'STOP'),
                proof_source_review_sha256=sha(proof/'source-review.json'),launcher_plan_sha256=arg.plan_sha,
                total_process_timeout_seconds=620,complete_target_bytes=plan['complete_target_bytes'],
                no_serialization_corpus_quality_or_adoption_admission=True))
        flags=dict(os.environ,**pp['environment'],PYTHONPATH=str(ROOT))
        args=[str(ROOT/'.venv/bin/python'),str(ROOT/plan['proof_source']),'--plan-sha',sha(proof/'plan.json')]
        with (R/'child.log').open('xb') as output:
            child=subprocess.Popen(args,cwd=ROOT,env=flags,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
            publish(R,'child.json',dict(pid=child.pid,argv=args))
            started=time.monotonic()
            while child.poll() is None:
                if time.monotonic()-started>=620:raise TimeoutError('Owned generated proof exceeded total620seconds')
                if (R/'child.log').stat().st_size>=1024**2:raise RuntimeError('Owned generated proof exceeded1MiB diagnostic log cap')
                if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Disk reserve during owned generated proof')
                time.sleep(.25)
            output.flush();os.fsync(output.fileno())
        publish(R,'completion.json',dict(returncode=child.returncode,seconds=time.monotonic()-started,
            child_log_sha256=sha(R/'child.log'),scope='one_original_generated_proof_only'))
        assert child.returncode==0,'Original generated proof failed; no retry'
        assert json.loads((proof/'STOP').read_text())['status']=='complete_original_generated_future_token_model_proof'
        assert json.loads((proof/'mps-guard.json').read_text())['status']=='finished'
        publish(R,'STOP',dict(status='complete_original_generated_launch',no_retry=True,no_replay=True))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if not (R/'STOP').exists():publish(R,'STOP',dict(status='closed_failed',error=repr(error),no_retry=True,no_replay=True))
        raise


if __name__=='__main__':main()
