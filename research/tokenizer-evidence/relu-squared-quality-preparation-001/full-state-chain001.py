"""One serial original chain of three independently guarded600s generated proofs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
B=Path(__file__).resolve().parent
R=B/'full-state001'
A=ROOT/'research_runs/tokenizer-relu-squared-full-state-v1-001'
ARMS=['native_swiglu','ungated_silu','ungated_relu_squared']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish(path,value):
    raw=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    assert sum(p.stat().st_size for p in R.rglob('*') if p.is_file())+len(raw)<=16*1024**2
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    assert sha(R/'plan.json')==arg.plan_sha and not A.exists() and not (R/'chain-attempt.json').exists(),'Unattempted exact original chain required'
    plan=json.loads((R/'plan.json').read_text())
    for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
    gate=json.loads(Path(plan['config_gate']['path']).read_text());assert sha(plan['config_gate']['path'])==plan['config_gate']['sha256'] and gate['status']=='passed_new_full_horizon_config_selection_statistics_fixture'
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);A.mkdir();publish(R/'chain-attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
        try:
            results=[]
            for arm in ARMS:
                assert not (A/'STOP').exists() and not (R/(arm+'-attempt.json')).exists()
                start=time.monotonic();argv=[sys.executable,str(B/'full-state001.py'),'--plan-sha',arg.plan_sha,'--arm',arm,'--lock-fd',str(lock.fileno())]
                with (R/(arm+'-process.log')).open('xb') as log:
                    process=subprocess.run(argv,cwd=ROOT,pass_fds=(lock.fileno(),),stdout=log,stderr=subprocess.STDOUT,timeout=620,check=False)
                publish(R/(arm+'-process.json'),dict(argv=argv,pid='see original attempt/guard',parent_pid=os.getpid(),returncode=process.returncode,elapsed_seconds=time.monotonic()-start,timeout_seconds=620,worker_mps_timeout_seconds=600))
                assert process.returncode==0,'Original generated arm failed;no retry'
                guard=json.loads((R/(arm+'-mps-guard.json')).read_text());assert guard['status']=='finished' and guard['reason'] is None
                result=json.loads((R/(arm+'-result.json')).read_text());assert result['status']=='accepted_genuine_generated_full_horizon_arm' and result['step']==1200 and result['actual_targets']==1228800 and result['ema_snapshots']==41
                results.append(dict(arm=arm,result=dict(path=str(R/(arm+'-result.json')),sha256=sha(R/(arm+'-result.json')))))
                print(json.dumps(dict(status='generated1200_arm_complete',arm=arm)),flush=True)
            publish(R/'completed.json',dict(status='all_three_original_generated1200_arms_complete',results=results,actual_optimizer_updates=3600,actual_generated_targets=3686400,
                serving_scope='Per-model generated H20083/45 protocol plus57/71cache;fullninefresh main consumer not yet executed',no_corpus_quality_or_adoption_claim=True))
            publish(A/'STOP',dict(status='complete_original_generated1200_FFN_state_chain',receipt=dict(path=str(R/'completed.json'),sha256=sha(R/'completed.json')),no_replay=True,no_retry=True))
            publish(R/'STOP',dict(status='complete_original_generated1200_FFN_state_chain',no_replay=True,no_retry=True))
        except BaseException as exc:
            for p in [A/'STOP',R/'STOP']:
                if not p.exists():publish(p,dict(status='closed_failed',error=repr(exc),no_retry=True))
            raise

if __name__=='__main__':main()
