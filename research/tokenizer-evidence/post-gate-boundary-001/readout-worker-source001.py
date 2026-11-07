"""One source/scope/output-bound prerequisite gate, no model or corpus calls."""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,fingerprint
from research.body_allocation_model_v1.evidence import reference
from research.readout_mixture_model_v1.contract import ARMS
from research.readout_mixture_resume_v1.data import config
from research.readout_mixture_worker_v1.training import validate_config,Counter
from research.readout_mixture_worker_v1.execution import sources
from research.readout_mixture_resume_v1.contract import runtime
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

E=Path(__file__).resolve().parent
R=E/'readout-worker-source001'


def publish(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    require(sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw)<=2*1024**2,'Metadata cap')
    with (R/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    require(reference(R/'plan.json')['sha256']==arg.plan_sha and set(p.name for p in R.iterdir())=={'plan.json'},'One pristine source gate attempt')
    torch.set_num_threads(4);runtime()
    plan=json.loads((R/'plan.json').read_text());require(sources()==plan['sources'],'Pinned source inventory differs')
    publish('attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
    try:
        with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
            initial=torch.get_rng_state().clone();roles=json.loads((E/'readout-worker-seed-census001/result.json').read_text())['roles']
            require(not json.loads((E/'readout-worker-seed-census001/result.json').read_text())['prior_matches'],'Actual fresh role audit required')
            rows=[];rejections=0
            for arm,cap in zip(ARMS,[560*1024**2,640*1024**2,640*1024**2]):
                for steps in [4,20]:
                    cfg=config(arm,steps,roles['backbone_initializer'],roles['private_train_and_global_rng'],roles['private_evaluation'])
                    cfg.readout_execution=dict(kind='readout_prerequisite_execution_v1',sources_sha256=fingerprint(plan['sources']),runner_sha256=reference(E/'readout-worker001.py')['sha256'])
                    require(validate_config(cfg)==steps,'Exact finite config rejected')
                    for kind in ['horizon','legacy','binding','clipping']:
                        bad=copy.deepcopy(cfg)
                        if kind=='horizon':bad.training.max_iters=1200
                        elif kind=='legacy':bad.model={}
                        elif kind=='binding':bad.readout_execution.kind='legacy'
                        else:bad.training.grad_clip=.5
                        try:validate_config(bad)
                        except ValueError:rejections+=1
                        else:raise AssertionError('Forbidden config accepted')
                    upper=5*ARMS[arm][0]*4+8*1024**2
                    require(upper+8*1024**2<=cap,'All-Adam tensor-byte upper bound plus serialization/history margin must fit pair cap')
                    rows.append(dict(arm=arm,steps=steps,parameters=ARMS[arm][0],pair_cap=cap,all_adam_payload_and_8MiB_overhead_bound=upper,remaining_margin=cap-upper))
            payload=dict(kind='generated_counter_only',x=torch.tensor([.25,-.5,1.5],dtype=torch.float32))
            counter=Counter(2*1024**2);torch.save(payload,counter);buffer=io.BytesIO();torch.save(payload,buffer)
            require(counter.size==len(buffer.getvalue()),'Serializer counter differs')
            small=Counter(4);require(small.write(b'1234')==4,'Counter boundary differs')
            try:small.write(b'x')
            except ValueError:pass
            else:raise AssertionError('Counter exceeded cap')
            require(torch.equal(initial,torch.get_rng_state()) and sources()==plan['sources'],'Source/global CPU RNG changed')
            refs={n:reference(E/n) for n in ['readout-resume001/result.json','readout-resume001/mps-guard.json','readout-resume001/STOP','readout-resume-review001.json']}
            publish('result.json',dict(status='passed_readout_worker_source_scope_and_output_bounds',sources=plan['sources'],prepared_builder=reference(E/'readout-worker001.py'),prerequisites=refs,
                rows=rows,forbidden_configs_rejected=rejections,counter_bytes=counter.size,counter_overflow_rejected=True,model_optimizer_corpus_checkpoint_weightfile_quality_seed_calls=0,
                private_generator_seed_calls=0,source_global_cpu_rng_unchanged=True,limit='Algebraic conservative bound; exact archive lengths measured before actual writes. Native validation and runtime remain original prerequisite checks.'))
        publish('STOP',dict(status='complete_readout_worker_source_scope_output_gate',no_replay=True))
    except BaseException as exc:
        if not (R/'STOP').exists():publish('STOP',dict(status='closed_failed',no_retry=True,exception=repr(exc)))
        raise


if __name__=='__main__':main()
