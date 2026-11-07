"""Finite own FFN config/role/output gate; no model, optimizer, corpus or files of tensors."""
import argparse
import copy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT, require, fingerprint, plain
from research.body_allocation_model_v1.evidence import reference
from research.relu_squared_model_v1.contract import ARMS, parameter_shapes
from research.relu_squared_resume_v1.data import config
from research.relu_squared_resume_v1.checkpoint import FILE_CAPS, RAW, H200
from research.relu_squared_resume_v1.contract import runtime
from research.relu_squared_worker_v1.training import validate_config, Counter
from research.relu_squared_worker_v1.execution import sources, checked, idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock

E=Path(__file__).resolve().parent
R=E/'relu-squared-worker-source001'
PREREQUISITES=['relu-squared-resume001/result.json','relu-squared-resume001/mps-guard.json',
    'relu-squared-resume001/STOP','relu-squared-resume001-review.json','relu-squared-resume001-archive-closure.json']


def publish(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    require(sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw)<=2*1024**2,'Metadata cap')
    with (R/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    require(reference(R/'plan.json')['sha256']==arg.plan_sha and set(p.name for p in R.iterdir())=={'plan.json'},'One pristine exact-plan source gate required')
    with session_lock(GLOBAL_LOCK.parent):
        idle();torch.set_num_threads(4);runtime()
        plan=json.loads((R/'plan.json').read_text());require(sources()==plan['sources'],'Pinned source inventory differs')
        census=E/'relu-squared-worker-seed-census001';receipt=checked(plan['census'])
        require(plan['census']==reference(census/'result.json') and not receipt['prior_matches']
            and receipt['status']=='new_ffn_worker_chain_roles_unused_before_construction'
            and checked(reference(census/'STOP'))['status']=='complete_new_ffn_worker_chain_role_census'
            and checked(reference(census/'guard.json'))['status']=='finished','Original successful twelve-role census required')
        require(checked(plan['admission'])['source_gate_source']==reference(__file__),'Exact prior source/case/resource admission required')
        publish('attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
        try:
            with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
                initial=torch.get_rng_state().clone();roles=receipt['roles'];rows=[];negative=[]
                module_spec=importlib.util.spec_from_file_location('new_ffn_prerequisite_builder',E/'relu-squared-worker001.py')
                builder=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(builder)
                for phase,steps in [('worker',4),('pilot',20)]:
                    prefix='four' if steps==4 else 'twenty'
                    prepared=builder.specs(plan['sources'],phase,roles)
                    require(len(prepared)==3 and [v['arm'] for v in prepared]==list(ARMS),'Exact three own arms required')
                    for slot,arm in enumerate(ARMS):
                        cfg=config(arm,steps,roles[prefix+'_initializer'],roles[prefix+'_train'],roles[prefix+'_eval'],
                            global_cpu_seed=roles[prefix+'_global_cpu'],global_mps_seed=roles[prefix+'_global_mps'])
                        cfg.training.eval_every=1 if steps==4 else 5;cfg.training.eval_iters=1
                        cfg.relu_squared_execution=dict(kind='ffn_prerequisite_execution_v1',sources_sha256=fingerprint(plan['sources']),runner_sha256=reference(E/'relu-squared-worker001.py')['sha256'])
                        path=builder.ROOTS[phase]/f'evidence/pilots/{slot:02d}'
                        cfg.training.checkpoint_dir=str(path/'checkpoints');cfg.training.metrics_file=str(path/'metrics.jsonl');cfg.training.summary_file=str(path/'worker.json')
                        saved=plain(cfg);require(validate_config(cfg)==steps and prepared[slot]['config']==saved,'Own literal-config/builder agreement required')
                        for kind in ['horizon','legacy','binding','lr','clipping','cadence','global_cpu_type','global_mps_range','global_missing']:
                            bad=copy.deepcopy(cfg)
                            if kind=='horizon':bad.training.max_iters=1200
                            elif kind=='legacy':bad.model={}
                            elif kind=='binding':bad.relu_squared_execution.kind='foreign'
                            elif kind=='lr':bad.training.learning_rate=.0025
                            elif kind=='clipping':bad.training.grad_clip=.5
                            elif kind=='cadence':bad.training.eval_every=2
                            elif kind=='global_cpu_type':bad.rng_identity.global_cpu=True
                            elif kind=='global_mps_range':bad.rng_identity.global_mps=-1
                            else:del bad.rng_identity['global_cpu']
                            try:validate_config(bad)
                            except ValueError:negative.append(dict(arm=arm,steps=steps,case=kind))
                            else:raise AssertionError('Forbidden config accepted: '+kind)
                            require(plain(cfg)==saved,'Caller valid configuration changed')
                        shapes=parameter_shapes(arm);muon={n:s for n,s in shapes.items() if len(s)==2 and n not in ('transformer.wte.weight','lm_head.weight')}
                        adam={n:s for n,s in shapes.items() if n not in muon};elements=sum(math.prod(s) for s in shapes.values())
                        require(elements==28492416 and len(muon)==(44 if arm=='native_swiglu' else 40) and len(adam)==44,'Own ordered scalar inventory differs')
                        learned=elements*4;model=learned+65536;moments=(sum(math.prod(s) for s in muon.values())+2*sum(math.prod(s) for s in adam.values()))*4+44*4
                        raw=model+moments+learned;h200=model;raw_upper=raw+16*1024**2;h200_upper=h200+8*1024**2
                        require(raw==444934320 and moments==216929456 and raw_upper<FILE_CAPS[RAW] and h200_upper<FILE_CAPS[H200]
                            and raw_upper+h200_upper<prepared[slot]['weight_cap']==640*1024**2,'Every archive and pair bound must fit')
                        rows.append(dict(arm=arm,steps=steps,raw_tensor_bytes=raw,h200_tensor_bytes=h200,raw_upper_bytes=raw_upper,h200_upper_bytes=h200_upper,
                            raw_cap=FILE_CAPS[RAW],h200_cap=FILE_CAPS[H200],pair_cap=640*1024**2,parameter_tensors=len(shapes),muon_tensors=len(muon),adam_tensors=len(adam)))
                for case,phase in [('missing','worker'),('duplicate','worker'),('bool','worker'),('private_four','worker'),('private_twenty','pilot'),('extra','worker')]:
                    bad=copy.deepcopy(roles)
                    if case=='missing':del bad['four_train']
                    elif case=='duplicate':bad['four_train']=bad['four_eval']
                    elif case=='bool':bad['four_global_mps']=True
                    elif case=='private_four':bad['four_private_ungated']=(bad['four_private_ungated']+1)%2**63
                    elif case=='private_twenty':bad['twenty_private_ungated']=(bad['twenty_private_ungated']+1)%2**63
                    else:bad['foreign']=1
                    try:builder.specs(plan['sources'],phase,bad)
                    except ValueError:negative.append(dict(case='roles_'+case))
                    else:raise AssertionError('Invalid role contract accepted: '+case)
                require(roles==receipt['roles'],'Census role input changed')
                payload=dict(kind='literal_ffn_counter_only',x=torch.tensor([.5625,-.875,2.25],dtype=torch.float32))
                counter=Counter(2*1024**2);torch.save(payload,counter);buffer=io.BytesIO();torch.save(payload,buffer)
                require(counter.size==len(buffer.getvalue()),'Counter/BytesIO actual stream differs')
                boundary=Counter(7);require(boundary.write(b'qrsuvwx')==7 and boundary.tell()==7,'Exact counter boundary differs')
                try:boundary.write(b'!')
                except ValueError:negative.append(dict(case='counter_overflow'))
                else:raise AssertionError('Counter overflow accepted')
                require(boundary.tell()==7,'Overflow changed prior counter count')
                require(builder.CHAIN_TARGET==15000928256 and builder.CHAIN_FLOOR==12853444608
                    and builder.forecast()['worker_seconds']==180. and builder.forecast()['acceptance_seconds']==240.
                    and builder.forecast()['complete_six_run_seconds']==2640.,'Implemented complete forecast differs')
                require(torch.equal(initial,torch.get_rng_state()) and sources()==plan['sources'],'Source/global CPU RNG changed')
                publish('result.json',dict(status='passed_ffn_worker_source_scope_and_output_bounds',sources=plan['sources'],prepared_builder=reference(E/'relu-squared-worker001.py'),
                    prerequisites={n:reference(E/n) for n in PREREQUISITES},census=plan['census'],admission=plan['admission'],rows=rows,negative_cases=negative,
                    config_rejections=54,role_rejections=6,counter_overflow_rejected=True,counter_bytes=counter.size,
                    model_optimizer_corpus_checkpoint_weightfile_quality_seed_calls=0,private_generator_seed_calls=0,mps_numerical_calls=0,
                    source_global_cpu_rng_unchanged=True,complete_chain_target_bytes=builder.CHAIN_TARGET,forecast=builder.forecast(),
                    limit='Finite config/role/scalar bounds plus NEW tiny in-memory serializer only; no worker, physical public scoring, actual-model archive or quality result.'))
            publish('STOP',dict(status='complete_ffn_worker_source_scope_output_gate',no_replay=True,no_retry=True))
        except BaseException as exc:
            if not (R/'STOP').exists():publish('STOP',dict(status='closed_failed',no_retry=True,error=repr(exc)))
            raise

if __name__=='__main__':main()
