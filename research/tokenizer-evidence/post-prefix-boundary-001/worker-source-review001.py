"""Finite config/report adversaries; no training, full scoring or quality seed."""
import copy
import gc
import importlib.util
import json
import math
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_resume_v1.data import config,load,architecture
from research.tokenizer_moe_resume_v1.contract import runtime
from research.tokenizer_moe_model_v1.model import GPT
from research.tokenizer_moe_worker_v1.execution import sources,idle
from research.tokenizer_moe_worker_v1.training import run
from research.tokenizer_moe_worker_v1.scoring import target_contract,validate_report
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock

E=Path(__file__).resolve().parent


def reject(fn):
    try:fn()
    except (ValueError,TypeError,RuntimeError):return
    raise AssertionError('Disallowed config/report accepted')


def publish(name,value):
    with (E/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def main():
    torch.set_num_threads(4)
    with session_lock(GLOBAL_LOCK.parent):
        idle();env=runtime();require(not (E/'worker-source-review001-plan.json').exists(),'New source gate only')
        pins=sources();builder=E/'worker001-build-run001.py'
        refs={n:reference(E/n) for n in ['resume001-result.json','resume001-guard.json','STOP-resume001','continuation-completed003.json']}
        publish('worker-source-review001-plan.json',dict(source=reference(__file__),prepared_builder=reference(builder),sources=pins,prerequisites=refs,training=False,scoring=False,scientific_seed=False))
        with CpuPrerequisiteGuard(E/'worker-source-review001-guard.json',ROOT),torch.random.fork_rng(devices=[]):
            rows=[]
            for arm in ['dense','wide','moe']:
                for steps in [4,20]:
                    for lr in [.000625,.00125,.0025]:require(architecture(config(arm,steps=steps,seed=600464,eval_seed=600465,lr=lr))['arm']==arm,'Finite genuine config route differs')
                reject(lambda:run(config(arm,steps=1200,seed=600464,eval_seed=600465),None,None))
                reject(lambda:config(arm,lr=.01))
                cfg=config(arm,seed=600464,eval_seed=600465);data=load(cfg,'cpu');model=GPT(plain(cfg.moe_model))
                reject(lambda:target_contract(model,data,cfg));model.eval();target=target_contract(model,data,cfg)
                fake=copy.copy(data);fake.val_data=data.val_data.clone();fake.val_data[0]+=1
                reject(lambda:target_contract(model,fake,cfg))
                oldcfg=copy.deepcopy(cfg);oldcfg['model']=oldcfg.backbone_template;reject(lambda:architecture(oldcfg))
                values=[.25+(i%19)*.07 for i in range(34278)];total=math.fsum(values)
                report=dict(target=target,window_loss_sums=[math.fsum(values[i:i+128]) for i in range(0,34278,128)],
                            block_loss_sums=[math.fsum(values[i:i+2048]) for i in range(0,34278,2048)],total_loss=total,ce=total/34278,
                            bits_per_byte=total/math.log(2)/target['target_bytes'],loss_dtype='FP32_cross_entropy_then_CPU_FP64_aggregation',holdout_scored=False,aux_loss_included=False)
                validate_report(report,target);cases=[]
                for mode in ['partial','target','ce','bytes','nan','block','holdout','aux']:
                    bad=copy.deepcopy(report)
                    if mode=='partial':bad['window_loss_sums'].pop()
                    elif mode=='target':bad['target']['target_ids_sha256']='0'*64
                    elif mode=='ce':bad['ce']+=.01
                    elif mode=='bytes':bad['bits_per_byte']+=.01
                    elif mode=='nan':bad['total_loss']=float('nan')
                    elif mode=='block':bad['block_loss_sums'][0]+=.1
                    elif mode=='holdout':bad['holdout_scored']=True
                    else:bad['aux_loss_included']=True
                    cases.append(bad)
                for bad in cases:reject(lambda bad=bad:validate_report(bad,target))
                rows.append(dict(arm=arm,finite_prerequisite_configs=6,full1200rejected_before_data_model=True,train_mode_mutated_target_legacycfg_rejected=True,score_forgeries_rejected=len(cases),native_targets=target['targets'],target_ids_sha256=target['target_ids_sha256']))
                del model,data,fake;gc.collect()
            spec=importlib.util.spec_from_file_location('new_prepared_moe_worker_builder',builder);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            batch=module.specs();require(len(batch)==3 and [v['arm'] for v in batch]==['dense','wide','moe'] and [v['weight_cap'] for v in batch]==[560*1024**2,640*1024**2,640*1024**2]
                and all(v['config']['training']['max_iters']==4 and 'model' not in v['config'] for v in batch),'Prepared exact three-arm scope differs')
            require(not module.RUN.exists() and not module.PLAN.exists() and sources()==pins,'Prepared builder attempted or sources changed')
        publish('worker-source-review001.json',dict(status='passed_finite_moe_worker_and_native_score_contracts',source=reference(__file__),prepared_builder=reference(builder),
            sources=pins,prerequisites=refs,environment=env,reports=rows,training=False,full_validation_scoring=False,weights_written=0,new_scientific_seed=False))
        print(json.dumps(dict(status='passed_finite_moe_worker_and_native_score_contracts',sources=len(pins),reports=rows)))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if not (E/'STOP-worker-source-review001').exists():publish('STOP-worker-source-review001',dict(error=repr(exc),closed=True,never_retry=True))
        raise
