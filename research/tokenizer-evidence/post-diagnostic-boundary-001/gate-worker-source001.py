"""Finite own gate source/config/report contracts; no worker,full scoring or quality."""
import copy,gc,importlib.util,json,math,os
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_mla_gate_resume_v1.data import config,load,architecture
from research.tokenizer_mla_gate_resume_v1.contract import runtime
from research.tokenizer_mla_gate_model_v1.model import GPT
from research.tokenizer_mla_gate_worker_v1.execution import sources,idle
from research.tokenizer_mla_gate_worker_v1.training import run
from research.tokenizer_mla_gate_worker_v1.scoring import target_contract,validate_report
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock

E=Path(__file__).resolve().parent
def reject(fn):
    try:fn()
    except (ValueError,TypeError,RuntimeError):return
    raise AssertionError('Disallowed gate config/report accepted')
def publish(p,v):
    raw=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<=2*1024**2,'Source proof metadata file cap')
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def main():
    torch.set_num_threads(4)
    with session_lock(GLOBAL_LOCK.parent):
        idle();env=runtime();require(not (E/'gate-worker-source001-plan.json').exists(),'Fresh source proof only;no closed retry')
        pins=sources();builder=E/'gate-worker001-build-run001.py';roles=json.loads((E/'fixture-role-census004.json').read_text())
        require(roles['no_prior_matches'] and not roles['scientific_seeds'] and roles['roles']['source_generated_initializer']==600530 and roles['roles']['source_config_private_train']==600531 and roles['roles']['source_config_private_evaluation']==600532,'Audited new source fixture roles required')
        refs={n:reference(E/n) for n in ['gate-resume002/result.json','gate-resume002/guard.json','gate-resume002/STOP','resume-review001.json']}
        plan=dict(source=reference(__file__),prepared_builder=reference(builder),sources=pins,prerequisites=refs,roles=reference(E/'fixture-role-census004.json'),training=False,full_validation_scoring=False,scientific_seed=False)
        publish(E/'gate-worker-source001-plan.json',plan)
        try:
            before=torch.get_rng_state().clone();rows=[]
            with CpuPrerequisiteGuard(E/'gate-worker-source001-guard.json',ROOT,timeout_seconds=120),torch.random.fork_rng(devices=[]):
                for arm in ('none','static_half','sigmoid_elementwise_zero'):
                    for steps in (4,20):
                        for lr in (.000625,.00125,.0025):require(architecture(config(arm,steps,600531,600532,lr))['arm']==arm,'Finite own config routing differs')
                    reject(lambda:run(config(arm,1200,600531,600532),None,None));reject(lambda:config(arm,4,600531,600532,.01))
                    cfg=config(arm,4,600531,600532);data=load(cfg,'cpu');torch.default_generator.manual_seed(600530);model=GPT(plain(cfg.gate_model))
                    reject(lambda:target_contract(model,data,cfg));model.eval();target=target_contract(model,data,cfg)
                    require(target['targets']==34278 and target['target_bytes']==111536 and target['target_ids_sha256']=='dad00eaab7c2076736432b83eb9f1cc404c87aa5e4820b5f7abebf384c7fcdaf','Full original native target contract required')
                    fake=copy.copy(data);fake.val_data=data.val_data.clone();fake.val_data[0]+=1;reject(lambda:target_contract(model,fake,cfg))
                    foreign=copy.deepcopy(cfg);foreign['model']=foreign.backbone_template;reject(lambda:architecture(foreign))
                    foreign=copy.deepcopy(cfg);foreign['moe_model']=foreign.pop('gate_model');reject(lambda:architecture(foreign))
                    values=[.25+(i%19)*.07 for i in range(34278)];total=math.fsum(values)
                    report=dict(target=target,window_loss_sums=[math.fsum(values[i:i+128]) for i in range(0,34278,128)],block_loss_sums=[math.fsum(values[i:i+2048]) for i in range(0,34278,2048)],total_loss=total,ce=total/34278,bits_per_byte=total/math.log(2)/target['target_bytes'],loss_dtype='FP32_cross_entropy_then_CPU_FP64_aggregation',holdout_scored=False,aux_loss_included=False)
                    validate_report(report,target);cases=[]
                    for mode in ('partial','target','ce','bytes','nan','block','holdout','aux'):
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
                    rows.append(dict(arm=arm,finite_prerequisite_configs=6,full1200_rejected_before_data_model=True,legacy_and_moe_cfg_rejected=True,train_mode_and_mutated_target_rejected=True,score_forgeries_rejected=8,native_targets=34278,target_ids_sha256=target['target_ids_sha256'],target_bytes=111536))
                    del model,data,fake;gc.collect()
                spec=importlib.util.spec_from_file_location('new_prepared_gate_worker_builder',builder);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);batch=module.specs()
                require(len(batch)==3 and [v['arm'] for v in batch]==['none','static_half','sigmoid_elementwise_zero'] and [v['weight_cap'] for v in batch]==[560*1024**2,560*1024**2,640*1024**2] and all(v['config']['training']['max_iters']==4 and 'model' not in v['config'] and 'gate_model' in v['config'] for v in batch),'Exact finite prepared worker scope required')
                require(not module.RUN.exists() and not module.PLAN.exists() and sources()==pins,'Prepared worker attempted or sources changed')
            require(torch.equal(before,torch.get_rng_state()),'Source proof changed global CPU RNG')
            out=dict(status='passed_finite_gate_worker_and_native_score_contracts',source=reference(__file__),prepared_builder=reference(builder),sources=pins,prerequisites=refs,roles=reference(E/'fixture-role-census004.json'),environment=env,reports=rows,global_cpu_rng_preserved=True,training=False,full_validation_scoring=False,weights_written=0,new_scientific_seed=False)
            publish(E/'gate-worker-source001.json',out);publish(E/'STOP-gate-worker-source001',dict(status='complete_gate_worker_source_contracts',never_rerun=True));print(json.dumps(dict(status=out['status'],sources=len(pins))),flush=True)
        except BaseException as exc:
            if not (E/'STOP-gate-worker-source001').exists():publish(E/'STOP-gate-worker-source001',dict(status='closed_failed',error=repr(exc),no_retry=True))
            raise

if __name__=='__main__':main()
