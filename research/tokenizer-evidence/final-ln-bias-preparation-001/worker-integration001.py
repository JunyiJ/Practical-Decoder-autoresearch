"""Read-only final LayerNorm bias adapter audit against both real final-horizon resume payloads."""
import copy,gc,hashlib,importlib.util,json,shutil
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.final_ln_bias_worker_v1 import evidence_contract as c,validation as v,inference as inf,artifact_acceptance as a
from research.final_ln_bias_model_v1.integration import identity,load_model_weights
from research.final_ln_bias_model_v1.runtime.utils.checkpoint import read_checkpoint
from research.final_ln_bias_worker_v1.training import _training_contract
from research.block_norm_v1.fixture_dispatch import read,reference as metadata_reference
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.block_norm_v1.pilot_execution import exclusive
from research.gradient_clip_quality_v2.quality_execution import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=Path(__file__).resolve().parent

def reference(path):
    path=Path(path).resolve();assert path.is_file() and not path.is_symlink() and path.stat().st_size<=512*1024**2
    with path.open('rb') as f:return dict(path=str(path),sha256=hashlib.file_digest(f,'sha256').hexdigest())

def clear():gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
def reject(fn):
    try:fn()
    except ValueError as e:return str(e)
    raise AssertionError('Invalid identity accepted')
def verify_ref(r):
    p=Path(r['path']);assert p.is_file() and not p.is_symlink()
    assert reference(p)==r

torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    idle();plan=read(reference(B/'worker-integration001-plan.json'));assert plan['runner']==reference(__file__)
    c.verify_sources(plan['sources'])
    for r in plan['bound_scripts']:verify_ref(r)
    prior=read(plan['resume_review']);assert prior['status']=='final_ln_bias_both_arms_serialized_resume_final_h200_passed'
    # Import only; no worker, scientific seed, optimizer update or old-fixture rerun.
    spec=importlib.util.spec_from_file_location('final_ln_bias_worker',B/'trainer-worker001.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    assert worker.ARMS==inf.ARMS==('learned','zero')
    bindings={};payloads=[];rows=[]
    try:
        with StrictMpsGuard(B/'worker-integration001-guard.json',c.ROOT):
            for row in prior['arms']:
                arm=row['arm'];root=Path(prior['root'])/arm
                for name,old in row['files'].items():
                    p=root/name;assert p.stat().st_size==old['size'] and reference(p)['sha256']==old['sha256']
                binding={k:reference(root/n) for k,n in [('config','config.json'),('parent','final.pt'),('h200','h200.pt')]};bindings[arm]=binding
                model,data,cfg=v.load(binding,'mps');raw=read_checkpoint(binding['parent']['path']);avg=read_checkpoint(binding['h200']['path'])
                expected=.00006;assert identity(cfg)['decay']==expected
                assert identity(cfg)['grad_clip']==1. and identity(cfg)['adamw']['beta2']==.999 and identity(cfg)['readout']['bias_kind']==dict(learned='learned',zero='fixed_zero')[arm] and identity(cfg)['schedule']['start_step']==2
                a.equal(raw['training_state']['contract'],_training_contract(cfg,'mps',data),'Real training contract differs')
                clipping=a.audit_clipping_history(cfg,raw['training_state'].get('clipping_observations'))
                history_rejections={}
                for mode in ('missing','threshold','sequence','coefficient','nonfinite','tensor_count'):
                    bad=copy.deepcopy(raw['training_state']['clipping_observations'])
                    if mode=='missing':bad.pop()
                    elif mode=='threshold':bad[0]['threshold']=.5 if cfg.training.grad_clip==1. else 1.
                    elif mode=='sequence':bad[0]['step']=2
                    elif mode=='coefficient':bad[0]['coefficient']=0.
                    elif mode=='tensor_count':bad[0]['gradient_tensors']=87 if arm=='learned' else 88
                    else:bad[0]['pre_norm']=float('nan')
                    history_rejections[mode]=reject(lambda:a.audit_clipping_history(cfg,bad))
                load_model_weights(model,raw,cfg,for_resume=True);load_model_weights(model,avg,cfg)
                assert raw['optimizer_state']['independent_decay']==expected and raw['step']==avg['step']==4
                ema=raw['training_state']['weight_averaging_state'];assert len(ema['parameter_names'])==(88 if arm=='learned' else 87) and ema['snapshots']==3
                assert raw['model_state'].keys()==avg['model_state'].keys()
                for name,tensor in avg['model_state'].items():
                    source=ema['averages']['200'][name] if name in ema['parameter_names'] else raw['model_state'][name]
                    assert torch.equal(tensor,source) and torch.isfinite(tensor).all()
                report=dict(binding=copy.deepcopy(binding),**v.score(model,data));v.validate_report(report,binding,data)
                forged=copy.deepcopy(report);forged['target']['targets']+=1
                coverage_rejection=reject(lambda:v.validate_report(forged,binding,data))
                foreign=copy.deepcopy(cfg);foreign.model.final_ln_bias_kind='fixed_zero' if arm=='learned' else 'learned'
                foreign_rejection=reject(lambda:load_model_weights(model,raw,foreign,for_resume=True))
                optimizer_groups=a.audit_optimizer_groups(cfg,raw['optimizer_state'])
                wrong_groups=copy.deepcopy(raw['optimizer_state'])
                wrong_groups['optimizers'][1]['param_groups'][0]['parameter_names']=['transformer.ln_f.bias'] if arm=='zero' else ['forged']
                if arm=='learned':
                    wrong_groups['optimizers'][1]['param_groups']=[g for g in wrong_groups['optimizers'][1]['param_groups'] if 'transformer.ln_f.bias' not in g.get('parameter_names',[])]
                bias_membership_rejection=reject(lambda:a.audit_optimizer_groups(cfg,wrong_groups))
                del wrong_groups
                changed=copy.deepcopy(raw['optimizer_state']);changed['optimizers'][1]['param_groups'][0]['betas']=(.9,.95)
                foreign_group_rejection=reject(lambda:a.audit_optimizer_groups(cfg,changed))
                bad_raw=dict(raw,optimizer_state=changed)
                foreign_resume_group_rejection=reject(lambda:load_model_weights(model,bad_raw,cfg,for_resume=True))
                del changed,bad_raw
                del model,data,raw,avg;clear()
                cpu=v.evaluate(binding,'cpu');assert cpu['target']==report['target'] and abs(cpu['ce']-report['ce'])<1e-5
                rows.append(dict(arm=arm,bias_membership_rejection=bias_membership_rejection,optimizer_groups=optimizer_groups,foreign_group_rejection=foreign_group_rejection,foreign_resume_group_rejection=foreign_resume_group_rejection,clipping=clipping,history_rejections=history_rejections,binding=binding,validation=report,cpu_mps_ce_abs=abs(cpu['ce']-report['ce']),ema_parameters=88 if arm=='learned' else 87,ema_snapshots=3,foreign_rejection=foreign_rejection,coverage_rejection=coverage_rejection))
                for kind in ('parent','h200'):
                    p=Path(binding[kind]['path']);payloads.append(dict(**binding[kind],size=p.stat().st_size))
                clear()
            measured=inf.measure_pair(bindings,0);review=inf.review(measured,bindings,0)
            exclusive(B/'worker-integration001-inference.json',measured)
            for row in rows:
                for r in row['binding'].values():verify_ref(r)
            c.verify_sources(plan['sources'])
            for r in plan['bound_scripts']:verify_ref(r)
            result=dict(status='final_ln_bias_actual_resume_payload_adapter_integration_passed',plan=reference(B/'worker-integration001-plan.json'),sources=plan['sources'],rows=rows,inference=reference(B/'worker-integration001-inference.json'),inference_review=review,
                worker_training_launched=False,scientific_training=False,holdout_read=False,eligible_successful_numerical_weights=payloads,
                scope='Read-only adapter integration of real four-update resume endpoints; no real run_training output acceptance claimed yet.',
                remaining='Exact audited obsolete resume-weight retirement if needed, then fresh real run_training/independent parent full validation and trained inference.')
        result['guard']=reference(B/'worker-integration001-guard.json');assert read(result['guard'])['status']=='finished'
        exclusive(B/'worker-integration001.json',result)
        exclusive(B/'resume-dependencies-closed001.json',dict(status='successful_resume_tensor_dependencies_closed',integration=reference(B/'worker-integration001.json'),source=reference(__file__),eligible_successful_numerical_weights=payloads,
            next_worker='Fresh generated input and initialization. Reads model-resume001 JSON success/config evidence; never reads the retired tensors. No scientific/selected/fresh weights eligible.'))
        print(json.dumps(dict(status=result['status'],arms=[r['arm'] for r in rows],eligible_bytes=sum(r['size'] for r in payloads))),flush=True)
    except Exception as exc:
        exclusive(B/'worker-integration001-failure.json',dict(error=repr(exc),runner=reference(__file__)));raise
