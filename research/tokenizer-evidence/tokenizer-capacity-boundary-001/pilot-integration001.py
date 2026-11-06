"""Bind the actual completed four-update pair to a new finite resource pilot."""
import copy,importlib.util,json,math,shutil
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,canonical
from research.body_allocation_model_v1.evidence import reference,environment
from research.gpt2_prefix_worker_v1.execution import sources,checked
from research.gpt2_prefix_worker_v1.validation import verify_artifact
from research.gpt2_prefix_data_v1.data import architecture
from omegaconf import OmegaConf
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent

def publish(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def reject(fn):
    try:fn()
    except (ValueError,RuntimeError):return
    raise AssertionError('Invalid projection scope accepted')
with session_lock(GLOBAL_LOCK.parent):
    torch.set_num_threads(4);idle();env=environment();require(not(E/'pilot-integration001-plan.json').exists(),'New integration proof only')
    pins=sources();builder=E/'pilot001-build-run001.py';worker_ref=reference(E/'worker001-review.json');review=checked(worker_ref);plan=checked(review['plan'])
    require(review['status']=='two_gpt2_prefix_generated_workers_independently_accepted' and plan['sources']==pins and len(review['rows'])==2,'Actual complete generated pair required')
    stop=reference(Path(plan['root'])/'STOP');require(checked(stop)['status']=='prerequisite_pair_complete','Actual terminal prerequisite STOP required')
    publish(E/'pilot-integration001-plan.json',dict(source=reference(__file__),prepared_builder=reference(builder),sources=pins,generated_worker_review=worker_ref,original_terminal_stop=stop,no_training_or_rescoring=True))
    try:
        with CpuPrerequisiteGuard(E/'pilot-integration001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
            reports=[];refs=[]
            for slot,row in enumerate(review['rows']):
                require(row['slot']==slot and row['arm']==['native','prefix8192'][slot],'Exact original pair scope required')
                actual=reference(Path(plan['root'])/f'evidence/pilots/{slot:02d}/accepted.json');require(checked(actual)==row,'Original accepted anchor changed');refs.append(actual)
                for n in ['worker_guard','parent_guard']:g=checked(row[n]);require(g['status']=='finished' and g['reason'] is None,'Actual completed original guard required')
                process=checked(row['process']);require(process['status']=='completed' and process['returncode']==0,'Actual clean worker process required')
                worker=checked(row['worker']);result=row['acceptance'];require(worker['step']==result['step']==4 and worker['targets']==result['targets']==4096 and worker['ema_snapshots']==result['ema_snapshots']==3 and result['parameter_tensors']==88 and result['complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked'],'Actual state acceptance incomplete')
                require(worker['architecture']==architecture(OmegaConf.create(worker['config'])) and result['architecture']==worker['architecture'],'Actual genuine architecture differs')
                for n in ['parent','h200']:verify_artifact(worker[n])
                for weight in ['raw','h200']:
                    scores=result['full_validation'][weight];require(scores['cpu_mps_mean_token_loss_difference']<=3e-5 and scores['cpu_mps_bpb_difference']<=3e-5,'Actual full CPU/MPS score gate incomplete')
                    for device in ['cpu','mps']:
                        score=scores[device];require(score['target_tokens']==[34257,39963][slot] and score['target_bytes']==111475 and score['target_sha256']=='ad7da8cdcacc24b5312461c23f2d00aead59d544fcc2fad4fb2de40da4964fe8' and math.isfinite(score['bits_per_byte']) and not score['holdout_scored'],'Actual common byte coverage differs')
                for device in ['cpu','mps']:
                    require([r['batch'] for r in result['cache'][device]]==[1,8] and all(r['cache_bytes_per_item']==196608 for r in result['cache'][device]),'Actual full/cache/reset acceptance incomplete')
                reports.append(dict(slot=slot,arm=row['arm'],accepted=actual,complete_actual_raw_h200_cpu_mps_common_byte_checks=True,original_weight_bytes=sum(worker[n]['size'] for n in ['parent','h200'])))
            spec=importlib.util.spec_from_file_location('prepared_prefix_pilot_builder',builder);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);new=module.specs()
            require(not module.RUN.exists() and not module.PLAN.exists() and len(new)==2,'New resource pair only')
            for slot,s in enumerate(new):
                cfg=OmegaConf.create(s['config']);require(s['arm']==['native','prefix8192'][slot] and s['slot']==slot and cfg.training.max_iters==20 and cfg.training.seed==600430 and cfg.training.eval_seed==600432 and cfg.training.weight_averaging.start_step==13 and cfg.training.weight_averaging.interval==1 and cfg.training.lr_schedule.start_step==13 and 'model' not in cfg,'Exact genuine resource config required');architecture(cfg)
            synthetic=dict(update_seconds=[.2]*20,step=20,ema_snapshots=8,training_seconds=5.)
            p=module.projection(synthetic,dict(elapsed_seconds=8.),dict(elapsed_seconds=7.));require(abs(p['full1200_projection_seconds']-620.)<1e-12,'Independent projection arithmetic differs')
            reject(lambda:module.projection(dict(synthetic,update_seconds=[.2]*19),dict(elapsed_seconds=8.),dict(elapsed_seconds=7.)))
            require(sources()==pins,'Proven sources changed')
            capacity=dict(full_pair_floor=module.FLOOR,first_wake=module.FLOOR+module.ENTRY,cleanup_target=module.FLOOR+2*1024**3,free_bytes=shutil.disk_usage(ROOT).free,no_previous_pair_or_future_retirement_credit=True)
        publish(E/'pilot-integration001.json',dict(status='passed_actual_worker_pair_and_twenty_update_resource_integration',source=reference(__file__),sources=pins,environment=env,total_source_bindings=993,prepared_builder=reference(builder),generated_worker_review=worker_ref,original_terminal_stop=stop,accepted_anchor_refs=refs,reports=reports,capacity=capacity,projection_arithmetic_oracle_passed=True,
            training=False,model_rescoring=False,new_scientific_seed=False,holdout_encoded_or_scored=False,weight_copies=0,next_action='Execute reviewed previously unattempted twenty-update resource builder once after actual idle/source/proof/full-pair capacity checks.'))
        print(json.dumps(dict(status='passed_actual_worker_pair_and_twenty_update_resource_integration',capacity=capacity)),flush=True)
    except Exception as exc:publish(E/'STOP-pilot-integration001',dict(error=repr(exc),closed=True,no_retry=True));raise
