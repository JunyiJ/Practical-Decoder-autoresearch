"""Finite twenty-update resource pair,only after actual generated-worker acceptance."""
import gc,json,shutil,statistics
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference,environment
from research.gpt2_prefix_data_v1.data import config
from research.gpt2_prefix_worker_v1.execution import sources,checked
from research.gpt2_prefix_worker_v1.validation import verify_artifact
from research.gpt2_prefix_worker_v1.dispatch import launch
from research.final_ln_scale_quality_v3.support import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
E=Path(__file__).resolve().parent;RUN=ROOT/'research_runs/tokenizer-gpt2-prefix-pilot-v1-001';PLAN=E/'pilot001-plan.json'
CAPS=[560*1024**2,160*1024**2];RESERVE=8*1024**3;METADATA=16*1024**2;EXTERNAL=1267904512;ENTRY=1458176;FLOOR=sum(CAPS)+RESERVE+METADATA+EXTERNAL+ENTRY

def specs():
    result=[]
    for i,kind in enumerate(['native','prefix8192']):
        cfg=config(kind,steps=20,seed=600430);cfg.training.eval_seed=600432;directory=RUN/f'evidence/pilots/{i:02d}'
        cfg.training.checkpoint_dir=str(directory/'checkpoints');cfg.training.metrics_file=str(directory/'metrics.jsonl');cfg.training.summary_file=str(directory/'worker.json')
        result.append(dict(stage='pilots',slot=i,arm=kind,config=plain(cfg),weight_cap=CAPS[i]))
    return result

def publish(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def projection(worker,process,parent_guard):
    times=worker['update_seconds'];require(len(times)==20 and worker['step']==20 and worker['ema_snapshots']==8,'Actual full pilot horizon/snapshots required')
    rate=max(statistics.mean(times),statistics.median(times[-10:]),worker['training_seconds']/20)
    fixed=max(0.,process['elapsed_seconds']-worker['training_seconds'])+parent_guard['elapsed_seconds']
    predicted=2*(1200*rate+fixed)
    return dict(full1200_projection_seconds=predicted,maximum_observed_pilot_average_update_seconds=rate,fixed_process_plus_parent_acceptance_seconds=fixed,safety_multiplier=2.,passes_original7200=predicted<7200,
        scope='Conservative planning estimate,not measured full-run speed or guarantee.Original live guards remain mandatory;CPU/MPS full scoring/serialization included as fixed measured overhead.')
def main():
    torch.set_num_threads(4)
    with session_lock(GLOBAL_LOCK.parent) as g:
        idle();env=environment();require(not RUN.exists() and not PLAN.exists(),'New unattempted resource pair only')
        pins=sources();ready=checked(reference(E/'pilot-integration001.json'));require(ready['status']=='passed_actual_worker_pair_and_twenty_update_resource_integration' and ready['sources']==pins and ready['prepared_builder']==reference(__file__),'Actual worker/pilot integration gate changed')
        prerequisite=checked(ready['generated_worker_review']);require(prerequisite['status']=='two_gpt2_prefix_generated_workers_independently_accepted','Complete independently accepted generated pair required')
        for row in prerequisite['rows']:
            worker=checked(row['worker'])
            for n in ['parent','h200']:verify_artifact(worker[n])
        require(shutil.disk_usage(ROOT).free>=FLOOR+ENTRY,'Complete new resource pair retained-output budget plus measured entry margin required')
        plan=dict(kind='gpt2_prefix_twenty_update_resource_pair_v1',root=str(RUN),runner=reference(__file__),sources=pins,environment=env,specs=specs(),seconds=600,compare_cpu=True,
            integration=reference(E/'pilot-integration001.json'),scientific_quality=False,new_scientific_seed=False,fixture_resource_seed=600430,fixture_evaluation_seed=600432,
            full_pair_floor=FLOOR,first_wake=FLOOR+ENTRY,cleanup_target=FLOOR+2*1024**3,weight_caps=CAPS,metadata_cap=METADATA,reserve=RESERVE,external_headroom=EXTERNAL,entry_headroom=ENTRY,
            no_conditional_retirement_or_hardlink_credit=True,no_retirement=True,scoring='RAW/H200 independent complete CPU/MPS common111475byte targets and H200cache/resetB1/8.')
        RUN.mkdir();publish(PLAN,plan)
        try:
            with session_lock(RUN) as r:
                writer=MetadataWriter(RUN,(g,r));rows=[];used=0;projections=[]
                for spec in plan['specs']:
                    require(shutil.disk_usage(ROOT).free>=max(RESERVE,FLOOR-used)+ENTRY,'Complete live remaining resource pair capacity required')
                    row=launch(writer,reference(PLAN),spec,max(RESERVE,FLOOR-used));rows.append(row);worker=checked(row['worker'])
                    for n in ['parent','h200']:verify_artifact(worker[n])
                    used+=sum(worker[n]['size'] for n in ['parent','h200']);projections.append(dict(arm=spec['arm'],**projection(worker,checked(row['process']),checked(row['parent_guard']))))
                    publish(E/f'pilot001-accepted-{spec["slot"]:02d}.json',dict(accepted=reference(RUN/f'evidence/pilots/{spec["slot"]:02d}/accepted.json'),scientific_quality=False));gc.collect();torch.mps.empty_cache()
                require(sources()==pins,'Runtime source changed')
                review=dict(status='two_prefix_resource_pilots_independently_accepted',plan=reference(PLAN),rows=rows,retained_weight_bytes=used,projections=projections,all_original7200_projections_pass=all(p['passes_original7200'] for p in projections),
                    scientific_quality=False,new_scientific_seed=False,holdout_encoded_or_scored=False,next_action='Use actual cost/serialization/acceptance and literature/local evidence to freeze a finite common-byte scientific design with equal bounded LR tuning,fresh native/8K confirmation,independent byte-grouped statistics and COMPLETE retained-output budget. No quality seed/plan/freeze/adoption yet;do not infer quality from pilots or retire weights without exact downstream closure audit.')
                publish(E/'pilot001-review.json',review);publish(RUN/'STOP',dict(status='resource_pair_complete',all_training_commands_closed=True,never_rerun=True));print(json.dumps(dict(status=review['status'],projections=projections,retained_weight_bytes=used)),flush=True)
        except Exception as exc:
            if not(RUN/'STOP').exists():publish(RUN/'STOP',dict(error=repr(exc),closed=True,no_retry=True))
            raise
if __name__=='__main__':main()
