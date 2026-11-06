"""New three-arm20-update resource panel after actual generated-worker acceptance."""
import gc
import json
import math
import shutil
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_resume_v1.data import config
from research.tokenizer_moe_resume_v1.contract import runtime
from research.tokenizer_moe_worker_v1.execution import sources,checked,idle
from research.tokenizer_moe_worker_v1.dispatch import launch
from research.tokenizer_moe_worker_v1.validation import verify_artifact
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import snapshot

E=Path(__file__).resolve().parent
WORKER=ROOT/'research_runs/tokenizer-moe-worker-v1-001'
RUN=ROOT/'research_runs/tokenizer-moe-pilot-v1-001'
PLAN=E/'pilot001-plan.json'
CAPS=[560*1024**2,640*1024**2,640*1024**2]
METADATA=16*1024**2;RESERVE=8*1024**3;EXTERNAL=1267904512;ENTRY=1458176
FLOOR=sum(CAPS)+METADATA+RESERVE+EXTERNAL+ENTRY


def publish(path,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<1024**2,'External receipt cap1MiB')
    with path.open('xb') as f:f.write(raw)


def specs():
    rows=[]
    for i,arm in enumerate(['dense','wide','moe']):
        cfg=config(arm,steps=20,seed=600470,eval_seed=600471)
        cfg.training.eval_every=5;cfg.training.eval_iters=2
        path=RUN/f'evidence/pilots/{i:02d}'
        cfg.training.checkpoint_dir=str(path/'checkpoints');cfg.training.metrics_file=str(path/'metrics.jsonl');cfg.training.summary_file=str(path/'worker.json')
        rows.append(dict(stage='pilots',slot=i,arm=arm,config=plain(cfg),weight_cap=CAPS[i]))
    return rows


def worker_prerequisites(pins):
    stop=checked(reference(WORKER/'STOP'));review=checked(reference(E/'worker001-review.json'));plan=checked(review['plan'])
    require(stop['status']=='three_arm_prerequisite_complete' and stop['review']==reference(E/'worker001-review.json')
        and review['status']=='all_three_moe_generated_workers_independently_accepted' and len(review['rows'])==3 and plan['sources']==pins,'Actual complete three-arm worker prerequisite required')
    rows=[];retained=0
    for i,arm in enumerate(['dense','wide','moe']):
        row=checked(reference(WORKER/f'evidence/pilots/{i:02d}/accepted.json'))
        require(row==review['rows'][i] and row['arm']==arm,'Original accepted worker ancestry differs')
        result=row['acceptance'];require(result['status']=='accepted' and result['step']==4 and result['targets']==4096 and result['ema_snapshots']==3
            and result['parameter_tensors']==(128 if arm=='moe' else 88) and result['complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked'] and result['holdout_scored'] is False,'Complete optimizer/EMA/routing worker acceptance required')
        for name in ['worker_guard','parent_guard']:
            guard=checked(row[name]);require(guard['status']=='finished' and guard['reason'] is None and guard['limits']==dict(rss_limit_bytes=4*1024**3,reserve_bytes=RESERVE,timeout_seconds=600,mps_driver_limit_bytes=6*1024**3),'Original worker/parent guard changed')
        process=checked(row['process']);require(process['status']=='completed' and process['returncode']==0,'Original child process incomplete')
        for endpoint in ['raw','h200']:
            report=result['full_validation'][endpoint]
            require(report['cpu_mps_ce_difference']<=3e-5 and report['cpu']['target']['targets']==report['mps']['target']['targets']==34278
                and report['cpu']['aux_loss_included'] is report['mps']['aux_loss_included'] is False,'Full CPU/MPS CE-only worker validation required')
        for device in ['cpu','mps']:require([v['batch'] for v in result['cache'][device]]==[1,8] and all(v['cache_bytes_per_item']==196608 for v in result['cache'][device]),'Actual both-batch cache/reset required')
        summary=checked(row['worker'])
        for name in ['parent','h200']:verify_artifact(summary[name]);retained+=summary[name]['size']
        rows.append(reference(WORKER/f'evidence/pilots/{i:02d}/accepted.json'))
    require(retained==review['retained_weight_bytes'],'Actual retained worker-byte total differs')
    return dict(worker_stop=reference(WORKER/'STOP'),worker_review=reference(E/'worker001-review.json'),worker_plan=review['plan'],worker_acceptances=rows,worker_retained_bytes=retained)


def metadata_bound():
    state=snapshot(RUN);used=sum(v['size'] for v in state['files'].values() if not v['group'].endswith('/weights'))
    require(used+sum(p.stat().st_size for p in E.rglob('*') if p.is_file())<=METADATA,'Complete pilot metadata cap exceeded')


def main():
    torch.set_num_threads(4)
    with session_lock(GLOBAL_LOCK.parent) as global_fd:
        idle();env=runtime();require(not RUN.exists() and not PLAN.exists(),'New unattempted resource panel only')
        pins=sources();prereqs=worker_prerequisites(pins)
        require(shutil.disk_usage(ROOT).free>=FLOOR+2*1024**3,'Complete pilot requirement plus2GiB required')
        plan=dict(kind='tokenizer_moe_three_arm_resource_pilot_v1',root=str(RUN),runner=reference(__file__),sources=pins,environment=env,specs=specs(),seconds=600,compare_cpu=True,
                  prerequisites=prereqs,scientific_quality=False,new_scientific_seed=False,fixture_train_seed=600470,fixture_evaluation_seed=600471,
                  full_floor=FLOOR,first_wake=FLOOR+ENTRY,cleanup_target=FLOOR+2*1024**3,weight_caps=CAPS,metadata_cap=METADATA,reserve=RESERVE,external_headroom=EXTERNAL,entry_headroom=ENTRY,
                  capacity_policy='Keepallworkerandpilotweights;creditonlyactualacceptedretainedCURRENTPILOTbytesonce.No oldworker/pilot/hardlink/futuredeletioncredit.Atomic temporary within cap.',
                  projection='Perarm2*((pilot_training_seconds/20)*1200+max(0,child_process_seconds-pilot_training_seconds)+parent_seconds).Descriptive conservative estimate,notfullrunstability/qualityguarantee.',
                  scoring='Complete34278originalnativeCPU AND MPS RAW/H200validation,CEexcludesaux,H200cache/resetB1/8bothdevices.',no_retirement=True)
        publish(E/'pilot001-preflight001.json',dict(status='actual_three_arm_workers_dependencies_and_new_pilot_scope_verified',prerequisites=prereqs,source_count=len(pins),runner=reference(__file__),floor=FLOOR,free_bytes=shutil.disk_usage(ROOT).free,new_training=False))
        RUN.mkdir();publish(PLAN,plan)
        try:
            with session_lock(RUN) as run_fd:
                writer=MetadataWriter(RUN,(global_fd,run_fd));rows=[];used=0;projections=[]
                for spec in plan['specs']:
                    require(shutil.disk_usage(ROOT).free>=max(RESERVE,FLOOR-used)+ENTRY,'Complete actual remaining pilot output requirement failed')
                    row=launch(writer,reference(PLAN),spec,max(RESERVE,FLOOR-used));rows.append(row);summary=checked(row['worker'])
                    for name in ['parent','h200']:verify_artifact(summary[name])
                    used+=sum(summary[name]['size'] for name in ['parent','h200'])
                    child=checked(row['process']);parent=checked(row['parent_guard'])
                    projected=2*(summary['training_seconds']/20*1200+max(0.,child['elapsed_seconds']-summary['training_seconds'])+parent['elapsed_seconds'])
                    require(math.isfinite(projected) and projected<7200,'Full-horizon resource projection does not pass original7200s guard')
                    projections.append(dict(arm=spec['arm'],pilot_training_seconds=summary['training_seconds'],child_seconds=child['elapsed_seconds'],parent_seconds=parent['elapsed_seconds'],conservative1200_seconds=projected))
                    publish(E/f'pilot001-accepted-{spec["slot"]:02d}.json',dict(accepted=reference(RUN/f'evidence/pilots/{spec["slot"]:02d}/accepted.json'),scientific_quality=False))
                    metadata_bound();gc.collect();torch.mps.empty_cache()
                require(sources()==pins and reference(__file__)==plan['runner'],'Pilot source/runner changed')
                review=dict(status='all_three_moe_resource_pilots_independently_accepted',plan=reference(PLAN),rows=rows,retained_weight_bytes=used,projections=projections,
                            scientific_quality=False,new_scientific_seed=False,holdout_encoded_or_scored=False,
                            next_action='Finite full-horizon source/serialization/statistical/selection contracts,actual dependency review,complete ALL-output quality budget and unused scientific-seed census before any quality freeze.No retirement without NEW exact audit.')
                publish(E/'pilot001-review.json',review);writer.publish_json('STOP',dict(status='three_arm_resource_pilot_complete',review=reference(E/'pilot001-review.json'),never_rerun=True,all_training_acceptance_commands_closed=True));metadata_bound()
                print(json.dumps(dict(status=review['status'],retained_weight_bytes=used,projections=projections)),flush=True)
        except BaseException as exc:
            if not (RUN/'STOP').exists():publish(RUN/'STOP',dict(status='CLOSED_FAILED_RESOURCE_PILOT_PANEL',error=repr(exc),never_retry=True))
            raise


if __name__=='__main__':main()
