"""New finite three-arm generated-worker panel; no implicit retry or quality."""
import gc
import json
import shutil
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_mla_gate_resume_v1.data import config
from research.tokenizer_mla_gate_resume_v1.contract import runtime
from research.tokenizer_mla_gate_worker_v1.execution import sources,checked,idle
from research.tokenizer_mla_gate_worker_v1.dispatch import launch
from research.tokenizer_mla_gate_worker_v1.validation import verify_artifact
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import snapshot

E=Path(__file__).resolve().parent
RUN=ROOT/'research_runs/tokenizer-mla-gate-worker-v1-001'
PLAN=E/'gate-worker001-plan.json'
CAPS=[560*1024**2,560*1024**2,640*1024**2]
METADATA=16*1024**2;RESERVE=8*1024**3;EXTERNAL=1267904512;ENTRY=1458176
FLOOR=sum(CAPS)+METADATA+RESERVE+EXTERNAL+ENTRY


def specs():
    rows=[]
    for i,arm in enumerate(['none','static_half','sigmoid_elementwise_zero']):
        cfg=config(arm,steps=4,seed=600533,eval_seed=600534);cfg.training.eval_every=1;cfg.training.eval_iters=1
        path=RUN/f'evidence/pilots/{i:02d}'
        cfg.training.checkpoint_dir=str(path/'checkpoints');cfg.training.metrics_file=str(path/'metrics.jsonl');cfg.training.summary_file=str(path/'worker.json')
        rows.append(dict(stage='pilots',slot=i,arm=arm,config=plain(cfg),weight_cap=CAPS[i]))
    return rows


def publish(path,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<1024**2,'External receipt cap1MiB')
    with path.open('xb') as f:f.write(raw)


def metadata_bound():
    state=snapshot(RUN)
    used=sum(v['size'] for v in state['files'].values() if not v['group'].endswith('/weights'))
    external=sum(p.stat().st_size for p in E.rglob('*') if p.is_file())
    require(used+external<=METADATA,'Complete root/external metadata cap exceeded')


def main():
    torch.set_num_threads(4)
    with session_lock(GLOBAL_LOCK.parent) as global_fd:
        idle();env=runtime();require(not RUN.exists() and not PLAN.exists(),'New unattempted root/plan required')
        pins=sources();gate=checked(reference(E/'gate-worker-source001.json'))
        require(gate['status']=='passed_finite_gate_worker_and_native_score_contracts' and gate['sources']==pins and gate['prepared_builder']==reference(__file__),'Actual source/builder gate changed')
        refs={n:reference(E/n) for n in ['gate-resume002/result.json','gate-resume002/guard.json','gate-resume002/STOP','resume-review001.json']}
        require(refs==gate['prerequisites'],'Actual transport prerequisite references changed')
        roles=checked(reference(E/'fixture-role-census004.json'));require(roles['no_prior_matches'] and not roles['scientific_seeds'] and roles['roles']['worker_model_and_private_train']==600533 and roles['roles']['worker_private_evaluation']==600534,'Explicit unused worker fixture-role audit required')
        design=checked(reference(E/'quality-design001.json'));future=design['complete_retained_budget']['target']
        require(shutil.disk_usage(ROOT).free>=max(FLOOR+2*1024**3,future),'Complete future retained target plus worker requirement required')
        plan=dict(kind='tokenizer_mla_gate_generated_three_arm_prerequisite_v1',root=str(RUN),runner=reference(__file__),sources=pins,environment=env,specs=specs(),seconds=600,compare_cpu=True,
                  prerequisites=refs,source_review=reference(E/'gate-worker-source001.json'),roles=reference(E/'fixture-role-census004.json'),design=reference(E/'quality-design001.json'),complete_prospective_target=future,scientific_quality=False,new_scientific_seed=False,
                  fixture_model_and_train_seed=600533,fixture_evaluation_seed=600534,full_floor=FLOOR,first_wake=FLOOR+ENTRY,cleanup_target=FLOOR+2*1024**3,
                  weight_caps=CAPS,metadata_cap=METADATA,reserve=RESERVE,external_headroom=EXTERNAL,entry_headroom=ENTRY,
                  capacity_policy='Only actual independently accepted retained current-root endpoint bytes credited once before next unstarted slot;no future deletion/pilot/hardlink credit.Atomic tmp within endpoint cap.',
                  scoring='Complete34278native targets/retainedtail,RAWandH200 onCPUandMPS,H200cache/resetB1/8bothdevices;own gate policy only.',no_retirement=True)
        forecast=dict(parent_rss_bytes=4116332544+16*1024**2,mps_driver_bytes=2719825920+256*1024**2,per_worker_and_acceptance_seconds=300)
        require(forecast['parent_rss_bytes']<4*1024**3 and forecast['mps_driver_bytes']<6*1024**3 and forecast['per_worker_and_acceptance_seconds']<600,'Complete worker/acceptance forecast must fit original guards')
        plan['forecast']=forecast;RUN.mkdir();publish(PLAN,plan)
        try:
            with session_lock(RUN) as run_fd:
                writer=MetadataWriter(RUN,(global_fd,run_fd));rows=[];used=0;hashes=set()
                for spec in plan['specs']:
                    require(shutil.disk_usage(ROOT).free>=max(RESERVE,FLOOR-used)+ENTRY,'Complete actual remaining output requirement failed')
                    row=launch(writer,reference(PLAN),spec,max(RESERVE,FLOOR-used));rows.append(row);worker=checked(row['worker'])
                    for name in ['parent','h200']:
                        verify_artifact(worker[name]);require(worker[name]['sha256'] not in hashes,'Duplicate accepted checkpoint bytes');hashes.add(worker[name]['sha256'])
                    used+=sum(worker[name]['size'] for name in ['parent','h200'])
                    publish(E/f'gate-worker001-accepted-{spec["slot"]:02d}.json',dict(accepted=reference(RUN/f'evidence/pilots/{spec["slot"]:02d}/accepted.json'),scientific_quality=False))
                    metadata_bound();gc.collect();torch.mps.empty_cache()
                require(sources()==pins,'Frozen sources changed')
                review=dict(status='all_three_gate_generated_workers_independently_accepted',plan=reference(PLAN),rows=rows,retained_weight_bytes=used,
                            scientific_quality=False,new_scientific_seed=False,holdout_encoded_or_scored=False,
                            next_action='Actual20-update three-arm resource pilots with complete LIVE budget/source/prerequisite gates;then finite quality contracts/full fresh-panel budget and unused scientific-seed census. No retirement without NEW exact downstream closure.')
                publish(E/'gate-worker001-review.json',review);writer.publish_json('STOP',dict(status='three_arm_prerequisite_complete',review=reference(E/'gate-worker001-review.json'),never_rerun=True,all_training_acceptance_commands_closed=True));metadata_bound()
                print(json.dumps(dict(status=review['status'],retained_weight_bytes=used)),flush=True)
        except BaseException as exc:
            if not (RUN/'STOP').exists():publish(RUN/'STOP',dict(status='CLOSED_FAILED_GENERATED_WORKER_PANEL',error=repr(exc),never_retry=True))
            raise


if __name__=='__main__':main()
