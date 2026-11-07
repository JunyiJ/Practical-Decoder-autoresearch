"""Distinct twenty-update readout resource pilots using the immutable worker."""
import gc
import json
import shutil
import hashlib
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain,fingerprint
from research.body_allocation_model_v1.evidence import reference
from research.readout_mixture_resume_v1.data import config
from research.readout_mixture_resume_v1.contract import runtime
from research.readout_mixture_worker_v1.execution import sources,checked,idle
from research.readout_mixture_worker_v1.dispatch import launch
from research.readout_mixture_worker_v1.validation import verify_artifact
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import snapshot

E=Path(__file__).resolve().parent
RUN=ROOT/'research_runs/tokenizer-readout-mixture-pilot-v1-001'
PLAN=E/'readout-pilot001-plan.json'
CAPS=[560*1024**2,640*1024**2,640*1024**2]
METADATA=64*1024**2;RESERVE=8*1024**3;EXTERNAL=2*1024**3;ENTRY=2*1024**2
CHAIN_FLOOR=26845642752;CHAIN_TARGET=28993126400
FLOOR=sum(CAPS)+METADATA+RESERVE+EXTERNAL+ENTRY


def specs(pins,execution):
    rows=[]
    roles=json.loads((E/'readout-pilot-seed-census002/result.json').read_text())['roles']
    for i,arm in enumerate(['native','context_mix2','probability_mix2']):
        cfg=config(arm,20,roles['backbone_initializer'],roles['private_train_and_global_rng'],roles['private_evaluation']);cfg.training.eval_every=5;cfg.training.eval_iters=1
        cfg.readout_execution=dict(kind='readout_prerequisite_execution_v1',sources_sha256=fingerprint(pins),runner_sha256=reference(__file__)['sha256'])
        cfg.readout_pilot_execution=dict(kind='readout_twenty_update_pilot_v1',execution_sources_sha256=fingerprint(execution),census_result_sha256=reference(E/'readout-pilot-seed-census002/result.json')['sha256'])
        path=RUN/f'evidence/pilots/{i:02d}'
        cfg.training.checkpoint_dir=str(path/'checkpoints');cfg.training.metrics_file=str(path/'metrics.jsonl');cfg.training.summary_file=str(path/'worker.json')
        rows.append(dict(stage='pilots',slot=i,arm=arm,config=plain(cfg),weight_cap=CAPS[i]))
    return rows


def execution_sources():
    pins=dict(checked(reference(E/'readout-pilot-seed-census002/plan.json'))['sources'])
    require(len(pins)==1111,'1109 predecessors plus closed failed and distinct new census required')
    pins[str(Path(__file__).relative_to(ROOT))]=reference(__file__)['sha256']
    require(len(pins)==1112,'Exact distinct pilot execution inventory required')
    for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Pilot execution source changed: '+p)
    return pins


def prior_weights():
    closure=checked(reference(E/'readout-worker001-closure001.json'));rows=[]
    require(closure['status']=='complete_three_arm_four_update_prerequisite_closure' and len(closure['weights'])==6,'All original four-update prerequisites required')
    for w in closure['weights']:
        verify_artifact(dict(path=w['path'],size=w['bytes'],sha256=w['sha256']))
        st=Path(w['path']).stat();rows.append(dict(w,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns))
    require(sum(w['bytes'] for w in rows)==1681766784,'Exact accepted current-chain prior credit required')
    return rows


def check_prior(rows):
    for w in rows:
        p=Path(w['path']);st=p.stat()
        require(not p.is_symlink() and st.st_nlink==1 and (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)==(w['dev'],w['inode'],w['bytes'],w['mtime_ns'],w['ctime_ns']),'Accepted current-chain prior changed')


def check_prepared(prepared):
    for ref in prepared['evidence']:
        require(reference(ref['path'])==ref,'Pilot prerequisite/failure/census evidence changed')


def forecasts():
    prior=checked(reference(E/'readout-worker001-review.json'));rows=[]
    for row in prior['rows']:
        w=checked(row['worker']);guard=checked(row['worker_guard']);parent=checked(row['parent_guard'])
        update=20*max(w['update_seconds']);evaluation=max(0.,w['training_seconds']-sum(w['update_seconds']))*5/len(w['metrics'])
        publication=max(0.,guard['elapsed_seconds']-w['training_seconds'])
        worker=2*(update+evaluation+publication)+15.;acceptance=2*parent['elapsed_seconds']+15.
        require(worker<600 and acceptance<600,'Measured conservative pilot forecast exceeds guard')
        rows.append(dict(arm=row['arm'],update_projection=update,evaluation_projection=evaluation,publication_observed=publication,
            conservative_worker_seconds=worker,conservative_acceptance_seconds=acceptance,interpretation='Four-update timing-derived working projection with2x+15s margin, not promise or full-horizon forecast'))
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
        pins=sources();gate=checked(reference(E/'readout-worker-source001/result.json'))
        require(gate['status']=='passed_readout_worker_source_scope_and_output_bounds' and gate['sources']==pins and gate['prepared_builder']==reference(E/'readout-worker001.py'),'Actual source/builder gate changed')
        refs={n:reference(E/n) for n in ['readout-resume001/result.json','readout-resume001/mps-guard.json','readout-resume001/STOP','readout-resume-review001.json']}
        require(refs==gate['prerequisites'],'Original own resume/scorer evidence changed')
        census=E/'readout-pilot-seed-census002';roles=checked(reference(census/'result.json'))
        require(not roles['prior_matches'] and checked(reference(census/'STOP'))['status']=='complete_new_pilot_role_census','Closed fresh role audit required')
        execution=execution_sources();prepared=checked(reference(E/'readout-pilot001-preflight.json'))
        require(prepared['status']=='distinct_twenty_pilot_source_resource_roles_reviewed' and prepared['execution_sources']==execution and prepared['worker_sources']==pins and prepared['runner']==reference(__file__),'Exact pilot preparation differs')
        check_prepared(prepared)
        prior=prior_weights();credit=sum(w['bytes'] for w in prior);remaining_floor=max(RESERVE,CHAIN_FLOOR-credit);future=remaining_floor+2*1024**3
        require(shutil.disk_usage(ROOT).free>=max(FLOOR+2*1024**3,future),'Complete remaining retained future plus pilot target required')
        plan=dict(kind='readout_mixture_three_arm_twenty_resource_pilot_v1',root=str(RUN),runner=reference(__file__),sources=pins,environment=env,specs=specs(pins,execution),seconds=600,compare_cpu=True,
                  prerequisites=refs,source_review=reference(E/'readout-worker-source001/result.json'),pilot_preflight=reference(E/'readout-pilot001-preflight.json'),execution_sources=execution,prior_retained_credit=prior,prior_accepted_bytes=credit,roles=reference(census/'result.json'),design=reference(E/'readout-mixture-design001.json'),complete_prospective_target=future,scientific_quality=False,new_scientific_seed=False,
                  initializer_private_roles=roles['roles'],full_floor=remaining_floor,prerequisite_root_floor=FLOOR,first_wake=remaining_floor+ENTRY,cleanup_target=future,
                  weight_caps=CAPS,metadata_cap=METADATA,reserve=RESERVE,external_headroom=EXTERNAL,entry_headroom=ENTRY,
                  capacity_policy='Only exact verified accepted current-chain four-update bytes and actual independently accepted current-pilot endpoint bytes credited ONCE;no old study/prerequisite double/future deletion/hardlink credit. Atomic tmp inside cap.',
                  scoring='Complete34278native targets/111536bytes,RAWandH200 own normalized-distributionNLL onCPUandMPS,H200cache/resetB1/8bothdevices.',no_retirement=True)
        forecast=forecasts()
        plan['forecast']=forecast;RUN.mkdir();publish(PLAN,plan)
        try:
            with session_lock(RUN) as run_fd:
                writer=MetadataWriter(RUN,(global_fd,run_fd));rows=[];used=0;hashes=set()
                for spec in plan['specs']:
                    require(execution_sources()==execution,'Pilot execution bindings changed');check_prior(prior);check_prepared(prepared)
                    require(shutil.disk_usage(ROOT).free>=max(RESERVE,remaining_floor-used)+ENTRY,'Complete actual remaining output requirement failed')
                    row=launch(writer,reference(PLAN),spec,max(RESERVE,remaining_floor-used));require(execution_sources()==execution,'Pilot execution changed during slot');check_prior(prior);check_prepared(prepared);rows.append(row);worker=checked(row['worker'])
                    for name in ['parent','h200']:
                        verify_artifact(worker[name]);require(worker[name]['sha256'] not in hashes,'Duplicate accepted checkpoint bytes');hashes.add(worker[name]['sha256'])
                    used+=sum(worker[name]['size'] for name in ['parent','h200'])
                    publish(E/f'readout-pilot001-accepted-{spec["slot"]:02d}.json',dict(accepted=reference(RUN/f'evidence/pilots/{spec["slot"]:02d}/accepted.json'),scientific_quality=False))
                    metadata_bound();gc.collect();torch.mps.empty_cache()
                require(sources()==pins,'Frozen sources changed')
                review=dict(status='all_three_readout_twenty_update_pilots_independently_accepted',plan=reference(PLAN),rows=rows,retained_weight_bytes=used,
                            scientific_quality=False,new_scientific_seed=False,holdout_encoded_or_scored=False,
                            next_action='Full1200state/history/41H200 serialization, equal LR-screen/selection/fresh-confirmation/statistics source contracts and independent review; complete retained-output/runtime forecast and NEWscientific-role census before any quality freeze. No retirement or quality selection from pilots.')
                publish(E/'readout-pilot001-review.json',review);writer.publish_json('STOP',dict(status='three_arm_twenty_pilot_complete',review=reference(E/'readout-pilot001-review.json'),never_rerun=True,all_training_acceptance_commands_closed=True));metadata_bound()
                print(json.dumps(dict(status=review['status'],retained_weight_bytes=used)),flush=True)
        except BaseException as exc:
            if not (RUN/'STOP').exists():publish(RUN/'STOP',dict(status='CLOSED_FAILED_READOUT_TWENTY_PILOT_PANEL',error=repr(exc),never_retry=True))
            raise


if __name__=='__main__':main()
