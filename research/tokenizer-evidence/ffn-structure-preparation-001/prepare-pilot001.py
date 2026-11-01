"""Bind scientific data, check finite output boundaries and freeze two resource pilots."""
import copy, hashlib, io, json, shutil
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.ffn_structure_v1 import pilot_contract as c,pilot_execution as e,pilot_artifacts as a,validation as v
from research.ffn_structure_v1.validation_data import load as view
from research.ffn_structure_v1.runtime.train.lr_schedule import schedule_contract
from research.ffn_structure_v1.runtime.train.weight_averaging import averaging_contract
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=c.EVIDENCE
def reject(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('Invalid boundary accepted')
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    e.idle();assert not c.RUN.exists() and not e.FREEZE.exists()
    with CpuPrerequisiteGuard(B/'pilot001-prepare-guard.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
        census=read(reference(B/'scientific-seed-census-001/result.json'))
        assert census['status']=='fresh_seeds_reserved' and not census['matches']
        inventory=B/'scientific-seed-census-001/inventory.json'
        assert hashlib.sha256(inventory.read_bytes()).hexdigest()==census['inventory_sha256']
        seeds=census['seeds'];sources=c.source_inventory()
        prior=read(reference(B/'trainer-worker003.json'))
        assert all(sources[p]==h for p,h in (prior['sources']|prior['frozen_sources']).items())
        prereqs={name:reference(B/name) for name in ['transport-worker-inference001-review.json','checkpoint-resume001.json','trainer-worker003.json','trained-inference001.json','scientific-seed-census-001/result.json']}
        assert read(prereqs['transport-worker-inference001-review.json'])['status']=='transport_worker_inference_prerequisites_passed'
        configs=[c.config(i,seeds) for i in range(2)];data=c.base.bind_data();checks=[]
        for i,cfg in enumerate(configs):
            c.validate_config(cfg,i,seeds);o=OmegaConf.create(cfg)
            schedule=schedule_contract(o.training);ema=averaging_contract(o.training)
            assert schedule['total_steps']==ema['total_steps']==20 and schedule['start_step']==ema['start_step']==13
            d=view(o);target=v.target_contract(d)
            assert set(d.splits)=={'train','val'} and target['targets']==data['validation_targets']
            assert target['target_ids_sha256']==data['validation_target_ids_sha256'] and target['window_counts']==data['validation_window_target_counts']
            assert target['target_bytes']==data['validation_target_bytes'] and target['tokenizer']==data['tokenizer']
            for key,value in [('max_iters',1200),('learning_rate',.00125),('seed',seeds['screen'])]:
                bad=copy.deepcopy(cfg);bad['training'][key]=value;reject(lambda:c.validate_config(bad,i,seeds));checks.append(f'slot{i}_reject_{key}')
            bad=copy.deepcopy(cfg);bad['data']['val_fraction']=.2;reject(lambda:c.validate_config(bad,i,seeds));checks.append(f'slot{i}_reject_split')
            sink=e.Sink(c.output(i),cfg)
            reject(lambda:sink.checkpoint(c.output(i)/'checkpoints/final_bad.pt',dict(step=19)))
            reject(lambda:sink.checkpoint(c.output(i)/'wrong.pt',dict(step=20)))
            reject(lambda:sink.metric(c.output(i)/'metrics.jsonl',dict(step=1)))
            checks.extend([f'slot{i}_reject_partial_checkpoint',f'slot{i}_reject_wrong_path',f'slot{i}_reject_metric_sequence'])
        for slot in (-1,2,True):reject(lambda:c.output(slot));checks.append(f'reject_slot_{slot}')
        buffer=io.BytesIO();writer=e.CappedWriter(buffer,8);writer.write(b'12345678');reject(lambda:writer.write(b'9'));assert buffer.getvalue()==b'12345678';checks.append('serialization_cap_before_write')
        reject(lambda:e.Guard(B/'not-created.json',0));checks.append('reject_unbounded_deadline')
        # Audit code is the previously tested payload arithmetic with only FFN imports and exact pilot request binding changed.
        historical=(c.ROOT/'research/block_norm_v1/artifact_acceptance.py').read_text()
        current=(c.ROOT/'research/ffn_structure_v1/pilot_artifacts.py').read_text()
        for start,end in [('def audit_metric_rows','def audit_payloads'),('def audit_payloads','def accept')]:
            assert historical[historical.index(start):historical.index(end)]==current[current.index(start):current.index(end)]
        weights_limit=c.RAW_LIMIT+c.EXPORT_LIMIT
        assert weights_limit+16*1024**2<c.TRAJECTORY_ALLOWANCE
        projection=dict(trajectory_allowance_bytes=c.TRAJECTORY_ALLOWANCE,raw_serialization_cap=c.RAW_LIMIT,h200_serialization_cap=c.EXPORT_LIMIT,
            metadata_and_log_allowance_per_trajectory_bytes=16*1024**2,round_metadata_allowance_bytes=256*1024**2,
            retained_pilots=2,peak_screens_plus_pilots=8,peak_selected_fresh_plus_pilots=10,reserve_bytes=8*1024**3,
            required_free_bytes=c.REQUIRED_FREE,free_bytes=shutil.disk_usage(c.ROOT).free,
            serialization='Direct exclusive capped raw and H200 writes: no full-size temporary/rename/copy; both files are charged simultaneously. Process journals have only bounded tiny replacement overlap.',
            retention='Keep both pilot pairs without cleanup credit. All six screens coexist through selection; four losing pairs require a separate exact audited deletion boundary before fresh runs. Keep two selected and all six fresh pairs. No failed or fixture cleanup credit.',
            quality_gate='Quality freeze must implement/test retention boundary and re-audit full-duration output/config acceptance; no quality dispatch authority here.')
        assert projection['free_bytes']>=c.REQUIRED_FREE
        e.exclusive(B/'pilot001-capacity.json',projection);prereqs['capacity']=reference(B/'pilot001-capacity.json')
        e.exclusive(B/'pilot001-boundary-tests.json',dict(status='passed',checks=checks,targets=data['validation_targets'],real_corpus_prior_and_train_validation_binding=True,holdout_tokenized=False,source_inventory=sources))
        prereqs['boundary_tests']=reference(B/'pilot001-boundary-tests.json')
        snap=B/'pilot001-source';snap.mkdir()
        for name in sources:
            if name.startswith('research/ffn_structure_v1/'):
                dest=snap/Path(name).relative_to('research/ffn_structure_v1');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((c.ROOT/name).read_bytes())
        plan=dict(kind='ffn_structure_two_resource_pilots_v1',root=str(c.RUN),sources=sources,environment=c.base.environment(),data=data,seeds=seeds,configs=configs,
            limits=c.base.LIMITS,required_free_bytes=c.REQUIRED_FREE,trajectory_allowance_bytes=c.TRAJECTORY_ALLOWANCE,prerequisites=prereqs,
            pilot_order=list(c.ARMS),lr=.0025,steps=20,quality_launch_permitted=False,
            failure_policy='A failed/partial attempt stops the pilot root; preserve original source/weights/logs. No automatic retries.',
            acceptance='Child strict MPS then independent parent raw/optimizer/EMA/export/schedule/tokenizer/metrics and full-validation recomputation under one600s deadline per slot; trained pair inference after both accepted.',
            holdout='Scoring uses train/validation-only view. Unchanged actual training loader materializes test tokens but never scores them.',
            scientific_quality_scope='These20step pilots are excluded from LR selection and quality conclusions. Proposed1200step3LR/arm plus3freshpairs still needs a separate final freeze.',
            prepare_source=reference(Path(__file__).resolve()))
        e.exclusive(B/'pilot001-plan.json',plan);c.RUN.mkdir();e.exclusive(e.FREEZE,dict(plan=reference(B/'pilot001-plan.json'),quality_launch_permitted=False))
        assert e.prefix(0)==([],0);assert e.bind(reference(e.FREEZE)['sha256'])==plan
    e.exclusive(B/'pilot001-preflight.json',dict(status='frozen_two_resource_pilots_only',freeze=reference(e.FREEZE),guard=reference(B/'pilot001-prepare-guard.json'),checks=len(checks),scientific_quality_launched=False))
    print(json.dumps(dict(status='frozen_two_resource_pilots_only',checks=len(checks),free_bytes=projection['free_bytes'],required_bytes=c.REQUIRED_FREE,freeze=reference(e.FREEZE))),flush=True)
