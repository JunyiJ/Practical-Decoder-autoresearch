"""One serial two-root chain: three four-update workers, then three twenty pilots."""
import gc
import json
import os
import shutil
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT, require, plain, fingerprint
from research.body_allocation_model_v1.evidence import reference
from research.relu_squared_resume_v1.data import config
from research.relu_squared_model_v1.contract import ungated_seed
from research.relu_squared_resume_v1.contract import runtime
from research.relu_squared_worker_v1.execution import sources, checked, idle
from research.relu_squared_worker_v1.dispatch import launch
from research.relu_squared_worker_v1.validation import verify_artifact
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import snapshot

E = Path(__file__).resolve().parent
ROOTS = {phase: ROOT/('research_runs/tokenizer-relu-squared-'+phase+'-v1-001') for phase in ('worker', 'pilot')}
CAPS = [640*1024**2]*3
METADATA_PER_ROOT = 64*1024**2
EXTERNAL_METADATA = 96*1024**2
RESERVE = 8*1024**3
ENTRY = 2*1024**2
CHAIN_FLOOR = 2*sum(CAPS)+2*METADATA_PER_ROOT+EXTERNAL_METADATA+RESERVE+ENTRY
CHAIN_TARGET = CHAIN_FLOOR+2*1024**3
PREREQUISITES = ['relu-squared-resume001/result.json', 'relu-squared-resume001/mps-guard.json',
    'relu-squared-resume001/STOP', 'relu-squared-resume001-review.json', 'relu-squared-resume001-archive-closure.json']


def publish(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    require(len(raw) < 1024**2, 'External receipt cap1MiB')
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def specs(pins, phase, roles):
    steps = 4 if phase == 'worker' else 20
    prefix = 'four' if steps == 4 else 'twenty'
    names=['initializer','private_ungated','train','eval','global_cpu','global_mps']
    require(set(roles)=={p+'_'+n for p in ['four','twenty'] for n in names}
        and all(type(v) is int and 0<=v<2**63 for v in roles.values())
        and len(set(roles.values()))==12,'Twelve distinct audited prerequisite roles required')
    require(roles[prefix+'_private_ungated']==ungated_seed(roles[prefix+'_initializer']),'Audited private ungated role differs')
    rows = []
    for slot, arm in enumerate(['native_swiglu', 'ungated_silu', 'ungated_relu_squared']):
        cfg = config(arm, steps, roles[prefix+'_initializer'], roles[prefix+'_train'], roles[prefix+'_eval'],
            global_cpu_seed=roles[prefix+'_global_cpu'], global_mps_seed=roles[prefix+'_global_mps'])
        cfg.training.eval_every = 1 if steps == 4 else 5; cfg.training.eval_iters = 1
        cfg.relu_squared_execution = dict(kind='ffn_prerequisite_execution_v1',
            sources_sha256=fingerprint(pins), runner_sha256=reference(__file__)['sha256'])
        path = ROOTS[phase]/f'evidence/pilots/{slot:02d}'
        cfg.training.checkpoint_dir = str(path/'checkpoints'); cfg.training.metrics_file = str(path/'metrics.jsonl')
        cfg.training.summary_file = str(path/'worker.json')
        rows.append(dict(stage='pilots', slot=slot, arm=arm, config=plain(cfg), weight_cap=CAPS[slot]))
    return rows


def metadata_bound():
    for root in ROOTS.values():
        if root.exists():
            state = snapshot(root)
            require(sum(v['size'] for v in state['files'].values() if not v['group'].endswith('/weights')) <= METADATA_PER_ROOT,
                    'Own root metadata cap exceeded')
    require(sum(p.stat().st_size for p in E.rglob('*') if p.is_file()) <= EXTERNAL_METADATA,
            'Complete external prerequisite evidence cap exceeded')


def retained_credit(retained):
    paths = set(); total = 0
    for record in retained:
        require(record['path'] not in paths, 'Current-chain endpoint cannot be credited twice')
        paths.add(record['path']); verify_artifact(record)
        st = Path(record['path']).stat()
        require((st.st_dev, st.st_ino) == (record['device'], record['inode']), 'Accepted current-chain inode changed')
        total += record['size']
    return total


def forecast():
    # Cost forecast only: prior own generated state68.825s and projection real
    # worker46.663s/acceptance16.708s. Include full physical CPU/MPS scoring,
    # parent archive re-reads, state validation and cold-start/source overhead.
    worker = 180.
    acceptance = 240.
    return dict(worker_seconds=worker, acceptance_seconds=acceptance, controller_margin_seconds=120.,
        complete_six_run_seconds=6*(worker+acceptance)+120., rss_bytes=int(3.5*1024**3), mps_bytes=int(3.5*1024**3),
        limit='Working serial cold-start/source-publication/fullCPU-MPS prior; not quality/cost adoption or full-horizon forecast.')


def main():
    torch.set_num_threads(4)
    with session_lock(GLOBAL_LOCK.parent) as global_fd:
        idle(); env = runtime()
        require(all(not p.exists() for p in ROOTS.values()) and all(not (E/('relu-squared-'+phase+'001-plan.json')).exists() for phase in ROOTS),
                'Both original roots/plans must be unattempted')
        pins = sources(); gate = checked(reference(E/'relu-squared-worker-source001/result.json'))
        require(gate['status'] == 'passed_ffn_worker_source_scope_and_output_bounds'
                and gate['sources'] == pins and gate['prepared_builder'] == reference(__file__), 'Exact source/builder gate required')
        require(checked(reference(E/'relu-squared-worker-source001/STOP'))['status'] == 'complete_ffn_worker_source_scope_output_gate'
                and checked(reference(E/'relu-squared-worker-source001/guard.json'))['status'] == 'finished', 'Original source gate must finish cleanly')
        refs = {n: reference(E/n) for n in PREREQUISITES}
        require(refs == gate['prerequisites'], 'Original serialized-state/scorer prerequisites changed')
        census = E/'relu-squared-worker-seed-census001'; roles = checked(reference(census/'result.json'))
        require(not roles['prior_matches'] and checked(reference(census/'STOP'))['status'] == 'complete_new_ffn_worker_chain_role_census',
                'Original successful twelve-role census required')
        require(gate['census'] == reference(census/'result.json') and checked(reference(census/'plan.json'))['sources'] == pins
                and checked(reference(census/'guard.json'))['status'] == 'finished', 'Same frozen successful census must bind this runner')
        require(shutil.disk_usage(ROOT).free >= CHAIN_TARGET, 'Complete six-run retained outputs/reserve/extra2GiB required')
        cost = forecast()
        require(cost['worker_seconds'] < 600 and cost['acceptance_seconds'] < 600
                and cost['rss_bytes'] < 4*1024**3 and cost['mps_bytes'] < 6*1024**3, 'Complete finite forecast must fit guards')
        retained = []; hashes = set(); phase_refs = []
        for phase in ('worker', 'pilot'):
            root = ROOTS[phase]; plan_path = E/('relu-squared-'+phase+'001-plan.json')
            if phase == 'pilot':
                prior = checked(phase_refs[0]); require(prior['status'] == 'all_three_ffn_four_update_workers_independently_accepted'
                    and len(prior['rows']) == 3 and checked(reference(ROOTS['worker']/'STOP'))['status'] == 'three_arm_ffn_worker_complete',
                    'All three original four-update acceptances must close before twenty pilots')
            plan = dict(kind='ffn_finite_prerequisite_phase_v1', phase=phase, root=str(root), runner=reference(__file__), sources=pins,
                environment=env, specs=specs(pins, phase, roles['roles']), seconds=600, compare_cpu=True,
                prerequisites=refs, source_review=reference(E/'relu-squared-worker-source001/result.json'),
                roles=reference(census/'result.json'), prior_phase_reviews=list(phase_refs), design=reference(E/'real-prerequisite-worker-source-design001.md'),
                complete_chain_floor=CHAIN_FLOOR, complete_chain_target=CHAIN_TARGET, weight_caps=CAPS,
                metadata_per_root=METADATA_PER_ROOT, external_metadata_cap=EXTERNAL_METADATA, reserve=RESERVE, entry_headroom=ENTRY,
                capacity_policy='Credit only actual independently accepted retained outputs from this six-run chain once; no old/proof/future-deletion/hardlink credit. Atomic.tmp within pair cap.',
                scientific_quality=False, new_scientific_seed=False, prerequisite_roles_only=True, no_retirement=True,
                forecast=cost, scoring='FULL34278native/111536bytes,RAW+H200 onCPUANDMPS,H200cache/resetB1B8bothdevices')
            root.mkdir(); publish(plan_path, plan)
            try:
                with session_lock(root) as run_fd:
                    writer = MetadataWriter(root, (global_fd, run_fd)); rows = []
                    for spec in plan['specs']:
                        require(sources() == pins and all(reference(v['path']) == v for v in refs.values()), 'Frozen inputs changed')
                        credit = retained_credit(retained); required = max(RESERVE, CHAIN_FLOOR-credit)
                        require(shutil.disk_usage(ROOT).free >= required+2*1024**3, 'Complete measured remaining target failed before unattempted slot')
                        row = launch(writer, reference(plan_path), spec, required); rows.append(row); summary = checked(row['worker'])
                        for name in ['parent', 'h200']:
                            endpoint = summary[name]; verify_artifact(endpoint)
                            require(endpoint['sha256'] not in hashes, 'Duplicate accepted checkpoint content forbidden')
                            hashes.add(endpoint['sha256']); st = Path(endpoint['path']).stat()
                            retained.append(dict(endpoint, device=st.st_dev, inode=st.st_ino))
                        publish(E/f'relu-squared-{phase}001-accepted-{spec["slot"]:02d}.json',
                                dict(accepted=reference(root/f'evidence/pilots/{spec["slot"]:02d}/accepted.json'), scientific_quality=False))
                        metadata_bound(); gc.collect(); torch.mps.empty_cache()
                    require(sources() == pins, 'Frozen sources changed')
                    status = 'all_three_ffn_four_update_workers_independently_accepted' if phase == 'worker' else 'all_three_ffn_twenty_update_pilots_independently_accepted'
                    review = dict(status=status, plan=reference(plan_path), rows=rows,
                        retained_weight_bytes=sum(checked(row['worker'])['serialized_bytes']['pair'] for row in rows),
                        scientific_quality=False, holdout_encoded_or_scored=False, no_retirement=True,
                        next_action='Next phase only after full acceptance. After twenty, separate full-horizon implementation/correctness/resource/new scientific-role gates; no quality launch or adoption admitted.')
                    review_path = E/('relu-squared-'+phase+'001-review.json'); publish(review_path, review)
                    writer.publish_json('STOP', dict(status='three_arm_ffn_'+phase+'_complete', review=reference(review_path), never_rerun=True))
                    phase_refs.append(reference(review_path)); metadata_bound()
                    print(json.dumps(dict(status=status, retained_weight_bytes=review['retained_weight_bytes'])), flush=True)
            except BaseException as exc:
                if not (root/'STOP').exists():
                    publish(root/'STOP', dict(status='CLOSED_FAILED_PROJECTION_PREREQUISITE_PHASE', error=repr(exc), never_retry=True))
                raise
        publish(E/'relu-squared-prerequisite-chain001-completed.json', dict(status='both_finite_phases_independently_accepted',
            phases=phase_refs, retained=retained, retained_weight_bytes=retained_credit(retained), scientific_quality=False, no_retirement=True))


if __name__ == '__main__':
    main()
