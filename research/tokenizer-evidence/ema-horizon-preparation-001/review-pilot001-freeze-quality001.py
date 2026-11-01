"""Record accepted pilot resources and freeze exactly three fresh EMA trajectories."""
import copy,hashlib,json,shutil
from pathlib import Path
import torch
from research.ema_horizon_v1 import contract as c,execution as e
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter

torch.set_num_threads(4)
B=e.EVIDENCE
with session_lock(GLOBAL_LOCK.parent) as global_fd,session_lock(e.ROOT) as campaign_fd:
    e.idle()
    with CpuPrerequisiteGuard(B/'quality001-freeze-guard.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
        pilot=read(read(reference(e.freeze_path('pilots')))['plan'])
        assert pilot['sources']==c.source_inventory()
        anchor=read(reference(e.anchor_path('pilots',0)));accepted=read(anchor['acceptance'])
        assert accepted['artifact_acceptance']['accepted'] and accepted['stage']=='pilots' and accepted['slot']==0
        parent=read(anchor['parent_guard']);worker=read(reference(e.ROOT/'control/worker-pilots-00.json'))
        process=read(accepted['process'])
        assert process['status']=='completed' and process['returncode']==0 and process['error'] is None
        for guard in (parent,worker):assert guard['status']=='finished' and guard['reason'] is None
        inference=read(reference(B/'pilot001-inference-result.json'));ig=read(reference(B/'pilot001-inference-guard.json'))
        assert inference['status']=='passed' and ig['status']=='finished' and ig['reason'] is None
        request=read(accepted['request'])['acceptance_request']
        assert inference['request_sha256']==c.inherited.fingerprint(request)
        weights=[];portable=B/'pilot001-evidence';portable.mkdir()
        for ref in anchor['files']:
            assert e.ref_file(ref['path'])==ref
            path=Path(ref['path'])
            if path.suffix=='.pt':weights.append(dict(ref,bytes=path.stat().st_size))
            else:
                target=portable/path.relative_to(e.ROOT)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        for path in sorted((e.ROOT/'control').glob('*.json')):
            target=portable/path.relative_to(e.ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        used=sum(w['bytes'] for w in weights);assert len(weights)==3 and used<c.TRAJECTORY_ALLOWANCE
        remaining=c.REQUIRED_FREE-used;free=shutil.disk_usage(c.ROOT).free;assert free>=remaining
        review=dict(status='accepted_resource_and_artifact_pilot',acceptance=reference(e.anchor_path('pilots',0)),
            worker_guard=worker,parent_guard=parent,process_elapsed_seconds=process['elapsed_seconds'],
            weights=weights,retained_weight_bytes=used,remaining_required_free_bytes=remaining,observed_free_bytes=free,
            validation_ce={half:row['ce'] for half,row in accepted['artifact_acceptance']['validation'].items()},
            audit=accepted['artifact_acceptance']['audit'],cache=accepted['artifact_acceptance']['cache'],
            inference=reference(B/'pilot001-inference-result.json'),inference_guard=reference(B/'pilot001-inference-guard.json'),
            quality_evidence=False,holdout_scored=False,cleanup_credit_bytes=0,
            note='One20step pilot passed real raw resume preflight, both exact EMA exports, independent full validation and cache/reset acceptance. No extra full-size fixture retained; original failures and all selected/fresh weights remain protected.')
        e.exclusive(B/'pilot001-review.json',review)
        plan=copy.deepcopy(pilot);seeds=plan['seeds']
        plan.update(stage='quality',status='frozen_three_fresh_trajectories',specs=[c.spec('quality',n,seeds) for n in range(3)],
            configs=[c.config(e.ROOT/f'evidence/quality/{n:02d}','quality',n,seeds) for n in range(3)],
            quality_plan_frozen=True,quality_launch_permitted=True,pilot_acceptance=reference(e.anchor_path('pilots',0)),
            free_bytes=free,remaining_required_free_bytes=remaining,
            real_artifact_gate='Accepted pilot anchor, independent raw/H100/H200 audit and inference prerequisites passed; every fresh slot must independently repeat artifact acceptance.',
            finite_design=dict(trajectories=3,screen_runs=0,screen_seed_reserved_unused=True,steps=1200,learning_rate=.00125,
                ema_start=800,ema_interval=10,ema_snapshots=41,control_half_life=200,candidate_half_life=100,
                paired_same_online_trajectory=True,quality_metric='full validation CE',
                statistical_primitive='research.block_norm_v1.campaign_rules.paired_statistics',
                input_arm_order=[200,100],input_shape=[3,2,17],base_block_tokens=2048,tail_tokens=1510,
                block_sizes=[2048,4096,8192],bootstrap_draws=2000,generator='PCG64',quantile_method='linear',
                gate_quantile=1-.05/3,mean_delta_required_below=-.02,each_seed_delta_required_below=0,
                each_simultaneous_text_upper_required_below=0,raw_selection=False,automatic_adoption=False,
                retry=False,replacement_seeds=False,expanded_grid=False),
            fresh_inference=dict(source=reference(B/'pilot001-inference.py'),function='measure',pair_indices=[0,1,2],
                half_lives=[200,100],batches=[1,8],warmups=2,repeats=8,prefill_tokens=32,decode_tokens=64,
                token_source='first96_training_ids',descriptive_only=True,timeout_seconds=600))
        plan['prerequisites'].update(pilot_review=reference(B/'pilot001-review.json'),
            pilot_inference=reference(B/'pilot001-inference-result.json'),pilot_inference_guard=reference(B/'pilot001-inference-guard.json'))
        e.exclusive(B/'quality001-plan.json',plan)
        e.exclusive(e.freeze_path('quality'),dict(plan=reference(B/'quality001-plan.json'),quality_launch_permitted=True))
        sha=reference(e.freeze_path('quality'))['sha256']
        bound=e.bind(MetadataWriter(e.ROOT,(global_fd,campaign_fd)),'quality',sha)
        prefix,retained=e.prefix('quality',0);assert retained==used and len(prefix)==1 and bound==plan
    e.exclusive(B/'quality001-preflight.json',dict(status='prepared_three_fresh_only',freeze=reference(e.freeze_path('quality')),
        guard=reference(B/'quality001-freeze-guard.json'),quality_started=False,next_slot=0))
    print(json.dumps(dict(freeze=reference(e.freeze_path('quality')),retained_bytes=used,free_bytes=free,required=remaining)))
