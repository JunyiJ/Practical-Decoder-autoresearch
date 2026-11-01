"""Read-only next-hypothesis and retained-output capacity review; no training."""
import hashlib,json,math,shutil
from pathlib import Path
from datetime import datetime,timezone
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[3]
B=ROOT/'research/tokenizer-evidence/ema-horizon-preparation-001'
N=ROOT/'research/tokenizer-evidence/block-norm-preparation-001'
def ref(path):
    path=Path(path);return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def load_ref(binding):
    path=Path(binding['path']);assert ref(path)==binding;return json.loads(path.read_text())
with session_lock(GLOBAL_LOCK.parent):
    idle()
    assert not (B/'review.json').exists()
    with CpuPrerequisiteGuard(B/'review-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
        closure=load_ref(ref(N/'quality001-closure.json'));load_ref(closure['stop'])
        final=load_ref(closure['review']);assert final['status']=='closed_not_adopted'
        frozen=load_ref(ref(N/'quality001-freeze.json'));plan=load_ref(frozen['plan'])
        for name,h in plan['sources'].items():assert ref(ROOT/name)['sha256']==h
        shape=load_ref(ref(N/'shape-storage-review.json'));arm=shape['arms']['rms_incumbent']
        p=arm['parameters'];buffers=shape['persistent_buffer_bytes'];opt=arm['optimizer_tensor_bytes']
        assert (p,buffers,opt)==(28492416,65536,216929456)
        curves=[]
        for slot in (6,9,10):
            anchor=load_ref(ref(N/f'quality001-accepted-{slot:02d}.json'));accepted=load_ref(anchor['acceptance'])
            request=accepted['request'];assert request['trial']['spec']['arm']=='rms_incumbent'
            metrics_ref=request['files']['metrics'];assert ref(metrics_ref['path'])==metrics_ref
            metrics=[json.loads(x) for x in Path(metrics_ref['path']).read_text().splitlines()]
            curves.append(dict(slot=slot,seed=request['trial']['spec']['seed'],metrics=metrics_ref,
                sampled_online_validation=[dict(step=x['step'],ce=x['val_ce_loss']) for x in metrics],
                accepted_h200_ce=accepted['artifact_acceptance']['validation']['ce'],
                limitation='Online sampled losses are diagnostics, not full-validation checkpoint-selection evidence.'))
        # One raw state includes online weights + two EMA arrays + optimizer.
        # Two exports each hold model parameters + persistent buffers.
        payload=20*p+opt+3*buffers
        raw_payload=12*p+opt+buffers
        # Preserve the earlier conservative 24P envelope; add 8P for the
        # extra EMA and export, one buffer, 2MiB headers and 15% margin.
        allowance=math.ceil((32*p+3*buffers+2*1024**2)*1.15)
        reserve=8*1024**3;metadata=256*1024**2;required=reserve+metadata+4*allowance
        free=shutil.disk_usage(ROOT).free
        refs=[ROOT/'research/PARALLEL_BLOCK_RESULTS.md',ROOT/'research/TOKENIZER_HORIZON_RESULTS.md',
              ROOT/'research/AVERAGING_QUALITY_RESULTS.md',ROOT/'research/TOKENIZER_MLA_RESULTS.md',
              ROOT/'research/TOKENIZER_TABLE_RESULTS.md',ROOT/'research/TOKENIZER_ARCHITECTURE_ROADMAP.md',
              N/'shape-storage-review.json',N/'quality001-final-review.json',
              ROOT/'research/block_norm_v1/runtime/train/weight_averaging.py',
              ROOT/'research/block_norm_v1/artifact_acceptance.py',ROOT/'research/block_norm_v1/validation.py']
        result=dict(status='hypothesis_selected_not_launch_frozen',observed_at=datetime.now(timezone.utc).isoformat(),
            runner=ref(__file__),references=[ref(x) for x in refs],frozen395_sources_unchanged=True,
            selected_hypothesis='H100 versus H200 returned weights on the same new 1200-update RMSNorm/SiLU MLA trajectories',
            initial_snapshot_weights=dict(H100=2**(-400/100),H200=2**(-400/200)),
            interpretation='The shorter 400-update EMA tail retains more initialization mass under H200 than the historical 1000-update character tail. Lower lag is a hypothesis, not an established quality benefit.',
            recent_rms_curves=curves,
            alternatives=dict(parallel_residual='Deferred: historical character deltas positive at both widths and no specific new MLA benefit established.',
                longer_training='Deferred: prior BPE GQA extension degraded; current MLA sampled loss still decreases, so neither early stopping nor extension is established.',
                table_tying='Closed negative BPE comparison; do not retry for capacity.',moe='Deferred: full capacity still unresolved.'),
            proposed_panel=dict(pilot_trajectories=1,quality_trajectories=3,new_fresh_seeds_required=True,
                outputs_per_trajectory=['raw_with_both_ema_states','H100','H200'],learning_rate=.00125,steps=1200,
                cooldown_start=800,ema_start=800,ema_interval=10,half_lives=[100,200],snapshots=41,
                selection='Fixed H100 candidate, no screen or LR reselection; identical online training and optimizer for both outputs.',
                adoption='Mean paired H100-minus-H200 CE below -0.02; each seed below zero; all simultaneous 2048/4096/8192 conditional-text upper bounds below zero. No cost-only route.'),
            capacity=dict(parameters=p,persistent_buffer_bytes=buffers,optimizer_tensor_bytes=opt,raw_payload_bytes=raw_payload,
                raw_plus_two_exports_payload_bytes=payload,per_trajectory_allowance_bytes=allowance,
                trajectories_retained_including_pilot=4,reserve_bytes=reserve,metadata_bytes=metadata,
                required_free_bytes=required,observed_free_bytes=free,headroom_bytes=free-required,
                fits_projection=free>=required,cleanup_credit_bytes=0,launch_permitted=False),
            concrete_contract_work=['Versioned averaging-only request and acceptance adapter outside frozen namespaces.',
                'Explicit bounded raw-file allowance above 512MiB: dual-EMA raw tensor payload alone is 558903984B; never loosen old 512MiB validators.',
                'Verify online weights/optimizer/RNG and H200 unchanged by extra shadow tracker; CPU recurrence and resume/export/parent/cache checks.',
                'Reuse unchanged scoring/statistical/guard/writer primitives with exact new output bindings; no new generic ledger.',
                'Fresh-seed census, finite pilot freeze, real guarded pilot then complete quality capacity/freeze.'],
            training_launched=False,checkpoints_loaded=False,holdout_scored=False,deletion_performed=False)
    result['guard']=ref(B/'review-guard.json')
    with (B/'review.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['capacity']))
