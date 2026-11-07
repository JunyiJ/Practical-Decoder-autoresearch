"""Final full-panel dependency/runtime/data/capacity check before source commit/freeze."""
import fcntl,json,os,shutil
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_mla_gate_quality_v1 import contract as C
from research.tokenizer_mla_gate_resume_v1.contract import runtime
from research.tokenizer_mla_gate_resume_v1.data import load
from research.tokenizer_mla_gate_worker_v1.validation import verify_artifact
from research.tokenizer_mla_gate_worker_v1.execution import idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
E=Path(__file__).resolve().parent;OUT=E/'gate-quality-preflight001.json';GUARD=E/'gate-quality-preflight001-guard.json'
def read(p):return json.loads(Path(p).read_text())
with session_lock(GLOBAL_LOCK.parent):
    idle();torch.set_num_threads(4);require(not OUT.exists() and not GUARD.exists() and not C.RUN.exists() and not C.PLAN.exists() and not C.FREEZE.exists(),'Fresh final preflight before production only')
    with CpuPrerequisiteGuard(GUARD,ROOT,timeout_seconds=120):
        review=read(E/'gate-quality-review001.json');proof=C.checked(review['proof']);require(review['status']=='full_gate_quality_correctness_independently_reviewed' and proof['status']=='passed_full_quality_bind_state_sources_and_final_contracts' and proof['sources']==C.sources() and proof['design']==C.DESIGN,'Complete actual independently reviewed runtime required')
        for p in [E/'gate-quality-contracts001-guard.json',E/'gate-quality-review001-guard.json',E/'gate-quality-seed-census001/guard.json']:
            g=read(p);require(g['status']=='finished' and g['reason'] is None,'Actual successful correctness/census guard required')
        census=read(E/'gate-quality-seed-census001/result.json');require(census['status']=='reserved_unused' and census['rg_exit_code']==1 and census['prior_matches']==[] and not census['scientific_training_started'],'Unused role census required');C.seeds_valid(census['seeds'])
        for r in census['inventory']:C.checked(r)
        retained=0;allrefs=[]
        for name in ['gate-worker-review001.json','gate-pilot-review001.json']:
            old=read(E/name)
            for r in old['weights']:
                verify_artifact(r);retained+=r['size'];allrefs.append({k:r[k] for k in ('path','sha256','size')})
        require(len(allrefs)==12 and retained==3363688308,'All original protected current prerequisites required')
        cfg=OmegaConf.create(C.screen_specs(census['seeds'])[0]['config']);view=load(cfg,'cpu');require(len(view.train_data)==267688 and len(view.val_data)==34279,'Actual original native train/validation data required')
        pilot=read(E/'gate-pilot001-review.json');trajectory_forecast=6*sum(p['conservative1200_seconds'] for p in pilot['projections'])
        time_budget=dict(trajectories_including_parent_and_serialization_seconds=trajectory_forecast,all_nine_fresh_serving_allowance_seconds=5400,independent_final_arithmetic_allowance_seconds=120,complete_remaining_forecast_seconds=trajectory_forecast+5400+120,not_a_promise=True)
        # Maintain the original full-chain target as well as the exact18-output floor.
        full_chain_target=read(E/'gate-prerequisite-capacity001.json')['target_with_2gib_extra'];target=max(C.TARGET,full_chain_target)
        with (ROOT/'research_state/external-cache-maintenance.lock').open('a+b') as lock:
            acquired=False
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True
            except BlockingIOError:pass
            free=shutil.disk_usage(ROOT).free;require(free>=target and free>=C.FLOOR+C.ENTRY,'Full retained target and measured entry margin must pass')
            capacity=dict(quality_weight_cap=6*sum(C.CAPS.values()),quality_floor=C.FLOOR,quality_target=C.TARGET,original_full_chain_remaining_target=full_chain_target,enforced_target=target,free_bytes=free,deficit_bytes=0,external_lock_acquired=acquired,deleted_bytes=0,inventory_or_deletion_needed=False,prerequisite_credit_against18output_floor=0,old_payload_hardlink_future_cleanup_credit=0)
        out=dict(status='complete_gate_quality_preflight_passed_before_freeze',source=reference(__file__),sources=C.sources(),source_bindings=1084,environment=runtime(),design=C.DESIGN,proof=reference(E/'gate-quality-contracts001.json'),independent_review=reference(E/'gate-quality-review001.json'),census=reference(E/'gate-quality-seed-census001/result.json'),reserved_scientific_roles=census['seeds'],all12prerequisites=allrefs,prerequisite_retained_bytes=retained,data=view.data_identity,capacity=capacity,time_budget=time_budget,quality_panel_frozen=False,training_started=False,source_snapshot_must_be_committed_before_freeze=True,holdout_encoded_or_scored=False)
        raw=(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<2*1024**2,'Final preflight metadata cap')
        with OUT.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        print(json.dumps(dict(status=out['status'],source_bindings=1084,quality_floor=C.FLOOR,target=target,free_bytes=free,remaining_seconds=time_budget['complete_remaining_forecast_seconds'])),flush=True)
