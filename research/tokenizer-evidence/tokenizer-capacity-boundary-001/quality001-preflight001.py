"""Final live source/environment/data/capacity audit before the new scientific freeze."""
import json,shutil
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.gpt2_prefix_data_v1.data import load
from research.gpt2_prefix_quality_v1 import contract as C
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent
with session_lock(GLOBAL_LOCK.parent):
 torch.set_num_threads(4);idle();env=environment();require(not(E/'quality001-preflight001-plan.json').exists() and not C.RUN.exists() and not C.PLAN.exists() and not C.FREEZE.exists(),'Unattempted new scientific preflight only')
 require(torch.backends.mps.is_available(),'Original local strict MPS device required');pins=C.sources();gate=C.checked(reference(E/'quality-contracts001.json'));require(gate['status']=='passed_full_quality_bind_state_sources_and_final_contracts' and gate['sources']==pins,'Actual full runtime source/state/numerical contracts required')
 census=C.checked(reference(E/'quality-seed-census001/result.json'));require(census['status']=='reserved_unused' and census['matches']==[],'Actual new nonuse census required');C.seeds_valid(census['seeds'])
 C.publish(E/'quality001-preflight001-plan.json',dict(source=reference(__file__),sources=pins,census=reference(E/'quality-seed-census001/result.json'),source_gate=reference(E/'quality-contracts001.json'),training=False,model_scoring=False,scientific_seed_rng_draws=False))
 try:
  with CpuPrerequisiteGuard(E/'quality001-preflight001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
   original=C.checked(reference(E/'pilot001-review.json'));reports=[];rng=torch.get_rng_state().clone()
   for arm in C.ARMS:
    sample=next(row for row in C.screen_specs(census['seeds']) if row['arm']==arm);cfg=OmegaConf.create(sample['config']);view=load(cfg,'cpu');prototype=C.checked(next(row for row in original['rows'] if row['arm']==arm)['worker'])
    require(view.data_identity==prototype['data'],'Actual corpus/tokenizer/library/train/val/common-byte identities changed')
    require(view.common_byte_plan['target_sha256']==C.DESIGN['common_target_sha256'] and view.common_byte_plan['target_bytes']==111475 and view.common_byte_plan['target_tokens']==C.DESIGN['token_targets'][arm],'Original full target boundary changed')
    require(cfg.training.max_iters==1200 and cfg.training.weight_averaging.start_step==800 and cfg.training.weight_averaging.interval==10 and cfg.training.lr_schedule.start_step==800 and cfg.training.device=='mps','Original numerical recipe/horizon differs')
    reports.append(dict(arm=arm,actual_data_identity=view.data_identity,private_sampler_draws=0,original_pilot_worker=next(row for row in original['rows'] if row['arm']==arm)['worker']))
   require(torch.equal(rng,torch.get_rng_state()),'Data-only preflight consumed global RNG');free=shutil.disk_usage(ROOT).free
   require(free>=C.FLOOR+C.ENTRY,'Complete current all-output floor plus entry margin required');require(C.sources()==pins,'Frozen scientific source changed')
  result=dict(status='passed_live_full_quality_source_environment_data_seed_and_capacity_preflight',source=reference(__file__),sources=pins,total_source_bindings=1004,source_gate=reference(E/'quality-contracts001.json'),census=reference(E/'quality-seed-census001/result.json'),environment=env,data_reports=reports,
   free_bytes=free,full_panel_floor=C.FLOOR,first_launch_wake=C.FLOOR+C.ENTRY,extra_headroom_target=C.TARGET,mandatory_deficit=0,target_deficit=max(0,C.TARGET-free),training=False,model_scoring=False,scientific_seed_rng_draws=False,holdout_encoded_or_scored=False,
   next_action='Verify this record/source refs,current idle/global lock and root/plan/freeze ABSENCE;invoke research.gpt2_prefix_quality_v1.execution prepare ONCE.Read NEW freezeSHA,verify bind/state prefix0,then run-all ONCE. Worker loads/binds actual original data and exact codec before ANY model initialization/update on every trajectory;do not bypass original guards.')
  C.publish(E/'quality001-preflight001.json',result);print(json.dumps(dict(status=result['status'],free_bytes=free,floor=C.FLOOR,target=C.TARGET,scientific_seeds=census['seeds'])),flush=True)
 except Exception as exc:C.publish(E/'STOP-quality001-preflight001',dict(error=repr(exc),closed=True,no_retry=True));raise
