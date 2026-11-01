"""Read-only actual payload bridge plus finite pilot configuration/resource gates."""
import copy,hashlib,json,subprocess,sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.local_mix_campaign_v1 import pilot as p,artifact_acceptance as a,evidence_contract as c
from research.local_mix_v1 import artifact_acceptance as old_a
from research.block_norm_v1.fixture_dispatch import read,reference
from research.block_norm_v1.pilot_execution import exclusive
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=Path(__file__).resolve().parent

def rejects(fn):
 try:fn()
 except (AssertionError,ValueError,TypeError):return
 raise RuntimeError('Invalid finite-pilot input accepted')

torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
 p.idle();assert not p.ROOT.exists() and not p.FREEZE.exists()
 plan=read(reference(B/'pilot-integration001-plan.json'))
 for n,h in plan['sources'].items():assert p.sha(c.ROOT/n)==h
 sources=c.source_inventory();prior=read(reference(B/'trainer-worker002.json'))
 assert len(sources)>len(prior['sources']) and all(sources[n]==h for n,h in prior['sources'].items())
 with p.Guard(B/'pilot-integration001-guard.json'):
  seeds=dict(pilot=90971,screen=90972,fresh=[90981,90982,90983],evaluation=90973,bootstrap=90974);data=p.data_contract.bind_data();configs=[];checks=[]
  for slot,arm in enumerate(p.ARMS):
   cfg=p.config(slot,seeds);model=p.model_contract(cfg['model']);t=OmegaConf.create(cfg).training
   assert model['ffn_input_mixing']['kind']==arm and model['score_scale']==.5
   assert cfg['training']['seed']==seeds['pilot'] and cfg['training']['eval_seed']==seeds['evaluation']
   assert cfg['training']['max_iters']==20 and cfg['training']['batch_size']==8 and cfg['model']['block_size']==128
   schedule=p.schedule_contract(t);ema=p.averaging_contract(t)
   assert schedule['total_steps']==ema['total_steps']==20 and schedule['start_step']==ema['start_step']==13
   assert ema['interval']==1 and ema['half_lives']==[200] and 1+(20-13)//ema['interval']==8
   target=p.v.target_contract(p.validation_data(OmegaConf.create(cfg)))
   assert target['targets']==data['validation_targets']==34278 and target['target_ids_sha256']==data['validation_target_ids_sha256']
   assert target['target_bytes']==data['validation_target_bytes'] and target['window_counts']==data['validation_window_target_counts']
   configs.append(cfg);checks.append(dict(arm=arm,target=target,schedule=schedule,ema=ema))
  shared=[copy.deepcopy(x) for x in configs]
  for cfg in shared:
   del cfg['model']['ffn_input_mixing']
   for k in ('checkpoint_dir','metrics_file','summary_file'):del cfg['training'][k]
  assert shared[0]==shared[1]
  for invalid in (True,-1,2,'0'):rejects(lambda:p.config(invalid,seeds))
  p.require_capacity(p.REQUIRED)
  rejects(lambda:p.require_capacity(p.REQUIRED-1))
  p.require_capacity(p.REQUIRED-p.WEIGHT_CAP,p.WEIGHT_CAP)
  rejects(lambda:p.require_capacity(p.REQUIRED-2*p.WEIGHT_CAP-1,2*p.WEIGHT_CAP))
  rejects(lambda:p.require_capacity(p.REQUIRED,2*p.WEIGHT_CAP+1))
  rejects(lambda:p.require_capacity(True))
  assert p.REQUIRED==8*(448+112)*1024**2+256*1024**2+8*1024**3
  assert p.QUALITY_DESIGN['fresh_trajectories']==6 and p.QUALITY_DESIGN['screens']==6 and p.QUALITY_DESIGN['mean_causal3_minus_none_strictly_below']==-.02
  cli=subprocess.run([sys.executable,'-m','research.local_mix_campaign_v1.pilot','--help'],cwd=c.ROOT,capture_output=True,text=True,timeout=20)
  assert cli.returncode==0 and '{prepare,run,worker,review}' in cli.stdout
  rows=[]
  for old in prior['arms']:
   old_request=read(old['worker'])['acceptance_request'];old_a.verify_request(old_request)
   request=copy.deepcopy(old_request);request['sources']=sources
   worker=read(old['worker']);accepted=a.accept(request,worker['validation']);p.clear()
   assert accepted['accepted'] and accepted['audit']==old['acceptance']['audit'] and accepted['validation']==old['acceptance']['validation']
   bad=copy.deepcopy(request);bad['config']['model']['ffn_input_mixing']='causal3' if old['arm']=='none' else 'none'
   rejects(lambda:a.verify_request(bad))
   rows.append(dict(arm=old['arm'],audit=accepted['audit'],old_acceptance=old['accepted'],new_source_sha256=accepted['source_sha256'],validation_exact=True))
  from research.local_mix_v1 import inference as inf
  bindings={row['arm']:row['binding'] for row in prior['arms']}
  measured=inf.measure_pair(bindings,0);reviewed=inf.review(measured,bindings,0)
  exclusive(B/'pilot-integration001-cache.json',measured)
  assert c.source_inventory()==sources and not p.ROOT.exists() and not p.FREEZE.exists()
  result=dict(status='finite_pilot_config_and_parent_bridge_passed',runner=reference(__file__),plan=reference(B/'pilot-integration001-plan.json'),sources=sources,configs=configs,checks=checks,parent_reaudits=rows,
    cache_review=reviewed,cache_measurements=reference(B/'pilot-integration001-cache.json'),exact_capacity_boundaries_passed=True,module_cli_entry_passed=True,scientific_training=False,scientific_seeds_consumed=False,holdout_scored=False,pilot_frozen=False,quality_frozen=False,
    quality_design=p.QUALITY_DESIGN,limitations='Read-only existing generated worker bundle re-audit and prospective config/capacity tests; no resource pilot dispatched.')
 result['guard']=reference(B/'pilot-integration001-guard.json');assert read(result['guard'])['status']=='finished'
 exclusive(B/'pilot-integration001.json',result);print(json.dumps(dict(status=result['status'],parent_reaudits=rows)),flush=True)
