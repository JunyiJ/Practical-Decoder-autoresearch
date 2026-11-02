"""Read-only actual payload bridge plus finite pilot configuration/resource gates."""
import copy,hashlib,json,subprocess,sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.parallel_residual_campaign_v1 import artifact_acceptance as a,evidence_contract as c
import importlib.util
_entry=Path(__file__).resolve().parent/'pilot002.py'
_spec=importlib.util.spec_from_file_location('parallel_resource_pilot002',_entry)
p=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(p)
from research.parallel_residual_worker_v2 import artifact_acceptance as old_a
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
 plan=read(reference(B/'pilot-integration002-plan.json'))
 for n,h in plan['sources'].items():assert p.sha(c.ROOT/n)==h
 sources=c.source_inventory();prior=read(reference(B/'trainer-worker003.json'))
 assert len(sources)>len(prior['sources']) and all(sources[n]==h for n,h in prior['sources'].items())
 with p.Guard(B/'pilot-integration002-guard.json'):
  seeds=dict(pilot=91111,fresh=[91121,91122,91123],evaluation=91113,bootstrap=91114);data=p.data_contract.bind_data();configs=[];checks=[]
  for slot,arm in enumerate(p.ARMS):
   cfg=p.config(slot,seeds);model=p.model_contract(cfg['model']);t=OmegaConf.create(cfg).training
   assert p.identity(OmegaConf.create(cfg))['residual']==p.TOPOLOGIES[arm] and model['score_scale']==.5
   assert t.learning_rate==.00125 and t.weight_decay==.00006
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
   del cfg['model']['residual_topology_kind']
   for k in ('checkpoint_dir','metrics_file','summary_file'):del cfg['training'][k]
  assert shared[0]==shared[1]
  for invalid in (True,-1,2,'0'):rejects(lambda:p.config(invalid,seeds))
  p.require_capacity(p.REQUIRED)
  rejects(lambda:p.require_capacity(p.REQUIRED-1))
  p.require_capacity(p.REQUIRED-p.WEIGHT_CAP,p.WEIGHT_CAP)
  rejects(lambda:p.require_capacity(p.REQUIRED-2*p.WEIGHT_CAP-1,2*p.WEIGHT_CAP))
  rejects(lambda:p.require_capacity(p.REQUIRED,2*p.WEIGHT_CAP+1))
  rejects(lambda:p.require_capacity(True))
  assert p.REQUIRED==6*(448+112)*1024**2+256*1024**2+8*1024**3
  assert p.QUALITY_DESIGN['fresh_trajectories']==6 and p.QUALITY_DESIGN['screens']==0 and p.QUALITY_DESIGN['mean_parallel_minus_serial_strictly_below']==-.02
  assert p.QUALITY_DESIGN['selection'] is None and p.QUALITY_DESIGN['lr_search_budget_per_arm']==0 and not p.QUALITY_DESIGN['lr_rescue']
  from research.parallel_residual_campaign_v1.rules import fresh_specs,validate_seeds
  specs=fresh_specs(seeds);assert len(specs)==6
  assert [s['arm'] for s in specs]==['serial','parallel','parallel','serial','serial','parallel']
  assert [s['seed'] for s in specs]==[seed for seed in seeds['fresh'] for _ in range(2)]
  assert all(s['learning_rate']==.00125 and s['steps']==1200 and s['topology']==p.TOPOLOGIES[s['arm']] for s in specs)
  for mutation in [lambda s:s.update(screen=91112),lambda s:s['fresh'].pop(),lambda s:s['fresh'].__setitem__(0,s['pilot']),lambda s:s.update(pilot=True)]:
   bad=copy.deepcopy(seeds);mutation(bad);rejects(lambda:validate_seeds(bad))
  cli=subprocess.run([sys.executable,str(_entry),'--help'],cwd=c.ROOT,capture_output=True,text=True,timeout=20)
  assert cli.returncode==0 and '{prepare,run,worker,review}' in cli.stdout
  rows=[]
  for old in prior['arms']:
   old_request=read(old['worker'])['acceptance_request'];old_a.verify_request(old_request)
   request=copy.deepcopy(old_request);request['sources']=sources
   worker=read(old['worker']);accepted=a.accept(request,worker['validation']);p.clear()
   assert accepted['accepted'] and accepted['audit']==old['acceptance']['audit'] and accepted['validation']==old['acceptance']['validation']
   bad=copy.deepcopy(request);bad['config']['model']['residual_topology_kind']='parallel_separate_norms' if old['arm']=='serial' else 'serial'
   rejects(lambda:a.verify_request(bad))
   rows.append(dict(arm=old['arm'],audit=accepted['audit'],old_acceptance=old['accepted'],new_source_sha256=accepted['source_sha256'],validation_exact=True))
  from research.parallel_residual_worker_v2 import inference as inf
  bindings={row['arm']:row['binding'] for row in prior['arms']}
  measured=inf.measure_pair(bindings,0);reviewed=inf.review(measured,bindings,0)
  exclusive(B/'pilot-integration002-cache.json',measured)
  assert c.source_inventory()==sources and not p.ROOT.exists() and not p.FREEZE.exists()
  result=dict(status='finite_pilot_config_and_parent_bridge_passed',runner=reference(__file__),plan=reference(B/'pilot-integration002-plan.json'),sources=sources,configs=configs,checks=checks,parent_reaudits=rows,
    cache_review=reviewed,cache_measurements=reference(B/'pilot-integration002-cache.json'),exact_capacity_boundaries_passed=True,module_cli_entry_passed=True,scientific_training=False,scientific_seeds_consumed=False,holdout_scored=False,pilot_frozen=False,quality_frozen=False,
    quality_design=p.QUALITY_DESIGN,limitations='Read-only existing generated worker bundle re-audit and prospective config/capacity tests; no resource pilot dispatched.')
 result['guard']=reference(B/'pilot-integration002-guard.json');assert read(result['guard'])['status']=='finished'
 exclusive(B/'pilot-integration002.json',result);print(json.dumps(dict(status=result['status'],parent_reaudits=rows)),flush=True)
