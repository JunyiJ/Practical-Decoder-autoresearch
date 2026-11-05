"""Review prepared finite launch/final sources after additive terminal verification.

No quality freeze, worker, tensor restoration, pilot retry or scientific bootstrap.
"""
import ast,copy,json,shutil
from pathlib import Path
import numpy as np,torch
from research.final_ln_scale_quality_v2 import campaign_contract as c,quality_execution as q
from research.final_ln_scale_quality_v2.pilot_retirement import verify_retirement,DEST
from research.final_ln_scale_quality_v1 import campaign_rules as r
from research.final_ln_scale_quality_v2.support import idle,exclusive
from research.block_norm_v1.fixture_dispatch import read,reference
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_metadata import budgets,snapshot
B=q.EVIDENCE;torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
 idle()
 with CpuPrerequisiteGuard(B/'quality-source-review002-guard.json',c.ROOT,timeout_seconds=120):
  sources=c.source_inventory();assert len(sources)==912;prior=read(reference(B/'pilot002-plan.json'));seeds=prior['seeds'];retired=reference(DEST/'completed.json');done=verify_retirement(retired)
  assert not q.ROOT.exists() and not q.FREEZE.exists()
  names=['quality001-freeze-build001.py','quality001-fresh-inference001.py','quality001-final-review001.py'];refs={n:reference(B/n) for n in names}
  for n in names:ast.parse((B/n).read_text())
  builder=(B/names[0]).read_text();assert builder.index("assert free>=q.FULL_FREE")<builder.index('q.ROOT.mkdir()')
  assert 'pilot002-final003-review.json' in builder and 'verification002.json' in builder and "assert sources==checked['sources']" not in builder
  configs=[]
  for spec in r.fresh_specs(seeds):configs.append(c.trial_config(q.ROOT/f"evidence/quality/{spec['slot']:02d}",spec,seeds))
  assert len(configs)==6 and [x['training']['seed'] for x in configs]==[s for s in seeds['fresh'] for _ in range(2)]
  active={k:v for k,v in budgets().items() if not k.startswith(('evidence/pilots/','evidence/quality/','evidence/.retention/')) or any(k.startswith(f'evidence/quality/{i:02d}/') for i in range(6))}
  assert sum(active.values())+32*1024**2<=256*1024**2
  # Existing full-horizon fixture is already closed: inspect its retained
  # allocation only, never republish or rerun it. Additive912 adds<2KiB sources.
  meta=snapshot(c.ROOT/'research_runs/tokenizer-final-ln-scale-quality-v1-disposable001-full-horizon')
  added_source_bytes=len(c.canonical(sources))-len(c.canonical(c.predecessor.source_inventory()))
  assert 0<added_source_bytes<2048
  for group in ('resources','events'):assert meta['groups']['evidence/quality/00/'+group]+added_source_bytes<2*1024**2
  # Verify the future inference source resolves compact receipts correctly.
  tree=ast.parse((B/'quality001-fresh-inference001.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='requests');ns={'q':q};exec(compile(ast.Module(body=[fn],type_ignores=[]),'prepared_requests','exec'),ns)
  fixture=B/'quality-source-review002-fixture';fixture.mkdir();receipts=[];expected=[]
  for spec in r.fresh_specs(seeds):
   req=dict(sources=sources,trial=dict(spec=spec,seeds=seeds),files={role:dict(path=f'synthetic-unexecuted-slot-{spec["slot"]}/{role}',sha256='0'*64) for role in ('config','parent','h200')})
   p=fixture/f'worker-{spec["slot"]:02d}.json';exclusive(p,dict(kind=q.KIND,acceptance_request=req,holdout_scored=False));receipts.append(dict(worker=reference(p),request_sha256=c.fingerprint(req)));expected.append(req)
  for pair in range(3):
   mapped=ns['requests'](dict(receipts=receipts),pair,dict(sources=sources))
   assert mapped=={req['trial']['spec']['arm']:req['files'] for req in expected[2*pair:2*pair+2]}
  bad=copy.deepcopy(receipts[0]);bad['request_sha256']='0'*64
  try:q.receipt_request(bad,dict(sources=sources))
  except ValueError:pass
  else:raise AssertionError('Foreign compact receipt accepted')
  # Independent nonconstant synthetic bootstrap oracle, with separate seeds.
  synthetic=dict(pilot=92001,fresh=[92011,92012,92013],evaluation=92003,bootstrap=92004)
  counts=np.array([2048]*16+[31]);control=np.tile(counts*4.,(3,1));deltas=np.array([-.035,-.045,-.055])[:,None]+np.linspace(-.013,.013,len(counts))[None,:]
  loss=np.stack([control,control+deltas*counts],axis=1);stats=r.paired_statistics(synthetic,loss,counts)
  independent={}
  for size in r.BLOCKS:
   factor=size//2048;totals=np.array([np.mean(loss[:,1,j:j+factor].sum(1)-loss[:,0,j:j+factor].sum(1)) for j in range(0,len(counts),factor)]);weights=np.array([counts[j:j+factor].sum() for j in range(0,len(counts),factor)])
   rng=np.random.Generator(np.random.PCG64(synthetic['bootstrap']));draws=[]
   for _ in range(2000):
    ix=rng.integers(0,len(weights),size=len(weights));draws.append(totals[ix].sum()/weights[ix].sum())
   got=np.quantile(draws,[.025,.975,1-.05/3],method='linear').tolist();want=stats['bounds'][str(size)]
   assert max(abs(a-b) for a,b in zip(got,want['descriptive_95pct']+[want['gate_upper']]))<1e-12;independent[str(size)]=got
  examined=[]
  for path in (c.ROOT/'research_runs').rglob('config.json'):
   cfg=json.loads(path.read_text());assert cfg.get('training',{}).get('seed') not in set(seeds['fresh']),'Reserved fresh seed used'
   examined.append(reference(path))
  c.verify_sources(sources);free=shutil.disk_usage(c.ROOT).free
  plan=dict(status='prepared_quality_source_review',sources=sources,runner=reference(Path(__file__).resolve()),prepared_sources=refs,configs=configs,seeds=seeds,prior_resource_plan=reference(B/'pilot002-plan.json'),retirement=retired,required_free_bytes=q.FULL_FREE,cleanup_target_bytes=q.FULL_FREE+2*1024**3)
  exclusive(B/'quality-source-review002-plan.json',plan)
  value=dict(status='passed_quality_source_ready_capacity_blocked' if free<q.FULL_FREE else 'passed_quality_source_ready_capacity_wake_eligible',plan=reference(B/'quality-source-review002-plan.json'),source_count=912,predecessors_preserved=906,original_numerical_contracts_preserved=898,prepared_sources=refs,finite_trajectories=6,exact_three_fresh_pairs=True,fixed_common_lr=.00125,lr_search_budget=0,compact_receipt_inference_mapping_passed=True,source_bridge_bytes=added_source_bytes,full_horizon_metadata_groups=meta['groups'],complete_active_group_budget=sum(active.values()),external_metadata_cap=32*1024**2,independent_synthetic_bootstrap_quantiles=independent,scientific_bootstrap_consumed=False,reserved_fresh_seeds_unused=True,checked_config_count=len(examined),checked_configs=examined,quality_frozen=False,quality_training_launched=False,retained_pilot_tensor_bytes=0,completed_retirement=retired,free_bytes=free,required_free_bytes=q.FULL_FREE,remaining_requirement_deficit=max(0,q.FULL_FREE-free),cleanup_target_bytes=q.FULL_FREE+2*1024**3,remaining_cleanup_target_deficit=max(0,q.FULL_FREE+2*1024**3-free),wake='actual local free >=12381585408B; recover toward14529069056B with any newly eligible action before dispatch; or human-supplied external APFS volume>=6GiB for verified protected-archive relocation. All known current recovery classes exhausted; no repeated broad scans for unchanged blocker. Recheck live complete capacity and fresh/bootstrap nonuse before executing prepared freeze builder.',next_action='On capacity wake, verify912 bindings/prepared source refs and strict environment, then execute NEW prepared quality001-freeze-build001.py once. Bind all six trajectories and actual three-pair final review before one serial worker. If capacity remains insufficient, hourly short free-space check only; no numerical launch/records/commits/notifications absent relevant change.',holdout_scored=False)
 exclusive(B/'quality-source-review002.json',value);print(json.dumps({k:value[k] for k in ('status','source_count','free_bytes','remaining_requirement_deficit','remaining_cleanup_target_deficit','checked_config_count')}),flush=True)
