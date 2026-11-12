"""Freeze and admit six native continuation runs before scientific scores."""
import datetime,fcntl,hashlib,json,math,shutil,subprocess,statistics
from pathlib import Path
from research.general_rsi_pseudo_worker_v1 import put
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-pseudo-repetition-001'
OLD=ROOT/'research/tokenizer-evidence/general-rsi-native-horizon-001'
Q=ROOT/'research/tokenizer-evidence/general-rsi-pseudo-integration-001'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not OUT.exists()
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  state=read(ROOT/'research_state/general-rsi-state.json');assert not state.get('stop_requested') and state['active_worker_pid'] is None
  assert read(Q/'completion-review.json')['accepted_engineering'] and read(Q/'result.json')['qualified']
  assert read(Q/'guard.json')['status']=='finished'
  assert read(ROOT/'research/tokenizer-evidence/general-rsi-capacity-008/completed.json')['remaining_deficit_bytes']==0
  scan=subprocess.run(['rg','-l','6211120523','research','research_state','research_runs','--glob','*.json','--glob','*.jsonl'],cwd=ROOT,capture_output=True,text=True)
  assert scan.returncode==1 and not scan.stdout and not scan.stderr
  check=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','research.general_rsi_pseudo_binding_check_v1'],cwd=ROOT,capture_output=True,text=True,check=True)
  old=read(OLD/'plan.json');sources=dict(read(Q/'plan.json')['sources'])
  for path,digest in sources.items():assert sha(ROOT/path)==digest,path
  for name in ['general_rsi_pseudo_worker_v1.py','general_rsi_pseudo_review_v1.py','general_rsi_pseudo_analysis_v1.py','general_rsi_pseudo_binding_check_v1.py','general_rsi_pseudo_prepare_v1.py','general_rsi_equal_time_analysis_v1.py','GENERAL_RSI_PSEUDO_REPETITION_PROTOCOL.md','GENERAL_RSI_PSEUDO_REPETITION_NEXT.md']:
   sources['research/'+name]=sha(ROOT/'research'/name)
  evidence={}
  for p in [Q/'completion-review.json',Q/'result.json',Q/'plan.json',Q/'guard.json',ROOT/'research/tokenizer-evidence/general-rsi-pseudo-spans-001/inventory.json',ROOT/'research/tokenizer-evidence/general-rsi-pseudo-spans-001/inventory.npz',ROOT/'research/tokenizer-evidence/general-rsi-pseudo-spans-001/component-attempt-01/result.json']:
   evidence[str(p.relative_to(ROOT))]=sha(p)
  sources.update(evidence)
  slots=[];parents=[];names=['native_initializer','composer_initializer','train_sampler','validation_sampler']
  for bundle,parent_slot in enumerate([1,3,5],1):
   d=read(OLD/f'slot-{parent_slot:02}/generation/result.json');parent=dict(d['checkpoint'],slot=parent_slot,binding=sha(OLD/'plan.json'))
   assert Path(parent['path']).stat().st_size==parent['size'] and sha(parent['path'])==parent['sha256'];parents.append(parent)
   roles=dict(zip(names,range(62111205230001+100*bundle,62111205230005+100*bundle)))
   for recipe in ['native_ce','native_pseudo_repetition']:
    slots.append(dict(slot=len(slots),phase='confirmation',arm='native',recipe=recipe,bundle=bundle,parent=parent,roles=roles,pseudo_seed=62111205230005+100*bundle,lr=.00025,budget_ns=1200*10**9,timeout_seconds=2400))
  q=[read(Q/f'{arm}-result.json') for arm in ['native_ce','native_pseudo_repetition']]
  measured={d['arm']:dict(mean_update_seconds=d['training_ns']/1e9/d['steps'],restart_bytes=d['checkpoint']['size']) for d in q}
  historical=[read(OLD/f'slot-{i:02}/result.json') for i in [1,3,5]]
  score=max(e['score_seconds'] for r in historical for e in r['endpoints']);generation=max(e['generation_seconds'] for r in historical for e in r['endpoints'])
  updates=sum(3600/x['mean_update_seconds'] for x in measured.values());checkpoints=6*(math.ceil(1200/120)+2)
  costs=dict(training_seconds=7200,endpoint_and_conservative_probe_seconds=24*score,generation_seconds=6*generation,checkpoint_seconds=checkpoints*3,initialization_seconds=6*30,outside_clock_logging_seconds=updates*.01)
  total=sum(costs.values());assert total<14400
  outputs=12*128*2**20;overlap=2*512*2**20;metadata=256*2**20;failed=5*2**30;reserve=10*2**30;incremental=outputs+overlap+metadata+failed;free=shutil.disk_usage(ROOT).free
  assert free>=incremental+reserve+2*2**30
  assert max(x['restart_bytes'] for x in measured.values())<512*2**20
  forecast=dict(measured=measured,measured_full_validation_seconds=score,measured_generation_seconds=generation,measured_final_weight_bytes=max(p['size'] for p in parents),final_weight_count=12,final_output_cap_bytes=outputs,active_restart_serialization_overlap_bytes=overlap,metadata_bytes=metadata,failed_write_allowance_bytes=failed,incremental_peak_bytes=incremental,reserve_bytes=reserve,required_free_bytes=incremental+reserve,extra_headroom_bytes=2*2**30,actual_free_bytes=free,predicted_updates=updates,checkpoint_count_allowance=checkpoints,cost_components_seconds=costs,estimated_seconds=total,tranche_seconds=14400,failure_and_variability_allowance_seconds=14400-total,
   assumptions='All retained payloads already deducted from actual free space.12 endpoint caps enforced128MiB; active restart and write overlap512MiB each. New measured restart445033525bytes leaves >91MB for longer history. Metadata256MiB covers steps, all generations, receipts, manifests. Checkpoint3s versus old native measured<2.1s mean; initialization30s and logging.01s/update are allowances. Two probes charged as full validation. All attempts share cumulative budgets.')
  plan=dict(kind='general_rsi_pseudo_repetition_panel_v1',sources=sources,qualification_evidence=evidence,torch_version=old['torch_version'],slots=slots,seed_role_audit='No prior6211120523 matches in recorded JSON/JSONL; paired roles62111205230101..105,62111205230201..205,62111205230301..305; bootstrap62111205230999.',ema_half_life_ns=old['ema_half_life_ns'],ema_rule=old['ema_rule'],checkpoint_interval_ns=120*10**9,probe_documents=old['probe_documents'],generation_protocol=old['generation_protocol'],acceptance=dict(endpoint='time_ema',candidate='native_pseudo_repetition',baseline='native_ce',minimum_mean_gain=.02,source_regression_max=.05,bootstrap_seed=62111205230999,replicates=10000,upper_quantile=.99,method='Average paired document loss sums then PCG64 resample108 IDs with replacement; resampled target denominator; linear99th percentile.',otherwise='inconclusive; no equivalence claim; no generation rescue'),max_attempts_per_slot=2,tranche_seconds=14400,rss_limit_bytes=4*2**30,mps_limit_bytes=6*2**30,reserve_bytes=reserve,incremental_peak_bytes=incremental,forecast=forecast,test_opened=False)
  plan['generation_protocol']['scope']='pseudo_repetition_continuation; unchanged prompts/decoders/rubric; retain144 first outputs'
  OUT.mkdir();put(OUT/'adaptation-review.json',dict(qualified=True,method='AST exact comparison after enumerated binding changes',output=check.stdout,check_source_sha256=sha(ROOT/'research/general_rsi_pseudo_binding_check_v1.py'),new_numerical_prerequisites_repeated=False))
  plan['sources'][str((OUT/'adaptation-review.json').relative_to(ROOT))]=sha(OUT/'adaptation-review.json')
  put(OUT/'plan.json',plan);put(OUT/'admission.json',dict(plan_sha256=sha(OUT/'plan.json'),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),scientific_scores_seen=False,shared_lock_held=True,forecast=forecast,all_six_runs_admitted=True))
  print(json.dumps(dict(binding=sha(OUT/'plan.json'),forecast=forecast),indent=2))
if __name__=='__main__':main()
