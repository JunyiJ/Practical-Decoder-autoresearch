"""Read-only closure of the complete input-feature dropout study; no replay."""
import json,hashlib,math,statistics,shutil
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.embedding_input_dropout_worker_v2.execution import sources,checked
from research.embedding_input_dropout_campaign_v1.pilot import campaign_sources
from research.embedding_input_dropout_quality_v1.contract import quality_sources,select
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.final_ln_scale_quality_v3.support import idle
B=Path(__file__).resolve().parent;RUN=ROOT/'research_runs/tokenizer-embedding-input-dropout-quality-v1-001'
def artifact(v):
 p=Path(v['path']);s=p.stat();require(p.is_file() and not p.is_symlink() and s.st_nlink==1 and s.st_size==v['size'],'Unique retained payload required')
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 t=p.stat();require(h==v['sha256'] and (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns),'Retained payload changed');return v

def publish(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
with session_lock(GLOBAL_LOCK.parent):
 idle();require(not(B/'quality001-closure001.json').exists(),'New read-only closure only')
 with CpuPrerequisiteGuard(B/'quality001-closure001-guard.json',ROOT,timeout_seconds=300,poll_seconds=.25):
  plan=checked(reference(B/'quality001-plan.json'));seal=checked(reference(B/'quality001-freeze.json'));confirm=checked(reference(B/'quality001-confirmation-plan.json'));selection=checked(reference(B/'quality001-selection.json'));report=checked(reference(B/'quality001-final-review.json'));stop=checked(reference(RUN/'STOP'))
  require(seal['plan']==reference(B/'quality001-plan.json') and report['freeze']==reference(B/'quality001-freeze.json') and report['selection']==reference(B/'quality001-selection.json') and report['confirmation']==reference(B/'quality001-confirmation-plan.json'),'Original binding chain required')
  require(stop['review']==reference(B/'quality001-final-review.json') and stop['never_rerun_training_inference_review'] is True,'Original terminal STOP required')
  require(plan['sources']==sources() and plan['pilot_sources']==campaign_sources() and plan['quality_sources']==quality_sources(),'Frozen sources changed');pins={**plan['sources'],**plan['pilot_sources'],**plan['quality_sources']};require(len(pins)==972,'Original972bindings required')
  require(report['status']=='complete_fresh_panel_reviewed' and checked(reference(B/'quality001-final-review-guard.json'))['status']=='finished','Original complete final review required')
  retirement=checked(reference(B/'quality001-loser-retirement001/completed.json'));manifest=checked(retirement['manifest']);require(retirement['deleted_files']==8 and retirement['all_selected_fresh_original_evidence_unchanged'],'Exact losing-screen closure required')
  selected={r['selected_slot'] for r in selection['audits'].values()};protected=[];rows=[];scores={a:[] for a in ('input_zero','input_feature_p10')};seconds={a:[] for a in scores};screens=[]
  for slot in range(12):
   anchor=checked(reference(B/f'quality001-accepted-{slot:02d}.json'));a=checked(anchor['accepted']);w=checked(a['worker']);expected=plan['specs'][slot] if slot<6 else confirm['specs'][slot-6]
   require(anchor['original_freeze']==reference(B/'quality001-freeze.json') and w['config']==expected['config'] and a['arm']==expected['arm'] and w['request']==a['request'],'Original configuration/request differs')
   ac=a['acceptance'];require(ac['status']=='accepted' and ac['step']==1200 and ac['targets']==1228800 and ac['ema_snapshots']==41 and ac['parameter_tensors']==88 and ac['complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked'] is True,'Complete original endpoint required')
   require(ac['holdout_scored'] is False,'Closed holdout required');process=checked(a['process']);require(process['status']=='completed' and process['returncode']==0,'Clean worker completion required')
   for name in ['parent_guard','worker_guard']:
    g=checked(a[name]);require(g['status']=='finished' and g['reason'] is None and g['limits']==dict(timeout_seconds=7200,rss_limit_bytes=4*1024**3,mps_driver_limit_bytes=6*1024**3,reserve_bytes=8*1024**3),'Original guard contract differs')
   kept=slot in selected or slot>=6
   for name in ['parent','h200']:
    v=w[name]
    if kept:protected.append(artifact(v))
    else:require(not Path(v['path']).exists() and any(r['path']==v['path'] and r['sha256']==v['sha256'] and r['size']==v['size'] for r in manifest['delete']),'Missing loser lacks exact retirement')
   row=dict(slot=slot,arm=a['arm'],seed=w['config']['training']['seed'],lr=w['config']['training']['learning_rate'],accepted=anchor['accepted'],worker=a['worker'],process=a['process'],ce=ac['full_validation']['ce'],training_seconds=w['training_seconds'],process_seconds=process['elapsed_seconds'],clipped_updates=ac['clipped_updates'],retained_weights=kept,metrics=w['metrics'],final_online_train_ce=w['final_online_train_ce']);rows.append(row)
   if slot<6:screens.append(dict(slot=slot,arm=a['arm'],lr=row['lr'],ce=row['ce']))
   else:scores[a['arm']].append(row['ce']);seconds[a['arm']].append(row['training_seconds'])
  choices,audits=select(screens);require(choices==selection['choices'] and audits==selection['audits'],'Original minimum/tie selection differs')
  require(len(protected)==16 and len({v['path'] for v in protected})==16,'Two selected and six fresh RAW/H200 pairs required')
  actual=[]
  for seed in plan['seeds']['fresh']:
   group={r['arm']:r for r in rows if r['slot']>=6 and r['seed']==seed};require(set(group)==set(scores),'Complete fresh pair required');actual.append(group['input_feature_p10']['ce']-group['input_zero']['ce'])
  contrast=report['statistics']['contrasts']['input_zero'];require(actual==contrast['deltas'] and abs(statistics.mean(actual)-contrast['mean_delta'])<1e-12,'Independent paired arithmetic differs')
  gates=dict(mean_below_minus_point02=statistics.mean(actual)<-.02,every_seed_negative=all(v<0 for v in actual),all_simultaneous_bounds_negative=all(v['upper']<0 for v in contrast['conditional_text_upper_bounds']));require(gates==contrast['gates'] and all(gates.values())==report['statistics']['all_strict_gates_pass'],'Original scientific gates differ')
  inference=checked(report['inference']);require(inference['all_fresh_pairs'] and len(inference['rows'])==6 and inference['selection'] is False and inference['holdout_scored'] is False,'All fresh serving review required')
  latency={}
  for batch in (1,8):
   full=[];decode=[]
   for seed in plan['seeds']['fresh']:
    group={r['arm']:r for r in inference['rows'] if r['seed']==seed};require(set(group)==set(scores),'All inference arms required');measure={}
    for arm in group:
     v=next(x for x in group[arm]['measurements'] if x['batch']==batch);require(v['cache_bytes_per_item']==196608 and len(v['full128_seconds'])==len(v['decode32_after_prefill96_seconds'])==5 and all(math.isfinite(x) and x>0 for x in v['full128_seconds']+v['decode32_after_prefill96_seconds']),'Original timing/cache scope differs');measure[arm]=v
    full.append(measure['input_feature_p10']['median_full128_seconds']/measure['input_zero']['median_full128_seconds']);decode.append(measure['input_feature_p10']['median_decode32_seconds']/measure['input_zero']['median_decode32_seconds'])
   latency[str(batch)]=dict(median_paired_full128_ratio=statistics.median(full),median_paired_decode32_ratio=statistics.median(decode))
  result=dict(status='CLOSED_NOT_ADOPTED' if not all(gates.values()) else 'CLOSED_GATE_PASSED_PENDING_SEPARATE_ADOPTION',source=reference(__file__),original_final_review=reference(B/'quality001-final-review.json'),original_stop=reference(RUN/'STOP'),
   sources=plan['sources'],pilot_sources=plan['pilot_sources'],quality_sources=plan['quality_sources'],all_972_bindings_unchanged=True,selection=choices,statistics=report['statistics'],fresh_h200_ce=scores,mean_h200_ce={a:statistics.mean(v) for a,v in scores.items()},fresh_mean_training_seconds={a:statistics.mean(v) for a,v in seconds.items()},median_paired_inference_ratios=latency,
   inference_scope='One resident model;2warm+5timed repetitions;FP32MPS;loading/cache allocation excluded. Descriptive,not a quality or cost-only adoption gate.',trajectories=rows,scientific_trajectories=12,fresh_trajectories=6,all_original_guards_passed=True,protected_original_payloads=protected,protected_weight_bytes=sum(v['size'] for v in protected),
   all_quality_seeds_used=True,holdout_scored=False,worker_active=False,free_bytes=shutil.disk_usage(ROOT).free,decision='Keep incumbent and input dropout0;do not change gates or run rescue/retry.',next_action='Continue evidence-grounded improvement: review local curves/cost/frequency evidence and primary input-representation papers,then perform a bounded train-only representation-capacity analysis before admitting another structural panel. No automatic dropout/beta1/horizon search.')
  publish(B/'quality001-closure001.json',result)
 print(json.dumps({k:v for k,v in result.items() if k in ['status','mean_h200_ce','fresh_mean_training_seconds','median_paired_inference_ratios','protected_weight_bytes','free_bytes']}))
