"""Independent source/artifact/resource review of completed warmup resource pilots."""
import json,shutil
from pathlib import Path
from research.warmup_campaign_v1 import pilot as p
from research.block_norm_v1.fixture_dispatch import read,reference
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
B=Path(__file__).resolve().parent
with session_lock(GLOBAL_LOCK.parent):
 p.idle()
 binding=read(reference(B/'pilot001-final002-plan.json'))
 for n,h in binding['sources'].items():assert p.sha(p.c.ROOT/n)==h
 plan=p.bind();review=read(reference(B/'pilot001-review.json'))
 assert review['status']=='both_resource_pilots_and_inference_passed' and not review['quality_launch_permitted']
 with CpuPrerequisiteGuard(B/'pilot001-final002-guard.json',p.c.ROOT,timeout_seconds=120):
  rows=[]
  for slot,arm in enumerate(p.ARMS):
   root=p.ROOT/f'evidence/pilots/{slot:02d}';accepted=read(reference(root/'accepted.json'))
   assert accepted['slot']==slot and accepted['arm']==arm and accepted['plan']==p.ref(B/'pilot001-plan.json')
   request=accepted['acceptance_request'];assert request['config']==plan['configs'][slot] and request['sources']==plan['sources']
   assert accepted['acceptance']['accepted'] and accepted['acceptance']['audit']==dict(completed_steps=120,tokens_processed=122880,ema_parameters=88,ema_snapshots=21)
   assert accepted['acceptance']['validation']==accepted['validation']
   for item in request['files'].values():assert p.ref(item['path'])==item,'Artifact changed'
   metrics=[json.loads(s) for s in Path(request['files']['metrics']['path']).read_text().splitlines()]
   assert [r['step'] for r in metrics]==[0,20,40,60,80,100,119]
   assert metrics[0]['lr_multiplier']==(1. if arm=='none' else .1)
   assert metrics[-2]['lr_multiplier']==1. and metrics[-1]['lr_multiplier']==.1
   process=read(accepted['process']);assert process['status']=='completed' and process['returncode']==0 and process['error'] is None
   guards=[read(reference(root/'worker-guard.json')),read(accepted['parent_guard'])]
   for guard in guards:
    assert guard['status']=='finished' and guard['reason'] is None
    assert guard['elapsed_seconds']<guard['limits']['timeout_seconds']<=600
    assert guard['observation']['peak_rss_bytes']<4*1024**3 and guard['observation']['peak_observed_mps_driver_bytes']<6*1024**3
    assert guard['observation']['free_bytes']>=8*1024**3
   sizes={k:Path(request['files'][k]['path']).stat().st_size for k in ('parent','h200')}
   assert sizes['parent']<=p.RAW_CAP and sizes['h200']<=p.EXPORT_CAP
   # Descriptive conservative projection, not a replacement for the full-trial watchdog.
   projected=20*(process['elapsed_seconds']+guards[1]['elapsed_seconds'])
   assert projected<7200
   rows.append(dict(slot=slot,arm=arm,accepted=reference(root/'accepted.json'),actual_weight_sizes=sizes,
    targets=122880,h200_snapshots=21,projected_1200step_seconds_with_2x_headroom=projected,
    worker_seconds=process['elapsed_seconds'],peak_rss_bytes=max(g['observation']['peak_rss_bytes'] for g in guards),
    peak_mps_driver_bytes=max(g['observation']['peak_observed_mps_driver_bytes'] for g in guards)))
  measured=read(review['inference']);checked=p.inf.review(measured,{row['arm']:{k:read(reference(p.ROOT/f'evidence/pilots/{row["slot"]:02d}/accepted.json'))['acceptance_request']['files'][k] for k in ('config','parent','h200')} for row in rows},0)
  assert checked==review['review'] and len(checked['cells'])==4
  for name in ('pilot001-review-guard.json','pilot001-inference-guard.json'):
   guard=read(reference(B/name));assert guard['status']=='finished' and guard['reason'] is None
  assert p.bind()==plan
  retained=sum(sum(r['actual_weight_sizes'].values()) for r in rows);free=shutil.disk_usage(p.c.ROOT).free
  assert free+retained>=p.REQUIRED
  result=dict(status='resource_pilots_independently_reviewed',runner=reference(__file__),plan=reference(B/'pilot001-plan.json'),review=reference(B/'pilot001-review.json'),sources=plan['sources'],
   rows=rows,retained_pilot_bytes=retained,free_bytes=free,conditional_free_after_exact_pilot_retirement=free+retained,required_free=p.REQUIRED,
   scope='Both real120-update pilots exercised the100-update warmup/none schedules. No pilot loss used for selection or adoption;1200-step quality behavior unmeasured.',
   next_action='Prepare finite quality orchestration outside frozen namespaces; verify accepted real pilot payloads/cache, close dependencies and audit exact successful-pilot retirement before first screen. Recheck capacity and unused screen/fresh seeds before final quality freeze.',
   quality_launch_permitted=False,quality_adopted=False,scientific_screen_or_fresh_seed_consumed=False,holdout_scored=False)
 result['guard']=reference(B/'pilot001-final002-guard.json');assert read(result['guard'])['status']=='finished'
 p.exclusive(B/'pilot001-final002-review.json',result)
 print(json.dumps({k:result[k] for k in ('status','retained_pilot_bytes','free_bytes','conditional_free_after_exact_pilot_retirement')}))
