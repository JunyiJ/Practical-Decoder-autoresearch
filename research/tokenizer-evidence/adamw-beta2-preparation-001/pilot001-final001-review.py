"""Independent source/artifact/resource review of completed fixed-LR adamw-beta2 resource pilots."""
import json,shutil
from omegaconf import OmegaConf
from pathlib import Path
from research.adamw_beta2_campaign_v1 import pilot as p
from research.block_norm_v1.fixture_dispatch import read,reference
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
B=Path(__file__).resolve().parent
with session_lock(GLOBAL_LOCK.parent):
 p.idle()
 binding=read(reference(B/'pilot001-final001-plan.json'))
 assert binding['runner']==reference(__file__)
 for n,h in binding['sources'].items():assert p.sha(p.c.ROOT/n)==h
 plan=p.bind();review=read(reference(B/'pilot001-review.json'))
 assert review['status']=='both_resource_pilots_and_inference_passed' and not review['quality_launch_permitted']
 with CpuPrerequisiteGuard(B/'pilot001-final001-guard.json',p.c.ROOT,timeout_seconds=120):
  rows=[]
  for slot,arm in enumerate(p.ARMS):
   root=p.ROOT/f'evidence/pilots/{slot:02d}';accepted=read(reference(root/'accepted.json'))
   assert accepted['slot']==slot and accepted['arm']==arm and accepted['plan']==p.ref(B/'pilot001-plan.json')
   request=accepted['acceptance_request'];assert request['config']==plan['configs'][slot] and request['sources']==plan['sources']
   assert accepted['acceptance']['accepted'] and {k:v for k,v in accepted['acceptance']['audit'].items() if k not in ('clipping','optimizer_groups')}==dict(completed_steps=20,tokens_processed=20480,ema_parameters=88,ema_snapshots=8)
   assert accepted['acceptance']['validation']==accepted['validation']
   assert accepted['acceptance']['audit']['optimizer_groups']==dict(beta1=.9,beta2=p.BETAS[arm],epsilon=1e-8,adamw_groups=44,muon_groups=44)
   for item in request['files'].values():assert p.ref(item['path'])==item,'Artifact changed'
   summary=json.loads(Path(request['files']['summary']['path']).read_text())
   clipping=p.a.audit_clipping_history(OmegaConf.create(request['config']),summary['clipping_observations'])
   assert clipping==accepted['acceptance']['audit']['clipping']
   metrics=[json.loads(s) for s in Path(request['files']['metrics']['path']).read_text().splitlines()]
   assert [r['step'] for r in metrics]==[0,19]
   assert metrics[0]['lr_multiplier']==1.
   assert all(m['base_learning_rate']==.00125 for m in metrics)
   assert request['config']['training']['weight_decay']==p.DECAYS[arm]
   assert request['config']['training']['adamw_beta2']==p.BETAS[arm]
   assert request['config']['training']['lr_schedule']['start_step']==p.PILOT_STARTS[arm]
   assert request['config']['training']['weight_averaging']['start_step']==13
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
   projected=120*(process['elapsed_seconds']+guards[1]['elapsed_seconds'])
   assert projected<7200
   rows.append(dict(slot=slot,arm=arm,clipping=clipping,accepted=reference(root/'accepted.json'),actual_weight_sizes=sizes,
    targets=20480,h200_snapshots=8,projected_1200step_seconds_with_2x_headroom=projected,
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
   scope='Both real20-update beta999/beta95 pilots with cooldown13 completed with8 H200 snapshots. No pilot loss used for selection or adoption;1200-step quality behavior unmeasured.',
   next_action='Prepare finite quality orchestration outside frozen namespaces; verify accepted real pilot payloads/cache, close dependencies and audit exact successful-pilot retirement before first fresh trajectory. Recheck capacity and unused fresh seeds before final quality freeze.',
   quality_launch_permitted=False,quality_adopted=False,scientific_fresh_seed_consumed=False,holdout_scored=False)
 result['guard']=reference(B/'pilot001-final001-guard.json');assert read(result['guard'])['status']=='finished'
 p.exclusive(B/'pilot001-final001-review.json',result)
 print(json.dumps({k:result[k] for k in ('status','retained_pilot_bytes','free_bytes','conditional_free_after_exact_pilot_retirement')}))
