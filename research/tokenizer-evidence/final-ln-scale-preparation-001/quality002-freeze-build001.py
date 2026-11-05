"""Prepared one-shot execution continuation of the SAME original scientific panel.

Imports exactly three accepted endpoints with zero retraining/weight copies.
Original quality001 STOP/entry failure is immutable. Never invoke below the new
measured complete remaining floor; this file is initially PREPARED/UNEXECUTED.
"""
import copy,json,shutil,subprocess,datetime
from pathlib import Path
import torch
from research.final_ln_scale_quality_v3 import quality_execution as q,campaign_contract as c
from research.final_ln_scale_quality_v3.continuation_contract import verify_imports
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import budgets
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
B=q.EVIDENCE;torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as global_fd:
 q.idle();integration=read(reference(B/'quality002-integration001.json'));checked=read(integration['plan'])
 assert integration['status']=='passed_same_panel_zero_copy_continuation_integration' and len(integration['imports'])==3 and integration['training_updates']==integration['weight_copies']==0
 sources=c.source_inventory();assert sources==checked['sources'];verify_imports(checked)
 retained=integration['imported_weight_bytes'];required=max(8*1024**3,q.FULL_FREE-retained)
 assert retained==1677774994 and required==10705268590
 # Capacity decline before this invocation creates no new fixture or STOP.
 assert shutil.disk_usage(c.ROOT).free>=required,'Prepared continuation remains blocked on complete capacity'
 assert not q.ROOT.exists() and not q.FREEZE.exists()
 with CpuPrerequisiteGuard(B/'quality002-freeze-guard.json',c.ROOT,timeout_seconds=120):
  original=read(checked['original_plan']);plan=copy.deepcopy(original)
  for item in integration['imports']:
   for binding in item['original_files']:
    assert q.file_ref(Path(binding['path']),binding['path'].endswith('.pt'))==binding
  for path in (c.ROOT/'research_runs').rglob('config.json'):
   assert json.loads(path.read_text()).get('training',{}).get('seed')!=200070013,'Final reserved fresh seed already used'
  active={k:v for k,v in budgets().items() if not k.startswith(('evidence/pilots/','evidence/quality/','evidence/.retention/')) or any(k.startswith(f'evidence/quality/{i:02d}/') for i in range(6))}
  legacy={k:v for k,v in budgets().items() if k.startswith(('evidence/meta','control','emergency')) or any(k.startswith(f'evidence/quality/{i:02d}/') for i in range(3))}
  assert sum(active.values())+sum(legacy.values())+32*1024**2<=256*1024**2
  free=shutil.disk_usage(c.ROOT).free;assert free>=required
  capacity=dict(full_floor=q.FULL_FREE,original_full_floor=12381585408,measured_entry_headroom=c.ENTRY_HEADROOM,retained_imported_weight_bytes=retained,remaining_requirement=required,free_bytes=free,margin=free-required,cleanup_target=required+2*1024**3,reserve=8*1024**3,raw_cap=448*1024**2,h200_cap=112*1024**2,per_trajectory_weight_cap=560*1024**2,per_trajectory_output_cap=576*1024**2,metadata_cap=256*1024**2,active_group_caps=active,legacy_group_caps=legacy,external_metadata_cap=32*1024**2,serialization='Original temporary-publication bounds unchanged;both roots and complete external evidence budget fit original256MiB metadata allowance. Exact imported weights credited once,no copies/hardlinks.')
  q.exclusive(B/'quality002-capacity.json',capacity)
  plan.update(kind=q.KIND,status='frozen_quality_only',root=str(q.ROOT),observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit_before_freeze=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=sources,added_sources=sorted(set(sources)-set(read(original['pilot_plan'])['sources'])),environment=c.environment(),data=c.bind_data(),limits=dict(c.LIMITS),allowances=q.ALLOWANCES,required_free_bytes=q.FULL_FREE,observed_free_bytes=free,capacity_accounting=capacity,
   original_plan=checked['original_plan'],entry_failure=checked['entry_failure'],original_STOP=checked['original_STOP'],original_root=checked['original_root'],original_anchors=checked['original_anchors'],imported_count=3,entry_headroom_bytes=c.ENTRY_HEADROOM,scientific_freeze=checked['scientific_freeze'],same_scientific_panel=True,
   prerequisites=dict(original_prerequisites=checked['original_plan'],integration=reference(B/'quality002-integration001.json'),entry_failure=checked['entry_failure'],source_review=reference(B/'quality002-source-review001.json'),capacity=reference(B/'quality002-capacity.json')),
   builder=reference(Path(__file__).resolve()),final_sources={n:reference(B/n) for n in ('quality002-fresh-inference001.py','quality002-final-review001.py')},quality_adopted=False,retention='All original3 and new3 fresh RAW/H200 pairs stay protected. Original failed run remains closed;metadata-only imports,no old fixture retry or scientific score pooling.')
  verify_imports(plan);q.ROOT.mkdir()
  with session_lock(q.ROOT) as run_fd:
   writer=MetadataWriter(q.ROOT,(global_fd,run_fd));path=writer.publish_json('evidence/meta/quality-plan.json',plan)
   with (B/'quality002-plan.json').open('xb') as f:f.write(path.read_bytes())
   q.exclusive(q.FREEZE,dict(kind=q.KIND,plan=reference(path),portable=reference(B/'quality002-plan.json'),builder=reference(Path(__file__).resolve()),same_scientific_freeze=checked['scientific_freeze']))
   freeze=reference(q.FREEZE);previous=freeze
   for item in integration['imports']:
    i=item['slot'];prior=read(item['original_acceptance']);base=f'evidence/quality/{i:02d}'
    receipt=dict(kind=q.KIND,slot=i,freeze=freeze,imported=True,original_acceptance=item['original_acceptance'],original_anchor=item['original_anchor'],worker=item['original_worker'],request_sha256=item['request_sha256'],artifact_acceptance=item['acceptance'],target_contract=item['acceptance']['validation']['target'],cache=item['cache'],process=prior['process'],selection=None,cleanup=None,adopted=False,retrained_updates=0,weight_copies=0)
    p=writer.publish_json(base+'/resource-events.jsonl',receipt)
    q.exclusive(q.anchor_path(i),dict(kind=q.KIND,slot=i,previous=previous,acceptance=reference(p),parent_guard=item['original_parent_guard'],files=item['original_files'],original_anchor=item['original_anchor']))
    previous=reference(q.anchor_path(i))
   assert q.bind(writer,freeze['sha256'])==plan;state=q.state(plan,freeze['sha256']);assert len(state['records'])==3 and state['used_weights']==retained and q.next_spec(plan,state)['slot']==3
   q.exclusive(B/'quality002-preflight.json',dict(status='frozen_same_panel_execution_continuation',freeze=freeze,accepted_prefix=3,imported_weight_bytes=retained,remaining_requirement=required,next_spec=q.next_spec(plan,state),training_updates=0,weight_copies=0,source_count=918,original_STOP_preserved=True))
 print(json.dumps(dict(status='execution_continuation_frozen',freeze=freeze,accepted_prefix=3,training_updates=0,weight_copies=0)),flush=True)
