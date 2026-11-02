"""Measure the complete retained-output high-water after exact lossless recovery."""
import datetime,json,shutil
from pathlib import Path
from research.cooldown_campaign_v1 import pilot as p,evidence_contract as c
from research.local_mix_v1.output import RAW_CAP,EXPORT_CAP,METADATA_CAP,WEIGHT_CAP
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
R=c.ROOT; B=p.B; E=R/'research/tokenizer-evidence/storage-recovery-007'
def read(q):return json.loads(q.read_text())
with session_lock(GLOBAL_LOCK.parent):
 p.idle()
 with CpuPrerequisiteGuard(B/'capacity002-guard.json',R,timeout_seconds=120):
  integration=read(B/'pilot-integration001.json');c.verify_sources(integration['sources'])
  assert len(integration['sources'])==615
  retirement=read(B/'worker-retirement001/completed.json');assert retirement['status']=='completed'
  recovery=read(E/'raw-completed.json');assert recovery['status']=='completed_lossless_raw_storage'
  assert recovery['all_raw_bytes_preserved'] and recovery['all_h200_unchanged'] and not recovery['raw_payloads_retired']
  assert read(E/'restoration-check.json')['status']=='exact_restoration_verified'
  manifest=read(B/'worker-retirement001/manifest.json');worker=read(B/'trainer-worker001.json');observed=[]
  for arm in worker['arms']:
   weights={k:next(r['size'] for r in manifest['delete_files'] if r['path']==arm['binding'][k]['path']) for k in ('parent','h200')}
   assert weights['parent']<=RAW_CAP and weights['h200']<=EXPORT_CAP and arm['observed_trial_metadata_bytes']<=METADATA_CAP
   observed.append(dict(arm=arm['arm'],raw_bytes=weights['parent'],h200_bytes=weights['h200'],metadata_bytes=arm['observed_trial_metadata_bytes'],audit=arm['acceptance']['audit']))
  inventory=[]
  for name in ('research_runs','research_archives','research','research_state'):
   for q in (R/name).rglob('*'):
    if '__pycache__' in q.parts or not q.is_file():continue
    n=q.stat().st_size
    if n<16*1024**2 and q.suffix not in ('.pt','.tmp','.zst','.gz'):continue
    assert not q.is_symlink()
    rel=str(q.relative_to(R));classification='unclassified_preserve'
    if 'research_archives' in q.parts:classification='archive_preserve_including_lossless_selected_fresh'
    elif '/evidence/quality/' in rel:classification='historical_quality_preserve'
    inventory.append(dict(path=rel,size=n,classification=classification))
  writer=R/'src/research/tokenizer_wide_hybrid_checkpoint_writer.py';source=writer.read_text()
  assert 'os.link(temporary,final)' in source and 'temporary.unlink()' in source
  free=shutil.disk_usage(R).free;p.require_capacity(free)
  result=dict(status='complete_fixed_lr_panel_capacity_ready',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,required_free_bytes=p.REQUIRED,deficit_bytes=0,
   projection=dict(retained_fresh_trajectories=6,fresh_pairs=3,screens=0,raw_cap=RAW_CAP,h200_cap=EXPORT_CAP,per_trajectory_weight_cap=WEIGHT_CAP,metadata_allowance=256*1024**2,reserve_bytes=8*1024**3,
    metadata_breakdown=dict(eight_pilot_and_fresh_trajectories=8*METADATA_CAP,shared_control_retention_inference_statistics=128*1024**2),pilot_stage_pairs=2,pilot_retirement_before_fresh_required=True,
    serialization='Bounded streaming temporary payload published by hardlink of same inode then unlink temporary name. Raw and H200 coexist inside weight cap; no extra whole-file on-disk publication copy. EMA uses one accumulator, not one file per snapshot.'),
   measured_generated_worker_outputs=observed,source_count=len(integration['sources']),source_sha256=c.fingerprint(integration['sources']),
   dependencies={n:p.ref(q) for n,q in dict(worker=B/'trainer-worker001.json',integration=B/'pilot-integration001.json',retention_test=B/'retention-boundary001.json',worker_retirement=B/'worker-retirement001/completed.json',lossless_recovery=E/'raw-completed.json',restoration=E/'restoration-check.json',writer=writer,output_caps=R/'research/local_mix_v1/output.py',runner=Path(__file__)).items()},
   size_based_inventory=inventory,incomplete_writes=[r for r in inventory if r['path'].endswith('.tmp')],inventory_scope='Includes size-selected files, checkpoints, incomplete .pt.tmp writes and archives. Inventory is not deletion eligibility; unknown files stay preserved.',
   scientific_seeds_reserved=False,pilot_frozen=False,quality_frozen=False,no_partial_panel=True,
   next_action='Fresh scientific seed census, freeze two finite resource pilots, run/accept both and paired inference plus independent resource review. Recheck actual capacity at every dispatch; final quality source and exact pilot retirement remain required.')
 p.exclusive(B/'capacity002.json',result)
 print(json.dumps({k:result[k] for k in ('status','free_bytes','required_free_bytes','source_count')}))
