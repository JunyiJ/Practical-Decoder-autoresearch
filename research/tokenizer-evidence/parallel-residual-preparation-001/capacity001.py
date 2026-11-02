"""Measured complete fixed-LR panel high-water; no dispatch or cleanup."""
import datetime,hashlib,json,shutil
from pathlib import Path
from research.parallel_residual_campaign_v1 import pilot as p,evidence_contract as c
from research.local_mix_v1.output import RAW_CAP,EXPORT_CAP,METADATA_CAP,WEIGHT_CAP
from research.block_norm_v1.pilot_execution import exclusive
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
R=c.ROOT;B=Path(__file__).resolve().parent;S=R/'research/tokenizer-evidence/storage-recovery-006'
def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
with session_lock(GLOBAL_LOCK.parent):
 p.idle()
 with CpuPrerequisiteGuard(B/'capacity001-guard.json',R,timeout_seconds=120):
  integration=json.loads((B/'pilot-integration002.json').read_text());c.verify_sources(integration['sources'])
  worker=json.loads((B/'trainer-worker003.json').read_text());ret=json.loads((B/'worker-retirement001/completed.json').read_text());manifest=json.loads((B/'worker-retirement001/manifest.json').read_text())
  assert ret['status']=='completed' and ret['deleted_bytes']==1118319188
  archived=json.loads((S/'raw-completed.json').read_text());restored=json.loads((S/'restoration-check.json').read_text())
  assert archived['status']=='completed_lossless_raw_storage' and archived['all_raw_bytes_preserved'] and archived['all_h200_unchanged'] and not archived['raw_payloads_retired']
  assert restored['status']=='exact_restoration_verified' and restored['owned_verified_temporary_copy_removed']
  for row in archived['files']:
   assert not Path(row['original']['path']).exists()
   assert ref(Path(row['archive']['path']))['sha256']==row['archive']['sha256']
   assert ref(Path(row['h200']['path']))['sha256']==row['h200']['sha256']
  observed=[]
  for arm in worker['arms']:
   weights={k:next(r['size'] for r in manifest['delete_files'] if r['path']==arm['binding'][k]['path']) for k in ('parent','h200')}
   assert weights['parent']<=RAW_CAP and weights['h200']<=EXPORT_CAP and arm['observed_trial_metadata_bytes']<=METADATA_CAP
   observed.append(dict(arm=arm['arm'],raw_bytes=weights['parent'],h200_bytes=weights['h200'],trial_metadata_bytes=arm['observed_trial_metadata_bytes'],audit=arm['acceptance']['audit']))
  rows=[]
  for name in ('research_runs','research_archives','research_state','research'):
   for q in (R/name).rglob('*'):
    if '__pycache__' in q.parts or not q.is_file():continue
    n=q.stat().st_size
    if q.suffix not in ('.pt','.tmp','.zst','.gz') and n<16*1024**2:continue
    assert not q.is_symlink();rows.append(dict(path=str(q.relative_to(R)),size=n,action='preserve'))
  free=shutil.disk_usage(R).free;p.require_capacity(free)
  result=dict(status='complete_fixed_lr_panel_capacity_ready',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,required_free_bytes=p.REQUIRED,deficit_bytes=0,reserve_bytes=8*1024**3,
   projection=dict(max_retained_pairs=6,raw_cap=RAW_CAP,h200_cap=EXPORT_CAP,per_pair_weight_cap=WEIGHT_CAP,metadata_allowance=256*1024**2,pilot_stage_pairs=2,pilot_retirement_before_fresh_required=True,screens=0,fresh_pairs=3,fresh_trajectories=6,
    metadata_breakdown=dict(eight_total_pilot_and_fresh_trajectories=8*METADATA_CAP,shared_control_retention_inference_statistics=128*1024**2),
    serialization='Bounded streaming writer writes one temporary payload then hardlinks the same inode and removes the temporary name under census lock. Raw and H200 coexist within weight cap; no second whole-payload copy. One EMA accumulator, not one saved file per snapshot.',
    no_partial_panel=True),measured_generated_worker_outputs=observed,source_count=len(integration['sources']),source_sha256=c.fingerprint(integration['sources']),
   dependencies={n:ref(B/n) for n in ['trainer-worker003.json','pilot-integration002.json','retention-boundary001.json','worker-retirement001/completed.json']},
   storage=dict(lossless_archive_manifest=ref(S/'raw-completed.json'),restoration_check=ref(S/'restoration-check.json'),logical_reclaimed_bytes=archived['logical_bytes_reclaimed'],all_selected_and_fresh_raw_bytes_preserved=True,all_h200_paths_unchanged=True,inventory=rows,incomplete_writes=[r for r in rows if r['path'].endswith('.tmp')]),
   scientific_seeds_reserved=False,pilot_frozen=False,quality_frozen=False,active_worker=None,
   next_action='Fresh scientific seed census, guarded finite20-step pilot prepare/run0/run1/review, then independent resource review. Quality source/plan freeze remains separate; remeasure capacity each dispatch.')
  result['dependencies']['writer']=ref(R/'src/research/tokenizer_wide_hybrid_checkpoint_writer.py');result['dependencies']['caps']=ref(R/'research/local_mix_v1/output.py')
  exclusive(B/'capacity001.json',result)
 print(json.dumps({k:result[k] for k in ['status','free_bytes','required_free_bytes','deficit_bytes']}),flush=True)
