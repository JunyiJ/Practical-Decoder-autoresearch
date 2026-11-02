import datetime,hashlib,json,shutil
from pathlib import Path
from research.decay_campaign_v1 import pilot as p,evidence_contract as c
from research.local_mix_v1.output import RAW_CAP,EXPORT_CAP,METADATA_CAP,TRAJECTORY_CAP
from research.block_norm_v1.pilot_execution import exclusive
R=Path.cwd();B=R/'research/tokenizer-evidence/decay-preparation-001'
def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
oldpath=R/'research/tokenizer-evidence/local-mix-preparation-001/capacity001.json';old=json.loads(oldpath.read_text());known={r['path']:r['size'] for r in old['tensor_and_archive_inventory']}
closurepath=R/'research/tokenizer-evidence/local-mix-preparation-001/quality001-closure.json';closure=json.loads(closurepath.read_text())
assert (R/'research_runs/tokenizer-local-mix-quality-v3-001/STOP').is_file()
ret=json.loads((B/'worker-retirement001/completed.json').read_text());assert ret['status']=='completed'
worker=json.loads((B/'trainer-worker001.json').read_text());plan=json.loads((B/'pilot-integration001.json').read_text());c.verify_sources(plan['sources'])
rows=[];groups={};unknown=[];tmp=[]
for root in ('research_runs','research_archives','research_state','research'):
 for q in (R/root).rglob('*'):
  if '__pycache__' in q.parts or not q.is_file():continue
  n=q.stat().st_size;rel=str(q.relative_to(R));is_payload=q.suffix in ('.pt','.tmp','.zst','.gz')
  if not is_payload and n<16*1024**2:continue
  assert not q.is_symlink()
  classification='unclassified_preserve'
  if rel in known and known[rel]==n:classification='prior_eligibility_review_retained_selected_or_fresh_or_small_unclassified'
  if rel.startswith('research_runs/tokenizer-local-mix-quality-v3-001/evidence/quality/'):
   slot=int(Path(rel).parts[4]);assert slot in (2,3,6,7,8,9,10,11)
   classification='local_mix_selected_or_fresh_protected_by_closed_campaign'
  if rel.startswith('research/tokenizer-evidence/decay-preparation-001/retention-boundary001/'):
   classification='new_disposable_retention_test_protected_sentinel'
  row=dict(path=rel,size=n,classification=classification);rows.append(row)
  if classification=='unclassified_preserve':unknown.append(row)
  if rel.endswith('.tmp'):tmp.append(row)
  if n>=16*1024**2:
   k='/'.join(Path(rel).parts[:2]);g=groups.setdefault(k,dict(files=0,bytes=0));g['files']+=1;g['bytes']+=n
free=shutil.disk_usage(R).free
observed=[]
for arm in worker['arms']:
 ck=arm['binding'];manifest=json.loads((B/'worker-retirement001/manifest.json').read_text())
 weights={k:next(r['size'] for r in manifest['delete_files'] if r['path']==ck[k]['path']) for k in ('parent','h200')}
 assert weights['parent']<=RAW_CAP and weights['h200']<=EXPORT_CAP and arm['observed_trial_metadata_bytes']<=METADATA_CAP
 observed.append(dict(arm=arm['arm'],raw_bytes=weights['parent'],h200_bytes=weights['h200'],trial_metadata_bytes=arm['observed_trial_metadata_bytes'],audit=arm['acceptance']['audit']))
result=dict(status='complete_fixed_lr_panel_capacity_ready' if free>=p.REQUIRED else 'complete_fixed_lr_panel_capacity_blocked_source_ready',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_bytes=free,required_free_bytes=p.REQUIRED,deficit_bytes=max(0,p.REQUIRED-free),reserve_bytes=8*1024**3,
 projection=dict(max_retained_pairs=6,raw_cap=RAW_CAP,h200_cap=EXPORT_CAP,per_pair_weight_cap=p.WEIGHT_CAP,metadata_allowance=256*1024**2,pilot_stage_pairs=2,pilot_retirement_before_fresh_required=True,screens=0,fresh_pairs=3,fresh_trajectories=6,
 metadata_breakdown=dict(eight_total_pilot_and_fresh_trajectories=8*METADATA_CAP,shared_control_retention_inference_statistics=128*1024**2),
 serialization='Actual bounded streaming CheckpointWriter writes one temporary payload, then hardlinks the same inode and unlinks the temporary name under the census lock. Raw+H200 coexist within per-pair weight cap. No additional whole-payload on-disk copy; fresh count reaches six. One EMA accumulator per half-life, not one checkpoint per snapshot.',
 final_free_requirement_rechecked=True,no_partial_panel=True),measured_generated_worker_outputs=observed,source_sha256=c.fingerprint(plan['sources']),source_count=len(plan['sources']),
 dependencies=dict(worker=ref(B/'trainer-worker001.json'),integration=ref(B/'pilot-integration001.json'),retention_test=ref(B/'retention-boundary001.json'),worker_retirement=ref(B/'worker-retirement001/completed.json'),writer=ref(R/'src/research/tokenizer_wide_hybrid_checkpoint_writer.py'),output_caps=ref(R/'research/local_mix_v1/output.py')),
 recovery_review=dict(prior_classification=ref(oldpath),new_local_mix_closure=ref(closurepath),large_groups=groups,inventory=rows,unclassified=unknown,incomplete_writes=tmp,
 already_recovered_bytes=ret['deleted_bytes'],retained_files_verified=ret['retained_files_verified'],remaining_large_eligible_candidates=[],
 scope='Size-based current inventory, including .pt.tmp and archives. Surviving large historical weights match earlier selected/fresh eligibility inventory; new local-mix paths are exact selected02/03 or fresh06-11. No remaining large failed/pilot/losing/numerical candidate identified. Small unknown payloads stay preserved. No unrelated maintenance touched.'),
 scientific_seeds_reserved=False,pilot_frozen=False,quality_frozen=False,active_worker=None,
 next_action='Wait for complete actual free capacity, then fresh census, guarded finite pilot prepare/run0/run1/review and independent resource review before final quality freeze.',
 wake_condition=f'Actual free bytes >= {p.REQUIRED}, or new eligible recovery evidence or user-authorized storage.',poll_interval_seconds=3600,
 external_action='Free another3GB on this volume to cover the measured deficit with headroom; selected models and all fresh confirmations remain protected.')
assert not any(r['size']>=16*1024**2 for r in unknown),'New large candidate requires independent eligibility review before declaring recovery exhausted'
exclusive(B/'capacity002.json',result)
print(json.dumps({k:result[k] for k in ['status','free_bytes','required_free_bytes','deficit_bytes','poll_interval_seconds']}));print('Inventory rows',len(rows),'unknown bytes',sum(x['size'] for x in unknown))
