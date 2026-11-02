"""Lossless cold storage of selected/fresh raw checkpoints; never retire models."""
import importlib.util,json,os,shutil,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];E=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('archive_primitives',R/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a);a.E=E

spec=importlib.util.spec_from_file_location('codec',E/'archive_codec.py');codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec);codec.install(a)
from research.readout_centering_quality_v1.quality_execution import idle
COLD=R/'research_archives/protected-raw-storage-014'
POLICY=R/'research/TOKENIZER_CHECKPOINT_RETENTION.md'
def all_paths():
 out=[]
 for name in ('research_runs','research_archives','research_state','research','src','data'):
  for p in sorted((R/name).rglob('*')):
   if E in p.parents or '__pycache__' in p.parts or 'maintenance-cache-cleanup-002' in p.parts or not p.is_file() or p.name=='.lock' or 'tokenizer-worker-lock' in p.parts or p.name=='tokenizer-quality-progress.json':continue
   assert not p.is_symlink();out.append(p)
 return sorted(set(out))
def plan():
 assert not COLD.exists() and not (E/'raw-plan.json').exists()
 probe=json.loads((E/'archive-probe.json').read_text());assert probe['status']=='verified_copy_only' and probe['original_unchanged'] and probe['logical_saving_bytes']>80*1024**2
 choices=[]
 root=R/'research_runs/tokenizer-gradient-clip-quality-v2-001';stop=root/'STOP'
 terminal=json.loads(stop.read_text());assert terminal['status']=='completed_closed'
 final=a.bound(terminal['final_review']);assert final['status']=='reviewed_not_adopted' and not final['quality_adopted'] and len(final['accepted_anchors'])==6
 for slot,anchorref in enumerate(final['accepted_anchors']):
  anchor=a.bound(anchorref);assert anchor['slot']==slot
  receipt=a.verify(anchor['acceptance']);d=json.loads(receipt.read_text());accepted=d['artifact_acceptance']
  assert d['slot']==slot and accepted['accepted'] and accepted['audit']['completed_steps']==1200 and accepted['audit']['ema_snapshots']==41
  binding=accepted['files']['parent'];h200=accepted['files']['h200'];p=Path(binding['path']);h=Path(h200['path'])
  expected=R/f'research_runs/tokenizer-gradient-clip-quality-v{1 if slot==0 else 2}-001/evidence/quality/{slot:02d}/checkpoints'
  assert p.parent==h.parent==expected and p.is_file() and h.is_file() and p.suffix=='.pt' and h.name=='h200.pt'
  if slot==0:assert anchor['import_only'] and anchor['repeat_training_updates']==0 and d['training_updates_this_import']==0
  assert not(p.stat().st_flags&32) and p.stat().st_size==445186149
  choices.append((slot,str(p),binding,receipt,stop,h200))
 assert len(choices)==6
 files=[]
 for index,(_,name,binding,receipt,stop,h200) in enumerate(sorted(choices)):
  p=Path(name);v=a.record(p);a.check(p,v);assert v['sha256']==binding['sha256']
  assert a.file_digest(Path(h200['path']))==h200['sha256']
  files.append(dict(original=v,member=str(p.relative_to(R)),archive=str(COLD/f'{index:02d}-{v["sha256"][:12]}.tar.zst'),acceptance=a.ref(receipt),terminal=a.ref(stop),h200=h200,classification='closed_gradient_clip_quality002_fresh_raw_preserved_losslessly_not_retired_including_imported_slot00'))
 selected={f['original']['path'] for f in files};protected=[a.record(p) for p in all_paths() if str(p) not in selected];shards=[]
 for i in range(0,len(protected),500):
  path=E/f'raw-protected-{i//500:03d}.json';a.publish(path,protected[i:i+500]);shards.append(a.ref(path))
 a.publish(E/'raw-plan.json',dict(operation='lossless_protected_raw_cold_storage_not_retirement',source=a.ref(Path(__file__)),codec=a.ref(E/'archive_codec.py'),materializer=a.ref(E/'materialize.py'),policy=a.ref(POLICY),primitives=a.ref(R/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py'),probe=a.ref(E/'archive-probe.json'),files=files,protected_shards=shards,original_bytes=sum(x['original']['size'] for x in files),free_before=shutil.disk_usage(R).free,required_remaining_panel=11263061680,all_h200_remain_loose=True,restoration='Use materialize.py with the original relative path and a capacity check; archive remains protected. No closed campaign is rerun.'))
 print(json.dumps(dict(status='planned',raw_files=len(files),logical_bytes=sum(x['original']['size'] for x in files),protected=len(protected))),flush=True)
def apply():
 plan=json.loads((E/'raw-plan.json').read_text());assert json.loads((E/'raw-plan-guard.json').read_text())['status']=='finished'
 assert not COLD.exists() and not (E/'journal.jsonl').exists()
 for ref in [plan['source'],plan['codec'],plan['materializer'],plan['policy'],plan['primitives'],plan['probe']]:a.verify(ref)
 protected=[r for shard in plan['protected_shards'] for r in a.bound(shard)]
 selected={x['original']['path'] for x in plan['files']}
 assert set(map(str,all_paths()))=={r['path'] for r in protected}|selected
 for r in protected:assert a.record(Path(r['path']))==r
 for x in plan['files']:
  a.verify(x['acceptance']);a.verify(x['terminal']);a.verify(x['h200']);a.check(Path(x['original']['path']),x['original'])
 COLD.mkdir();a.syncdir(COLD.parent);rows=[]
 for x in plan['files']:
  src=Path(x['original']['path']);archive=Path(x['archive']);v=x['original'];a.check(src,v)
  files={x['member']:dict(size=v['size'],sha256=v['sha256'])}
  a.create_archive(archive,files);a.verify_archive(archive,files)
  ar=a.record(archive);assert ar['size']<v['size']
  assert a.record(src)==v
  row=dict(original=v,member=x['member'],archive=ar,acceptance=x['acceptance'],terminal=x['terminal'],h200=x['h200'],retired=False,logical_saving_bytes=v['size']-ar['size'])
  a.journal(dict(event='exact_raw_bytes_preserved_in_verified_archive',**row))
  a.check(src,v);src.unlink();a.syncdir(src.parent)
  a.journal(dict(event='duplicate_loose_copy_removed',original_path=str(src),archive=ar));rows.append(row)
 assert set(map(str,all_paths()))=={r['path'] for r in protected}|{x['archive']['path'] for x in rows}
 for row in rows:
  assert not Path(row['original']['path']).exists();assert a.record(Path(row['archive']['path']))==row['archive']
  a.verify_archive(Path(row['archive']['path']),{row['member']:dict(size=row['original']['size'],sha256=row['original']['sha256'])})
 for r in protected:assert a.record(Path(r['path']))==r
 for ref in [plan['source'],plan['codec'],plan['materializer'],plan['policy'],plan['primitives'],plan['probe']]:a.verify(ref)
 a.publish(E/'raw-completed.json',dict(status='completed_lossless_raw_storage',plan=a.ref(E/'raw-plan.json'),files=rows,all_raw_bytes_preserved=True,all_h200_unchanged=True,protected_files_verified=len(protected),logical_bytes_reclaimed=sum(x['logical_saving_bytes'] for x in rows),free_after=shutil.disk_usage(R).free,required_remaining_panel=11263061680,scientific_training=False,raw_payloads_retired=False))
 print(json.dumps(dict(status='completed_lossless_raw_storage',files=len(rows),logical_bytes_reclaimed=sum(x['logical_saving_bytes'] for x in rows),free_after=shutil.disk_usage(R).free)),flush=True)
if __name__=='__main__':
 mode=sys.argv[1];assert mode in ('plan','apply')
 with a.session_lock(R/'research_state/tokenizer-worker-lock'),a.CpuPrerequisiteGuard(E/f'raw-{mode}-guard.json',R,timeout_seconds=900):idle();plan() if mode=='plan' else apply()
