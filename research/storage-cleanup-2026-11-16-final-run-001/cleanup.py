"""Exact retirement of accepted short fixtures after downstream dependency closure."""
import datetime,fcntl,json,os,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'storage-cleanup-2026-11-09'))
from common import ROOT,sha,row,save,opens,remove_entries,protected_current
OUT=Path(__file__).resolve().parent
E=ROOT/'research/tokenizer-evidence'; PANEL=E/'general-rsi-depth-context-quality-001'
def read(p):return json.loads(p.read_text())
def main():
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('r+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  ps=subprocess.check_output(['ps','-axo','pid=,args='],text=True)
  for line in ps.splitlines():
   parts=line.split()
   if len(parts)>2 and int(parts[0])!=os.getpid() and 'python' in Path(parts[1]).name.lower():
    assert not any(s in line for s in ('research.general_rsi_','research.byte_input_','train.py')),line
  assert read(PANEL/'slot-10/guard-01.json')['status']=='finished'
  assert read(PANEL/'slot-10/review.json')['accepted_execution']
  assert read(PANEL/'slot-10/supervisor-01-exit.json')['returncode']==0
  q=E/'general-rsi-depth-context-integration-003'; b=E/'byte-composition-boundary-001'
  assert read(q/'completion-review.json')['accepted']
  result=read(q/'attempt-01/result.json');assert result['accepted'] and not result['scientific_training']
  pilot=read(b/'pilot-review003/review.json');assert pilot['accepted'] and not pilot['quality_selected']
  for i in range(18):
   root='quality001' if i<14 else 'quality-recovery001'
   assert read(b/root/f'slots/{i:02d}/acceptance.json')['accepted']
  receipts=[r for s in result['summaries'] for r in s['endpoint_receipts'].values()]
  receipts += [r for r in pilot['artifacts'] if r['path'].endswith('-h200.pt')]
  assert len(receipts)==11
  entries=[]
  for receipt in receipts:
   e=row(Path(receipt['path']));assert e['size']==receipt['size'] and e['sha256']==receipt['sha256'];entries.append(e)
  plan=read(PANEL/'plan.json');protected=protected_current()
  for p,h in plan['sources'].items():
   target=ROOT/p;assert sha(target)==h;protected[str(target)]=h
  for d in [q,b,PANEL,E/'general-rsi-depth-context-integration-002']:
   for p in d.rglob('*'):
    if p.is_file():protected[str(p)]=sha(p)
  for study in ['general-rsi-depth-context-quality-001','general-rsi-depth-context-integration-002','byte-input-quality-v1-001','byte-input-quality-v1-recovery001']:
   for p in (ROOT/'research_runs'/study).rglob('*.pt'):protected[str(p)]=sha(p)
  for p in (ROOT/'research_runs/byte-input-pilot-math002').glob('*-raw.pt'):protected[str(p)]=sha(p)
  targets={e['path'] for e in entries};assert not targets.intersection(protected)
  refs=subprocess.run(['rg','-l','general-rsi-depth-context-integration-003|byte-input-pilot-math002','research','src','research_state','--glob','*.py','--glob','*.md','--glob','*.json'],cwd=ROOT,text=True,capture_output=True)
  assert refs.returncode in (0,1);assert not targets.intersection(opens())
  save(OUT/'manifest.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),entries=entries,protected_sha256=protected,consumer_reference_files=refs.stdout.splitlines(),source_sha256=sha(__file__),helper_sha256=sha(ROOT/'research/storage-cleanup-2026-11-09/common.py'),dependency_closure='Combined-shape003 completion review accepted all four shapes, eight inference endpoints and96 first outputs. Scientific preparer consumes its retained JSON receipts, dimensions, timing and guards; worker imports only host_memory and initializes fresh. All5269 frozen source bindings exclude these payloads. Eleven scientific runs now pass execution review. Byte pilot math002 is a closed accepted20-update resource fixture, with accepted saved-only review003. All18 later fresh quality slots have accepted execution receipts. The quality driver initializes fresh and consumes pilot acceptance metadata, not H200 weights. The three RAW pilot parents implicated in the historical backend-spelling diagnosis remain retained. Only closed fixture/review replays require the exact eleven retired payloads; accepted original receipts remain immutable.',policy_exception='Standing human authorization for obsolete concluded-study numerical payloads applies narrowly to eight accepted003 engineering inference endpoints and three accepted byte-pilot H200 inference payloads. Retain repaired003 source, receipts, metrics and96 first outputs; preserve original002 RNG-harness failure and every payload, all selected/fresh scientific models, pilot RAW restart evidence and all first outputs. No scientific protocol or immutable original evidence is rewritten.'))
  for e in entries:assert row(e['path'])==e
  before=shutil.disk_usage(ROOT).free
  removed,skipped=remove_entries(entries,OUT/'deletion-journal.jsonl')
  for p,h in protected.items():assert sha(p)==h,p
  endpoints=[p for p in (ROOT/'research_runs/general-rsi-depth-context-quality-001').glob('slot-*/*.pt') if p.name in ('raw.pt','time_ema.pt')];assert len(endpoints)==22
  retained=sum(p.stat().st_size for p in endpoints)
  required=plan['reserve_bytes']+max(0,plan['incremental_peak_bytes']-retained);free=shutil.disk_usage(ROOT).free
  result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files_removed=len(removed),skipped=skipped,logical_bytes_removed=sum(e['size'] for e in removed),allocated_bytes_removed=sum(e['allocated'] for e in removed),free_before=before,free_after=free,required_remaining_free_bytes=required,margin_bytes=free-required,retained_endpoint_bytes=retained,protected_files_verified=len(protected),manifest_sha256=sha(OUT/'manifest.json'),reserve_10GiB_plus_extra_2GiB_included=True)
  save(OUT/'completed.json',result);print(json.dumps(result))
if __name__=='__main__':main()
