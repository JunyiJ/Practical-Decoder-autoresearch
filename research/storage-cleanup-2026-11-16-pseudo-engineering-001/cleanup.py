"""Retire two accepted, obsolete pseudo-objective engineering restart payloads."""
import fcntl,json,os,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'storage-cleanup-2026-11-09'))
from common import ROOT,sha,row,save,opens,remove_entries,protected_current
OUT=Path(__file__).resolve().parent
E=ROOT/'research/tokenizer-evidence'
PANEL=E/'general-rsi-depth-context-quality-001'
def read(p):return json.loads(p.read_text())
def main():
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('r+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  ps=subprocess.check_output(['ps','-axo','pid=,args='],text=True)
  for line in ps.splitlines():
   parts=line.split()
   if len(parts)>2 and int(parts[0])!=os.getpid():assert not (parts[1].lower().endswith('/python') and '-m research.general_rsi_' in line),line
  assert read(PANEL/'slot-05/guard-01.json')['status']=='finished'
  assert read(PANEL/'slot-05/review.json')['accepted_execution']
  q=E/'general-rsi-pseudo-integration-001';review=read(q/'completion-review.json')
  assert review['accepted_engineering'] and all(x['exact_full_restore_and_replay'] for x in review['arms'])
  closed=read(E/'general-rsi-pseudo-repetition-001/completion-review.json')
  assert closed['all_six_execution_reviews_accepted'] and closed['first_generation_count']==144
  entries=[]
  for arm in review['arms']:
   receipt=arm['restart'];e=row(Path(receipt['path']))
   assert e['size']==receipt['size'] and e['sha256']==receipt['sha256'];entries.append(e)
  assert len(entries)==2
  plan=read(PANEL/'plan.json');protected=protected_current()
  for p,h in plan['sources'].items():
   target=ROOT/p;assert sha(target)==h;protected[str(target)]=h
  protected[str(PANEL/'plan.json')]=sha(PANEL/'plan.json')
  for d in [q,E/'general-rsi-pseudo-repetition-001',PANEL,E/'general-rsi-depth-context-integration-002']:
   for p in d.rglob('*'):
    if p.is_file():protected[str(p)]=sha(p)
  for d in ['general-rsi-pseudo-repetition-001','general-rsi-depth-context-quality-001','general-rsi-depth-context-integration-002']:
   for p in (ROOT/'research_runs'/d).rglob('*.pt'):protected[str(p)]=sha(p)
  targets={e['path'] for e in entries};assert not targets.intersection(protected)
  refs=subprocess.run(['rg','-l','general-rsi-pseudo-integration-001|native_pseudo_repetition-restart.pt|native_ce-restart.pt','research','src','research_state','--glob','*.py','--glob','*.md','--glob','*.json'],cwd=ROOT,text=True,capture_output=True)
  assert refs.returncode in (0,1)
  assert not targets.intersection(opens())
  save(OUT/'manifest.json',dict(entries=entries,protected_sha256=protected,consumer_reference_files=refs.stdout.splitlines(),source_sha256=sha(__file__),helper_sha256=sha(ROOT/'research/storage-cleanup-2026-11-09/common.py'),dependency_closure='Original engineering review accepted both exact full-state replay tests. The six-run pseudo-repetition panel is complete, with all12 fresh endpoints and144 first outputs retained. Its preparer reads engineering JSON sizes/timings, not these payloads, and loads scientific parents from native-horizon. Only the closed integration fixture reads these restart files. Other references bind retained JSON/source evidence. Current depth/context worker starts fresh and its5269 source bindings exclude these two payloads. No worker is active.',policy_exception='Standing human authorization for obsolete accepted engineering payloads supersedes historical retain-both wording for these exact two restart files only. Preserve original sources, logs, configs, metrics, receipts, all scientific endpoints and first outputs; original RNG-harness failure and payloads remain protected.'))
  for e in entries:assert row(e['path'])==e
  before=shutil.disk_usage(ROOT).free
  removed,skipped=remove_entries(entries,OUT/'deletion-journal.jsonl')
  for p,h in protected.items():assert sha(p)==h,p
  retained=sum(p.stat().st_size for p in (ROOT/'research_runs/general-rsi-depth-context-quality-001').glob('slot-*/*.pt') if p.name in ('raw.pt','time_ema.pt'))
  required=plan['reserve_bytes']+max(0,plan['incremental_peak_bytes']-retained)
  free=shutil.disk_usage(ROOT).free
  result=dict(files_removed=len(removed),skipped=skipped,allocated_bytes_removed=sum(e['allocated'] for e in removed),free_before=before,free_after=free,required_remaining_free_bytes=required,margin_bytes=free-required,retained_endpoint_bytes=retained,protected_files_verified=len(protected),manifest_sha256=sha(OUT/'manifest.json'),reserve_10GiB_plus_extra_2GiB_included=True)
  save(OUT/'completed.json',result);print(json.dumps(result))
if __name__=='__main__':main()
