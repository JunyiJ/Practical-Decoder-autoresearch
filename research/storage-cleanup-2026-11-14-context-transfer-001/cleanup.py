"""Retire exact accepted engineering payloads after all consumers closed."""
import fcntl,json,os,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'storage-cleanup-2026-11-09'))
from common import ROOT,sha,row,save,opens,remove_entries,protected_current
OUT=Path(__file__).resolve().parent
E=ROOT/'research/tokenizer-evidence'
NAMES={
 'general-rsi-cross-size-pilot-001':['d4-native-raw.pt','d4-raw_bytes-raw.pt','d8-native-raw.pt','d8-raw_bytes-raw.pt','d16-native-raw.pt','d16-raw_bytes-raw.pt'],
 'general-rsi-cross-size-integration-002':['slot-00/raw.pt','slot-00/h200.pt'],
 'general-rsi-context-length-pilot-001':['c128-raw.pt','c256-step24.pt','c256-step25-reference.pt'],
 'general-rsi-context-length-integration-001':['active-ema-reference.pt','active-ema-restart.pt','raw.pt','time_ema.pt'],
 'general-rsi-joint-integration-001':['ce_full-restart.pt','ce_half-restart.pt','ce_pseudo_mean-restart.pt'],
 'general-rsi-continuation-decay-integration-001':['all-restart.pt','readout_only-restart.pt','other_only-restart.pt','none-restart.pt']}
CLOSED=['general-rsi-cross-size-quality-001','general-rsi-context-length-quality-001','general-rsi-joint-diagnostic-001','general-rsi-continuation-decay-diagnostic-001']
COST=E/'general-rsi-cross-size-attribution-001/context-transfer-planning-cost.json'

def walk(x):
 if isinstance(x,dict):
  yield x
  for y in x.values():yield from walk(y)
 elif isinstance(x,list):
  for y in x:yield from walk(y)

def closure():
 state=json.loads((ROOT/'research_state/general-rsi-state.json').read_text())
 assert state['active_worker_pid'] is None and state['active_slot'] is None
 for name in list(NAMES)+CLOSED:
  j=json.loads((E/name/'completion-review.json').read_text())
  assert any(j.get(k) is True for k in ('accepted','accepted_execution','accepted_engineering','accepted_completion')),name
 assert json.loads(COST.read_text())['scientific_training_admitted'] is False

def prepare():
 closure();entries=[];proof={};protected=protected_current()
 for name,files in NAMES.items():
  receipts=[]
  for q in (E/name).rglob('*.json'):
   for x in walk(json.loads(q.read_text())):
    if 'path' in x and 'sha256' in x and ('size' in x or 'bytes' in x):receipts.append((q,x))
  for file in files:
   e=row(ROOT/'research_runs'/name/file)
   matches=[(q,x) for q,x in receipts if x['path']==e['path'] and x['sha256']==e['sha256'] and x.get('size',x.get('bytes'))==e['size']]
   assert matches,e['path']
   proof[e['path']]=[str(q.relative_to(ROOT)) for q,x in matches]
   entries.append(e)
  for q in (E/name).rglob('*'):
   if q.is_file() and not q.is_symlink():protected[str(q)]=sha(q)
 for name in CLOSED:
  for file in ('completion-review.json','plan.json'):
   q=E/name/file;protected[str(q)]=sha(q)
 for name in ('general-rsi-cross-size-quality-001','general-rsi-context-length-quality-001'):
  for q in (ROOT/'research_runs'/name).glob('slot-*/*.pt'):protected[str(q)]=sha(q)
 targets={e['path'] for e in entries};assert len(targets)==22 and not targets.intersection(protected)
 assert not targets.intersection(opens())
 consumers={}
 for name in NAMES:
  r=subprocess.run(['rg','-l',name,'research','src','research_state','--glob','*.py','--glob','*.md','--glob','*.json'],cwd=ROOT,text=True,capture_output=True)
  assert r.returncode in (0,1);consumers[name]=r.stdout.splitlines()
 save(OUT/'manifest.json',dict(entries=entries,original_receipt_proof=proof,protected_sha256=protected,consumer_reference_files=consumers,
  dependency_closure='All six engineering studies passed original reviews. Their cross-size/context panels and joint/decay diagnostics have completed reviews. References are historical qualification, completed consumers, immutable receipts/source bindings, and cleanup/state history. No worker is active. The proposed native depth/context study starts fresh and will use source math plus new narrowly required qualification, not these payloads. No accepted scientific confirmation or selected model is targeted.',
  policy_exception='Standing human authorization retires obsolete concluded-study engineering payloads after exact acceptance/consumer checks. Supersedes historical retain-all archive wording only for these22 accepted engineering files. Failed attempts and all first outputs, source/log/config/receipts remain. Cross-size integration001 failure is excluded.',
  planning_cost_sha256=sha(COST),sources={str(Path(__file__).resolve()):sha(__file__),str(ROOT/'research/storage-cleanup-2026-11-09/common.py'):sha(ROOT/'research/storage-cleanup-2026-11-09/common.py')},allocated_bytes=sum(e['allocated'] for e in entries)))
 print(json.dumps(dict(files=len(entries),allocated_bytes=sum(e['allocated'] for e in entries),protected=len(protected))))

def execute():
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('r+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);closure()
  ps=subprocess.check_output(['ps','-axo','pid=,args='],text=True)
  for line in ps.splitlines():
   parts=line.strip().split(None,2)
   if len(parts)>=2 and int(parts[0])!=os.getpid():assert not Path(parts[1]).name.lower().startswith('python'),line
  m=json.loads((OUT/'manifest.json').read_text())
  for p,h in {**m['sources'],**m['protected_sha256']}.items():assert sha(p)==h,p
  assert sha(COST)==m['planning_cost_sha256']
  opened=opens()
  for e in m['entries']:assert row(e['path'])==e and e['path'] not in opened
  before=shutil.disk_usage(ROOT).free
  removed,skipped=remove_entries(m['entries'],OUT/'deletion-journal.jsonl')
  for p,h in m['protected_sha256'].items():assert sha(p)==h,p
  free=shutil.disk_usage(ROOT).free;required=json.loads(COST.read_text())['total_required_free_bytes']
  result=dict(files_removed=len(removed),skipped=skipped,allocated_bytes_removed=sum(e['allocated'] for e in removed),protected_files_verified=len(m['protected_sha256']),free_before=before,free_after=free,required_free_bytes=required,margin_bytes=free-required,reserve_10GiB_plus_extra_2GiB_included=True,manifest_sha256=sha(OUT/'manifest.json'),scientific_training_admitted=False)
  save(OUT/'completed.json',result);print(json.dumps(result))

if __name__=='__main__':{'prepare':prepare,'execute':execute}[sys.argv[1]]()
