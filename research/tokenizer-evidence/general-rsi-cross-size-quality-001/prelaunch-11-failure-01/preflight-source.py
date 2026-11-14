import json,hashlib,pathlib,fcntl,subprocess,os,shutil,datetime
R=pathlib.Path.cwd();P=R/'research/tokenizer-evidence/general-rsi-cross-size-quality-001';S=R/'research_state/general-rsi-state.json'
def sha(p):
 with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,j):
 with p.open('x') as f:json.dump(j,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
plan=json.loads((P/'plan.json').read_text());binding=sha(P/'plan.json');assert binding=='5db6e40f41fe52af3479e2358fa615a69b9d9050411d3637f814a5c9f0dab080'
assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='codex/tokenizer-quality'
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert head==subprocess.check_output(['git','rev-parse','origin/codex/tokenizer-quality'],text=True).strip()
assert not json.loads(S.read_text()).get('stop_requested',False)
assert not (P/'slot-11').exists()
with (R/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 processes=subprocess.check_output(['ps','-axo','pid=,args='],text=True)
 workers=[]
 for line in processes.splitlines():
  parts=line.strip().split(None,2)
  if len(parts)>=2 and int(parts[0])!=os.getpid() and pathlib.Path(parts[1]).name.lower().startswith('python'):workers.append(line)
 assert not workers,workers
 for p,h in plan['sources'].items():assert sha(R/p)==h,p
 completed=[];retained=0
 for i in range(11):
  d=P/f'slot-{i:02d}';v=json.loads((d/'review.json').read_text());r=json.loads((d/'result.json').read_text())
  assert v['accepted_execution'] and v['result_sha256']==sha(d/'result.json') and r['binding']==binding
  for e in r['endpoints']:
   w=e['weights'];assert pathlib.Path(w['path']).stat().st_size==w['size'] and sha(w['path'])==w['sha256'];retained+=w['size']
  ep={e['endpoint']:e for e in r['endpoints']}
  completed.append(dict(slot=i,depth=r['slot']['depth'],arm=r['slot']['arm'],bundle=r['slot']['bundle'],steps=r['steps'],h200_ce=ep['h200']['ce'],raw_ce=ep['raw']['ce'],training_seconds=r['training_seconds'],worker_seconds=v['guard_seconds'],review=str((d/'review.json').relative_to(R)),first_outputs=v['first_outputs']))
 free=shutil.disk_usage(R).free;need=plan['reserve_bytes']+max(0,plan['incremental_peak_bytes']-retained);assert free>=need
 spent=sum(json.loads(p.read_text())['elapsed_seconds'] for p in P.glob('slot-*/guard-*.json'))
 forecast=spent+sum(x['timeout_seconds'] for x in plan['slots'][11:])+1.3*plan['completed_engineering_seconds'];assert forecast<=plan['full_comparison_tranche_seconds']
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 audit=dict(at=now,binding=binding,source_commit=head,slot=11,free_bytes=free,required_free_bytes=need,margin_bytes=free-need,retained_endpoints_bytes=retained,shared_lock_available=True,active_workers=[],conservative_total_seconds_with_remaining_slot_ceilings=forecast)
 save(P/'launch-admission-10.json',audit)
 print(json.dumps(audit))
