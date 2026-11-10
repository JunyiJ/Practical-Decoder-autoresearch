"""One-shot slot05 recovery: retain first attempt and restore canonical log prefix."""
import datetime,fcntl,hashlib,json,os,shutil,subprocess
from pathlib import Path
from research.general_rsi_training_v2 import read_checkpoint,sha
ROOT=Path.cwd();P=ROOT/'research/tokenizer-evidence/general-rsi-exposure-001';D=P/'slot-05'
OUT=ROOT/'research/tokenizer-evidence/general-rsi-exposure-recovery-002'
def put(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:os.kill(58135,0)
 except ProcessLookupError:pass
 else:raise AssertionError('prior process still exists')
 plan=json.loads((P/'plan.json').read_text());assert sha(P/'plan.json')=='f4bacb74f0b82627e1a5639890f53b5911d2d2bba05fa0c8b0276c17fde19ad6'
 for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
 guard=json.loads((D/'guard-01.json').read_text());assert guard['status']=='failed' and guard['reason']=='disk_reserve'
 receipt=json.loads((D/'latest.json').read_text());payload=read_checkpoint(receipt);step=payload['step'];assert step==6144
 assert payload['config']['binding']==sha(P/'plan.json') and payload['config']['roles']==plan['slots'][5]['roles'] and payload['config']['horizon']==20480
 raw=(D/'steps.jsonl').read_bytes();lines=raw.splitlines(keepends=True);rows=[json.loads(x) for x in lines]
 assert [x['step'] for x in rows]==list(range(1,6706))
 for historical,logged in zip(payload['history'],rows):
  assert all(logged[k]==v for k,v in historical.items())
 assert len(payload['history'])==step
 retained=sum(f.stat().st_size for f in (ROOT/'research_runs/general-rsi-exposure-001').glob('slot-*/*.pt') if f.name in ('raw.pt','h200.pt'))
 required=plan['reserve_bytes']+plan['incremental_peak_bytes']-retained
 assert shutil.disk_usage(ROOT).free>=required+len(raw)*2
 OUT.mkdir();archive=OUT/'attempt-01-original';archive.mkdir();evidence={}
 for f in sorted(D.iterdir()):
  assert f.is_file() and not f.is_symlink()
  target=archive/f.name
  with target.open('xb') as stream:stream.write(f.read_bytes());stream.flush();os.fsync(stream.fileno())
  assert sha(target)==sha(f);evidence[f.name]={'size':f.stat().st_size,'sha256':sha(f)}
 diagnosis={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'classification':'resource_failure_not_scientific_negative','exit_code':90,'guard':guard,'checkpoint':receipt,'checkpoint_step':step,'last_first_attempt_logged_step':6705,'replayed_updates':561,'original_files':evidence,'source_bindings_verified':len(plan['sources']),'canonical_log_policy':'Archive all first-attempt files unchanged; retain canonical steps1..6144 matching checkpoint history, replay6145..6705 with identical roles/optimizer/RNG/EMA; compare replay numerics to original before accepting endpoint. Original guard and attempt remain in slot directory.','root_cause_limits':'Guard observed free10732765184 below10737418240 reserve. No checkpoint write at failure (step6705); unrelated filesystem consumption cause not established. Initial inspected free10730217472 was747109308 below full requirement11477326780. A size-based research/cache inventory identified retained historical checkpoint families and excluded updater/shared-runtime payloads. Capacity subsequently recovered above the full requirement without deletion; external consumer not established.','required_free_bytes':required,'free_bytes':shutil.disk_usage(ROOT).free,'failed_seconds_charged':guard['elapsed_seconds'],'scientific_contract_changed':False,'test_opened':False}
 put(OUT/'preparation.json',diagnosis)
 # Durable first-attempt archive and intent precede canonical prefix replacement.
 temp=D/'steps.recovery.tmp'
 with temp.open('xb') as stream:stream.write(b''.join(lines[:step]));stream.flush();os.fsync(stream.fileno())
 os.replace(temp,D/'steps.jsonl')
 for name,record in evidence.items():assert sha(archive/name)==record['sha256']
 assert sha(receipt['path'])==receipt['sha256']
 put(OUT/'prepared.json',{'canonical_prefix_steps':step,'canonical_steps_sha256':sha(D/'steps.jsonl'),'archived_first_attempt_verified':True,'complete_checkpoint_verified':True})
 print(json.dumps(diagnosis,default=str))
