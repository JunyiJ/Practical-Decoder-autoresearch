"""Verify the real interrupted-run replay without changing the active worker."""
import datetime,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
original=(OUT/'attempt-01-original/steps.jsonl').read_bytes()
current=(ROOT/'research/tokenizer-evidence/general-rsi-exposure-001/slot-05/steps.jsonl').read_bytes().splitlines(keepends=True)[:6705]
a=[json.loads(x) for x in original.splitlines()]
b=[json.loads(x) for x in current]
assert len(a)==len(b)==6705
keys=sorted(set(a[0])-{'seconds','mps_driver_bytes'})
assert keys==['batch_sha256','clip','loss','lr_ratio','step']
assert [x['step'] for x in b]==list(range(1,6706))
for old,new in zip(a,b):
 assert all(old[k]==new[k] for k in keys),old['step']
prep=json.loads((OUT/'preparation.json').read_text())
for name,row in prep['original_files'].items():
 p=OUT/'attempt-01-original'/name
 assert p.stat().st_size==row['size'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],name
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'replay_start_step':6145,'replay_end_step':6705,'replay_updates':561,'canonical_prefix_records':6705,'exactly_equal_fields':keys,'excluded_execution_fields':['seconds','mps_driver_bytes'],'original_steps_sha256':hashlib.sha256(original).hexdigest(),'resumed_prefix_sha256':hashlib.sha256(b''.join(current)).hexdigest(),'original_failure_archive_hashes_verified':True,'conclusion':'All561 replayed numerical updates match exactly. Complete restart preserves this observed trajectory; endpoint acceptance and remaining third-pair confirmation are still pending.','test_opened':False}
with (OUT/'replay-verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(result))
