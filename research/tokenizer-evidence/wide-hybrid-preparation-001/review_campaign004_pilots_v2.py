"""Read-only immutable pilot evidence review while serial quality worker runs."""
import base64,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
E=Path(__file__).resolve().parent;RUN=ROOT/'research_runs/tokenizer-wide-hybrid-004';R=RUN/'evidence'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
a=read(E/'campaign004-pilots-v2-archive.json');b=read(E/'campaign004-pilot-boundary-v2.json');freeze=read(E/'campaign004-freeze-v2-review.json')
assert b['review']['pilot_boundary_verified'] is True
ref=b['reference'];assert sha(R/ref['path'])==ref['sha256']
terminal=read(R/ref['path']);assert terminal['plan_sha256']==freeze['plan_sha256'] and terminal['adopted'] is False
manifest=b['review']['chain_stage']['manifest'];deleted=manifest['delete'];assert len(deleted)==6
for row in deleted:
 assert not (R/row['path']).exists()
 assert a['files']['evidence/'+row['path']]==row['sha256']
for n,h in manifest['evidence'].items():assert sha(R/n)==h
execution=b['review']['chain_stage']['execution_ref'];assert sha(R/execution['path'])==execution['sha256']
for n,encoded in a['noncheckpoint_bytes_base64'].items():
 data=base64.b64decode(encoded);assert hashlib.sha256(data).hexdigest()==a['files'][n]
 # Mutable control samples are archival snapshots; immutable evidence must match.
 if n.startswith('evidence/'):assert (RUN/n).read_bytes()==data
rows=[]
for i in range(3):
 p=R/'pilots'/f'{i:02d}';audit=read(p/'audit.json');res=read(p/'resource-audit.json');proc=read(p/'process/process.json')
 assert proc['status']=='completed' and proc['returncode']==0 and proc['error'] is None
 assert res['status']=='completed' and not res['violations'] and res['cache_audit_sha256']==sha(p/'audit.json')
 assert audit['adopted'] is False and audit['status']=='pilot_cache_checks_complete_pending_resource_review'
 assert sha(p/res['events']['path'])==res['events']['sha256']
 rows.append(dict(slot=i,family=audit['spec']['family'],steps=audit['spec']['steps'],elapsed_seconds=proc['elapsed_seconds'],
  samples=len(res['samples']),peak_rss_bytes=max(v['peak_rss_bytes'] for v in res['samples']),peak_mps_driver_bytes=max(v['mps_driver_bytes'] for v in res['samples']),
  minimum_free_bytes=min(v['free_bytes'] for v in res['samples']),cache_audit_sha256=sha(p/'audit.json'),resource_audit_sha256=sha(p/'resource-audit.json')))
result=dict(status='three_pilots_and_recorded_boundary_verified',plan_sha256=freeze['plan_sha256'],boundary_ref=ref,pilots=rows,
 deleted_successful_pilot_weights=6,deleted_logical_bytes=sum(v['size'] for v in deleted),
 archived_noncheckpoint_files=len(a['noncheckpoint_bytes_base64']),immutable_metadata_verified=True,
 current_quality_slot=0,quality_result_available=(R/'quality/00/result.json').exists(),quality_or_adoption_claim=False,
 qualification='Read-only check of immutable completed pilot records and prior audited cleanup receipts. Did not reacquire active worker locks, load removed weights, rerun pilots or modify the live campaign. Sampled resource peaks are not instantaneous upper bounds.')
with (E/'campaign004-pilots-reviewed-v2.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result),flush=True)
