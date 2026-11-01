"""Account disjoint enforced weight/metadata budgets; no dispatch authority."""
import hashlib,json,shutil
from pathlib import Path
from src.research.tokenizer_wide_hybrid_metadata import budgets
from research.mla_score_v1.output import RAW_CAP,EXPORT_CAP,WEIGHT_CAP,METADATA_CAP
ROOT=Path.cwd();B=Path(__file__).resolve().parent

def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 archive=json.loads((B/'worker-archive001/completed.json').read_text());assert archive['status']=='completed'
 worker=json.loads((B/'trainer-worker001.json').read_text());assert worker['status']=='real_bounded_worker_parent_acceptance_passed'
 metadata=budgets();assert sum(metadata.values())==256*1024**2
 per_trial=sum(v for k,v in metadata.items() if k.startswith('evidence/quality/00/'))
 assert per_trial==8650752 and per_trial<=METADATA_CAP
 assert WEIGHT_CAP==RAW_CAP+EXPORT_CAP==560*1024**2
 phases={'pilots_before_retirement':2,'screens_after_pilot_retirement':6,'selected_screens_plus_fresh_pairs':8}
 reserve=8*1024**3
 requirements={k:n*WEIGHT_CAP+sum(metadata.values())+reserve for k,n in phases.items()}
 required=max(requirements.values());assert required==13555990528
 old=13690208256;assert old-required==8*METADATA_CAP
 free=shutil.disk_usage(ROOT).free
 result=dict(status='measured_disjoint_budget_projection',required_free_bytes=required,observed_free_bytes=free,margin_bytes=free-required,
  raw_cap_bytes=RAW_CAP,export_cap_bytes=EXPORT_CAP,trajectory_weight_cap_bytes=WEIGHT_CAP,
  all_campaign_metadata_cap_bytes=sum(metadata.values()),per_trajectory_metadata_included_bytes=per_trial,reserve_bytes=reserve,
  phase_retained_trajectory_counts=phases,phase_required_free_bytes=requirements,
  previous_requirement_bytes=old,removed_duplicate_metadata_charge_bytes=old-required,
  serialization='Streaming temporary is charged within active trajectory weight cap. Same-inode link/unlink publication does not create a second payload. Existing checkpoint bytes and partial files count against aggregate cap. Metadata temporary overlap is charged by MetadataWriter within the complete 256MiB budget.',
  guards_relaxed=False,confirmation_pairs_reduced=False,dispatch_authorized=False,
  outstanding=['Prospectively test exact successful-pilot and losing-screen retirement boundaries before source/finite pilot freeze.','Fresh seed census and immutable finite pilot plan; guarded pilots then final quality plan.','All further prerequisite outputs and external disk changes must be recharged at dispatch.'],
  worker=ref(B/'trainer-worker001.json'),archive=ref(B/'worker-archive001/completed.json'),
  sources={str(p):ref(p)['sha256'] for p in [Path(__file__).resolve(),ROOT/'research/mla_score_v1/output.py',ROOT/'src/research/tokenizer_wide_hybrid_metadata.py',ROOT/'src/research/tokenizer_wide_hybrid_metadata_writer.py',ROOT/'src/research/tokenizer_wide_hybrid_checkpoint_writer.py']})
 with (B/'capacity003.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ['status','required_free_bytes','observed_free_bytes','margin_bytes','dispatch_authorized']}))
if __name__=='__main__':main()
