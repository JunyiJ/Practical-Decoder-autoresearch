"""Publish the reviewed prospective005 bytes once, after actual full capacity fits.

No old campaign retry, numerical rerun or scientific trajectory reuse. The exact
preview plan and metadata ledger are preserved; only its proper output location
becomes runnable after prerequisite and capacity verification.
"""
import base64,gzip,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research import tokenizer_wide_hybrid_plan_v2 as p
from src.research.protocol import digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.tokenizer_wide_hybrid_retention_audit import numerical_review
from src.research.tokenizer_wide_hybrid_metadata import budgets,MAX_FILES
E=Path(__file__).resolve().parent;PREVIEW=E/'census-campaign005-preview';RUN=ROOT/'research_runs/tokenizer-wide-hybrid-005'

def read(pth):return json.loads(pth.read_text())
def save(pth,v):
 with pth.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def prepare():
 assert not RUN.exists()
 review=read(E/'census-campaign005-preview-review.json');transport=read(E/'census-preview-transport-review-v2.json')
 for name in ('census-campaign005-preview-guard.json','census-preview-transport-review-v2-guard.json'):
  g=read(E/name);assert g['status']=='finished' and g['reason'] is None
 assert review['status']=='preview_only_not_frozen' and transport['status']=='prospective_transport_verified_not_frozen'
 assert transport['preview_sha256']==file_digest(E/'census-campaign005-preview-review.json')
 assert transport['plan_sha256']==review['plan_sha256']==file_digest(PREVIEW/'meta/plan.json')
 source,runtime=p.source_fingerprint(),p.runtime_identity()
 assert digest(source)==review['source_sha256'] and digest(runtime)==review['runtime_sha256']
 plan=read(PREVIEW/'meta/plan.json');assert plan['output']==str(RUN) and plan['source']==source and plan['runtime']==runtime
 assert plan['requests']==p.request_identity(p.request_options(RUN))
 for row in plan['files']:
  f=Path(row['path']);local=PREVIEW/f.relative_to(RUN/'evidence') if f.is_relative_to(RUN/'evidence') else f
  assert file_digest(local)==row['sha256']
 trace=json.loads(gzip.decompress((PREVIEW/'meta/source-applicability.json.gz').read_bytes()))
 assert trace['not_a_new_numerical_or_input_run'] and trace['archived_originals_preserved']
 for n,v in trace['originals'].items():assert (ROOT/n).read_bytes()==base64.b64decode(v['bytes_base64']) and file_digest(ROOT/n)==v['sha256']
 component=read(E/'confirmation-census-v2-review.json')
 for n,h in component['source_and_test_hashes'].items():assert file_digest(ROOT/n)==h
 for name,v in component['tests'].items():
  assert file_digest(E/('confirmation-census-v2-'+name+'-tests.log'))==v['log_sha256']
  assert read(E/('confirmation-census-v2-'+name+'-guard.json'))==v['guard']
 archival=read(E.parent/'checkpoint-compaction-002/logical-evidence-review.json')
 for n,v in archival['archives'].items():assert (ROOT/n).stat().st_size==v['size'] and file_digest(ROOT/n)==v['sha256']
 for n in archival['archived_members']:assert not (ROOT/n).exists()
 for group,refs in archival['receipt_hashes'].items():
  for name,h in refs.items():assert file_digest(E.parent/group/name)==h
 compression=E.parent/'filesystem-compression-001';completed=read(compression/'completed.json');guard=read(compression/'continuation-guard.json')
 assert completed['status']=='completed' and completed['all_original_paths_and_readable_bytes_preserved']
 assert guard['status']=='finished' and guard['reason'] is None
 assert file_digest(compression/'plan.json')==completed['manifest_sha256'] and file_digest(compression/'journal.jsonl')==completed['journal_sha256']
 assert file_digest(compression/'continuation-journal.jsonl')==completed['continuation_journal_sha256']
 numerical=read(PREVIEW/'meta/numerical.json')
 numerical_review(PREVIEW,plan['numerical_ref'],dict(source_sha256=digest(source),runtime_sha256=digest(runtime),corpus_sha256=numerical['corpus_sha256']))
 assert review['metadata_fits'] and review['conservative_file_count']<MAX_FILES
 for group,v in review['groups'].items():assert v['bound_bytes']<=budgets()[group] and v['cap_bytes']==budgets()[group]
 required=p.storage_envelope(read(p.PROJECTION))['required_free_bytes']+128*1024
 assert required==review['required_free_bytes'] and shutil.disk_usage(ROOT).free>=required
 assert p.source_fingerprint()==source and p.runtime_identity()==runtime
 # Exact existing candidate bytes, no mutation of the preview or historical run.
 shutil.copytree(PREVIEW/'meta',RUN/'evidence/meta')
 assert file_digest(RUN/'evidence/meta/plan.json')==review['plan_sha256']
 bound=p.bind_campaign(RUN/'evidence/meta/plan.json',review['plan_sha256'])
 assert bound['numerical_review_bound'] and bound['training_input_bound'] and not bound['dispatch_authorized']
 assert shutil.disk_usage(ROOT).free>=required
 result=dict(status='frozen_pending_new_resource_pilots',campaign=str(RUN),plan_sha256=review['plan_sha256'],source_sha256=digest(source),runtime_sha256=digest(runtime),preview_review_sha256=file_digest(E/'census-campaign005-preview-review.json'),transport_review_sha256=file_digest(E/'census-preview-transport-review-v2.json'),script_sha256=file_digest(Path(__file__)),compression_receipt_sha256=file_digest(compression/'completed.json'),archival_review_sha256=file_digest(E.parent/'checkpoint-compaction-002/logical-evidence-review.json'),ledger=review['ledger'],groups=review['groups'],conservative_file_count=review['conservative_file_count'],required_free_bytes=required,observed_free_bytes=shutil.disk_usage(ROOT).free,pilot_requests=3,quality_trajectories=24,inference_panels=3,training_launched=False,quality_launch_authorized=False,closed_campaigns_reopened=False,numerical_or_input_rerun=False,old_process_or_cleanup_receipts_reused=False,prior_quality_results_not_pooled=True,screen_seed_reused_not_independent=True,qualification='Exact prospective005 plan bytes now published after measured storage recovery. Original numerical/input provenance and archive identities verified. New serial pilots and each production acceptance gate still required; no old scientific result is pooled.')
 save(E/'campaign005-freeze-v2-review.json',result)
 shutil.copytree(RUN/'evidence/meta',E/'campaign005-frozen/meta')
 return result
if __name__=='__main__':
 with session_lock(ROOT/'research_state/tokenizer-worker-lock'),CpuPrerequisiteGuard(E/'campaign005-freeze-v2-guard.json',ROOT,timeout_seconds=600):r=prepare()
 print(json.dumps({k:r[k] for k in ('status','plan_sha256','required_free_bytes','observed_free_bytes')}),flush=True)
