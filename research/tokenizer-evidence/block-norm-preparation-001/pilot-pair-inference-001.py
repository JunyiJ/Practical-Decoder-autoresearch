"""Read-only timing of the accepted LN and separately recovered RMS pilot pair."""
import argparse,copy,json,shutil,sys
from datetime import datetime,timezone
from pathlib import Path
import torch
from research.block_norm_v1 import campaign_contract as c,inference as i,pilot_continuation as p,artifact_acceptance as a
from research.block_norm_v1.fixture_dispatch import reference,read
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
b=p.EVIDENCE;stem='pilot-pair-inference-001';plan_path=b/(stem+'-plan.json');guard_path=b/(stem+'-guard.json');result_path=b/(stem+'-result.json');review_path=b/(stem+'-review.json')
parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);parser.add_argument('--plan-sha256');args=parser.parse_args();torch.set_num_threads(4)

def verify(plan):
 c.require(reference(Path(__file__).absolute())==plan['runner'],'Runner changed')
 c.verify_sources(plan['sources']);c.frozen_predecessors(plan['sources']);c.require(c.environment()==plan['environment'],'Environment changed')
 continuation=read(plan['continuation']);p.verify_carried(continuation)
 accepted=read(plan['layer_acceptance']);anchor=read(plan['layer_anchor']);c.require(anchor['acceptance']==plan['layer_acceptance'] and accepted['artifact_acceptance']['accepted'],'Layer parent acceptance changed')
 for root_ref in plan['root_inventories']:
  inventory=read(root_ref);root=Path(inventory['root']);actual={str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()}
  c.require(actual==set(inventory['inventory']),'Pilot root inventory changed')
  for n,row in inventory['inventory'].items():
   path=root/n;c.require(path.stat().st_size==row['size'] and c.file_sha(path,max_bytes=512*1024**2)==row['sha256'],'Pilot file changed')
 for req in plan['requests'].values():a.verify_request(req)
 c.require(i.identity(plan['requests'],0)==plan['identity'],'Pair identity changed')

with session_lock(GLOBAL_LOCK.parent):
 idle()
 if args.mode=='prepare':
  c.require(not plan_path.exists(),'No plan replacement')
  with CpuPrerequisiteGuard(b/(stem+'-prepare-guard.json'),c.ROOT,timeout_seconds=120,poll_seconds=.25):
   continuation=read(reference(b/'pilot002-plan.json'));p.verify_carried(continuation)
   rms=read(reference(b/'validation-recovery-plan-001.json'))['pilot_request'];ln=read(reference(p.ROOT/'evidence/pilots/01/worker.json'))['acceptance_request'];sources=c.source_inventory()
   c.require(sources==continuation['sources'],'Continuation source changed')
   original_refs=dict(rms=reference(b/'validation-recovery-plan-001.json'),layer=reference(p.ROOT/'evidence/pilots/01/worker.json'))
   rms=copy.deepcopy(rms);rms['sources']=sources
   requests=dict(rms_incumbent=rms,layer_matched_epsilon=ln)
   plan=dict(status='frozen_readonly_pilot_pair_inference',observed_at=datetime.now(timezone.utc).isoformat(),runner=reference(Path(__file__).absolute()),sources=sources,environment=c.environment(),requests=requests,identity=i.identity(requests,0),
    original_requests=original_refs,continuation=reference(b/'pilot002-plan.json'),layer_acceptance=reference(p.ROOT/'evidence/pilots/01/accepted.json'),layer_anchor=reference(b/'pilot002-accepted-01.json'),root_inventories=[reference(b/'pilot001-failure.json'),reference(b/'pilot002-result.json')],
    source_rebinding='RMS request copied for new read-only measurement only; all recovery source hashes verified unchanged, additive continuation module only. Original request/process preserved.',
    limits=dict(seconds=600,cpu_threads=4,rss_bytes=4*1024**3,driver_bytes=6*1024**3,reserve_bytes=8*1024**3),free_bytes=shutil.disk_usage(c.ROOT).free,quality_adoption_allowed=False,training_allowed=False,deletion_allowed=False,holdout_scored=False)
   verify(plan);p.exclusive(plan_path,plan)
  print(json.dumps(dict(status='prepared',plan=reference(plan_path))))
 else:
  plan=read(dict(path=str(plan_path),sha256=args.plan_sha256));c.require(not any(f.exists() for f in (guard_path,result_path,review_path)),'Existing attempt cannot retry')
  error=None
  try:
   with StrictMpsGuard(guard_path,c.ROOT):
    verify(plan);result=i.measure_pair(plan['requests'],0);review=i.review(result,plan['requests'],0);verify(plan)
    p.exclusive(result_path,result)
   guard=read(reference(guard_path));c.require(guard['status']=='finished' and guard['reason'] is None,'Guard did not complete')
   value=dict(status='passed',plan=reference(plan_path),result=reference(result_path),guard=reference(guard_path),values_review=review,original_pilot_roots_unchanged=True,original_rms_process_reclassified=False,quality_adopted=False,training_launched=False,holdout_scored=False,deletion_performed=False)
  except Exception as exc:
   error=repr(exc);value=dict(status='failed',error=error,plan=reference(plan_path),no_retry=True)
  p.exclusive(review_path,value);print(json.dumps(value));sys.exit(1 if error else 0)
