"""Exercise smaller source-reference receipts with the unchanged metadata caps."""
import copy,json
from pathlib import Path
import torch
from research.final_ln_scale_campaign_v2 import pilot as p
from src.research.tokenizer_wide_hybrid_metadata import snapshot,budgets
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=Path(__file__).resolve().parent;FIX=B/'pilot002-publication-fixture'
def reject(fn):
 try:fn()
 except (ValueError,AssertionError):return
 raise AssertionError('Forged reference accepted')
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as global_fd:
 p.idle();assert not FIX.exists() and not p.ROOT.exists()
 binding=json.loads((B/'pilot002-source-plan.json').read_text());p.c.verify_sources(binding['sources'])
 assert binding['publication_runner']==p.ref(__file__)
 old=p.OLDROOT/'evidence/pilots/00';worker=p.read(p.ref(old/'worker.json'));original=worker['acceptance_request'];sources=binding['sources']
 assert all(sources.get(n)==h for n,h in original['sources'].items())
 request=copy.deepcopy(original);request['sources']=sources
 with p.Guard(B/'pilot002-publication-preflight-guard.json'):
  accepted=p.a.accept(request,worker['validation']);p.clear()
  row=dict(slot=0,arm='learned',plan=p.ref(B/'pilot001-plan.json'),worker=p.ref(old/'worker.json'),acceptance_request_sha256=p.c.fingerprint(request),validation=worker['validation'],acceptance=accepted,process=p.ref(old/'process/process.json'),worker_guard=p.ref(old/'worker-guard.json'),parent_guard=p.ref(old/'parent-guard.json'),imported=True,repeated_training_updates=0,checkpoint_copies=0)
  assert p.request_for(row,dict(sources=sources))==request
  bad=copy.deepcopy(row);bad['acceptance_request_sha256']='0'*64;reject(lambda:p.request_for(bad,dict(sources=sources)))
  foreign=dict(sources=sources.copy());foreign['sources'][next(iter(original['sources']))]='0'*64;reject(lambda:p.request_for(row,foreign))
  FIX.mkdir()
  with session_lock(FIX) as run_fd:
   writer=MetadataWriter(FIX,(global_fd,run_fd))
   # Exact completed numerical metadata is copied only as a disposable transport fixture.
   for name in ['config.json','summary.json','metrics.jsonl','request.json','worker-guard.json','parent-guard.json']:
    value=json.loads((old/name).read_text()) if name!='metrics.jsonl' else None
    if value is not None:writer.publish_json('evidence/pilots/00/'+name,value)
   cloned=copy.deepcopy(worker);cloned['acceptance_request']=request
   writer.publish_json('evidence/pilots/00/worker.json',cloned)
   writer.publish_json('evidence/pilots/00/accepted.json',row)
   observed=snapshot(FIX);group='evidence/pilots/00/other';assert observed['groups'][group]+16*1024<budgets()[group]
   assert not any(n.endswith('.tmp') for n in observed['files']) and observed['checkpoint_logical_bytes']==0
  result=dict(status='source_reference_receipt_publication_passed',runner=p.ref(__file__),source_plan=p.ref(B/'pilot002-source-plan.json'),sources=sources,receipt_bytes=len(p.c.canonical(row).encode())+1,observed_group_bytes=observed['groups'][group],unchanged_group_cap=budgets()[group],additional_margin_bytes=16*1024,reference_forgery_rejected=True,foreign_source_bridge_rejected=True,original_worker=p.ref(old/'worker.json'),read_only_parent_acceptance=accepted,training_updates=0,checkpoint_copies=0,fixture_scope='Disposable metadata publication only; original completed learned endpoint independently re-audited. No second-arm scientific result claimed.',original_failed_round_unchanged=True)
 result['guard']=p.ref(B/'pilot002-publication-preflight-guard.json');assert p.read(result['guard'])['status']=='finished'
 p.exclusive(B/'pilot002-publication-preflight.json',result)
 print(json.dumps({k:result[k] for k in ['status','receipt_bytes','observed_group_bytes','unchanged_group_cap','training_updates','checkpoint_copies']}))
