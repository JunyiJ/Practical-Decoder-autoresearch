"""Validate a separate execution continuation; no new training or scientific panel."""
import copy,json
from pathlib import Path
import torch
from research.final_ln_scale_quality_v3 import campaign_contract as c,quality_execution as q,artifact_acceptance as a
from research.final_ln_scale_quality_v3.continuation_contract import verify_imports,resolve_request,OLDROOT
from research.final_ln_scale_quality_v3.support import idle,cache_audit,exclusive
from research.final_ln_scale_quality_v2 import artifact_acceptance as previous_a,quality_execution as previous_q
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=q.EVIDENCE

def rejects(fn):
 try:fn()
 except (ValueError,AssertionError,TypeError):return
 raise AssertionError('Invalid continuation contract accepted')

torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
 idle();sources=c.source_inventory();assert len(sources)==918
 failure=read(reference(B/'quality001-slot03-entry-failure001.json'));original_ref=reference(B/'quality001-plan.json');original=read(original_ref)
 assert all(sources[n]==h for n,h in original['sources'].items()) and len(original['sources'])==912
 assert not q.ROOT.exists() and not q.FREEZE.exists()
 plan=dict(kind='same_panel_execution_continuation_source_integration',sources=sources,original_plan=original_ref,entry_failure=reference(B/'quality001-slot03-entry-failure001.json'),original_STOP=failure['original_STOP'],original_root=str(OLDROOT),original_anchors=failure['accepted_anchors'],imported_count=3,entry_headroom_bytes=c.ENTRY_HEADROOM,seeds=original['seeds'],fresh=original['fresh'],design=original['design'],scientific_panel=original['builder'],scientific_freeze=failure['original_freeze'],runner=reference(Path(__file__).resolve()),quality_training_launched=False,weight_copies=0)
 verify_imports(plan);exclusive(B/'quality002-integration001-plan.json',plan)
 with q.QualityGuard(B/'quality002-integration001-guard.json',c.ROOT,600):
  assert Path(a.__file__).read_bytes()==Path(previous_a.__file__).read_bytes()
  assert q.FULL_FREE==previous_q.FULL_FREE+1458176 and q.SECONDS==previous_q.SECONDS==7200 and q.ALLOWANCES==previous_q.ALLOWANCES and c.LIMITS==previous_q.c.LIMITS
  for slot in (0,1,2,6,-1,True):rejects(lambda:q.run_once(None,slot,'0'*64))
  future=[]
  for spec in original['fresh'][3:]:
   cfg=c.trial_config(q.ROOT/f"evidence/quality/{spec['slot']:02d}",spec,original['seeds']);old=previous_q.c.trial_config(OLDROOT/f"evidence/quality/{spec['slot']:02d}",spec,original['seeds'])
   left,right=copy.deepcopy(cfg),copy.deepcopy(old)
   for d in (left,right):
    for k in ('checkpoint_dir','metrics_file','summary_file'):del d['training'][k]
   assert left==right;future.append(cfg)
  for mutate in [lambda p:p.update(imported_count=2),lambda p:p['fresh'][3].update(seed=200070099),lambda p:p.update(entry_headroom_bytes=0),lambda p:p['original_anchors'][0].update(sha256='0'*64)]:
   bad=copy.deepcopy(plan);mutate(bad);rejects(lambda:verify_imports(bad))
  imports=[]
  for slot,anchor_ref in enumerate(plan['original_anchors']):
   anchor=read(anchor_ref);prior=read(anchor['acceptance']);worker=read(prior['worker']);req=copy.deepcopy(worker['acceptance_request']);req['sources']=sources
   descriptor=dict(imported=True,worker=prior['worker'],original_acceptance=anchor['acceptance'],request_sha256=c.fingerprint(req))
   assert resolve_request(descriptor,plan,q.KIND)==req
   bad=copy.deepcopy(descriptor);bad['request_sha256']='0'*64;rejects(lambda:resolve_request(bad,plan,q.KIND))
   bad=copy.deepcopy(descriptor);bad['original_acceptance']=read(plan['original_anchors'][(slot+1)%3])['acceptance'];rejects(lambda:resolve_request(bad,plan,q.KIND))
   accepted=a.accept(req,prior['artifact_acceptance']['validation']);cached=cache_audit(req)
   assert accepted['accepted'] and accepted['audit']==prior['artifact_acceptance']['audit'] and accepted['validation']==prior['artifact_acceptance']['validation']
   assert c.canonical(cached)==c.canonical(prior['cache'])
   imports.append(dict(slot=slot,spec=original['fresh'][slot],original_anchor=anchor_ref,original_acceptance=anchor['acceptance'],original_worker=prior['worker'],original_parent_guard=anchor['parent_guard'],original_files=anchor['files'],request_sha256=c.fingerprint(req),acceptance=accepted,cache=cached,retrained_updates=0,weight_copies=0))
   import gc;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
  retained=sum(Path(x['path']).stat().st_size for item in imports for x in item['original_files'] if x['path'].endswith('.pt'));assert retained==1677774994
  verify_imports(plan);c.verify_sources(sources);assert not q.ROOT.exists() and not q.FREEZE.exists()
 result=dict(status='passed_same_panel_zero_copy_continuation_integration',plan=reference(B/'quality002-integration001-plan.json'),guard=reference(B/'quality002-integration001-guard.json'),source_count=918,predecessors_preserved=912,acceptance_implementation_byte_identical=True,imports=imports,future_configs=future,imported_weight_bytes=retained,original_partial_entry_closed=True,original_STOP_preserved=True,remaining_numerical_trajectories=3,new_scientific_panel=False,new_scientific_seeds=False,training_updates=0,weight_copies=0,quality_frozen=False,quality_training_launched=False,quality_adopted=False,holdout_scored=False,scope='Same original frozen six-trajectory scientific panel. Recompute actual accepted prefix under an explicitly additive source bridge;new output paths only for the three trajectories that never started. No old campaign score pooling or selection.')
 exclusive(B/'quality002-integration001.json',result);print(json.dumps(dict(status=result['status'],sources=918,imported=3,training_updates=0,weight_copies=0)),flush=True)
