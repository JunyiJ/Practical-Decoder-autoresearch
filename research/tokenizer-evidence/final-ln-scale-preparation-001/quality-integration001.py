"""Finite quality-source and real cross-root pilot audit; no training or retirement."""
import copy,json,os
from pathlib import Path
import torch,pytest
from research.final_ln_scale_quality_v1 import campaign_contract as c,quality_execution as q,artifact_acceptance as a
from research.final_ln_scale_quality_v1.support import cache_audit
from research.final_ln_scale_quality_v1.pilot_retirement import targets
from research.final_ln_scale_campaign_v2 import pilot
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=q.EVIDENCE

def inventory(root):return {str(p.relative_to(root)):q.file_ref(p,p.suffix=='.pt') for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('.lock','.metadata.lock')}
class Report:
 def __init__(self):self.tests=[]
 def pytest_runtest_logreport(self,report):
  if report.when=='call' or report.failed:self.tests.append(dict(nodeid=report.nodeid,phase=report.when,outcome=report.outcome,details=str(report.longrepr) if report.failed else None))

torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g:
 q.idle();sources=c.source_inventory();prior=read(reference(B/'pilot002-plan.json'));terminal=read(reference(B/'pilot002-terminal-review.json'));roots={};requests={};validation={}
 assert len(sources)==906 and len(prior['sources'])==898 and all(sources[n]==h for n,h in prior['sources'].items())
 assert not q.ROOT.exists() and not q.FREEZE.exists()
 assert (pilot.OLDROOT/'STOP').exists() and not (pilot.OLDROOT/'evidence/pilots/01').exists()
 original_stop=reference(pilot.OLDROOT/'STOP');failure=reference(pilot.OLDROOT/'evidence/pilots/00/accepted.json.tmp')
 expected_targets=targets()
 for item in terminal['accepted']:
  row=read(item['acceptance']);req=pilot.request_for(row,prior);req.update(scope=a.SCOPE,sources=sources)
  arm=item['arm'];requests[arm]=req;validation[arm]=row['validation'];root=Path(req['files']['config']['path']).parent;roots[str(root)]=inventory(root)
 plan=dict(sources=sources,added_sources=sorted(set(sources)-set(prior['sources'])),seeds=prior['seeds'],requests=requests,validation=validation,protected_pilot_roots=roots,
  disposable_root=str(c.ROOT/'research_runs/tokenizer-final-ln-scale-quality-v1-disposable001'),runner=reference(Path(__file__).resolve()),original_stop=original_stop,original_failed_partial_receipt=failure,eligible_pilot_weights=expected_targets,quality_training_launched=False,
  quality_design=q.rules.QUALITY_DESIGN,full_required_bytes=q.FULL_FREE,cleanup_target_bytes=q.FULL_FREE+2*1024**3)
 q.exclusive(B/'quality-integration001-plan.json',plan)
 os.environ['FINAL_LN_SCALE_INTEGRATION_PLAN']=str(B/'quality-integration001-plan.json');os.environ['FINAL_LN_SCALE_GLOBAL_FD']=str(g)
 report=Report();audits={}
 with q.QualityGuard(B/'quality-integration001-guard.json',c.ROOT,600):
  code=pytest.main(['-q','-p','no:cacheprovider','research/final_ln_scale_quality_v1/test_contracts.py'],plugins=[report])
  if code==0:
   for arm,req in requests.items():
    accepted=a.accept(req,validation[arm]);cached=cache_audit(req);audits[arm]=dict(audit=accepted['audit'],cache=cached,validation_exact=accepted['validation']==validation[arm])
    assert accepted['audit']['ema_parameters']==(88 if arm=='learned' else 87)
    assert accepted['audit']['optimizer_groups']==dict(beta1=.9,beta2=.999,epsilon=1e-6,adamw_groups=44 if arm=='learned' else 43,muon_groups=44)
    assert accepted['audit']['clipping']['updates']==20
    forged=copy.deepcopy(req);forged['scope']=a.TRIAL_SCOPE;forged['trial']=dict(spec=next(s for s in q.rules.fresh_specs(prior['seeds']) if s['arm']==arm),seeds=prior['seeds'])
    try:a.verify_request(forged)
    except ValueError:pass
    else:raise AssertionError('Pilot bundle accepted as quality endpoint')
    audits[arm]['pilot_cannot_impersonate_quality']=True
    bad=copy.deepcopy(req);bad['sources']=dict(sources);bad['sources'][next(iter(sources))]='0'*64
    try:a.verify_request(bad)
    except ValueError:pass
    else:raise AssertionError('Mutated source accepted')
    del accepted;pilot.clear()
  unchanged=all(inventory(Path(root))==inv for root,inv in roots.items());c.verify_sources(sources)
  assert reference(pilot.OLDROOT/'STOP')==original_stop and reference(pilot.OLDROOT/'evidence/pilots/00/accepted.json.tmp')==failure
 result=dict(status='passed' if code==0 and unchanged and len(audits)==2 else 'failed',exit_code=int(code),tests=report.tests,actual_pilot_audits=audits,protected_pilot_roots_unchanged=unchanged,plan=reference(B/'quality-integration001-plan.json'),guard=reference(B/'quality-integration001-guard.json'),scientific_quality_launched=False,scientific_fresh_seeds_consumed=False,holdout_scored=False,training_updates=0,weight_copies=0,source_count=len(sources),predecessors_preserved=898)
 q.exclusive(B/'quality-integration001-review.json',result)
 assert result['status']=='passed';print(json.dumps(dict(status='passed',tests=len(report.tests),actual_pilot_arms=list(audits))),flush=True)
