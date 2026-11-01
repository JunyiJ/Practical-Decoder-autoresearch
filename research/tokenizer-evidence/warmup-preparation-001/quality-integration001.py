"""Test finite quality contracts and independently audit real retained pilots."""
import json,os,copy
from pathlib import Path
import torch,pytest
from research.warmup_quality_v1 import campaign_contract as c,quality_execution as q,artifact_acceptance as a,quality_retention as r
from research.warmup_quality_v1.support import cache_audit
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=q.EVIDENCE
class Report:
 def __init__(self):self.tests=[]
 def pytest_runtest_logreport(self,report):
  if report.when=='call' or report.failed:self.tests.append(dict(nodeid=report.nodeid,phase=report.when,outcome=report.outcome,details=str(report.longrepr) if report.failed else None))
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g:
 q.idle();sources=c.source_inventory();prior=read(reference(B/'pilot001-plan.json'));roots={};requests={};validation={}
 for slot,arm in enumerate(q.rules.ARMS):
  row=read(reference(c.ROOT/f'research_runs/tokenizer-warmup-pilot-001/evidence/pilots/{slot:02d}/accepted.json'))
  req=dict(row['acceptance_request'],scope=a.SCOPE,sources=sources);requests[arm]=req;validation[arm]=row['validation'];root=Path(req['files']['config']['path']).parent;roots[str(root)]=r.inventory(root)
 plan=dict(sources=sources,added_sources=sorted(set(sources)-set(prior['sources'])),seeds=prior['seeds'],requests=requests,validation=validation,protected_pilot_roots=roots,
  disposable_root=str(c.ROOT/'research_runs/tokenizer-warmup-quality-v1-disposable001'),runner=reference(Path(__file__).resolve()),quality_training_launched=False)
 q.exclusive(B/'quality-integration001-plan.json',plan)
 os.environ['WARMUP_INTEGRATION_PLAN']=str(B/'quality-integration001-plan.json');os.environ['WARMUP_GLOBAL_FD']=str(g)
 report=Report();audits={}
 with q.QualityGuard(B/'quality-integration001-guard.json',c.ROOT,600):
  code=pytest.main(['-q','-p','no:cacheprovider','research/warmup_quality_v1/test_campaign_rules.py','research/warmup_quality_v1/test_execution.py'],plugins=[report])
  if code==0:
   for arm,req in requests.items():
    accepted=a.accept(req,validation[arm]);cached=cache_audit(req);audits[arm]=dict(audit=accepted['audit'],cache=cached)
    assert accepted['audit']['ema_parameters']==88
    forged=copy.deepcopy(req);forged['scope']=a.TRIAL_SCOPE;forged['trial']=dict(spec=next(s for s in q.rules.screen_specs(prior['seeds']) if s['arm']==arm),seeds=prior['seeds'])
    try:a.verify_request(forged)
    except ValueError:pass
    else:raise AssertionError('Pilot bundle accepted as quality endpoint')
    audits[arm]['pilot_cannot_impersonate_quality']=True
    bad=copy.deepcopy(req);bad['sources']=dict(sources);bad['sources'][next(iter(sources))]='0'*64
    try:a.verify_request(bad)
    except ValueError:pass
    else:raise AssertionError('Mutated source accepted')
  unchanged=all(r.inventory(Path(root))==inv for root,inv in roots.items());c.verify_sources(sources)
 result=dict(status='passed' if code==0 and unchanged and len(audits)==2 else 'failed',exit_code=int(code),tests=report.tests,actual_pilot_audits=audits,protected_pilot_roots_unchanged=unchanged,plan=reference(B/'quality-integration001-plan.json'),guard=reference(B/'quality-integration001-guard.json'),scientific_quality_launched=False)
 q.exclusive(B/'quality-integration001-review.json',result)
 assert result['status']=='passed';print(json.dumps(dict(status='passed',tests=len(report.tests),actual_pilot_arms=list(audits))),flush=True)
