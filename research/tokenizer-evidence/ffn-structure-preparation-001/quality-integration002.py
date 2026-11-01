"""Finite-rule/retention checks and actual accepted-pilot audits with additive quality code."""
import json,os,shutil
from pathlib import Path
import pytest,torch
from research.ffn_quality_v1 import campaign_contract as c,quality_execution as q,quality_retention as r,artifact_acceptance as a
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=q.EVIDENCE
class Report:
    def __init__(self):self.tests=[]
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed:self.tests.append(dict(nodeid=report.nodeid,phase=report.when,outcome=report.outcome,properties=json.loads(json.dumps(report.user_properties)),details=str(report.longrepr) if report.failed else None))
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g:
    q.idle();sources=c.source_inventory();prior=read(reference(B/'pilot001-plan.json'));requests={};validation={}
    roots={}
    for slot,arm in enumerate(q.rules.ARMS):
        anchor=read(reference(B/f'pilot001-accepted-{slot:02d}.json'));receipt=read(anchor['acceptance'])['artifact_acceptance']
        requests[arm]=dict(scope=a.SCOPE,sources=sources,config=json.loads(Path(receipt['files']['config']['path']).read_text()),files=receipt['files'])
        validation[arm]=receipt['validation'];root=Path(receipt['files']['config']['path']).parent;roots[str(root)]=r.inventory(root)
    plan=dict(sources=sources,added_sources=sorted(set(sources)-set(prior['sources'])),seeds=prior['seeds'],requests=requests,validation=validation,
        disposable_root=str(c.ROOT/'research_runs/tokenizer-ffn-quality-v1-disposable002'),disposable_retention=str(B/'quality-retention-disposable002'),
        protected_pilot_roots=roots,runner=reference(Path(__file__).resolve()),quality_training_launched=False)
    q.exclusive(B/'quality-integration002-plan.json',plan)
    snap=B/'quality-integration002-source';snap.mkdir()
    for name in plan['added_sources']:
        dst=snap/Path(name).relative_to('research/ffn_quality_v1');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((c.ROOT/name).read_bytes())
    os.environ['FFN_QUALITY_PLAN_PATH']=str(B/'quality-integration002-plan.json');os.environ['FFN_QUALITY_GLOBAL_FD']=str(g)
    report=Report()
    with q.QualityGuard(B/'quality-integration002-guard.json',c.ROOT,600):
        code=pytest.main(['-q','-p','no:cacheprovider','research/ffn_quality_v1/test_quality_execution.py','research/ffn_quality_v1/test_campaign_rules.py'],plugins=[report])
        unchanged=all(r.inventory(Path(root))==inv for root,inv in roots.items());c.verify_sources(sources)
    result=dict(status='passed' if code==0 and unchanged else 'failed',exit_code=int(code),tests=report.tests,protected_pilot_roots_unchanged=unchanged,
        plan=reference(B/'quality-integration002-plan.json'),guard=reference(B/'quality-integration002-guard.json'),quality_training_launched=False,scientific_weights_deleted=False)
    q.exclusive(B/'quality-integration002-review.json',result)
    assert result['status']=='passed';print(json.dumps(dict(status='passed',checks=len(report.tests),pilots_unchanged=unchanged)))
