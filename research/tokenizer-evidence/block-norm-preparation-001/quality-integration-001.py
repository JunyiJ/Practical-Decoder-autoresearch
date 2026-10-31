"""Bounded real-file retention and saved-model parent acceptance integration."""
import argparse,contextlib,copy,json,os,shutil
from pathlib import Path
from datetime import datetime,timezone
import pytest,torch
from research.block_norm_v1 import campaign_contract as c,quality_execution as q
from research.block_norm_v1.fixture_dispatch import read,reference
from research.block_norm_v1.pilot_execution import exclusive
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
b=q.EVIDENCE;stem='quality-integration-001';pp=b/(stem+'-plan.json');parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args();torch.set_num_threads(4)
class Results:
 def __init__(self):self.tests=[]
 def pytest_runtest_logreport(self,report):
  if report.when=='call' or report.failed:self.tests.append(dict(nodeid=report.nodeid,outcome=report.outcome,observations=dict(report.user_properties)))
with session_lock(GLOBAL_LOCK.parent) as fd:
 idle()
 if args.mode=='prepare':
  sources=c.source_inventory();prior=read(reference(b/'pilot002-plan.json'));assert all(sources.get(k)==v for k,v in prior['sources'].items())
  added=sorted(set(sources)-set(prior['sources']));assert added==['research/block_norm_v1/quality_execution.py','research/block_norm_v1/quality_retention.py','research/block_norm_v1/test_quality_execution.py']
  inference=read(reference(b/'pilot-pair-inference-001-plan.json'));requests=copy.deepcopy(inference['requests'])
  for r in requests.values():r['sources']=sources
  rms=read(reference(b/'validation-recovery-001-rms-result.json'))['receipt']['validation'];ln=read(reference(q.ROOT.parent/'tokenizer-block-norm-v2-pilots-002/evidence/pilots/01/accepted.json'))['artifact_acceptance']['validation']
  snapshot=b/(stem+'-source');snapshot.mkdir()
  for n in added:shutil.copy2(c.ROOT/n,snapshot/Path(n).name)
  plan=dict(scope='quality_execution_integration_not_launch_authorization',sources=sources,added_sources=added,runner=reference(Path(__file__).absolute()),seeds=prior['seeds'],requests=requests,validation=dict(rms_incumbent=rms,layer_matched_epsilon=ln),disposable_root=str(c.ROOT/'research_runs/tokenizer-block-norm-v2-retention-test-001'),disposable_retention=str(b/'quality-retention-disposable-001'),protected=[reference(b/'pilot001-failure.json'),reference(b/'pilot002-result.json')],environment=c.environment(),training_allowed=False,scientific_checkpoint_deletion_allowed=False)
  exclusive(pp,plan);print('Integration plan frozen')
 else:
  plan=read(reference(pp));assert reference(Path(__file__).absolute())==plan['runner'];os.environ['BLOCK_NORM_PLAN_PATH']=str(pp);os.environ['BLOCK_NORM_GLOBAL_FD']=str(fd);result=Results();error=None;code=1
  def verify():
   c.verify_sources(plan['sources']);c.frozen_predecessors(plan['sources']);assert c.environment()==plan['environment']
   for ref in plan['protected']:
    saved=read(ref);root=Path(saved['root']);assert {str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()}==set(saved['inventory'])
    for n,row in saved['inventory'].items():assert c.file_sha(root/n,max_bytes=512*1024**2)==row['sha256']
  try:
   with StrictMpsGuard(b/(stem+'-guard.json'),c.ROOT):
    verify()
    with (b/(stem+'.log')).open('x') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):code=pytest.main(['-q','-x','-p','no:cacheprovider','research/block_norm_v1/test_quality_execution.py'],plugins=[result])
    verify()
  except Exception as exc:error=repr(exc)
  value=dict(status='passed' if code==0 and error is None else 'failed',error=error,tests=result.tests,plan=reference(pp),guard=reference(b/(stem+'-guard.json')),log=reference(b/(stem+'.log')),training_launched=False,scientific_checkpoint_deletion=False,protected_pilot_roots_unchanged=error is None,quality_plan_frozen=False)
  exclusive(b/(stem+'-review.json'),value);print(json.dumps(dict(status=value['status'],error=error,tests=len(result.tests))));raise SystemExit(0 if value['status']=='passed' else 1)
