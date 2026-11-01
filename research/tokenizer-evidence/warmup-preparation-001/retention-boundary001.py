"""Exercise exact new two-arm eligibility using the bounded historical executor."""
import copy,hashlib,json
from pathlib import Path
from research.warmup_campaign_v1 import retention as r
from research.ffn_norm_quality_v1.support import idle
from research.block_norm_v1.pilot_execution import exclusive
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.protocol import digest,file_digest
B=Path(__file__).resolve().parent;OUT=B/'retention-boundary001'
def reject(fn):
 try:fn()
 except (ValueError,FileExistsError):return
 raise AssertionError('Invalid retirement accepted')
def file(p):return dict(size=p.stat().st_size,sha256=file_digest(p))
def test(boundary,gfd):
 root=OUT/boundary;root.mkdir();e=root/'evidence';e.mkdir();(e/'meta').mkdir()
 records=[dict(slot=i,arm=a,lr=lr,status='completed',h200_ce=2.+(0 if lr==.00125 else .1)) for i,(a,lr) in enumerate([(a,.0025) for a in r.ARMS] if boundary=='pilot' else [(a,lr) for lr,order in zip(r.RATES,(r.ARMS,r.ARMS[::-1],r.ARMS)) for a in order])]
 selection=None if boundary=='pilot' else {a:.00125 for a in r.ARMS};stage='pilots' if boundary=='pilot' else 'quality';pairs={}
 for row in records:
  slot=row['slot'];p=e/f'{stage}/{slot:02d}/checkpoints';p.mkdir(parents=True);pairs[slot]={}
  for role in ('raw','h200'):
   q=p/(role+'.pt');q.write_bytes(f'disposable {boundary} {slot} {role}'.encode());pairs[slot][role]=dict(path=str(q.relative_to(e)),**file(q))
 marker='pilot-inference.json' if boundary=='pilot' else 'selection.json';exclusive(e/'meta'/marker,dict(disposable=True,selection=selection))
 protected=e/'quality/06/checkpoints/fresh-protected.pt';protected.parent.mkdir(parents=True);protected.write_bytes(b'protected disposable fresh pair sentinel')
 exclusive(e/'meta/failed.json',dict(status='failed',protected=True))
 evidence={f'meta/{marker}':file_digest(e/'meta'/marker),'meta/failed.json':file_digest(e/'meta/failed.json')}
 args=dict(boundary=boundary,records=records,selection=selection,pairs=pairs,evidence=evidence);manifest=r.planned_manifest(**args)
 for mutate in [lambda d:d.update(boundary='confirm'),lambda d:d['records'].pop(),lambda d:d['records'][0].update(status='failed'),lambda d:d['pairs'].update({6:d['pairs'][0]}),lambda d:d['pairs'][0]['raw'].update(path='quality/06/checkpoints/fresh-protected.pt'),lambda d:d['pairs'][0]['h200'].update(path=d['pairs'][0]['raw']['path']),lambda d:d.update(evidence={'meta/unrelated.json':'0'*64})]:
  bad=copy.deepcopy(args);mutate(bad);reject(lambda:r.planned_manifest(**bad))
 if boundary=='lr':
  bad=copy.deepcopy(args);bad['selection']['none']=.0025;reject(lambda:r.planned_manifest(**bad))
  tie=copy.deepcopy(args)
  for row in tie['records']:row['h200_ce']=2.
  tie['selection']={a:.000625 for a in r.ARMS};assert {v['index'] for v in r.planned_manifest(**tie)['retain']}=={0,1}
 inventory={str(p.relative_to(e)):file(p) for p in e.rglob('*') if p.is_file()}
 source=file_digest(Path(r.__file__));context={'source':source,'boundary':boundary,'selection':selection,'records':records}
 def revalidate():
  idle();assert file_digest(Path(r.__file__))==source and not (root/'STOP').exists()
  for n,h in evidence.items():assert file_digest(e/n)==h
  return context
 with session_lock(root) as fd:
  result=r.execute_boundary(e,boundary,(gfd,fd),lambda:dict(inputs=args,inventory=inventory),revalidate)
  ref=dict(path=f'.retention/{boundary}/execution-review.json',sha256=file_digest(e/f'.retention/{boundary}/execution-review.json'))
  checked=r.recheck_execution(e,manifest,ref,digest(context));assert checked['deleted_files']==(4 if boundary=='pilot' else 8)
  assert all(not (e/x['path']).exists() for x in manifest['delete']);assert file(protected)==inventory[str(protected.relative_to(e))]
  reject(lambda:r.execute_boundary(e,boundary,(gfd,fd),lambda:dict(inputs=args,inventory=inventory),revalidate))
 return dict(boundary=boundary,execution=result,recheck=checked,complete_stage_selection_protection_rejections=True,no_retry=True)
with session_lock(GLOBAL_LOCK.parent) as gfd:
 idle()
 plan=json.loads((B/'retention-boundary001-plan.json').read_text())
 for n,h in plan['sources'].items():assert file_digest(Path(n))==h
 OUT.mkdir()
 with CpuPrerequisiteGuard(B/'retention-boundary001-guard.json',Path.cwd(),timeout_seconds=120):rows=[test(s,gfd) for s in ('pilot','lr')]
 exclusive(B/'retention-boundary001.json',dict(status='exact_pilot_and_losing_screen_boundaries_passed',rows=rows,source_sha256=file_digest(Path(r.__file__)),runner_sha256=file_digest(Path(__file__)),scientific_artifacts_modified=False,remaining='Real pilot inference/payload and screen selection re-audits must authorize these boundaries; no scientific dispatch authority from disposables.'))
 print('passed pilot and losing-screen exact retention; protected fresh sentinel and all metadata intact')
