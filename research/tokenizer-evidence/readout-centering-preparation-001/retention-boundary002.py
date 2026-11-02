"""Exercise exact new two-arm eligibility using the bounded historical executor."""
import copy,hashlib,json
from pathlib import Path
from research.readout_centering_campaign_v2 import retention as r
from research.mla_score_quality_v1.support import idle,exclusive
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.protocol import digest,file_digest
B=Path(__file__).resolve().parent;OUT=B/'retention-boundary002'
def reject(fn):
 try:fn()
 except (ValueError,FileExistsError):return
 raise AssertionError('Invalid retirement accepted')
def file(p):return dict(size=p.stat().st_size,sha256=file_digest(p))
def test(boundary,gfd):
 root=OUT/boundary;root.mkdir();e=root/'evidence';e.mkdir();(e/'meta').mkdir()
 records=[dict(slot=i,arm=a,lr=lr,status='completed',h200_ce=2.+(0 if lr==.00125 else .1)) for i,(a,lr) in enumerate([(a,.00125) for a in r.ARMS])]
 selection=None;stage='pilots';pairs={}
 for row in records:
  slot=row['slot'];p=e/f'{stage}/{slot:02d}/checkpoints';p.mkdir(parents=True);pairs[slot]={}
  for role in ('raw','h200'):
   q=p/(role+'.pt');q.write_bytes(f'disposable {boundary} {slot} {role}'.encode());pairs[slot][role]=dict(path=str(q.relative_to(e)),**file(q))
 marker='pilot-inference.json';exclusive(e/'meta'/marker,dict(disposable=True,selection=selection))
 protected=e/'quality/06/checkpoints/fresh-protected.pt';protected.parent.mkdir(parents=True);protected.write_bytes(b'protected disposable fresh pair sentinel')
 exclusive(e/'meta/failed.json',dict(status='failed',protected=True))
 evidence={f'meta/{marker}':file_digest(e/'meta'/marker),'meta/failed.json':file_digest(e/'meta/failed.json')}
 args=dict(boundary=boundary,records=records,selection=selection,pairs=pairs,evidence=evidence);manifest=r.planned_manifest(**args)
 for mutate in [lambda d:d.update(boundary='confirm'),lambda d:d['records'].pop(),lambda d:d['records'][0].update(status='failed'),lambda d:d['pairs'].update({6:d['pairs'][0]}),lambda d:d['pairs'][0]['raw'].update(path='quality/06/checkpoints/fresh-protected.pt'),lambda d:d['pairs'][0]['h200'].update(path=d['pairs'][0]['raw']['path']),lambda d:d.update(evidence={'meta/unrelated.json':'0'*64})]:
  bad=copy.deepcopy(args);mutate(bad);reject(lambda:r.planned_manifest(**bad))
 for forbidden in ('lr','quality','fresh'):
  bad=copy.deepcopy(args);bad['boundary']=forbidden;reject(lambda:r.planned_manifest(**bad))
 bad=copy.deepcopy(args);bad['selection']={a:.00125 for a in r.ARMS};reject(lambda:r.planned_manifest(**bad))
 bad=copy.deepcopy(args);bad['records'][0]['lr']=.0025;reject(lambda:r.planned_manifest(**bad))
 inventory={str(p.relative_to(e)):file(p) for p in e.rglob('*') if p.is_file()}
 source=file_digest(Path(r.__file__));context={'source':source,'boundary':boundary,'selection':selection,'records':records}
 def revalidate():
  idle();assert file_digest(Path(r.__file__))==source and not (root/'STOP').exists()
  for n,h in evidence.items():assert file_digest(e/n)==h
  return context
 with session_lock(root) as fd:
  result=r.execute_boundary(e,boundary,(gfd,fd),lambda:dict(inputs=args,inventory=inventory),revalidate)
  ref=dict(path=f'.retention/{boundary}/execution-review.json',sha256=file_digest(e/f'.retention/{boundary}/execution-review.json'))
  checked=r.recheck_execution(e,manifest,ref,digest(context));assert checked['deleted_files']==4
  assert all(not (e/x['path']).exists() for x in manifest['delete']);assert file(protected)==inventory[str(protected.relative_to(e))]
  reject(lambda:r.execute_boundary(e,boundary,(gfd,fd),lambda:dict(inputs=args,inventory=inventory),revalidate))
 return dict(boundary=boundary,execution=result,recheck=checked,complete_stage_selection_protection_rejections=True,no_retry=True)
with session_lock(GLOBAL_LOCK.parent) as gfd:
 idle();OUT.mkdir()
 from research.readout_centering_campaign_v2 import evidence_contract as c
 plan=json.loads((B/'retention-boundary002-plan.json').read_text());c.verify_sources(plan['sources']);assert file_digest(Path(__file__))==plan['runner_sha256']
 with CpuPrerequisiteGuard(B/'retention-boundary002-guard.json',Path.cwd(),timeout_seconds=120):rows=[test(s,gfd) for s in ('pilot',)]
 c.verify_sources(plan['sources'])
 exclusive(B/'retention-boundary002.json',dict(status='exact_fixed_lr_pilot_boundary_passed',rows=rows,source_sha256=file_digest(Path(r.__file__)),runner_sha256=file_digest(Path(__file__)),scientific_artifacts_modified=False,remaining='Real pilot inference/payload re-audits must authorize these boundaries; no scientific dispatch authority from disposables.'))
 print('passed pilot-only exact retention; protected fresh sentinel and all metadata intact')
