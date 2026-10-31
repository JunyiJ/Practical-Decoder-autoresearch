"""Read-only selection/retention audit; never deletes or launches a worker."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
from src.research.protocol import ROOT,digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import bind,idle,session_lock,GLOBAL_LOCK,MetadataWriter,Session,reference,require,require_capacity,prefix
from src.research.tokenizer_activation_retention import verify_boundaries
from src.research.tokenizer_wide_hybrid_metadata import snapshot
p=argparse.ArgumentParser();p.add_argument('phase',choices=('selected','released'));phase=p.parse_args().phase
base=ROOT/'research/tokenizer-evidence/activation-preparation-001';root=ROOT/'research_runs/tokenizer-activation-001'
label='campaign001-screen-'+phase;dest=base/(label+'-portable');guard=base/(label+'-guard.json')
with CpuPrerequisiteGuard(guard,ROOT,timeout_seconds=180):
 with session_lock(GLOBAL_LOCK.parent) as g,session_lock(root) as c:
  idle();before=snapshot(root);ledger=Session(MetadataWriter(root,(g,c)),reference(root,'evidence/meta/activation-plan.json'))
  plan=bind(ledger,capacity=True);state=ledger.inspect();prefix(ledger,state);verify_boundaries(ledger,state)
  require(len(state['records'])==6 and state['pending'] is None,'Complete screen without fresh attempts required')
  rows=[]
  for i in range(6):
   stem=f'evidence/quality/{i:02d}'
   rq=ledger.document(reference(root,stem+'/request.json'))['spec'];m=ledger.document(reference(root,stem+'/measurements.json'))
   rows.append(dict(slot=i,**rq,h200_ce=m['h200']['mean_ce'],raw_ce=m['raw']['mean_ce']))
  expected={f:min((r for r in rows if r['family']==f),key=lambda r:(r['h200_ce'],r['lr']))['lr'] for f in ('silu','gelu_exact')}
  require(state['selection']==expected==dict(silu=0.00125,gelu_exact=0.00125),'Persisted minimum/tie-break selection differs')
  selected={ref['path']:ref for i in (2,3) for ref in state['pairs'][i+2]}
  require(len(selected)==4,'Two selected raw/H200 pairs required')
  for ref in selected.values():ledger.verify(ref)
  if phase=='selected':
   require(state['next']['kind']=='review_screen_boundary','Cleanup must not have started')
   candidates=ledger._eligible(state,'screen');require(len(candidates)==8 and not set(candidates)&set(selected),'Exact losing-only allowlist required')
   require({int(Path(n).parts[2]) for n in candidates}=={0,1,4,5},'Wrong losing slots')
   execution=None
  else:
   require(state['screen_boundary'] and state['next']['kind']=='quality' and state['next']['spec']['seed']==3611,'Fresh panel must be next')
   execution=ledger.document(reference(root,'evidence/.retention/lr/execution.json'))
   require(execution['deleted_files']==8,'All eight losing weights must be journaled')
   require({str(q.relative_to(root)) for q in root.rglob('*.pt')}==set(selected),'Unexpected retained/missing weights')
   candidates=None
  copied={}
  for name,entry in before['files'].items():
   if entry['group'].endswith('/weights'):continue
   if not (name.startswith('evidence/meta/') or name.startswith('evidence/.retention/lr/')):continue
   ref=reference(root,name);target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f:f.write((root/name).read_bytes())
   require(file_digest(target)==ref['sha256'],'Portable metadata changed');copied[name]=ref
  capacity=require_capacity(ledger);require(snapshot(root)==before,'Review mutated campaign')
value=dict(status='screen_'+phase+'_verified',observed_at=datetime.now(timezone.utc).isoformat(),plan=ledger.plan,source_sha256=digest(plan['source']),rows=rows,
 selection=state['selection_ref'],choices=expected,protected_selected=selected,eligible_losing=candidates,execution=execution,capacity=capacity,next=state['next'],
 copied=copied,guard=reference(ROOT,str(guard.relative_to(ROOT))),procedure_sha256=file_digest(Path(__file__)),holdout_scored=False,adoption_claim=False)
with (base/(label+'-review.json')).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in value.items() if k not in ('copied','rows','protected_selected','eligible_losing')}))
