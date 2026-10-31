"""Guarded read-only accepted quality prefix audit and incremental text snapshot."""
import argparse,json
from pathlib import Path
from datetime import datetime,timezone
from src.research.protocol import ROOT,digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import bind,idle,session_lock,GLOBAL_LOCK,MetadataWriter,Session,reference,require,require_capacity,prefix
from src.research.tokenizer_activation_retention import verify_boundaries
from src.research.tokenizer_wide_hybrid_metadata import snapshot
p=argparse.ArgumentParser();p.add_argument('--count',type=int,required=True);args=p.parse_args()
require(1<=args.count<=12,'Finite accepted quality count')
base=ROOT/'research/tokenizer-evidence/activation-preparation-001';root=ROOT/'research_runs/tokenizer-activation-001'
label=f'campaign001-quality-prefix-{args.count:02d}';dest=base/(label+'-portable');guard=base/(label+'-guard.json')
with CpuPrerequisiteGuard(guard,ROOT,timeout_seconds=180):
 with session_lock(GLOBAL_LOCK.parent) as g,session_lock(root) as c:
  idle();before=snapshot(root);ledger=Session(MetadataWriter(root,(g,c)),reference(root,'evidence/meta/activation-plan.json'))
  plan=bind(ledger,capacity=True);state=ledger.inspect();dispatches=prefix(ledger,state);verify_boundaries(ledger,state)
  require(len(state['records'])==args.count and state['pending'] is None,'Exact completed prefix without pending worker required')
  rows=[]
  for i in range(args.count):
   stem=f'evidence/quality/{i:02d}'
   def read(n):return ledger.document(reference(root,stem+'/'+n))
   proc=read('process/process.json');measured=read('measurements.json');train=read('training-summary.json')
   rows.append(dict(slot=i,spec=read('request.json')['spec'],raw_ce=measured['raw']['mean_ce'],h200_ce=measured['h200']['mean_ce'],
       training_seconds=train['training_time_sec'],worker_seconds=proc['elapsed_seconds'],resources=read('resources-audit.json')['detail'],dispatch=dispatches[i+2]))
  copied={};i=args.count-1;stem=f'evidence/quality/{i:02d}/'
  for name,entry in before['files'].items():
   if entry['group'].endswith('/weights'):continue
   if not (name.startswith(stem) or name.startswith('evidence/meta/session/') or (f'quality-{i:02d}' in name)):continue
   ref=reference(root,name);target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f:f.write((root/name).read_bytes())
   require(file_digest(target)==ref['sha256'],'Copy changed');copied[name]=ref
  cap=require_capacity(ledger);require(snapshot(root)==before,'Review changed campaign contents')
value=dict(status='accepted_quality_prefix',observed_at=datetime.now(timezone.utc).isoformat(),accepted_quality=args.count,
    plan=ledger.plan,source_sha256=digest(plan['source']),rows=rows,next=state['next'],capacity=cap,copied=copied,
    guard=reference(ROOT,str(guard.relative_to(ROOT))),procedure_sha256=file_digest(Path(__file__)),holdout_scored=False,adoption_claim=False)
with (base/(label+'-review.json')).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in value.items() if k!='copied'}))
