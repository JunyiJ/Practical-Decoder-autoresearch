"""Static dependency/source applicability inventory; no training or model import."""
import ast,hashlib,importlib.util,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research/tokenizer-evidence/block-norm-preparation-001'
OUT=BASE/'runtime-dependency-map.json'
assert not OUT.exists(),'No overwrite'
seeds=['src.models.gpt','src.attention.cache','src.train.train','src.train.optimizers','src.train.weight_averaging','src.utils.checkpoint']
def path_for(module):
 p=ROOT.joinpath(*module.split('.'))
 if p.with_suffix('.py').is_file():return p.with_suffix('.py')
 if (p/'__init__.py').is_file():return p/'__init__.py'
 if p.is_dir():return None
 raise ValueError('Missing local module '+module)
pending=list(seeds);seen={};external=set();edges=[];hazards=[]
while pending:
 name=pending.pop()
 if name in seen:continue
 path=path_for(name)
 if path is None:continue
 body=path.read_text();tree=ast.parse(body);seen[name]=dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),size=path.stat().st_size)
 parts=name.split('.')
 for n in range(1,len(parts)):
  parent='.'.join(parts[:n]);p=ROOT.joinpath(*parent.split('.'))/'__init__.py'
  if p.exists() and parent not in seen:pending.append(parent)
 package=name if path.name=='__init__.py' else name.rsplit('.',1)[0]
 for node in ast.walk(tree):
  targets=[]
  if isinstance(node,ast.Import):targets=[a.name for a in node.names]
  elif isinstance(node,ast.ImportFrom):
   target=importlib.util.resolve_name('.'*node.level+(node.module or ''),package) if node.level else node.module
   targets=[target]
   for a in node.names:
    candidate=target+'.'+a.name
    if ROOT.joinpath(*candidate.split('.')).with_suffix('.py').exists():targets.append(candidate)
   if node.level==0 and target.startswith('src'):hazards.append(dict(module=name,line=node.lineno,kind='absolute_local_import'))
  elif isinstance(node,ast.Call):
   func=ast.unparse(node.func)
   if func in ('__import__','importlib.import_module'):hazards.append(dict(module=name,line=node.lineno,kind='dynamic_import'))
  for target in targets:
   if target=='src' or target.startswith('src.'):
    edges.append(dict(source=name,target=target,line=node.lineno));pending.append(target)
   else:external.add(target.split('.')[0])
assert not hazards,hazards
assert all(not n.startswith('src.research') for n in seen)
refs=['src/models/tokenizer_architecture.py','src/models/blocks.py','src/utils/checkpoint.py','src/train/weight_averaging.py','src/train/train.py']
functions={}
for name in refs:
 tree=ast.parse((ROOT/name).read_text());functions[name]=[
  dict(name=n.name,line=n.lineno,end_line=n.end_lineno) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
value=dict(status='static_relative_import_closure_mapped',observed_at=datetime.now(timezone.utc).isoformat(),
 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 roots=seeds,modules=dict(sorted(seen.items())),module_count=len(seen),source_bytes=sum(x['size'] for x in seen.values()),
 edges=edges,external_import_roots=sorted(external),absolute_or_dynamic_local_import_hazards=hazards,functions=functions,
 chosen_namespace='research.block_norm_v1.runtime',
 strategy='Copy this exact closure into a new package, preserving relative imports and keeping old src unchanged. Baseline parity before named-policy changes. No sys.modules/sys.path redirect, monkeypatch or old-validator suppression.',
 constraints=['Static analysis is a conservative import closure, not proof of all executed dependencies.',
 'train.train Hydra CLI refers to ../config; do not invoke the copied Hydra entry. A separately frozen controller must pass explicit configs.',
 'Future candidate changes require exact changed-file allowlist and complete new source/applicability digest; baseline copy remains auditable via origin hashes.'],
 source_modified=False,model_constructed=False,training_launched=False,holdout_scored=False)
with OUT.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({k:value[k] for k in ('status','module_count','source_bytes','external_import_roots')}))
