"""Independent saved-array and minimal source-delta review; no Torch imports."""
import ast,fcntl,hashlib,json,os
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path.cwd();BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001';OUT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(1048576):h.update(x)
 return h.hexdigest()
def save(name,x):
 with (OUT/name).open('x') as f:json.dump(x,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def normalized(p):return ast.dump(ast.parse(p.read_text()),include_attributes=False)
with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert {p.name for p in OUT.iterdir()}=={'review.py','plan.json'}
 plan=json.loads((OUT/'plan.json').read_text());save('attempt.json',dict(pid=os.getpid(),plan_sha256=sha(OUT/'plan.json')))
 try:
  with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4294967296,reserve_bytes=8589934592):
   for p in plan['pins']:assert Path(p['path']).stat().st_size==p['size'] and sha(Path(p['path']))==p['sha256']
   import numpy as np
   records=[];count=0
   proof=BASE/'embedding-primitive002';result=json.loads((proof/'result.json').read_text())
   assert result['accepted'] and json.loads((proof/'guard.json').read_text())['status']=='finished'
   assert sha(proof/'gradients.npz')==result['gradients_sha256']
   with np.load(proof/'gradients.npz') as actual:
    assert len(actual.files)==12
    for number in (0,1):
     source=BASE/f'embedding-accumulation-diagnosis00{number+1}'
     reported=json.loads((source/'result.json').read_text());assert sha(source/'arrays.npz')==reported['arrays_sha256']
     with np.load(source/'arrays.npz') as saved:
      ix=saved['index'];g=saved['upstream'].astype('float64');reference=np.stack([g[ix==row].sum(axis=0) for row in range(259)])
      assert np.array_equal(reference,saved['reference'])
      native=[saved[f'native_{k}']for k in range(3)];grouped=[saved[f'grouped_{k}']for k in range(3)]
      assert all(np.array_equal(grouped[0],a)for a in grouped[1:])
      assert reported['native']['repeat_exact']==all(np.array_equal(native[0],a)for a in native[1:])
      for backend in ('cpu','mps'):
       values=[actual[f'case{number}-{backend}-{k}']for k in range(3)]
       assert all(np.array_equal(values[0],a)for a in values[1:])
       for a in values:
        assert a.shape==(259,15) and np.isfinite(a).all()
        assert (np.abs(a.astype('float64')-reference)<=4e-6+4e-5*np.abs(reference)).all();count+=a.size
     records.append(dict(case=number,verified=True,native_repeat_exact=reported['native']['repeat_exact']))
   assert len(result['rejections'])==14 and all(r['error'] and r['unchanged']for r in result['rejections'])
   old=(ROOT/'research/byte_composer_v1.py').read_text()
   expected=old.replace('from torch.nn import functional as F','from torch.nn import functional as F\nfrom research.byte_embedding_v1 import embedding').replace("KIND = 'rsi_byte_composer_math_v1'","KIND = 'rsi_byte_composer_math_v2'").replace('F.embedding(rows, self.byte_weight)','embedding(self.byte_weight, rows)')
   assert (ROOT/'research/byte_composer_v2.py').read_text()==expected
   for p in (ROOT/'research/byte_input_model_v2').glob('*.py'):
    expected=p.read_text()
    if p.name=='model.py':expected=expected.replace('from research.byte_composer_v1 import ByteComposer','from research.byte_composer_v2 import ByteComposer')
    if p.name=='contract.py':expected=expected.replace("POLICY = 'rsi_byte_input_model_math_v2'","POLICY = 'rsi_byte_input_model_math_v3'").replace("else 'rsi_byte_composer_math_v1'","else 'rsi_byte_composer_math_v2'")
    target=ROOT/'research/byte_input_model_v3'/p.name;assert target.read_text()==expected;compile(ast.parse(expected),str(target),'exec')
   expected=(ROOT/'research/byte_input_state_v1.py').read_text().replace('research.byte_input_model_v2','research.byte_input_model_v3').replace("KIND = 'byte_input_numerical_checkpoint_v1'","KIND = 'byte_input_numerical_checkpoint_v2'")
   assert (ROOT/'research/byte_input_state_v2.py').read_text()==expected
   for p in [ROOT/'research/byte_input_state_v2.py',ROOT/'research/byte_composer_v2.py',ROOT/'research/byte_embedding_v1.py']:compile(ast.parse(p.read_text()),str(p),'exec')
   failure=BASE/'state-math002';prefix='mps-raw_bytes'
   a=json.loads((failure/(prefix+'-control-next-gradients.json')).read_text());b=json.loads((failure/(prefix+'-restored-next-gradients.json')).read_text())
   changed=[k for k in a if a[k]!=b[k]];assert changed==['composer.byte_weight']
   for suffix in ['restore','rng']:
    v=json.loads((failure/(prefix+'-'+suffix+'.json')).read_text());assert v['expected']==v['actual']
   assert json.loads((failure/(prefix+'-control-next-forward.json')).read_text())==json.loads((failure/(prefix+'-restored-next-forward.json')).read_text())
   save('review.json',dict(accepted_primitive_and_source_delta=True,verified_primitive_gradient_coordinates=count,
    controls=records,changed_full_model_gradient_names=changed,model_v3_numerically_executed=False,
    state_v2_numerically_executed=False,checkpoint_full_panel_qualified=False,
    scope='Standalone primitive numerical proof plus exact source-composition delta. Existing model-v2 finite integration proof preserved, but not relabeled as model-v3 checkpoint acceptance. Next test remains six full-size CPU/MPS checkpoint cases with all 234 rejections and exact next-update/RNG gates.',
    conclusion='Native MPS lookup accumulation is nonrepeatable for mixed-magnitude repeated rows. Fixed-order grouped reduction passed independent FP64 error/repeatability checks; actual-model recovery remains to be executed.'))
  save('STOP',dict(status='complete_original_primitive_and_source_review',no_replay=True))
 except BaseException as exc:
  save('STOP',dict(status='closed_failed_primitive_and_source_review',error=repr(exc),no_replay=True));raise
