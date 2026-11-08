"""Original AST-only state adapter review; no candidate/state/Torch imports."""
import ast,fcntl,hashlib,json,os
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path.cwd();BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001';OUT=BASE/'state-source001'
def save(name,value):
 raw=(json.dumps(value,sort_keys=True)+'\n').encode();assert sum(p.stat().st_size for p in OUT.iterdir())+len(raw)<2097152
 with (OUT/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert {p.name for p in OUT.iterdir()}=={'review.py','plan.json'}
 plan=json.loads((OUT/'plan.json').read_text());save('attempt.json',dict(pid=os.getpid(),plan_sha256=hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest()))
 try:
  with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
   for pin in plan['pins']:
    raw=Path(pin['path']).read_bytes();assert len(raw)==pin['size'] and hashlib.sha256(raw).hexdigest()==pin['sha256']
   accepted=json.loads((BASE/'model-saved-review004/review.json').read_text());assert accepted['accepted'] and accepted['complete_records']==5392 and accepted['all_coordinates']==1412427220
   assert json.loads((BASE/'model-saved-review004/guard.json').read_text())['status']=='finished'
   assert json.loads((BASE/'model-saved-review004/STOP').read_text())['status']=='complete_original_saved_review'
   source=ROOT/'research/byte_input_state_v1.py';tree=ast.parse(source.read_text());compile(tree,str(source),'exec')
   functions={n.name:ast.unparse(n) for n in tree.body if isinstance(n,ast.FunctionDef)}
   assert all(n in functions for n in ['capture','validate','restore_pristine','write_new','read_bound'])
   restore=functions['restore_pristine'];assert restore.index('step = validate(')<restore.index('staged_parameters =')<restore.index('parameter.copy_(')<restore.index('optimizer.validate()')<restore.index('torch.mps.set_rng_state(')
   assert 'p.grad is None' in functions['live'] and 'optimizer.model is model' in functions['live']
   assert "same_typed(payload['immutable_buffer_hashes'], model._own_expected_buffer_hashes)" in functions['validate']
   assert 'model.state_dict(' not in source.read_text() and 'optimizer.state_dict(' not in source.read_text() and 'load_state_dict(' not in source.read_text()
   assert 'os.link(temporary, path)' in functions['write_new'] and "temporary.open('xb')" in functions['write_new'] and 'weights_only=True' in functions['read_bound']
   assert 'torch.Generator(' in functions['validate'] and 'finally:' in functions['validate']
   save('review.json',dict(accepted_source_only=True,numerical_admission=False,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_bytes=source.stat().st_size,
    scope='Own numerical model/optimizer/RNG checkpoint source; no sampler/schedule/H200/training/quality qualification',
    next_action='Prepare bounded all-three-arm CPU/MPS serialization/restore/next-update proof, complete cost/source/runtime/role freeze and static admission before importing or calling the new adapter.',
    roles='UNASSIGNED',no_candidate_state_or_Torch_import=True))
  save('STOP',dict(status='complete_original_source_review',no_replay=True))
 except BaseException as exc:
  save('STOP',dict(status='closed_failed_original_source_review',error=repr(exc),no_replay=True));raise
