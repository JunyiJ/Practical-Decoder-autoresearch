"""AST-only next training-context review, never imports candidate/Torch/data."""
import ast,fcntl,hashlib,json,os
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path.cwd();BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001';OUT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert {p.name for p in OUT.iterdir()}=={'review.py','plan.json'}
 plan=json.loads((OUT/'plan.json').read_text());save('attempt.json',dict(pid=os.getpid(),plan_sha256=sha(OUT/'plan.json')))
 try:
  with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4294967296,reserve_bytes=8589934592):
   for pin in plan['pins']:
    p=Path(pin['path']);assert p.stat().st_size==pin['size'] and sha(p)==pin['sha256']
   accepted=json.loads((BASE/'state-review003/review.json').read_text())
   assert accepted['accepted'] and len(accepted['cases'])==6 and accepted['invalid_transactions']==234
   assert json.loads((BASE/'state-review003/guard.json').read_text())['status']=='finished'
   assert json.loads((BASE/'state-review003/STOP').read_text())['status']=='complete_original_state_saved_review'
   p=ROOT/'research/byte_input_training_v1.py';tree=ast.parse(p.read_text());compile(tree,str(p),'exec')
   functions={n.name:ast.unparse(n)for n in tree.body if isinstance(n,ast.FunctionDef)}
   context=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Context')
   methods={n.name:ast.unparse(n)for n in context.body if isinstance(n,ast.FunctionDef)}
   step=methods['step']
   assert step.index('self.schedule.apply(step)')<step.index('self.view.get_batch(')<step.index('result.loss.backward()')
   assert step.index('self.optimizer.validate(require_gradients=True)')<step.index('clip_gradients(parameters.values(), 1.0)')<step.index('self.optimizer.step()')<step.index('self.averages.update(step + 1)')
   restore=functions['restore_pristine']
   assert restore.index('step = validate(context, value)')<restore.index('staged.load_state_dict')<restore.index('numerical.restore_pristine')<restore.index('context.averages = staged')
   assert 'gradient_tensors' in functions['history'] and 'len(names)' in functions['history'] and '88' not in functions['history']
   assert "('train', text[:a])" in functions['load_view'] and "('val', text[a:b])" in functions['load_view']
   assert 'text[b:]' not in functions['load_view'] and 'test_data' not in functions['load_view']
   assert "{'train', 'val'}" in functions['check_view'] and 'initial_seed()' in functions['validate']
   assert 'CanonicalParameters' in methods['__init__'] and 'numerical.capture' in functions['capture']
   assert 'numerical._tensor' in functions['validate'] and 'multiplier(context.schedule.spec, step - 1)' in functions['validate']
   assert 'os.link(temporary, target)' in functions['write_new'] and 'weights_only=True' in functions['read_bound']
   save('review.json',dict(accepted_source_only=True,source_sha256=sha(p),source_bytes=p.stat().st_size,
     numerical_import_or_call=False,roles_assigned=False,data_read_or_encoded=False,
     scope='Complete RAW training-context source: native train/val sampling identity, canonical clipping history, fixed schedule, H200 state, nested numerical state, private samplers and bounded serialization. No numerical training-resume, H200 export, cache/scorer/worker or quality acceptance.',
     next_action='Prepare a bounded full-size CPU/MPS training-context resume fixture across native/raw/nonlexical arms, checking real sampled IDs, clipping, cooldown and EMA before/after restoration. Freeze complete costs and fresh five-role identity before admission. Preserve scientific horizon and all prior failure evidence.'))
  save('STOP',dict(status='complete_original_training_source_review',no_replay=True))
 except BaseException as exc:
  save('STOP',dict(status='closed_failed_training_source_review',error=repr(exc),no_replay=True));raise
