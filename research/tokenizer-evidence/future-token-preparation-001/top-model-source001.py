"""Original TOP model SOURCE gate: AST/literal/metadata only, no model imports."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    assert sum(p.stat().st_size for p in R.glob('*') if p.is_file()) + len(data) + 65536 < 2097152
    with (R / name).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def source_review(plan):
    assert all(sha(ROOT / path) == digest for path, digest in plan['sources'].items())
    design = json.loads((ROOT / plan['source_design']).read_text())
    assert sha(ROOT / plan['source_design']) == plan['source_design_sha256']
    texts = {name: (ROOT / path).read_text() for name, path in plan['model_sources'].items()}
    trees = {name: ast.parse(text) for name, text in texts.items()}
    # No execution/compile/exec/eval of parsed source, including decorators or extracted functions.
    for name, tree in trees.items():
        assert all(isinstance(n, (ast.Expr, ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef, ast.ClassDef))
                   for n in tree.body)
        for n in tree.body:
            if isinstance(n, ast.Expr): assert isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
            if isinstance(n, ast.Assign): ast.literal_eval(n.value)
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id in {'exec', 'eval', 'compile', '__import__'} for n in ast.walk(tree))
        assert '.backward(' not in texts[name] and 'manual_seed' not in texts[name]
        assert 'future_token_model_v1' not in texts[name]
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in trees['contract'].body if isinstance(n, ast.Assign)}
    assert json.loads(constants['NATIVE_MODEL_JSON']) == design['native_config']
    assert constants['VARIANTS'] == tuple(design['variants'])
    spec = next(n for n in trees['contract'].body if isinstance(n, ast.ClassDef) and n.name == 'Spec')
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'dataclass'
               and any(k.arg == 'frozen' and ast.literal_eval(k.value) is True for k in n.keywords) for n in spec.decorator_list)
    top = next(n for n in trees['model'].body if isinstance(n, ast.ClassDef) and n.name == 'TopModel')
    methods = {n.name: n for n in top.body if isinstance(n, ast.FunctionDef)}
    assert set(methods) == {'__init__', 'identity', 'validate', 'ordered_parameters', '_ids', '_hidden',
                            'forward_train', 'primary_logits', 'forward', 'state_dict', 'load_state_dict'}
    init = ast.get_source_segment(texts['model'], methods['__init__'])
    assert init.index('self.native = Native(C.native_config())') < init.index('self.aux_weight = nn.Parameter')
    assert 'self.native.lm_head.weight.detach().clone()' in init and 'nn.Linear(' not in texts['model']
    assert all(isinstance(n, ast.Raise) for name in ('forward', 'state_dict', 'load_state_dict') for n in methods[name].body)
    validation = ast.get_source_segment(texts['model'], methods['validate'])
    assert all(x in validation for x in ('C.same_typed(cfg, C.native_model_dict())', 'validate_architecture_model(self.native)',
                                        '28492416', '41358208', '(50257, 256)', 'torch.float32', 'len(set(pointers))'))
    hidden = ast.get_source_segment(texts['model'], methods['_hidden'])
    assert 'self.native.transformer.wte(inputs)' in hidden and 'for block in self.native.transformer.h:' in hidden
    assert 'return self.native.transformer.ln_f(hidden)' in hidden and 'detach' not in hidden and 'self.native(' not in hidden
    training = ast.get_source_segment(texts['model'], methods['forward_train'])
    assert training.index('self._ids(raw_ids') < training.index('sequences =') < training.index('inputs =')
    assert all(x in training for x in ('rows(sequence, context=context, window=self.spec.window, vocabulary=50257)',
            'probabilities(row)', 'raw_ids[:, :context].contiguous()', 'raw_ids[:, 1:context + 1].contiguous()',
            'self.native(inputs, labels)', 'hidden = self._hidden(inputs)', 'self.native.lm_head(hidden) / self.native.width_mult',
            'F.linear(hidden, self.aux_weight) / self.native.width_mult', "reduction='mean'", 'sparse_proximity_ce(auxiliary_logits, targets)',
            'total_loss = .5 * primary_loss + .5 * auxiliary_loss', "'top_model_training_loss_v1'"))
    assert training.count('self._hidden(inputs)') == 1
    inference = ast.get_source_segment(texts['model'], methods['primary_logits'])
    assert 'logits, _ = self.native(ids)' in inference and 'aux_weight' not in inference
    opt = texts['optimizer']
    assert all(x in opt for x in ('self._items = model.ordered_parameters()', "name.startswith('transformer.h.')",
              'scale = .5 if hidden else 1.', 'len(matrices) == 44', 'else 45', "adjust_lr_fn='match_rms_adamw'",
              'betas=(.9, .999), eps=1e-6, weight_decay=0.', 'C.same_typed(options, policy)',
              'self.validate(require_gradients=True)', 'parameter.mul_(1. - .00006)', 'optimizer.step()'))
    assert opt.index('self.validate(require_gradients=True)', opt.index('    def step(')) < opt.index('parameter.mul_(')
    assert 'inherited_build' not in opt and 'add_param_group' not in opt
    imports = [n for t in trees.values() for n in ast.walk(t) if isinstance(n, ast.ImportFrom)]
    assert any(n.module == 'research.top_proximity_targets_v1' for n in imports)
    assert any(n.module == 'research.top_proximity_objective_v1' for n in imports)
    assert any(n.module == 'research.final_ln_scale_model_v1.runtime.models.gpt' for n in imports)
    assert plan['future_source_bindings'] == 1260 and plan['predecessor_bindings'] == 1255
    assert plan['limits'] == dict(CPU_seconds=120, RSS_bytes=4294967296, reserve_bytes=8589934592, metadata_bytes=2097152)
    assert plan['forecast']['RSS_bytes'] == 134217728 and plan['forecast']['seconds'] == 10
    assert plan['no_model_functional_or_scientific_admission'] and plan['no_replay'] and plan['no_retry']
    write('review.json', dict(status='complete_TOP_model_SOURCE_AST_review_only', accepted_source_only=True,
                              no_Torch_model_target_objective_optimizer_or_extracted_function_execution=True,
                              model_source_bytes={name: len(value.encode()) for name, value in texts.items()},
                              sources=plan['sources'], future_source_bindings=1260,
                              no_model_functional_or_scientific_admission=True))


def main():
    assert len(sys.argv) == 2 and sha(R / 'plan.json') == sys.argv[1]
    plan = json.loads((R / 'plan.json').read_text())
    assert not any((R / name).exists() for name in ('attempt.json', 'STOP', 'review.json'))
    assert all(sha(ROOT / path) == digest for path, digest in plan['sources'].items())
    assert shutil.disk_usage(ROOT).free >= 10739515392
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write('attempt.json', dict(pid=os.getpid(), plan_sha256=sha(R / 'plan.json'), no_retry=True, no_replay=True))
        try:
            with CpuPrerequisiteGuard(R / 'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
                source_review(plan)
        except BaseException as error:
            write('STOP', dict(status='closed_failed_SOURCE_review_only', error=repr(error), no_retry=True, no_replay=True))
            raise
        else:
            write('STOP', dict(status='complete_TOP_model_SOURCE_gate_only', no_retry=True, no_replay=True))


if __name__ == '__main__':
    main()
