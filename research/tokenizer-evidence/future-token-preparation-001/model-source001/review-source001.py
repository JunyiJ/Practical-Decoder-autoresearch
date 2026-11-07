"""One bounded SOURCE review: AST plus extracted scalar shape function, no Torch."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[4]
R = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw) <= 2*1024**2
    with (R/name).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def main():
    assert len(sys.argv) == 2 and sha(R/'plan.json') == sys.argv[1]
    assert not (R/'attempt.json').exists() and not (R/'STOP').exists()
    plan = json.loads((R/'plan.json').read_text())
    publish('attempt.json', dict(pid=os.getpid(), plan_sha256=sys.argv[1]))
    try:
        with CpuPrerequisiteGuard(R/'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
            assert 'torch' not in sys.modules and 'numpy' not in sys.modules
            sources = {}
            for name, expected in plan['new_sources'].items():
                p = ROOT/name; assert sha(p) == expected
                text = p.read_text(); tree = ast.parse(text); compile(tree, str(p), 'exec')
                sources[p.name] = (text, tree)
            assert set(sources) == {'__init__.py', 'contract.py', 'model.py', 'backward.py', 'optimizer.py'}
            contract = sources['contract.py'][1]
            fn = next(n for n in contract.body if isinstance(n, ast.FunctionDef) and n.name == 'parameter_shapes')
            # Execute ONLY the real source's pure integer shape function. No module imports/config/model stubs.
            module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
            def require(value, message):
                if not value:
                    raise ValueError(message)
            scope = dict(ARMS=('native_next', 'duplicate_next', 'future_two'), require=require)
            exec(compile(module, '<extracted_scalar_parameter_shapes>', 'exec'), scope)
            inventories = {}
            for arm in scope['ARMS']:
                shapes = scope['parameter_shapes'](arm)
                matrices = [n for n, shape in shapes.items() if len(shape) == 2
                            and (n.startswith('native.transformer.h.') or n.startswith('auxiliary.block.'))]
                value = dict(parameters=sum(math.prod(s) for s in shapes.values()), tensors=len(shapes),
                             muon_groups=len(matrices), adam_groups=len(shapes)-len(matrices))
                assert value == plan['inventories'][arm]
                inventories[arm] = value
            texts = {name: value[0] for name, value in sources.items()}
            model, backward, optimizer = texts['model.py'], texts['backward.py'], texts['optimizer.py']
            calls = [ast.unparse(n.func) for _, tree in sources.values() for n in ast.walk(tree) if isinstance(n, ast.Call)]
            forbidden = [n for n in calls if any(p in n for p in ('torch.rand', 'torch.load', 'torch.save', 'Generator'))]
            assert not forbidden and calls.count('Native') == 1
            assert calls.count('torch.random.default_generator.manual_seed') == 1
            assert 'self.native = Native(copy.deepcopy(self.cfg[\'native_template\']))' in model
            assert model.index('self.native = Native(') < model.index('self.auxiliary = nn.ModuleDict(')
            assert "block=copy.deepcopy(self.native.transformer.h[3])" in model
            assert "norm=copy.deepcopy(self.native.transformer.ln_f)" in model
            assert 'nn.Linear(' not in model and 'nn.Parameter(' not in model
            assert 'future_token_native_weights_v1' in model and 'future_token_primary_native_logits' in model
            assert 'cache.owner() is self' in model and 'cache.epoch == self.weight_epoch' in model
            assert 'torch.equal(mask, expected_mask)' in model
            assert 'x.shape[0] <= 2 and x.shape[1] <= 16' in backward
            assert backward.index('(primary*.5).backward()') < backward.index('del primary, logits')
            assert backward.index('del primary, logits') < backward.index('logits = head(model, leaf, auxiliary=True)')
            assert backward.count('h.backward(leaf.grad)') == 1
            assert not any(n.endswith(('.step', '.zero_grad', '.clip_grad_norm_')) for n in
                           [ast.unparse(n.func) for n in ast.walk(sources['backward.py'][1]) if isinstance(n, ast.Call)])
            assert "'future_token_signature'" in optimizer and 'relu_squared_signature' not in optimizer
            assert "name.startswith('auxiliary.block.')" in optimizer and "name.startswith('native.transformer.h.')" in optimizer
            assert 'owned = copy.deepcopy(state)' in optimizer
            for name, expected in plan['new_sources'].items():
                assert sha(ROOT/name) == expected
            assert 'torch' not in sys.modules and 'numpy' not in sys.modules
            publish('review.json', dict(status='passed_source_review_only_actual_model_runtime_unqualified',
                inventories=inventories, sources=plan['new_sources'], review_scope=plan['review_scope'],
                no_torch_numpy_or_model_module_imports=True, no_models_tensors_optimizer_rng_corpus_checkpoints=True,
                no_census_or_new_role_assignment_or_runtime_admission=True))
        publish('STOP', dict(status='complete_original_future_token_model_source_review', no_replay=True, no_retry=True))
    except BaseException as error:
        if not (R/'STOP').exists():
            publish('STOP', dict(status='closed_failed', error=repr(error), no_retry=True))
        raise


if __name__ == '__main__':
    main()
