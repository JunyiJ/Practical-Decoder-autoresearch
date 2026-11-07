"""Distinct semantic-AST source review; executes no extracted functions or Torch."""
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
            trees = {}
            for name, expected in plan['sources'].items():
                path = ROOT/name; assert sha(path) == expected
                trees[path.name] = ast.parse(path.read_text())
                compile(trees[path.name], str(path), 'exec')
            assert set(trees) == {'__init__.py', 'contract.py', 'model.py', 'backward.py', 'optimizer.py'}
            calls = [(name, n, ast.unparse(n.func)) for name, tree in trees.items()
                     for n in ast.walk(tree) if isinstance(n, ast.Call)]
            # Classify complete AST attribute names, never substrings like 'torch.rand'.
            random_terminals = {'rand', 'randn', 'randint', 'rand_like', 'randn_like', 'normal',
                                'bernoulli', 'multinomial', 'uniform_', 'random_'}
            assert not [f for _, _, f in calls if f.rsplit('.', 1)[-1] in random_terminals]
            assert not [f for _, _, f in calls if f in {'torch.load', 'torch.save', 'torch.Generator'}]
            seeds = [(name, node) for name, node, f in calls if f.endswith('.manual_seed')]
            assert len(seeds) == 1 and seeds[0][0] == 'model.py'
            constructors = [node for name, node, f in calls if name == 'model.py' and f == 'Native']
            assert len(constructors) == 1
            gpt = next(n for n in trees['model.py'].body if isinstance(n, ast.ClassDef) and n.name == 'GPT')
            init = next(n for n in gpt.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
            fork = next(n for n in init.body if isinstance(n, ast.With))
            assert ast.unparse(fork.items[0].context_expr.func) == 'torch.random.fork_rng'
            assert seeds[0][1] in list(ast.walk(fork)) and constructors[0] in list(ast.walk(fork))
            assert constructors[0].lineno > seeds[0][1].lineno
            copies = [node for name, node, f in calls if name == 'model.py' and f == 'copy.deepcopy'
                      and ast.unparse(node.args[0]) in {'self.native.transformer.h[3]', 'self.native.transformer.ln_f'}]
            assert len(copies) == 2 and min(n.lineno for n in copies) > fork.end_lineno
            assert not [f for name, _, f in calls if name == 'model.py' and f in {'nn.Linear', 'nn.Parameter'}]
            # Independent integer algebra on literal projection/FFN dimensions, not parameter_shapes execution.
            shape_fn = next(n for n in trees['contract.py'].body if isinstance(n, ast.FunctionDef) and n.name == 'parameter_shapes')
            attention = next(n.value for n in shape_fn.body if isinstance(n, ast.Assign)
                             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'attention')
            dims = {k.arg: ast.literal_eval(k.value) for k in attention.keywords}
            assert dims == dict(dkv=(64,256),uk=(256,64),uv=(256,64),dq=(64,256),uq=(256,64),qr=(256,64),kr=(32,256),out=(256,256))
            block = 2*256+sum(math.prod(s)+s[0] for s in dims.values())+3*672*256
            native = 2*50257*256+4*block+2*256
            extra = block+2*256
            algebra = dict(native_parameters=native, auxiliary_extra_parameters=extra,
                           auxiliary_total_parameters=native+extra, native_tensors=1+4*(2+16+3)+2+1,
                           auxiliary_extra_tensors=2+16+3+2, native_muon=4*(8+3), auxiliary_muon=5*(8+3),
                           native_adam=44, auxiliary_adam=56)
            assert algebra == plan['independent_scalar_forecast']
            backward = trees['backward.py']
            seq = next(n for n in backward.body if isinstance(n, ast.FunctionDef) and n.name == 'sequential_backward')
            seq_calls = [(n, ast.unparse(n.func)) for n in ast.walk(seq) if isinstance(n, ast.Call)]
            assert len([n for n, f in seq_calls if f == 'h.backward']) == 1
            half = [n for n, f in seq_calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'backward'
                    and isinstance(n.func.value, ast.BinOp) and isinstance(n.func.value.op, ast.Mult)
                    and isinstance(n.func.value.right, ast.Constant) and n.func.value.right.value == .5]
            assert len(half) == 2
            deletes = [n for n in ast.walk(seq) if isinstance(n, ast.Delete)
                       and [ast.unparse(t) for t in n.targets] == ['primary', 'logits']]
            assert len(deletes) == 1
            aux_call = next(n for n, f in seq_calls if f == 'head' and any(k.arg == 'auxiliary' and isinstance(k.value, ast.Constant) and k.value.value is True for k in n.keywords))
            assert min(n.lineno for n in half) < deletes[0].lineno < aux_call.lineno
            assert not [f for _, _, f in calls if f.rsplit('.',1)[-1] in {'step', 'clip_grad_norm_'}
                        and f not in {'super().step'}]
            literals = {v.value for t in trees.values() for v in ast.walk(t) if isinstance(v, ast.Constant) and isinstance(v.value, str)}
            assert {'future_token_signature', 'future_token_native_weights_v1', 'future_token_primary_native_logits'} <= literals
            assert 'relu_squared_signature' not in literals
            for name, expected in plan['sources'].items():
                assert sha(ROOT/name) == expected
            assert 'torch' not in sys.modules and 'numpy' not in sys.modules
            publish('review.json', dict(status='passed_semantic_AST_source_review_only_runtime_unqualified',
                sources=plan['sources'], independent_scalar_forecast=algebra,
                original_review001_remains_closed_failed=True, five_implementations_byte_identical=True,
                no_extracted_functions_or_model_modules_executed=True, no_torch_numpy_tensor_model_optimizer_rng_calls=True,
                no_fixture_roles_or_runtime_or_quality_admitted=True))
        publish('STOP', dict(status='complete_original_semantic_AST_future_token_source_review', no_replay=True, no_retry=True))
    except BaseException as error:
        if not (R/'STOP').exists():
            publish('STOP', dict(status='closed_failed', error=repr(error), no_retry=True))
        raise


if __name__ == '__main__':
    main()
