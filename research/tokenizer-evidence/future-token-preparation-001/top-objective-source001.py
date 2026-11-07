"""NEW AST-only review of supplied-logit source; no Torch or component import."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    case_bytes = (R.parent / 'top-objective-literal-source-plan001.json').stat().st_size
    assert case_bytes + sum(p.stat().st_size for p in R.iterdir() if p.is_file()) + len(raw) + 65536 <= 2 * 1024**2
    with (R / name).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan-sha', required=True)
    arg = ap.parse_args()
    assert sha(R / 'plan.json') == arg.plan_sha and {p.name for p in R.iterdir()} == {'plan.json'}
    plan = json.loads((R / 'plan.json').read_text())
    publish('attempt.json', dict(pid=os.getpid(), plan_sha256=arg.plan_sha))
    try:
        with CpuPrerequisiteGuard(R / 'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
            assert shutil.disk_usage(ROOT).free >= plan['complete_target_bytes']
            assert not any(k.split('.')[0] in {'torch', 'numpy', 'tiktoken', 'triton', 'random'} for k in sys.modules)
            for path, wanted in plan['sources'].items():
                assert sha(ROOT / path) == wanted, path
            assert sha(ROOT / plan['case_plan']) == plan['case_plan_sha256']
            assert sha(ROOT / plan['wrapper_reference']) == plan['wrapper_reference_sha256']
            cases = json.loads((ROOT / plan['case_plan']).read_text())
            labels = [x['id'] for x in cases['positive_fixtures']] + cases['negative_family_ids']
            assert len(cases['positive_fixtures']) == 10 and len(cases['negative_family_ids']) == 41
            assert labels == cases['unique_design_IDs'] and len(labels) == len(set(labels)) == 51
            assert cases['no_functional_execution_or_model_science_admission'] is True
            assert cases['functional_case_counts'] == 'UNFROZEN_until_complete_harness_and_concrete_backend_specific_payloads'
            for case in cases['positive_fixtures']:
                B, T, V = case['batch'], case['time'], case['vocabulary']
                assert 1 <= B <= 2 and 1 <= T <= 8 and 2 <= V <= 31
                assert len(case['logits']) == B and all(len(x) == T and all(len(z) == V for z in x) for x in case['logits'])
                assert all(type(z) is float and math.isfinite(z) for b in case['logits'] for t in b for z in t)
                assert len(case['targets']) == B * T
                for row in case['targets']:
                    assert 1 <= len(row) <= 4 and len({x[0] for x in row}) == len(row)
                    assert all(type(i) is int and 0 <= i < V and type(w) is float and 0 < w <= 1 and math.isfinite(w) for i, w in row)
                    assert abs(math.fsum(w for _, w in row) - 1.) <= 1e-14
            source = ROOT / plan['objective_source']
            tree = ast.parse(source.read_text())
            compile(tree, str(source), 'exec')
            imports = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names] + [
                n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
            assert imports == ['math', 'torch', 'torch.nn']
            fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
            assert set(fns) == {'_require', 'sparse_proximity_ce'}
            top = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
            assert all(isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) for n in top)
            fn = fns['sparse_proximity_ce']
            calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
            assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in
                           {'backward', 'zero_grad', 'step', 'manual_seed', 'rand', 'randn', 'randint', 'save', 'load'} for n in calls)
            assert not any(isinstance(n, ast.Attribute) and n.attr == 'grad' for n in ast.walk(tree))
            tensors = [n for n in calls if ast.unparse(n.func) == 'torch.tensor']
            assert len(tensors) == 2
            options = [{k.arg: ast.unparse(k.value) for k in n.keywords} for n in tensors]
            assert dict(dtype='torch.int64', device='logits.device') in options
            assert dict(dtype='logits.dtype', device='logits.device') in options
            logsm = [n for n in calls if ast.unparse(n.func) == 'F.log_softmax']
            assert len(logsm) == 1 and len(logsm[0].args) == 1 and ast.unparse(logsm[0].args[0]) == 'logits'
            assert {k.arg: ast.literal_eval(k.value) for k in logsm[0].keywords} == {'dim': -1}
            staging = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and
                           any(ast.unparse(t) == 'indices' for t in n.targets))
            target_loop = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.For))
            assert target_loop < staging
            represented = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(ast.unparse(t) == 'represented' for t in n.targets)]
            assert len(represented) == 1 and ast.unparse(represented[0].value.func) == 'torch.tensor'
            loss = next(n.value for n in fn.body if isinstance(n, ast.Assign) and any(ast.unparse(t) == 'loss' for t in n.targets))
            assert isinstance(loss, ast.BinOp) and isinstance(loss.op, ast.Div) and ast.unparse(loss.right) == 'positions'
            assert isinstance(fn.body[-1], ast.Return) and ast.unparse(fn.body[-1].value) == 'loss'
            assert cases['predicates'] == dict(CPU64_loss_atol=1e-11, CPU64_gradient_atol=1e-10, CPU64_rtol=1e-10,
                    FP32_loss_atol=4e-6, FP32_gradient_atol=4e-6, FP32_rtol=4e-5,
                    CPU_MPS_same_input_loss_atol=3e-5, CPU_MPS_same_input_loss_rtol=0,
                    all_coordinates=True, report_before_assert=True)
            assert plan['complete_target_bytes'] == 2 * 1024**2 + 10 * 1024**3
            for path, wanted in plan['sources'].items():
                assert sha(ROOT / path) == wanted, path
            assert not any(k.split('.')[0] in {'torch', 'numpy', 'tiktoken', 'triton', 'random'} for k in sys.modules)
            publish('review.json', dict(status='complete_original_TOP_objective_SOURCE_review_only',
                    source_sha256=plan['sources'][plan['objective_source']], literal_fixture_designs=10,
                    negative_family_designs=41, unique_design_IDs=51, represented_mass_cast_once_no_renormalization=True,
                    full_target_preflight_before_staging=True, mean_over_all_input_positions=True,
                    no_backward_gradient_slot_mutation_model_or_scorer_hooks=True,
                    scope='AST_and_literal_input_metadata_ONLY_NO_Torch_objective_target_or_oracle_execution',
                    no_functional_model_science_admission=True))
        publish('STOP', dict(status='complete_original_TOP_objective_SOURCE_review_only', no_retry=True, no_replay=True))
    except BaseException as error:
        if not (R / 'STOP').exists():
            publish('STOP', dict(status='closed_failed', error=repr(error), no_retry=True, no_replay=True))
        raise


if __name__ == '__main__':
    main()
