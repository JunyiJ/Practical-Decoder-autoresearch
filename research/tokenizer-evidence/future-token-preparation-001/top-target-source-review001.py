"""NEW AST/literal/resource gate; never imports or executes the target/harness."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')
P = R.parent / 'top-target-component001'
CASES = R.parent / 'top-target-component-source-plan001.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(folder, name, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    retained = CASES.stat().st_size + sum(p.stat().st_size for d in (R, P)
                   for p in d.iterdir() if p.is_file())
    assert retained + len(raw) + 65536 <= 2 * 1024**2
    with (folder / name).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def imports(tree):
    return [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names] + [
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan-sha', required=True)
    arg = ap.parse_args()
    assert sha(R / 'plan.json') == arg.plan_sha and {p.name for p in R.iterdir()} == {'plan.json'}
    assert {p.name for p in P.iterdir()} == {'plan.json'}
    plan = json.loads((R / 'plan.json').read_text())
    publish(R, 'attempt.json', dict(pid=os.getpid(), plan_sha256=arg.plan_sha))
    try:
        with CpuPrerequisiteGuard(R / 'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
            assert shutil.disk_usage(ROOT).free >= plan['complete_target_bytes']
            assert not any(k.split('.')[0] in {'torch', 'numpy', 'tiktoken', 'triton', 'random'} for k in sys.modules)
            for path, wanted in plan['sources'].items():
                assert sha(ROOT / path) == wanted, path
            assert sha(CASES) == plan['case_plan_sha256'] and sha(P / 'plan.json') == plan['proof_plan_sha256']
            cases = json.loads(CASES.read_text())
            proof = json.loads((P / 'plan.json').read_text())
            assert proof['sources'] == plan['sources'] and proof['case_plan_sha256'] == plan['case_plan_sha256']
            assert proof['complete_target_bytes'] == plan['complete_target_bytes'] == 2 * 1024**2 + 10 * 1024**3
            assert proof['limits'] == dict(CPU_seconds=120, RSS_bytes=4 * 1024**3, reserve_bytes=8 * 1024**3,
                                          whole_metadata_bytes=2 * 1024**2)
            assert proof['forecast']['seconds'] == 30 and proof['forecast']['RSS_bytes'] == 256 * 1024**2
            labels = [c['id'] for c in cases['positive_cases'] + cases['negative_cases']]
            assert len(labels) == len(set(labels)) == len(cases['unique_output_labels']) == 37
            assert labels == cases['unique_output_labels'] and len(cases['positive_cases']) == 12 and len(cases['negative_cases']) == 25
            for case in cases['positive_cases']:
                assert len(case['sequence']) == case['context'] + case['window'] and len(case['expected_rows']) == case['context']
                assert all(type(x) is int and 0 <= x < case['vocabulary'] for x in case['sequence'])
                for row in case['expected_rows']:
                    assert row and row[0][1] == 1 and len({v for v, _ in row}) == len(row)
                    assert all(type(v) is int and 0 <= v < case['vocabulary'] and type(k) is int and 1 <= k <= case['window'] for v, k in row)
                    assert [k for _, k in row] == sorted(set(k for _, k in row))
            target = ast.parse((ROOT / plan['target_source']).read_text())
            harness = ast.parse((ROOT / plan['proof_source']).read_text())
            compile(target, plan['target_source'], 'exec')
            compile(harness, plan['proof_source'], 'exec')
            assert imports(target) == ['math']
            assert {n.name for n in target.body if isinstance(n, ast.FunctionDef)} == {'_require', 'rows', 'probabilities'}
            top = [n for n in target.body if not isinstance(n, (ast.FunctionDef, ast.Import))]
            assert all(isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) for n in top)
            allowed = {'argparse', 'copy', 'decimal', 'fcntl', 'hashlib', 'json', 'math', 'os', 'pathlib',
                       'shutil', 'sys', 'src.research.cpu_prerequisite_guard', 'research'}
            assert set(imports(harness)) <= allowed
            fns = {n.name: n for n in harness.body if isinstance(n, ast.FunctionDef)}
            payloads = ast.literal_eval(fns['negative_payloads'].body[0].value)
            assert list(payloads) == labels[12:] and len(payloads) == 25
            for label, (api, args, kwargs) in payloads.items():
                declared = next(c for c in cases['negative_cases'] if c['id'] == label)
                assert api == declared['api'] and type(args) is tuple and len(args) == 1 and type(kwargs) is dict
                assert set(kwargs) == ({'context', 'window', 'vocabulary'} if api == 'rows' else set())
            oracle = fns['decimal_oracle']
            assert not any(isinstance(n, ast.Name) and n.id == 'C' for n in ast.walk(oracle))
            assert any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'prec' for t in n.targets)
                       and isinstance(n.value, ast.Constant) and n.value.value == 80 for n in ast.walk(oracle))
            main_fn = fns['main']
            guard = next(n for n in ast.walk(main_fn) if isinstance(n, ast.With) and any(
                isinstance(x.context_expr, ast.Call) and ast.unparse(x.context_expr.func) == 'CpuPrerequisiteGuard' for x in n.items))
            component_imports = [n for n in ast.walk(harness) if isinstance(n, ast.ImportFrom) and n.module == 'research']
            assert len(component_imports) == 1 and component_imports[0] in list(ast.walk(guard))
            assert [(a.name, a.asname) for a in component_imports[0].names] == [('top_proximity_targets_v1', 'C')]
            record = fns['record'].body
            assert next(i for i, n in enumerate(record) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                        and ast.unparse(n.value.func) == 'publish') < next(i for i, n in enumerate(record) if isinstance(n, ast.Assert))
            text = (ROOT / plan['proof_source']).read_text()
            assert "f'{SERIAL:03d}-{label}.json'" in text and 'fcntl.LOCK_EX | fcntl.LOCK_NB' in text
            assert "decimal_oracle(row) for row in case['expected_rows']" in text
            assert '1e-14 + 4e-14 * abs(b[1])' in text and 'far_relative_error <= 4e-14' in text
            assert all(c['expect'] == 'ValueError_and_input_unchanged' for c in cases['negative_cases'])
            for path, wanted in plan['sources'].items():
                assert sha(ROOT / path) == wanted, path
            assert not any(k.split('.')[0] in {'torch', 'numpy', 'tiktoken', 'triton', 'random'} for k in sys.modules)
            publish(R, 'review.json', dict(status='complete_original_TOP_target_source_gate', sources=plan['sources'],
                    positive_cases=12, negative_payloads=25, unique_case_ids=37, Decimal_precision=80,
                    source_case_resource_scope='AST_literal_metadata_only_NO_target_harness_or_oracle_execution',
                    proof_plan_sha256=plan['proof_plan_sha256']))
            publish(P, 'source-review.json', dict(status='source_reviewed', review_sha256=sha(R / 'review.json')))
            publish(P, 'admission.json', dict(scope='literal_target_math_only', plan_sha256=plan['proof_plan_sha256'],
                    review_sha256=sha(R / 'review.json'), no_retry=True, no_replay=True))
        publish(R, 'STOP', dict(status='complete_original_TOP_target_source_gate', no_retry=True, no_replay=True))
    except BaseException as error:
        if not (R / 'STOP').exists():
            publish(R, 'STOP', dict(status='closed_failed', error=repr(error), no_retry=True, no_replay=True))
        raise


if __name__ == '__main__':
    main()
