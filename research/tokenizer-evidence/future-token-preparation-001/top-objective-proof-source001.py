"""Original AST/literal admission gate. Never imports or executes the proof/oracle."""
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


def write(name, record):
    data = (json.dumps(record, sort_keys=True, allow_nan=False) + '\n').encode()
    assert sum(p.stat().st_size for p in R.glob('*') if p.is_file()) + len(data) < 2097152
    with (R / name).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def review(plan):
    proof_path = ROOT / plan['proof_plan']
    assert sha(proof_path) == plan['proof_plan_sha256']
    proof = json.loads(proof_path.read_text())
    original_path = ROOT / proof['literal_design']
    assert sha(original_path) == proof['literal_design_sha256']
    original = json.loads(original_path.read_text())
    assert proof['fixtures'] == original['positive_fixtures']
    assert proof['families'] == original['negative_family_ids']
    assert proof['predicates'] == original['predicates']
    assert len(proof['fixtures']) == 10 and len(proof['families']) == 41
    assert all(sha(ROOT / path) == digest for path, digest in proof['sources'].items())
    tree = ast.parse((ROOT / proof['harness']).read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert set(functions) == {'sha', 'write', 'represented', 'scalar_oracle', 'close', 'tensor_values',
                              'snapshot', 'rng_snapshot', 'rng_equal', 'positive', 'invalid_payload',
                              'negative', 'run', 'main'}
    special = next(ast.literal_eval(n.value) for n in tree.body
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SPECIAL' for t in n.targets))
    assert special == {'pn_sparse_layout': ('CPU32',), 'pn_device_meta': ('CPU32',),
                       'pn_cpu_autocast_active': ('CPU64', 'CPU32'), 'pn_mps_autocast_active': ('MPS32',)}
    backends = ('CPU64', 'CPU32', 'MPS32')
    positive_ids = [b + '_' + f['id'] for b in backends for f in proof['fixtures']]
    negative_ids = [b + '_' + family for b in backends for family in proof['families']
                    if b in special.get(family, backends)]
    paired_ids = ['PAIR_' + f['id'] for f in proof['fixtures']]
    assert (len(positive_ids), len(negative_ids), len(paired_ids)) == (30, 116, 10)
    ids = positive_ids + negative_ids + paired_ids
    assert len(ids) == len(set(ids)) == 156 and sorted(ids) == sorted(proof['output_ids'])
    assert proof['counts'] == {'positive': 30, 'negative': 116, 'paired': 10, 'unique_output': 156,
                               'CPU64_negative': 38, 'CPU32_negative': 40, 'MPS32_negative': 38}
    payload = ast.get_source_segment((ROOT / proof['harness']).read_text(), functions['invalid_payload'])
    assert all(repr(family) in payload for family in proof['families'])
    oracle = functions['scalar_oracle']
    # No tensor/tested-function access or numerical oracle calls by this gate.
    names = {n.id for n in ast.walk(oracle) if isinstance(n, ast.Name)}
    assert not names.intersection({'torch', 'objective', 'F', 'positive', 'tensor_values'})
    oracle_text = ast.unparse(oracle)
    assert all(part in oracle_text for part in ('math.fsum', 'math.exp', 'math.log', 'by_id.get', 'mass * e / normalizer'))
    rounding = ast.unparse(functions['represented'])
    assert 'struct.unpack' in rounding and 'struct.pack' in rounding
    transfer = ast.unparse(functions['tensor_values'])
    assert transfer.index("to('cpu')") < transfer.index('to(dtype=torch.float64)')
    proof_text = (ROOT / proof['harness']).read_text()
    assert all(part in proof_text for part in ('torch.autograd.grad', 'F.cross_entropy', "reduction='mean'",
            "write(report['id'] + '.json', report)\n    assert all(checks.values())", 'fcntl.LOCK_EX | fcntl.LOCK_NB',
            "R / 'startup-guard.json'", "R / 'oracle-guard.json'", "R / 'mps-guard.json'", 'finally:',
            'torch.set_rng_state(saved_rng[0])', 'torch.mps.set_rng_state(saved_rng[1])'))
    assert '.backward(' not in proof_text
    assert all(not isinstance(n, (ast.Import, ast.ImportFrom)) or
               all(not a.name.startswith(('numpy', 'random', 'tiktoken', 'triton')) for a in n.names)
               for n in ast.walk(tree))
    top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all('torch' not in ast.unparse(n) for n in top_imports)
    # Resource arithmetic is static literal counting, never scalar-oracle math.
    max_elements = max(f['batch'] * f['time'] * f['vocabulary'] for f in proof['fixtures'])
    assert max_elements == 496
    assert max(len(row) for f in proof['fixtures'] for row in f['targets']) == 4
    assert sum(len(f['targets']) * f['vocabulary'] for f in proof['fixtures']) == proof['cost']['positive_gradient_coordinates_per_backend']
    assert proof['cost']['max_invalid_dense_elements'] == 50258
    assert proof['limits'] == dict(CPU_seconds=120, MPS_seconds=600, RSS_bytes=4294967296,
                                   MPS_bytes=6442450944, reserve_bytes=8589934592, metadata_bytes=2097152)
    assert proof['entry_target_bytes'] == 10739515392
    assert proof['cost']['forecast_RSS_bytes'] == 1073741824
    assert proof['cost']['forecast_MPS_bytes'] == 268435456
    assert proof['no_retry'] and proof['no_replay'] and proof['finite_supplied_logits_only']
    write('review.json', dict(accepted=True, proof_plan_sha256=sha(proof_path), source_gate_only=True,
                              no_Torch_objective_or_oracle_execution=True, sources=proof['sources'],
                              counts=proof['counts'], unique_output_ids=ids,
                              scope='finite supplied-logit proof admission only; no model, full-vocab, state or science'))


def main():
    assert len(sys.argv) == 2 and sha(R / 'plan.json') == sys.argv[1]
    plan = json.loads((R / 'plan.json').read_text())
    assert sha(__file__) == plan['reviewer_sha256']
    assert not any((R / name).exists() for name in ('attempt.json', 'STOP', 'review.json'))
    assert shutil.disk_usage(ROOT).free >= 10739515392
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write('attempt.json', dict(pid=os.getpid(), plan_sha256=sha(R / 'plan.json'), no_retry=True, no_replay=True))
        try:
            with CpuPrerequisiteGuard(R / 'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
                review(plan)
        except BaseException as error:
            write('STOP', dict(status='closed_failed', error=repr(error), no_retry=True, no_replay=True))
            raise
        else:
            write('STOP', dict(status='complete_source_gate_only', no_retry=True, no_replay=True))


if __name__ == '__main__':
    main()
