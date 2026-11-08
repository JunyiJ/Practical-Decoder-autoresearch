"""Bound representative metadata check only; no model/tensor/weight loading."""
import ast
import json
from pathlib import Path
import time

from research import byte_input_checkpoint_consumer_v1 as consumer
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
Q = OUT.parent/'quality001'
started = time.perf_counter()
with CpuPrerequisiteGuard(OUT/'guard.json', ROOT, timeout_seconds=60,
                         rss_limit_bytes=1024**3, reserve_bytes=8*1024**3):
    import torch
    state = torch.random.get_rng_state().clone()
    def receipt(path):
        return dict(path=str(path.resolve()), size=path.stat().st_size, sha256=consumer.sha(path))
    plan, acceptance = receipt(Q/'plan.json'), receipt(Q/'slots/10/acceptance.json')
    prepared = consumer.prepare(plan, acceptance)
    assert prepared['spec'] == dict(slot=10, stage='confirmation', bundle='fresh0', variant='raw_bytes', learning_rate=.0025)
    expected = prepared['identity']
    assert type(expected['model']) is tuple and type(expected['model'][3]) is tuple
    assert expected['model'][2] == 62100500007 and expected['model'][3] == prepared['codebook'].identity
    assert torch.equal(state, torch.random.get_rng_state())
    source = Path(consumer.__file__); tree = ast.parse(source.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    calls = [n for n in ast.walk(functions['load']) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'load']
    assert len(calls) == 1 and ast.unparse(calls[0]) == "inference.load(prepared['h200'], prepared['identity'], prepared['parent'], prepared['codebook'], backend)"
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('step', 'manual_seed')
                   for n in ast.walk(tree))
    result = dict(accepted_metadata_preparation=True, runtime_loading_qualified=False,
        no_model_instantiated=True, no_weight_payload_loaded=True, cpu_rng_unchanged=True,
        plan_record=plan, representative_acceptance_record=acceptance,
        representative_rule='First raw-byte fresh candidate in original frozen order, not loss selected; adoption remains conditional on complete final review.',
        expected_identity_json=json.loads(json.dumps(expected)),
        model_signature_type=type(expected['model']).__name__, codebook_signature_type=type(expected['model'][3]).__name__,
        parent=prepared['parent'], h200=prepared['h200'], consumer_sha256=consumer.sha(source),
        check_source_sha256=consumer.sha(Path(__file__)), elapsed_seconds=time.perf_counter()-started,
        remaining='After full-panel independent acceptance and worker-lock release, perform one bounded loading integration through this helper before publishing an adopted recipe. Existing model/inference qualifications are reused.')
assert json.loads((OUT/'guard.json').read_text())['status'] == 'finished'
with (OUT/'result.json').open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
print(json.dumps({k: result[k] for k in ('accepted_metadata_preparation', 'runtime_loading_qualified', 'cpu_rng_unchanged', 'consumer_sha256', 'elapsed_seconds')}))
