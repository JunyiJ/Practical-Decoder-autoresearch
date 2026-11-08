"""Compare complete validation and fixed invalid states on accepted weights."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024): digest.update(chunk)
    return digest.hexdigest()


def read(path): return json.loads(path.read_text())


def publish(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def method(source, class_name):
    tree = ast.parse(source); cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'validate')
    return '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'


def run():
    assert not (OUT/'guard.json').exists() and not (OUT/'result.json').exists()
    plan = read(OUT/'plan.json'); assert sha(Path(__file__)) == plan['runner_sha256']
    for record in plan['bindings']:
        path = ROOT/record['path']; assert path.stat().st_size == record['size'] and sha(path) == record['sha256']
    mapping = read(OUT.parent/'finite-batch-source001/source-map.json')
    for record in (mapping['original'], mapping['candidate']):
        path = ROOT/record['path']; assert path.stat().st_size == record['size'] and sha(path) == record['sha256']
    restored = method((ROOT/mapping['candidate']['path']).read_text(), 'BatchedFiniteModel')
    for old, new in reversed(mapping['exact_replacements']): restored = restored.replace(new, old)
    assert restored == method((ROOT/mapping['original']['path']).read_text(), 'ByteInputModel')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from research.byte_input_quality_recovery_v1.guard import Guard
        with Guard(OUT/'guard.json', ROOT, timeout_seconds=90, rss_limit_bytes=4*1024**3,
                   reserve_bytes=8*1024**3, weight_roots=[ROOT/'research_runs/byte-input-quality-v1-001',
                       ROOT/'research_runs/byte-input-quality-recovery-v1-001']) as guard:
            import torch
            from research import byte_input_checkpoint_consumer_v1 as original
            torch.set_num_threads(4); assert torch.backends.mps.is_available(); guard.torch = torch
            before_cpu = torch.get_rng_state().clone(); before_mps = torch.mps.get_rng_state().clone()
            recipe = read(OUT.parent/'direct-hash-adoption001/recipe.json')
            implementation = original.read_bound(recipe['implementation'])
            for record in implementation['sources']: original.verify(record)
            prepared = original.prepare(recipe['loader']['plan_record'], recipe['loader']['acceptance_record'])
            from research import byte_input_inference_v2 as inference
            from research.byte_input_model_v3 import Spec
            from research.byte_input_batched_finite_v1 import BatchedFiniteModel
            reference = inference.load(prepared['h200'], prepared['identity'], prepared['parent'], prepared['codebook'], 'cpu')
            signature = prepared['identity']['model']
            try:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(prepared['identity']['roles']['native_initializer'])
                    candidate = BatchedFiniteModel(Spec(signature[1], signature[2], signature[3]), prepared['codebook'])
            finally: torch.mps.set_rng_state(before_mps)
            with torch.no_grad():
                for (name, p), (old_name, old) in zip(candidate.ordered_parameters(), reference.ordered_parameters(), strict=True):
                    assert name == old_name and p.shape == old.shape
                    p.copy_(old); assert torch.equal(p, old)
            candidate.eval()
            def digest(tensor): return hashlib.sha256(tensor.detach().to('cpu').numpy().tobytes()).hexdigest()
            expected = {name: digest(p) for name, p in reference.ordered_parameters()}
            cases = []
            for backend in ('cpu', 'mps'):
                reference.to(backend); candidate.to(backend)
                models = dict(reference=reference, batched=candidate)
                timings = {key: [] for key in models}; rejected = []
                for model in models.values():
                    for _ in range(2): model.validate()
                for repeat in range(5):
                    for key in (list(models) if repeat % 2 == 0 else list(reversed(models))):
                        if backend == 'mps': torch.mps.synchronize()
                        started = time.perf_counter(); device = models[key].validate()
                        if backend == 'mps': torch.mps.synchronize()
                        timings[key].append(time.perf_counter()-started); assert device.type == backend
                for label, model in models.items():
                    parameters = model.ordered_parameters(); assert len(parameters) == 105
                    for index, poison in ((0, float('nan')), (52, float('inf')), (104, -float('inf'))):
                        name, parameter = parameters[index]; value = parameter.reshape(-1)[0]; saved = value.detach().clone()
                        try:
                            with torch.no_grad(): value.fill_(poison)
                            try: model.validate()
                            except ValueError as error:
                                assert str(error) == 'Finite dense trainable FP32 parameters'
                                rejected.append(dict(implementation=label, case=f'nonfinite_parameter_{index}', parameter=name, error=str(error)))
                            else: raise AssertionError('Nonfinite parameter accepted')
                        finally:
                            with torch.no_grad(): value.copy_(saved)
                        model.validate()
                    parameter = parameters[-1][1]
                    try:
                        parameter.requires_grad_(False)
                        try: model.validate()
                        except ValueError as error:
                            assert str(error) == 'Finite dense trainable FP32 parameters'
                            rejected.append(dict(implementation=label, case='frozen_parameter', error=str(error)))
                        else: raise AssertionError('Frozen parameter accepted')
                    finally: parameter.requires_grad_(True)
                    model.validate()
                    saved = model.code_ids[0, 0].clone()
                    try:
                        model.code_ids[0, 0] += 1
                        try: model.validate()
                        except ValueError as error:
                            assert str(error) == 'Codebook bytes differ or common buffer mutated'
                            rejected.append(dict(implementation=label, case='changed_codebook', error=str(error)))
                        else: raise AssertionError('Changed codebook accepted')
                    finally: model.code_ids[0, 0].copy_(saved)
                    model.validate()
                    assert expected == {name: digest(p) for name, p in model.ordered_parameters()}
                    assert model._own_expected_buffer_hashes == reference._own_expected_buffer_hashes
                medians = {key: statistics.median(values) for key, values in timings.items()}
                cases.append(dict(backend=backend, timings_seconds=timings, median_seconds=medians,
                    speedup=medians['reference']/medians['batched'], rejections=rejected,
                    paired_faster=sum(b < a for a, b in zip(timings['reference'], timings['batched']))))
            assert torch.equal(before_cpu, torch.get_rng_state()) and torch.equal(before_mps, torch.mps.get_rng_state())
            gpu = next(c for c in cases if c['backend'] == 'mps')
            useful = gpu['median_seconds']['batched'] <= .9*gpu['median_seconds']['reference'] and gpu['paired_faster'] >= 4
            result = dict(correctness_passed=True, useful_mps_improvement=useful, eligible_for_integration=useful,
                adopted=False, cases=cases, plan_sha256=sha(OUT/'plan.json'), runner_sha256=sha(Path(__file__)),
                weights_unchanged=True, cpu_rng_unchanged=True, mps_rng_unchanged=True,
                training_updates=0, quality_evaluations=0, limits=plan['interpretation_limits'])
        assert read(OUT/'guard.json')['status'] == 'finished'
        result['guard_sha256'] = sha(OUT/'guard.json'); publish('result.json', result)
        publish('STOP', dict(status='complete_candidate_pending_integration' if useful else 'complete_no_useful_mps_gain', no_replay=True))
    print(json.dumps(dict(correctness_passed=True, useful_mps_improvement=useful,
        cases=[{k: c[k] for k in ('backend', 'median_seconds', 'speedup', 'paired_faster')} for c in cases])))


if __name__ == '__main__':
    try: run()
    except Exception:
        publish('failure.json', dict(traceback=traceback.format_exc(), no_automatic_replay=True))
        raise
