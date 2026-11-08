"""Measure actual-input padding removal; exact equivalence is a fixed gate."""
import fcntl
import gc
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


def run():
    assert not (OUT/'guard.json').exists() and not (OUT/'result.json').exists()
    plan = read(OUT/'plan.json'); assert sha(Path(__file__)) == plan['runner_sha256']
    for record in plan['bindings']:
        path = ROOT/record['path']
        assert path.is_file() and not path.is_symlink() and path.stat().st_size == record['size'] and sha(path) == record['sha256']
    mapping = read(OUT/'source-map.json')
    for record in (mapping['original'], mapping['candidate']):
        path = ROOT/record['path']; assert path.stat().st_size == record['size'] and sha(path) == record['sha256']
    assert (ROOT/mapping['candidate']['path']).read_text().replace(mapping['exact_insertion'], '') == (ROOT/mapping['original']['path']).read_text()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from research.byte_input_quality_recovery_v1.guard import Guard
        with Guard(OUT/'guard.json', ROOT, timeout_seconds=120, rss_limit_bytes=4*1024**3,
                   reserve_bytes=8*1024**3, weight_roots=[ROOT/'research_runs/byte-input-quality-v1-001',
                       ROOT/'research_runs/byte-input-quality-recovery-v1-001']) as guard:
            import torch
            from research import byte_input_checkpoint_consumer_v2 as consumer
            torch.set_num_threads(4); assert torch.backends.mps.is_available(); guard.torch = torch
            before_cpu = torch.get_rng_state().clone(); before_mps = torch.mps.get_rng_state().clone()
            recipe = read(OUT.parent/'direct-hash-adoption001/recipe.json'); loader = recipe['loader']
            model, identity, receipt = consumer.load(loader['implementation_record'], loader['plan_record'], loader['acceptance_record'], 'cpu')
            from research.byte_composer_trim_v1 import ByteComposer
            from research import byte_input_training_v3 as training
            reference = model.composer; candidate = ByteComposer(identity['roles']['composer_initializer'])
            with torch.no_grad():
                for (name, parameter), (old_name, old) in zip(candidate.named_parameters(), reference.named_parameters(), strict=True):
                    assert name == old_name and parameter.shape == old.shape
                    parameter.copy_(old); assert torch.equal(parameter, old)
            codes, lengths = model.code_ids, model.code_lengths
            view = training.load_view(identity['roles'], 'cpu')
            train = view.train_data; train_lengths = lengths.index_select(0, train)
            longest_position = int(train_lengths.argmax())
            start = min(max(0, longest_position-512), train.numel()-1024)
            cases = [('training_prefix1024', train[:1024]), ('first_longest_training_window1024', train[start:start+1024])]
            for length in (3, 4, 5, 6):
                cases.append((f'first_regular_length{length}', torch.nonzero(lengths[:50256] == length)[0].reshape(1)))
            cases += [('first_maximum_vocabulary_length', lengths.argmax().reshape(1)),
                      ('special_token', torch.tensor([50256], dtype=torch.int64))]
            del model, view; gc.collect()

            def digest(value):
                value = value.detach().to('cpu').contiguous().reshape(-1)
                return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()

            def metrics(old, new):
                assert old.shape == new.shape and old.dtype == new.dtype
                delta = (old.double()-new.double()).abs()
                return dict(exact=torch.equal(old, new), original_sha256=digest(old), candidate_sha256=digest(new),
                    coordinates=old.numel(), unequal_coordinates=int((old != new).sum()),
                    max_abs=float(delta.max()), original_max_abs=float(old.double().abs().max()))

            expected_weights = {name: digest(p) for name, p in reference.named_parameters()}
            records = []; rejections = []
            for backend in ('cpu', 'mps'):
                reference.to(backend); candidate.to(backend)
                models = dict(reference=reference, trimmed=candidate)
                for case_name, ids in cases:
                    selected = codes.index_select(0, ids).to(backend)
                    size = lengths.index_select(0, ids).to(backend)
                    upstream = (((torch.arange(ids.numel()*256, dtype=torch.int64) % 17)-8).to(torch.float32)/8).reshape(ids.numel(), 256).to(backend)
                    def execute(component):
                        component.zero_grad(set_to_none=True)
                        if backend == 'mps': torch.mps.synchronize()
                        started = time.perf_counter()
                        output = component(selected, size); output.backward(upstream)
                        if backend == 'mps': torch.mps.synchronize()
                        elapsed = time.perf_counter()-started
                        saved = output.detach().to('cpu').clone()
                        gradients = {name: p.grad.detach().to('cpu').clone() for name, p in component.named_parameters()}
                        return elapsed, saved, gradients
                    first = {key: execute(component) for key, component in models.items()}
                    output = metrics(first['reference'][1], first['trimmed'][1])
                    gradients = {name: metrics(first['reference'][2][name], first['trimmed'][2][name]) for name in first['reference'][2]}
                    assert len(gradients) == 18
                    timings = None
                    if case_name == 'training_prefix1024':
                        timings = {key: [] for key in models}
                        for repeat in range(3):
                            for key in (list(models) if repeat % 2 == 0 else list(reversed(models))):
                                elapsed, actual, grads = execute(models[key])
                                assert torch.equal(actual, first[key][1]) and all(torch.equal(grads[n], first[key][2][n]) for n in grads)
                                timings[key].append(elapsed)
                    row = dict(backend=backend, case=case_name, token_ids=ids.tolist(), token_ids_sha256=digest(ids),
                        rows=ids.numel(), original_width=130, trimmed_width=max(6, int(size.max())),
                        output=output, gradients=gradients, exact=output['exact'] and all(g['exact'] for g in gradients.values()),
                        timings_seconds=timings)
                    records.append(row); publish(backend+'-'+case_name+'.json', row)
                invalid = codes.index_select(0, cases[0][1]).to(backend).clone()
                invalid[0, -1] = 0; size = lengths.index_select(0, cases[0][1]).to(backend)
                for label, component in models.items():
                    saved_grads = {name: digest(p.grad) for name, p in component.named_parameters()}
                    try:
                        component(invalid, size)
                    except ValueError as error:
                        assert str(error) == 'All trailing positions are pad256'
                        rejections.append(dict(backend=backend, implementation=label, error=str(error)))
                    else: raise AssertionError('Invalid discarded padding was accepted')
                    assert saved_grads == {name: digest(p.grad) for name, p in component.named_parameters()}
                    assert expected_weights == {name: digest(p) for name, p in component.named_parameters()}
            assert torch.equal(before_cpu, torch.get_rng_state()) and torch.equal(before_mps, torch.mps.get_rng_state())
            exact = all(r['exact'] for r in records)
            timings = {r['backend']: {key: statistics.median(values) for key, values in r['timings_seconds'].items()}
                       for r in records if r['timings_seconds'] is not None}
            result = dict(execution_completed=True, exact_equivalence=exact, eligible_for_transparent_integration=exact,
                adopted=False, plan_sha256=sha(OUT/'plan.json'), runner_sha256=sha(Path(__file__)),
                representative_h200=receipt, case_count=len(records), inexact_cases=[dict(backend=r['backend'], case=r['case']) for r in records if not r['exact']],
                max_output_abs=max(r['output']['max_abs'] for r in records),
                max_gradient_abs=max(g['max_abs'] for r in records for g in r['gradients'].values()),
                prefix_forward_backward_median_seconds=timings, invalid_padding_rejections=rejections,
                weights_unchanged=True, cpu_rng_unchanged=True, mps_rng_unchanged=True,
                optimizer_updates=0, quality_evaluations=0, limits=plan['interpretation_limits'])
        assert read(OUT/'guard.json')['status'] == 'finished'
        result['guard_sha256'] = sha(OUT/'guard.json')
        result['case_evidence'] = {p.name: sha(p) for p in sorted(OUT.glob('*.json')) if p.name.startswith(('cpu-', 'mps-'))}
        publish('result.json', result)
        publish('STOP', dict(status='complete_exact_component_pending_integration' if exact else 'complete_not_exact_not_adopted', no_replay=True))
    print(json.dumps({key: result[key] for key in ('execution_completed', 'exact_equivalence', 'inexact_cases', 'max_output_abs', 'max_gradient_abs', 'prefix_forward_backward_median_seconds')}))


if __name__ == '__main__':
    try:
        run()
    except Exception:
        publish('failure.json', dict(traceback=traceback.format_exc(), no_automatic_replay=True))
        publish('STOP', dict(status='closed_execution_failure', no_replay=True))
        raise
