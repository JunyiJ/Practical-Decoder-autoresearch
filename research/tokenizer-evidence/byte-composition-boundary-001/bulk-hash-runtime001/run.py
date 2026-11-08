"""Actual-buffer engineering comparison; no training or quality evaluation."""
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
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text())


def publish(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def run():
    assert not (OUT/'guard.json').exists() and not (OUT/'result.json').exists()
    plan = read(OUT/'plan.json')
    for rec in plan['bindings']:
        path = ROOT/rec['path']
        assert not path.is_symlink() and path.stat().st_size == rec['size'] and sha(path) == rec['sha256']
    assert sha(Path(__file__)) == plan['runner_sha256']
    recipe = read(OUT.parent/'optimizer-state-adoption001/recipe.json')
    assert recipe['status'] == 'adopted_equivalent_batched_optimizer_state_implementation'
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from research.byte_input_quality_recovery_v1.guard import Guard
        with Guard(OUT/'guard.json', ROOT, timeout_seconds=90, rss_limit_bytes=4*1024**3,
                   reserve_bytes=8*1024**3, weight_roots=[ROOT/'research_runs/byte-input-quality-v1-001',
                       ROOT/'research_runs/byte-input-quality-recovery-v1-001']) as guard:
            import torch
            from research import byte_input_checkpoint_consumer_v1 as consumer
            torch.set_num_threads(4); assert torch.backends.mps.is_available(); guard.torch = torch
            cpu_rng = torch.get_rng_state().clone(); mps_rng = torch.mps.get_rng_state().clone()
            args = recipe['loader']
            implementation = consumer.read_bound(recipe['implementation'])
            for record in implementation['sources']: consumer.verify(record)
            training_implementation = consumer.read_bound(recipe['training_implementation'])
            for record in training_implementation['sources']: consumer.verify(record)
            prepared = consumer.prepare(args['plan_record'], args['acceptance_record'])
            # prepare admits the complete frozen source/assets before these imports.
            from research import byte_input_inference_v4 as inference
            from research.byte_input_model_v3 import Spec
            from research.byte_input_bulk_hash_v1 import BulkHashModel
            reference = inference.load(prepared['h200'], prepared['identity'], prepared['parent'], prepared['codebook'], 'cpu')
            signature = prepared['identity']['model']
            # Reconstruct historical identity solely to install the accepted weights.
            try:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(prepared['identity']['roles']['native_initializer'])
                    candidate = BulkHashModel(Spec(signature[1], signature[2], signature[3]), prepared['codebook'])
            finally:
                torch.mps.set_rng_state(mps_rng)
            with torch.no_grad():
                for (name, target), (old_name, source) in zip(candidate.ordered_parameters(), reference.ordered_parameters(), strict=True):
                    assert name == old_name and target.shape == source.shape
                    target.copy_(source); assert torch.equal(target, source)
            candidate.eval()
            assert candidate.identity == reference.identity and candidate._own_expected_buffer_hashes == reference._own_expected_buffer_hashes
            records = []
            for backend in plan['backends']:
                reference.to(backend); candidate.to(backend)
                models = {'reference': reference, 'bulk_bytes': candidate}
                def sync():
                    if backend == 'mps': torch.mps.synchronize()
                def timed(fn):
                    sync(); start = time.perf_counter(); value = fn(); sync()
                    return time.perf_counter()-start, value
                inventories = {key: dict(model.named_buffers()) for key, model in models.items()}
                assert list(inventories['reference']) == list(inventories['bulk_bytes'])
                buffers = []
                for name in inventories['reference']:
                    expected = reference._own_expected_buffer_hashes[name]
                    timings = {key: [] for key in models}
                    for key, model in models.items():
                        assert model._buffer_digest(inventories[key][name]) == expected
                    for repeat in range(plan['paired_repeats']):
                        for key in (list(models) if repeat % 2 == 0 else list(reversed(models))):
                            model = models[key]
                            elapsed, digest = timed(lambda: model._buffer_digest(inventories[key][name]))
                            assert digest == expected; timings[key].append(elapsed)
                    rejected = {}
                    for key, model in models.items():
                        flat = inventories[key][name].reshape(-1); index = flat.numel()//2
                        original = flat[index].clone()
                        try:
                            with torch.no_grad():
                                if flat.dtype == torch.bool: flat[index].logical_not_()
                                else: flat[index].add_(1)
                            changed = model._buffer_digest(inventories[key][name])
                            assert changed != expected
                            try:
                                model.validate()
                            except ValueError as error:
                                assert str(error) == 'Codebook bytes differ or common buffer mutated'
                                rejected[key] = dict(error=str(error), mutated_sha256=changed)
                            else:
                                raise AssertionError('Changed real buffer was accepted')
                        finally:
                            with torch.no_grad(): flat[index].copy_(original)
                        model.validate()
                        assert model._own_expected_buffer_hashes[name] == expected
                    assert rejected['reference']['mutated_sha256'] == rejected['bulk_bytes']['mutated_sha256']
                    value = inventories['reference'][name]
                    buffers.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
                        bytes=value.numel()*value.element_size(), sha256=expected, mutation_rejections=rejected,
                        timings_seconds=timings))
                edge_checks = []
                for index in (0, 131071, 131072, reference.code_ids.numel()-1):
                    mutations = {}
                    for key, model in models.items():
                        flat = model.code_ids.reshape(-1); saved = flat[index].clone()
                        try:
                            flat[index].add_(1)
                            changed = model._buffer_digest(model.code_ids)
                            assert changed != model._own_expected_buffer_hashes['code_ids']
                            try: model.validate()
                            except ValueError as error:
                                assert str(error) == 'Codebook bytes differ or common buffer mutated'
                                mutations[key] = changed
                            else: raise AssertionError('Edge mutation accepted')
                        finally: flat[index].copy_(saved)
                        model.validate()
                    assert mutations['reference'] == mutations['bulk_bytes']
                    edge_checks.append(dict(index=index, mutated_digests=mutations))
                full_timings = {key: [] for key in models}
                for model in models.values(): model.validate()
                for repeat in range(plan['paired_repeats']):
                    for key in (list(models) if repeat % 2 == 0 else list(reversed(models))):
                        elapsed, _ = timed(models[key].validate); full_timings[key].append(elapsed)
                medians = {key: statistics.median(value) for key, value in full_timings.items()}
                records.append(dict(backend=backend, buffers=buffers, validation_seconds=full_timings,
                    validation_median_seconds=medians, validation_speedup=medians['reference']/medians['bulk_bytes'],
                    paired_faster=sum(b<a for a,b in zip(full_timings['reference'],full_timings['bulk_bytes'])), edge_checks=edge_checks))
            assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(mps_rng, torch.mps.get_rng_state())
            gpu = next(r for r in records if r['backend']=='mps')
            useful = gpu['validation_speedup'] >= 1/.9 and gpu['paired_faster'] >= 5
            for (_, a), (_, b) in zip(reference.ordered_parameters(),candidate.ordered_parameters(),strict=True):
                assert torch.equal(a,b)
            result = dict(accepted=True, useful_mps_improvement=useful, plan_sha256=sha(OUT/'plan.json'), runner_sha256=sha(Path(__file__)),
                cases=records, cpu_rng_unchanged=True, mps_rng_unchanged=True, training_updates=0,
                forward_backward_calls=0, quality_evaluations=0, adopted=False,
                conclusion='Actual immutable-buffer digest equality and complete-validator mutation rejection preserved on CPU/MPS.',
                limits=plan['interpretation_limits'])
        assert read(OUT/'guard.json')['status'] == 'finished'
        result['guard_sha256'] = sha(OUT/'guard.json'); publish('result.json', result)
        publish('STOP', dict(status='complete_candidate_pending_integration' if useful else 'complete_no_useful_mps_gain', no_replay=True))
    print(json.dumps(dict(accepted=True, useful_mps_improvement=useful, cases=[dict(backend=r['backend'], buffers=len(r['buffers']),
        validation_speedup=r['validation_speedup']) for r in records])))


if __name__ == '__main__':
    try:
        run()
    except Exception:
        publish('failure.json', dict(traceback=traceback.format_exc(), no_automatic_replay=True))
        publish('STOP', dict(status='closed_failed_actual_buffer_comparison', no_replay=True))
        raise
