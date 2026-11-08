"""Check the public loader once, then publish equivalent implementation adoption."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024): digest.update(chunk)
    return digest.hexdigest()


def record(path):
    return dict(path=str(path), size=path.stat().st_size, sha256=sha(path))


def read(path):
    return json.loads(path.read_text())


def publish(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def run():
    assert not (OUT/'guard.json').exists() and not (OUT/'recipe.json').exists()
    manifest = record(OUT/'implementation.json')
    previous_path = OUT.parent/'adoption001/recipe.json'
    previous = read(previous_path)
    assert previous['status'] == 'adopted_under_original_three_bundle_gates'
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from research.byte_input_quality_recovery_v1.guard import Guard
        with Guard(OUT/'guard.json', ROOT, timeout_seconds=60, rss_limit_bytes=4*1024**3,
                   reserve_bytes=8*1024**3, weight_roots=[ROOT/'research_runs/byte-input-quality-v1-001',
                       ROOT/'research_runs/byte-input-quality-recovery-v1-001']) as guard:
            import torch
            from research import byte_input_checkpoint_consumer_v2 as consumer
            torch.set_num_threads(4); assert torch.backends.mps.is_available(); guard.torch = torch
            before_cpu = torch.get_rng_state().clone(); before_mps = torch.mps.get_rng_state().clone()
            loader = previous['loader']; started = time.monotonic()
            model, identity, receipt = consumer.load(manifest, loader['plan_record'], loader['acceptance_record'], 'mps')
            torch.mps.synchronize(); elapsed = time.monotonic()-started
            assert receipt == previous['representative']['h200'] and model.identity == identity['model']
            assert not model.training and next(model.parameters()).device.type == 'mps'
            assert type(model).__module__ == 'research.byte_input_model_v4.model'
            assert sum(p.numel() for p in model.parameters()) == 16351859
            assert torch.equal(before_cpu, torch.get_rng_state()) and torch.equal(before_mps, torch.mps.get_rng_state())
            assert record(OUT/'implementation.json') == manifest
            loaded = dict(accepted=True, implementation=manifest, consumer=record(Path(consumer.__file__).resolve()),
                source=record(Path(__file__).resolve()), representative_slot=10, h200=receipt,
                loading_seconds=elapsed, backend='mps', cpu_rng_unchanged=True, mps_rng_unchanged=True,
                model_class=type(model).__module__+'.'+type(model).__name__, training_updates=0, quality_evaluations=0)
        guard_value = read(OUT/'guard.json')
        assert guard_value['status'] == 'finished' and guard_value['reason'] is None
        loaded['guard'] = record(OUT/'guard.json'); publish('load.json', loaded)
        implementation = read(OUT/'implementation.json')
        comparison = read(Path(implementation['runtime_evidence']['path']))
        recipe = copy.deepcopy(previous)
        recipe.update(status='adopted_equivalent_direct_hash_implementation', previous_recipe=record(previous_path),
            implementation=manifest, consumer=loaded['consumer'], runtime_loading=record(OUT/'load.json'),
            loader=dict(module='research.byte_input_checkpoint_consumer_v2', function='load',
                implementation_record=manifest, plan_record=loader['plan_record'], acceptance_record=loader['acceptance_record'],
                qualified_backends=['cpu', 'mps'], caller_requirements=loader['caller_requirements']),
            implementation_timings=dict(evidence=implementation['runtime_evidence'], cases=comparison['cases'],
                scope='Two paired training-context updates per implementation/backend, including first optimizer state and initial averaging. Descriptive, not sustained throughput or a confidence interval.'),
            quality_evidence_unchanged=True,
            future_baseline='Use this source-bound hash-only implementation for the adopted raw-byte model. Model math, artifact formats, quality evidence and selected weights are unchanged. New scientific campaigns require fresh roles and their own source/resource admission.')
        recipe['original_quality_implementation_costs'] = recipe.pop('measured_costs')
        publish('recipe.json', recipe)
        publish('closure.json', dict(status='closed_adopted_equivalent_implementation', at=datetime.now(timezone.utc).isoformat(),
            recipe=record(OUT/'recipe.json'), previous_recipe=record(previous_path), implementation=manifest,
            runtime_loading=record(OUT/'load.json'), numerical_comparison=implementation['runtime_evidence'],
            same_selected_weights=True, quality_panel_replayed=False, quality_evidence_unchanged=True,
            no_scientific_tolerance_change=True, next_action='Use the faster baseline; examine the next measured composition bottleneck, starting from the existing padding work analysis, before choosing a bounded implementation experiment.'))
        publish('STOP', dict(status='completed_closed_adopted_equivalent_implementation', recipe=record(OUT/'recipe.json'), no_replay=True))
    print(json.dumps(dict(adopted=True, loading_seconds=elapsed, recipe=str(OUT/'recipe.json'))))


if __name__ == '__main__':
    try:
        run()
    except Exception:
        publish('failure.json', dict(traceback=traceback.format_exc(), no_automatic_replay=True))
        raise
