"""One bounded integration of the prepared consumer with the fixed representative.

Existing inference.load owns tensor validation. This checks actual caller wiring,
device placement and RNG preservation; it adds no quality evaluation or training.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def bound(path, digest):
    assert path.is_file() and not path.is_symlink() and sha(path) == digest
    value = json.loads(path.read_text())
    assert sha(path) == digest
    return value


def publish(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def run():
    assert not (OUT/'result.json').exists() and not (OUT/'guard.json').exists()
    consumer_path = ROOT/'research/byte_input_checkpoint_consumer_v1.py'
    assert sha(consumer_path) == 'fa2bb6b4023c6232a6e4a0927c73adfa34dfd2fca157a769cec2663e293e16ba'
    preparation = bound(BASE/'checkpoint-consumer-preparation001/result.json',
        '0374a62343b0bda96083e14054965122669601e5fb26972b19a35b840c9325b7')
    review = bound(BASE/'recovery-final-review001/review.json',
        '84c027105e908d5e9415dcfadedd030b00cadb4943cee246aaf7dfa5f85401cc')
    assert review['accepted'] and review['practical_eligible'] and review['adopted'] is False
    assert sha(BASE/'quality-recovery001/final.json') == review['final_sha256']
    review_guard = json.loads((BASE/'recovery-final-review001/review.guard.json').read_text())
    assert review_guard['status'] == 'finished' and review_guard['reason'] is None
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from research.byte_input_quality_recovery_v1.guard import Guard
        with Guard(OUT/'guard.json', ROOT, timeout_seconds=120, rss_limit_bytes=4*1024**3,
                   reserve_bytes=8*1024**3, weight_roots=[
                       ROOT/'research_runs/byte-input-quality-v1-001',
                       ROOT/'research_runs/byte-input-quality-recovery-v1-001']) as guard:
            import torch
            from research import byte_input_checkpoint_consumer_v1 as consumer
            assert torch.backends.mps.is_available()
            torch.set_num_threads(4); guard.torch = torch
            before_cpu = torch.get_rng_state().clone()
            before_mps = torch.mps.get_rng_state().clone()
            started = time.monotonic()
            model, identity, receipt = consumer.load(preparation['plan_record'],
                preparation['representative_acceptance_record'], 'mps')
            torch.mps.synchronize()
            elapsed = time.monotonic()-started
            assert json.loads(json.dumps(identity)) == preparation['expected_identity_json']
            assert identity['model'][1] == 'raw_bytes' and identity['roles']['composer_initializer'] == 62100500007
            assert receipt == preparation['h200'] and sha(Path(receipt['path'])) == receipt['sha256']
            assert not model.training and all(p.device.type == 'mps' for p in model.parameters())
            assert torch.equal(before_cpu, torch.get_rng_state())
            assert torch.equal(before_mps, torch.mps.get_rng_state())
            parameters = list(model.parameters())
            assert len(parameters) == 105 and sum(p.numel() for p in parameters) == 16351859
            result = dict(accepted=True, runtime_loading_qualified=True, backend='mps',
                representative_slot=10, representative_rule=preparation['representative_rule'],
                consumer_sha256=sha(consumer_path), runner_sha256=sha(Path(__file__)),
                final_review_sha256=sha(BASE/'recovery-final-review001/review.json'),
                plan_record=preparation['plan_record'], acceptance_record=preparation['representative_acceptance_record'],
                h200=receipt, parent=preparation['parent'], expected_identity_json=preparation['expected_identity_json'],
                parameter_tensors=len(parameters), parameters=sum(p.numel() for p in parameters),
                cpu_rng_unchanged=True, mps_rng_unchanged=True, loading_seconds=elapsed,
                training_updates=0, quality_evaluations=0, adopted=False,
                scope='Actual prepared caller to qualified inference loader, complete saved parameter/buffer validation, MPS placement and RNG preservation. Reuses accepted model/inference/cache proofs.')
        finished = json.loads((OUT/'guard.json').read_text())
        assert finished['status'] == 'finished' and finished['reason'] is None
        result['guard_sha256'] = sha(OUT/'guard.json')
        publish('result.json', result)
    print(json.dumps({k: result[k] for k in ('accepted', 'runtime_loading_qualified', 'loading_seconds', 'parameters')}))


if __name__ == '__main__':
    try:
        run()
    except Exception:
        publish('failure.json', dict(traceback=traceback.format_exc(), no_automatic_replay=True))
        raise
