"""Twenty new-data updates per arm; execution evidence, never quality selection."""
import argparse
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/tokenizer-evidence/general-rsi-pilot-001'
WEIGHTS = ROOT / 'research_runs/general-rsi-pilot-001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare():
    import torch
    prior = json.loads((ROOT / 'research/tokenizer-evidence/public-inference-correction-001/plan.json').read_text())
    sources = dict(prior['sources'])
    for name in ('research/general_rsi_data_v1.py', 'research/general_rsi_pilot_v1.py',
                 'research/GENERAL_RSI_PLAN.md', 'src/research/cpu_prerequisite_guard.py'):
        sources[name] = sha(ROOT / name)
    assets = json.loads((ROOT / 'research/tokenizer-evidence/byte-residual-readout-001/pilot-plan.json').read_text())['codebook_assets']
    plan = dict(kind='general_rsi_execution_pilot_v1', sources=sources, codebook_assets=assets,
        torch_version=torch.__version__, python_version=platform.python_version(),
        roles=dict(zip(('native_initializer', 'composer_initializer', 'train_sampler', 'validation_sampler'),
                       range(62110100001, 62110100005))),
        prior_role_audit='rg on research/research_state JSON/JSONL before assigning: exit1, no matches',
        arms=[dict(name='native', learning_rate=.00125), dict(name='raw_bytes', learning_rate=.0025)],
        steps=20, timeout_seconds=600, rss_limit_bytes=4*1024**3,
        mps_limit_bytes=6*1024**3, weight_cap_bytes=256*1024**2,
        metadata_cap_bytes=8*1024**2, reserve_bytes=10*1024**3,
        logits_atol=3e-5, logits_rtol=3e-4, quality_selection=False)
    assert not OUT.exists() and not WEIGHTS.exists()
    OUT.mkdir()
    put(OUT / 'plan.json', plan)
    print(sha(OUT / 'plan.json'), flush=True)


def run(plan):
    import torch
    from research.general_rsi_data_v1 import GeneralData
    from research.byte_input_model_v6 import ByteInputModel, Spec, build_codebook, build_optimizer
    from research.gradient_clip_model_v1.clipping import clip_gradients
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    assert torch.backends.mps.is_available()
    assert torch.__version__ == plan['torch_version'] and platform.python_version() == plan['python_version']
    torch.set_num_threads(4)
    roles = plan['roles']
    data = GeneralData(roles['train_sampler'], roles['validation_sampler'], 'mps')
    # Check actual scoring inventory and isolation without calculating any loss.
    validation_before = data.view.generators['val'].get_state().clone()
    counts = [0] * len(data.indexes['val'])
    for doc, source, x, y in data.validation_chunks():
        assert 1 <= y.numel() <= 128 and torch.equal(x[:, 1:], y[:, :-1])
        counts[doc] += y.numel()
    assert counts == [r['tokens'] - 1 for r in data.indexes['val']]
    assert sum(counts) == 100416
    raw = []
    for asset in plan['codebook_assets']:
        assert sha(asset['path']) == asset['sha256']
        raw.append(Path(asset['path']).read_bytes())
    book = build_codebook(*raw, 'raw_bytes')
    expected_batches = None
    common_weights = None
    reports = []
    for arm in plan['arms']:
        name, lr = arm['name'], arm['learning_rate']
        data.view.generators['train'].manual_seed(roles['train_sampler'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(roles['native_initializer'])
            spec = Spec('native') if name == 'native' else Spec(name, roles['composer_initializer'], book.identity)
            model = ByteInputModel(spec, None if name == 'native' else book).to('mps')
        common = {n: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                  for n, p in model.ordered_parameters() if n.startswith(('transformer.h.', 'transformer.ln_f.', 'lm_head.'))}
        if common_weights is None:
            common_weights = common
        else:
            assert common == common_weights
        optimizer = build_optimizer(model, lr)
        rows, batches = [], []
        peak_mps = 0
        for step in range(plan['steps']):
            torch.mps.synchronize()
            started = time.perf_counter()
            x, y = data.batch()
            batches.append(hashlib.sha256(x.cpu().numpy().tobytes()+y.cpu().numpy().tobytes()).hexdigest())
            optimizer.zero_grad()
            result = model.forward_train(x, y)
            loss = float(result.loss)
            result.loss.backward()
            clip = clip_gradients((p for _, p in model.ordered_parameters()), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            del result
            torch.mps.synchronize()
            memory = torch.mps.driver_allocated_memory()
            peak_mps = max(peak_mps, memory)
            assert memory < plan['mps_limit_bytes']
            row = dict(step=step+1, loss=loss, seconds=time.perf_counter()-started,
                       mps_driver_bytes=memory, clip=clip)
            rows.append(row)
            with (OUT / f'{name}-steps.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n'); stream.flush()
            if step in (0, 9, 19):
                print(json.dumps(dict(arm=name, **row)), flush=True)
        assert torch.equal(validation_before, data.view.generators['val'].get_state())
        if expected_batches is None:
            expected_batches = batches
        else:
            assert batches == expected_batches
        items = model.ordered_parameters()
        parameter_bytes = sum(p.numel()*p.element_size() for _, p in items)
        optimizer_bytes = sum(v.numel()*v.element_size() for child in optimizer._optimizers
                              for fields in child.state.values() for v in fields.values())
        # Pilot endpoints are inference weights only; no claim of complete resume.
        payload = dict(kind='general_rsi_pilot_weights_v1', identity=model.identity,
                       parameters={n: p.detach().cpu().clone() for n, p in items})
        path = WEIGHTS / f'{name}.pt'
        with path.open('xb') as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        assert sum(p.stat().st_size for p in WEIGHTS.iterdir()) <= plan['weight_cap_bytes']
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['identity'] == model.identity
        assert list(saved['parameters']) == list(payload['parameters'])
        assert all(torch.equal(v, saved['parameters'][n]) for n, v in payload['parameters'].items())
        model.eval()
        probe = data.view.train_data[:128][None, :]
        with torch.no_grad(), FP32RMSNormInference():
            mps_logits = model.primary_logits(probe.to('mps')).values.cpu()
        del optimizer, items, payload
        model.to('cpu')
        with torch.no_grad():
            for n, p in model.ordered_parameters():
                p.copy_(saved['parameters'][n])
        with torch.no_grad(), FP32RMSNormInference():
            cpu_logits = model.primary_logits(probe).values
        torch.testing.assert_close(cpu_logits, mps_logits, atol=plan['logits_atol'], rtol=plan['logits_rtol'])
        report = dict(arm=name, steps=20, parameter_bytes=parameter_bytes,
            optimizer_state_tensor_bytes=optimizer_bytes,
            median_update_seconds=statistics.median(r['seconds'] for r in rows[5:]),
            training_seconds=sum(r['seconds'] for r in rows), peak_mps_driver_observed_bytes=peak_mps,
            max_cpu_mps_logit_error=float((cpu_logits-mps_logits).abs().max()),
            exact_saved_weights=True, checkpoint=dict(path=str(path), size=path.stat().st_size, sha256=sha(path)),
            batch_hashes=batches, validation_sampler_unchanged=True)
        put(OUT / f'{name}-result.json', report)
        reports.append(report)
        del model, saved, cpu_logits, mps_logits, x, y
        gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()
    return dict(arms=reports, paired_batches=True, common_body_initialization=True,
                validation_target_count=sum(counts), quality_selection=False,
                test_opened=False, full_training_resume_qualified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('plan_sha', nargs='?')
    args = parser.parse_args()
    if args.action == 'prepare':
        return prepare()
    assert sha(OUT / 'plan.json') == args.plan_sha
    plan = json.loads((OUT / 'plan.json').read_text())
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free >= plan['reserve_bytes'] + plan['weight_cap_bytes'] + plan['metadata_cap_bytes']
        for name, digest in plan['sources'].items():
            assert sha(ROOT / name) == digest, name
        WEIGHTS.mkdir()
        put(OUT / 'attempt.json', dict(pid=os.getpid(), plan_sha256=args.plan_sha, started=time.time()))
        try:
            with CpuPrerequisiteGuard(OUT / 'guard.json', ROOT, timeout_seconds=plan['timeout_seconds'],
                    rss_limit_bytes=plan['rss_limit_bytes'], reserve_bytes=plan['reserve_bytes']):
                result = run(plan)
            for name, digest in plan['sources'].items():
                assert sha(ROOT / name) == digest, name
            assert sum(p.stat().st_size for p in OUT.iterdir()) < plan['metadata_cap_bytes']
            put(OUT / 'result.json', result)
            print(json.dumps(dict(status='complete', arms=[{k:r[k] for k in ('arm','median_update_seconds','training_seconds','max_cpu_mps_logit_error')} for r in result['arms']])), flush=True)
        except BaseException:
            put(OUT / 'failure.json', dict(traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
