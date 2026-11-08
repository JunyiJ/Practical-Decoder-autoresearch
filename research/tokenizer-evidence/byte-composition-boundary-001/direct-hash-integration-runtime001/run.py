"""Bounded differential check of additive hash-only model/consumer integration."""
import copy
import fcntl
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024): digest.update(chunk)
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
    assert sha(Path(__file__)) == plan['runner_sha256']
    for rec in plan['bindings']:
        path = ROOT/rec['path']
        assert path.is_file() and not path.is_symlink() and path.stat().st_size == rec['size'] and sha(path) == rec['sha256']
    mapping = read(OUT.parent/'direct-hash-integration-source001/source-map.json')
    for row in mapping['pairs']:
        for rec in (row['original'], row['candidate']):
            path = ROOT/rec['path']; assert path.stat().st_size == rec['size'] and sha(path) == rec['sha256']
        restored = (ROOT/row['candidate']['path']).read_text()
        for before, after in reversed(row['exact_replacements']): restored = restored.replace(after, before)
        assert restored == (ROOT/row['original']['path']).read_text()
    recipe = read(OUT.parent/'adoption001/recipe.json')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from research.byte_input_quality_recovery_v1.guard import Guard
        with Guard(OUT/'guard.json', ROOT, timeout_seconds=180, rss_limit_bytes=4*1024**3,
                   reserve_bytes=8*1024**3, weight_roots=[ROOT/'research_runs/byte-input-quality-v1-001',
                       ROOT/'research_runs/byte-input-quality-recovery-v1-001']) as guard:
            import torch
            from research import byte_input_checkpoint_consumer_v1 as consumer
            torch.set_num_threads(4); assert torch.backends.mps.is_available(); guard.torch = torch
            cpu_rng = torch.get_rng_state().clone(); mps_rng = torch.mps.get_rng_state().clone()
            prepared = consumer.prepare(recipe['loader']['plan_record'], recipe['loader']['acceptance_record'])
            roles = copy.deepcopy(prepared['identity']['roles']); roles.update(plan['fresh_sampler_roles'])
            runtime = dict(python=platform.python_version(), torch=torch.__version__, threads=torch.get_num_threads(), platform=platform.platform())
            binding = dict(source_sha256=sha(OUT/'plan.json'), runtime_sha256=hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest())

            def fingerprint(tensor):
                value = tensor.detach().to('cpu').contiguous()
                return dict(dtype=str(value.dtype), shape=list(value.shape),
                    sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())

            def tree(value):
                if isinstance(value, torch.Tensor): return dict(tensor=fingerprint(value))
                if type(value) is dict: return {key: tree(item) for key, item in value.items()}
                if type(value) is tuple: return dict(tuple=[tree(item) for item in value])
                if type(value) is list: return [tree(item) for item in value]
                assert value is None or type(value) in (str, int, float, bool), type(value)
                return value

            def arm(backend, label):
                versions = plan['implementations'][label]
                inference = importlib.import_module('research.'+versions['inference'])
                training = importlib.import_module('research.'+versions['training'])
                cache = importlib.import_module('research.'+versions['cache'])
                model_api = importlib.import_module('research.'+versions['model'])
                torch.mps.synchronize(); started = time.perf_counter()
                model = inference.load(prepared['h200'], prepared['identity'], prepared['parent'], prepared['codebook'], backend)
                torch.mps.synchronize(); loading = time.perf_counter()-started
                view = training.load_view(roles, backend)
                assert view.data_identity == prepared['identity']['data']
                prefix = view.train_data[:128].to(backend)
                cached = {}
                for batch in (1, 8):
                    session = cache.Session(model, batch)
                    tokens = prefix.unsqueeze(0).repeat(batch, 1)
                    cached[str(batch)] = [fingerprint(session.logits(tokens[:, :96], 0)),
                                          fingerprint(session.logits(tokens[:, 96:], 96))]
                    assert session.position == 128
                    session.reset(); assert session.position == 0
                    del session, tokens
                # Fixed training-prefix derivative comparison; no sampler draws here.
                inputs = view.train_data[:1024].reshape(8, 128).to(backend)
                labels = view.train_data[1:1025].reshape(8, 128).to(backend)
                model.train(); prediction = model.forward_train(inputs, labels)
                forward = dict(logits=fingerprint(prediction.primary.values), loss=fingerprint(prediction.loss))
                prediction.loss.backward(); del prediction
                gradients = {name: fingerprint(parameter.grad) for name, parameter in model.ordered_parameters()}
                optimizer = model_api.build_optimizer(model, .0025)
                optimizer.zero_grad(set_to_none=True)
                context = training.Context(model, optimizer, view, 4, .0025, roles, binding)
                initial = training.capture(context); training.validate(context, initial)
                state = [tree(initial)]; del initial
                timings = []
                for _ in range(2):
                    torch.mps.synchronize(); started = time.perf_counter()
                    context.step()
                    torch.mps.synchronize(); timings.append(time.perf_counter()-started)
                    captured = training.capture(context); training.validate(context, captured)
                    state.append(tree(captured)); del captured
                assert context.optimizer._steps == 2 and context.averages.snapshots == 1
                assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(mps_rng, torch.mps.get_rng_state())
                value = dict(backend=backend, implementation=label, loading_seconds=loading,
                    cache=cached, forward=forward, gradients=gradients, states=state,
                    step_seconds=timings, input_ids=fingerprint(inputs), label_ids=fingerprint(labels),
                    completed_steps=2, averaging_snapshots=1)
                publish(f'{backend}-{label}.json', value)
                del context, optimizer, model, view, inputs, labels
                gc.collect(); torch.mps.empty_cache()
                return value

            results = []
            for backend in ('cpu', 'mps'):
                values = {label: arm(backend, label) for label in plan['orders'][backend]}
                old, new = values['reference'], values['direct_bytes']
                for key in ('cache', 'forward', 'gradients', 'states', 'input_ids', 'label_ids', 'completed_steps', 'averaging_snapshots'):
                    assert old[key] == new[key], (backend, key, 'Exact differential equality failed')
                results.append(dict(backend=backend, exact=True, parameter_gradients=len(new['gradients']),
                    state_boundaries=3, cached_batches=[1, 8], updates_per_implementation=2,
                    original_step_seconds=old['step_seconds'], optimized_step_seconds=new['step_seconds'],
                    paired_step_speedup=[a/b for a, b in zip(old['step_seconds'], new['step_seconds'])],
                    total_two_step_speedup=sum(old['step_seconds'])/sum(new['step_seconds'])))
            result = dict(accepted=True, runner_sha256=sha(Path(__file__)), plan_sha256=sha(OUT/'plan.json'),
                runtime=runtime, binding=binding, roles=roles, cases=results,
                total_numerical_updates=8, quality_evaluations=0, new_weight_archives=0,
                cpu_rng_unchanged=True, mps_rng_unchanged=True, numerical_integration_qualified=True,
                optimized_implementation_adopted=False, limits=plan['interpretation_limits'])
        assert read(OUT/'guard.json')['status'] == 'finished'
        result['guard_sha256'] = sha(OUT/'guard.json')
        result['case_evidence'] = {p.name: sha(p) for p in sorted(OUT.glob('*-*.json')) if p.name.startswith(('cpu-', 'mps-'))}
        publish('result.json', result); publish('STOP', dict(status='complete_hash_only_integration_comparison', no_replay=True))
    print(json.dumps(dict(accepted=True, cases=results)))


if __name__ == '__main__':
    try:
        run()
    except Exception:
        publish('failure.json', dict(traceback=traceback.format_exc(), no_automatic_replay=True))
        publish('STOP', dict(status='closed_failed_hash_only_integration_comparison', no_replay=True))
        raise
