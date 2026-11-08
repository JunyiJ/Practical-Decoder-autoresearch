"""Profile one warmed step of the adopted implementation; no quality test."""
import copy
import cProfile
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pstats
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
    assert not (OUT/'guard.json').exists() and not (OUT/'profile.pstats').exists()
    plan = read(OUT/'plan.json'); assert sha(Path(__file__)) == plan['runner_sha256']
    for record in plan['bindings']:
        path = ROOT/record['path']; assert path.stat().st_size == record['size'] and sha(path) == record['sha256']
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
            recipe = read(OUT.parent/'direct-hash-adoption001/recipe.json'); loader = recipe['loader']
            model, identity, receipt = consumer.load(loader['implementation_record'], loader['plan_record'], loader['acceptance_record'], 'mps')
            from research import byte_input_training_v3 as training
            from research.byte_input_model_v4 import build_optimizer
            roles = copy.deepcopy(identity['roles']); roles.update(plan['fresh_sampler_roles'])
            view = training.load_view(roles, 'mps'); optimizer = build_optimizer(model, .0025)
            binding = dict(source_sha256=sha(OUT/'plan.json'), runtime_sha256=hashlib.sha256(torch.__version__.encode()).hexdigest())
            context = training.Context(model, optimizer, view, 1200, .0025, roles, binding)
            warmups = []
            for _ in range(2):
                torch.mps.synchronize(); started = time.perf_counter(); context.step(); torch.mps.synchronize()
                warmups.append(time.perf_counter()-started)
            profiler = cProfile.Profile()
            torch.mps.synchronize(); started = time.perf_counter()
            profiler.enable(); row = context.step(); profiler.disable()
            torch.mps.synchronize(); elapsed = time.perf_counter()-started
            assert context.optimizer._steps == 3 and context.averages.snapshots == 0
            assert torch.equal(before_cpu, torch.get_rng_state()) and torch.equal(before_mps, torch.mps.get_rng_state())
            profiler.dump_stats(str(OUT/'profile.pstats'))
            stats = pstats.Stats(profiler)
            functions = []
            for (path, line, name), (primitive, total, exclusive, cumulative, callers) in stats.stats.items():
                functions.append(dict(path=path, line=line, function=name, primitive_calls=primitive,
                    total_calls=total, exclusive_seconds=exclusive, cumulative_seconds=cumulative,
                    callers=[dict(path=k[0], line=k[1], function=k[2], statistics=list(v) if isinstance(v, tuple) else v)
                             for k, v in callers.items()]))
            result = dict(completed=True, plan_sha256=sha(OUT/'plan.json'), runner_sha256=sha(Path(__file__)),
                profile_sha256=sha(OUT/'profile.pstats'), roles=roles, h200=receipt, warmup_seconds=warmups,
                profiled_step_seconds=elapsed, profiler_total_seconds=stats.total_tt, completed_updates=3,
                measured_update=3, averaging_snapshots=0, gradient_tensors=row['gradient_tensors'],
                cpu_rng_unchanged=True, mps_rng_unchanged=True, quality_evaluations=0,
                functions=sorted(functions, key=lambda f: f['cumulative_seconds'], reverse=True),
                limits=plan['interpretation_limits'])
        assert read(OUT/'guard.json')['status'] == 'finished'
        result['guard_sha256'] = sha(OUT/'guard.json')
        publish('result.json', result); publish('STOP', dict(status='complete_warmed_step_profile', no_replay=True))
    print(json.dumps(dict(completed=True, warmup_seconds=warmups, profiled_step_seconds=elapsed,
        top_cumulative=[{k: f[k] for k in ('path','function','total_calls','exclusive_seconds','cumulative_seconds')}
                        for f in result['functions'][:16]])))


if __name__ == '__main__':
    try: run()
    except Exception:
        publish('failure.json', dict(traceback=traceback.format_exc(), no_automatic_replay=True))
        raise
