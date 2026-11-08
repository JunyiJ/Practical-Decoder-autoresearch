"""Read-only audit of the completed RAW payload left by the guard failure."""
import fcntl
import hashlib
import json
from pathlib import Path

from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
Q = OUT.parent/'quality001'


def read(path):
    return json.loads(path.read_text())


with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with CpuPrerequisiteGuard(OUT/'raw-audit-guard.json', ROOT, timeout_seconds=120,
                             rss_limit_bytes=2*1024**3, reserve_bytes=8*1024**3):
        import torch
        torch.set_num_threads(4)
        from research import byte_input_training_v2 as training
        failure = read(OUT/'failure.json')
        for record in failure['original_evidence']:
            data = Path(record['path']).read_bytes()
            assert len(data) == record['size'] and hashlib.sha256(data).hexdigest() == record['sha256']
        plan = read(Q/'plan.json')
        assert hashlib.sha256((Q/'plan.json').read_bytes()).hexdigest() == 'cd0dcc19ae94e09e3d14aa071043f6c89b498e482e942bc0bfeb65bda15d8ddd'
        spec = read(Q/'slots/14/attempt.json')['spec']
        assert spec == dict(bundle='fresh1', learning_rate=0.00125, slot=14, stage='confirmation', variant='native')
        raw = training.read_bound(failure['raw_receipt'])
        identity = raw['identity']; numerical = raw['numerical']; averages = raw['averages']
        assert identity['roles'] == plan['bundles']['fresh1'] and identity['binding'] == plan['binding']
        assert identity['data'] == plan['data'] and identity['recipe'] == plan['recipes']['0.00125']
        assert identity['model'] == ('rsi_byte_input_model_math_v3', 'native', None, None, 256, 50257, 128)
        assert numerical['completed_steps'] == averages['completed_steps'] == 1200
        assert numerical['backend'] == 'mps:0' and averages['snapshots'] == 41
        assert averages['spec']['start_step'] == 800 and averages['spec']['interval'] == 10
        assert averages['spec']['half_lives'] == [200] and set(averages['averages']) == {'200'}
        parameters = numerical['parameters']; names = list(parameters)
        assert len(names) == 88 and sum(t.numel() for t in parameters.values()) == 28492416
        assert averages['parameter_names'] == list(averages['averages']['200']) == names
        for name, tensor in parameters.items():
            avg = averages['averages']['200'][name]
            assert tensor.dtype == avg.dtype == torch.float32 and tensor.shape == avg.shape
            assert bool(torch.isfinite(tensor).all()) and bool(torch.isfinite(avg).all())
        assert len(raw['history']) == 1200
        for i, history in enumerate(raw['history'], 1):
            row = read(Q/'slots/14'/f'step{i:04}.json')
            assert history == {k: row[k] for k in history} and history['step'] == i
            assert history['active_parameter_names'] == names and history['gradient_tensors'] == 88
        for step in (1, 301, 601, 901, 1200):
            diagnostic = read(Q/'slots/14'/f'diagnostic{step:04}.json')
            assert diagnostic['step'] == step and diagnostic['diagnostic_only']
            assert all(diagnostic[k] for k in ('train_sampler_unchanged', 'global_rng_unchanged', 'parameters_unchanged'))
        samplers = {name: hashlib.sha256(state.numpy().tobytes()).hexdigest() for name, state in raw['samplers'].items()}
        assert samplers == read(Q/'slots/12/review.json')['sampler_hashes'] == read(Q/'slots/13/review.json')['sampler_hashes']
        result = dict(accepted_read_only_integrity=True, original_scientific_slot_accepted=False,
                      raw_receipt=failure['raw_receipt'], plan_sha256=hashlib.sha256((Q/'plan.json').read_bytes()).hexdigest(),
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      completed_steps=1200, snapshots=41, matched_step_records=1200,
                      parameter_tensors=88, parameter_elements=28492416, sampler_hashes=samplers,
                      no_training_or_model_execution=True,
                      remaining='Restore into a pristine qualified context, export existing H200 averages, run unchanged full validation/cache/saved review under corrected guard in a separately identified recovery.')
    assert read(OUT/'raw-audit-guard.json')['status'] == 'finished'
    with (OUT/'raw-audit.json').open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
    print(json.dumps(result, indent=2))
