"""Two finite width512 GQA/MLA pilots; correctness/resources, never quality selection."""
import argparse
import gc
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from ..models.tokenizer_architecture import architecture_contract
from .tokenizer_wide_config import config as wide_config
KINDS = ('gqa', 'mla')
from .protocol import digest, file_digest
from .runner import atomic_json, session_lock, source_fingerprint
from .tokenizer_study import ROOT, runtime_identity

RECIPE = ROOT/'research/tokenizer-evidence/horizon-001/adopted-width256-gqa-recipe.json'
PROTOCOL = ROOT/'research/TOKENIZER_WIDE_ATTENTION_PILOT_PROTOCOL.md'
MODULE = 'src.research.tokenizer_wide_pilot'
RESERVE = 8*1024**3
ALLOWANCE = dict(gqa=1737836135, mla=1725319424)
PARAMETERS = dict(gqa=62877696, mla=62424192)


def config(root, kind):
    if kind not in KINDS:
        raise ValueError('Undeclared wider attention arm')
    cfg = wide_config(root, kind, width=512, seed=3391, lr=.00125, steps=20)
    t = cfg['training']; t.update(eval_every=20)
    t['lr_schedule'].update(start_step=10)
    t['weight_averaging'].update(start_step=10, interval=5)
    architecture_contract(cfg['model'])
    return cfg


def required_free(completed):
    if not isinstance(completed, list) or completed != list(KINDS[:len(completed)]):
        raise ValueError('Invalid completed pilot prefix')
    # Every quality trial remains reserved; pilots never consume that headroom.
    return RESERVE + 4*1024**2 + 6*sum(ALLOWANCE.values()) + sum(ALLOWANCE[k] for k in KINDS if k not in completed)


def prepare(root):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit clean source before freezing pilot')
    if shutil.disk_usage(ROOT).free < required_free([]):
        raise RuntimeError('Full pilot plus quality panel does not fit')
    requests = {k: dict(kind=k, config=config(root/k, k)) for k in KINDS}
    plan = dict(source=source_fingerprint(), runtime=runtime_identity(), requests=requests,
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                recipe_sha256=file_digest(RECIPE), protocol_sha256=file_digest(PROTOCOL),
                corpus_sha256=file_digest(Path(requests[KINDS[0]]['config']['data']['path'])),
                job_seconds=600, reserve_bytes=RESERVE, allowances=ALLOWANCE,
                max_pilot_output_bytes=sum(ALLOWANCE.values())+4*1024**2,
                full_panel_required_free_bytes=required_free([]))
    root.mkdir(exist_ok=False)
    for k, req in requests.items():
        (root/k).mkdir(); atomic_json(root/k/'request.json', req)
    atomic_json(root/'plan.json', plan)
    atomic_json(root/'state.json', dict(status='prepared', completed=[], active=None, plan_sha256=digest(plan)))
    return plan


def verify(root):
    plan = json.loads((root/'plan.json').read_text())
    state = json.loads((root/'state.json').read_text())
    if state['plan_sha256'] != digest(plan) or source_fingerprint() != plan['source'] or runtime_identity() != plan['runtime']:
        raise ValueError('Frozen source/runtime/plan changed')
    for path, key in [(RECIPE, 'recipe_sha256'), (PROTOCOL, 'protocol_sha256'),
                      (Path(plan['requests'][KINDS[0]]['config']['data']['path']), 'corpus_sha256')]:
        if file_digest(path) != plan[key]:
            raise ValueError('Frozen recipe/protocol/corpus changed')
    for k in KINDS:
        expected = dict(kind=k, config=config(root/k, k))
        if plan['requests'][k] != expected or json.loads((root/k/'request.json').read_text()) != expected:
            raise ValueError('Frozen request changed')
    required_free(state['completed'])
    return plan, state


def worker(root, kind):
    import logging
    import torch
    from omegaconf import OmegaConf
    from ..attention.cache import build_caches
    from ..models.gpt import GPT
    from ..train.train import run_training
    from ..train.weight_averaging import export_averaged_checkpoint
    from ..utils.checkpoint import checkpoint_loader, load_model_weights, read_checkpoint
    from .memory_guard import start_memory_guard
    from .tokenizer_attention_bench import buffer_bytes, check_cached
    plan, _ = verify(root)
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('Strict MPS required')
    torch.set_num_threads(4); start_memory_guard()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    cfg = OmegaConf.create(plan['requests'][kind]['config']); trial = root/kind
    started = time.monotonic(); training = run_training(cfg)
    if training['completed_steps'] != 20:
        raise ValueError('Incomplete pilot')
    raw = Path(training['final_checkpoint'])
    avg = export_averaged_checkpoint(raw, trial/'checkpoints/h200.pt', 200)
    rows = []
    for name, path in [('raw', raw), ('h200', Path(avg['checkpoint']))]:
        payload = read_checkpoint(path)
        if payload['step'] != 20:
            raise ValueError('Wrong checkpoint step')
        if name == 'h200' and payload['averaged_weights']['snapshots'] != 3:
            raise ValueError('Wrong short-tail snapshot count')
        loader = checkpoint_loader(cfg, payload, cfg.data.path, 'cpu')
        model = GPT(cfg.model).to('mps').eval(); load_model_weights(model, payload)
        del payload
        if sum(p.numel() for p in model.parameters()) != PARAMETERS[kind]:
            raise ValueError('Unexpected parameter count')
        storages = {p.untyped_storage().data_ptr(): p.untyped_storage().nbytes() for p in model.parameters()}
        if sum(storages.values()) != 4*PARAMETERS[kind]:
            raise ValueError('Unexpected unique FP32 parameter storage')
        checks = []
        for batch in (1, 8):
            ids = loader.train_data[:12].unsqueeze(0).repeat(batch, 1).to('mps')
            error = check_cached(model, cfg, ids)
            caches = build_caches(cfg, 'mps', batch_size=batch, dtype=torch.float32)
            cache_bytes = buffer_bytes(caches)
            if cache_bytes != batch*4*128*4*(512 if kind == 'gqa' else 160):
                raise ValueError('Unexpected wider attention cache allocation')
            checks.append(dict(batch=batch, max_absolute_error=error, cache_bytes=cache_bytes))
            del caches, ids
        rows.append(dict(kind=name, checkpoint=str(path), sha256=file_digest(path), bytes=path.stat().st_size,
                         unique_parameter_bytes=sum(storages.values()), cache_checks=checks,
                         observed_mps_driver_bytes=torch.mps.driver_allocated_memory()))
        del model, loader; gc.collect(); torch.mps.empty_cache()
    verify(root)
    atomic_json(trial/'result.json', dict(status='completed', kind=kind, training=training, checkpoints=rows,
                request_sha256=file_digest(trial/'request.json'), wall_seconds=time.monotonic()-started,
                max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                holdout_consumed=False, quality_selection=False))


def size(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def run(root):
    plan, state = verify(root)
    # Never restart a failed, running, interrupted or completed controller.
    if state['status'] != 'prepared':
        raise ValueError('Pilot controller is not freshly prepared; review required')
    def stop(signum, frame):
        (root/'STOP').write_text(f'Stop signal {signum}\n')
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    with session_lock(root) as fd, session_lock(ROOT/'research_state/tokenizer-worker-lock') as gfd:
        proc = None
        try:
            for kind in KINDS:
                verify(root)
                if (root/'STOP').exists():
                    raise RuntimeError('Stop requested')
                free = shutil.disk_usage(root).free
                if free < required_free(state['completed']):
                    raise RuntimeError('Remaining full-panel storage guard')
                trial = root/kind
                atomic_json(trial/'prelaunch.json', dict(free_bytes=free, required_free_bytes=required_free(state['completed'])))
                state.update(status='running', active=kind, controller_pid=os.getpid())
                atomic_json(root/'state.json', state)
                with (trial/'worker.log').open('x') as log:
                    proc = subprocess.Popen([sys.executable, '-m', MODULE, 'worker', '--output', str(root), '--kind', kind],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(fd, gfd))
                    state['worker_pid'] = proc.pid; atomic_json(root/'state.json', state)
                    started = time.monotonic()
                    while proc.poll() is None:
                        reason = ('stop' if (root/'STOP').exists() else 'timeout' if time.monotonic()-started >= plan['job_seconds']
                                  else 'disk' if shutil.disk_usage(root).free < RESERVE or size(trial) > ALLOWANCE[kind]
                                  or size(root) > plan['max_pilot_output_bytes'] else None)
                        if reason:
                            atomic_json(trial/'guard-failure.json', dict(reason=reason))
                            os.killpg(proc.pid, signal.SIGTERM)
                            try: proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                            raise RuntimeError('Pilot guard: '+reason)
                        try: proc.wait(timeout=2)
                        except subprocess.TimeoutExpired: pass
                if proc.returncode or not (trial/'result.json').is_file():
                    raise RuntimeError(f'Pilot worker failed: {proc.returncode}')
                result = json.loads((trial/'result.json').read_text())
                if (result['status'] != 'completed' or result['kind'] != kind or result['holdout_consumed']
                    or result['quality_selection'] or result['request_sha256'] != file_digest(trial/'request.json')
                    or size(trial) > ALLOWANCE[kind]):
                    raise ValueError('Pilot result or resource mismatch')
                verify(root)
                state['completed'].append(kind); state.update(active=None, worker_pid=None)
                atomic_json(root/'state.json', state)
                print('DONE '+kind, flush=True)
            state.update(status='completed', controller_pid=None)
            atomic_json(root/'state.json', state)
        except BaseException as error:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            state.update(status='failed', error=f'{type(error).__name__}: {error}')
            atomic_json(root/'state.json', state)
            raise


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare', 'run', 'worker'])
    p.add_argument('--output', required=True, type=Path); p.add_argument('--kind', choices=KINDS)
    a = p.parse_args(); root = a.output.absolute()
    if a.command == 'prepare': prepare(root)
    elif a.command == 'worker': worker(root, a.kind)
    else: run(root)


if __name__ == '__main__':
    main()
