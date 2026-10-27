"""One source-frozen width64 hybrid feasibility pilot; no quality selection."""
import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from src.models.tokenizer_architecture import architecture_contract
from src.models.tokenizer_hybrid import GEOMETRY, LAYOUTS, POLICY
from src.research.protocol import digest, file_digest
from src.research.runner import atomic_json, session_lock, source_fingerprint
from src.research.tokenizer_study import ROOT, runtime_identity

RECIPE = ROOT/'research/tokenizer-evidence/horizon-001/adopted-width256-gqa-recipe.json'
PROTOCOL = ROOT/'research/TOKENIZER_HYBRID_PILOT_PROTOCOL.md'
MODULE = 'research.tokenizer-evidence.hybrid-preparation-001.run_pilot'


def config(root):
    cfg = copy.deepcopy(json.loads(RECIPE.read_text())['config'])
    cfg['model'].update(attn_type='hybrid', dim=64, n_heads=2, rope=0,
                        tokenizer_architecture_policy=POLICY, attention_layout=list(LAYOUTS[0]),
                        attention_width_policy=GEOMETRY, depth_width_policy='depth4_hybrid_unscaled')
    t = cfg['training']
    t.update(seed=2991, learning_rate=.00125, max_iters=20, eval_every=20,
             checkpoint_dir=str(root/'checkpoints'), metrics_file=str(root/'metrics.jsonl'),
             summary_file=str(root/'training-summary.json'))
    t['lr_schedule'].update(total_steps=20, start_step=10)
    t['weight_averaging'].update(total_steps=20, start_step=10, interval=5)
    architecture_contract(cfg['model'])
    return cfg


def verify(root):
    plan = json.loads((root/'plan.json').read_text())
    assert source_fingerprint() == plan['source'] and runtime_identity() == plan['runtime']
    for path, key in [(RECIPE, 'recipe_sha256'), (PROTOCOL, 'protocol_sha256'),
                      (Path(__file__), 'helper_sha256'),
                      (Path(plan['config']['data']['path']), 'corpus_sha256')]:
        assert file_digest(path) == plan[key]
    assert plan['config'] == config(root)
    return plan


def worker(root):
    import logging
    import torch
    from omegaconf import OmegaConf
    from src.attention.cache import build_caches
    from src.models.gpt import GPT
    from src.research.memory_guard import start_memory_guard
    from src.research.tokenizer_attention_bench import buffer_bytes, check_cached
    from src.train.train import run_training
    from src.train.weight_averaging import export_averaged_checkpoint
    from src.utils.checkpoint import checkpoint_loader, load_model_weights, read_checkpoint
    plan = verify(root)
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('MPS fallback forbidden')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    torch.set_num_threads(4)
    start_memory_guard()
    started = time.monotonic()
    cfg = OmegaConf.create(plan['config'])
    training = run_training(cfg)
    assert training['completed_steps'] == 20
    raw = Path(training['final_checkpoint'])
    avg = export_averaged_checkpoint(raw, root/'checkpoints/h200.pt', 200)
    payload = read_checkpoint(avg['checkpoint'])
    loader = checkpoint_loader(cfg, payload, cfg.data.path, 'cpu')
    model = GPT(cfg.model).to('mps').eval()
    load_model_weights(model, payload)
    rows = []
    for batch in (1, 8):
        ids = loader.train_data[:12].unsqueeze(0).repeat(batch, 1).to('mps')
        error = check_cached(model, cfg, ids)
        caches = build_caches(cfg, 'mps', batch_size=batch, dtype=torch.float32)
        sizes = [buffer_bytes([c]) for c in caches]
        assert sizes == [batch*41*64*4]*3 + [batch*128*(16+32)*4]
        rows.append(dict(batch=batch, max_absolute_error=error, per_layer_cache_bytes=sizes,
                         allocated_cache_bytes=sum(sizes)))
        del caches
    verify(root)
    atomic_json(root/'result.json', dict(status='completed', training=training,
                raw_checkpoint=str(raw), raw_sha256=file_digest(raw),
                h200_checkpoint=avg['checkpoint'], h200_sha256=file_digest(Path(avg['checkpoint'])),
                cache_checks=rows, wall_seconds=time.monotonic()-started, holdout_consumed=False,
                scope='20-update width64 feasibility only. Sampled validation is diagnostic; no full-validation ranking, no quality or latency adoption.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'worker'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); root = args.output.resolve()
    if args.command == 'prepare':
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
        assert shutil.disk_usage(ROOT).free > (8*1024+256)*1024**2
        root.mkdir(exist_ok=False)
        cfg = config(root)
        plan = dict(source=source_fingerprint(), runtime=runtime_identity(), config=cfg,
                    job_seconds=600, min_free_bytes=8*1024**3, max_output_bytes=256*1024**2,
                    recipe_sha256=file_digest(RECIPE), protocol_sha256=file_digest(PROTOCOL),
                    helper_sha256=file_digest(Path(__file__)), corpus_sha256=file_digest(Path(cfg['data']['path'])),
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        atomic_json(root/'plan.json', plan)
        atomic_json(root/'state.json', dict(status='prepared', plan_sha256=digest(plan)))
    elif args.command == 'worker':
        worker(root)
    else:
        plan = verify(root); state = json.loads((root/'state.json').read_text())
        assert state['plan_sha256'] == digest(plan) and state['status'] == 'prepared'
        with session_lock(root) as fd, session_lock(ROOT/'research_state/tokenizer-worker-lock') as gfd:
            if (root/'STOP').exists():
                raise RuntimeError('Pilot stopped')
            assert shutil.disk_usage(root).free > plan['min_free_bytes']+plan['max_output_bytes']
            with (root/'worker.log').open('x') as log:
                proc = subprocess.Popen([sys.executable, '-m', MODULE, 'worker', '--output', str(root)],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True, pass_fds=(fd, gfd))
                def stop(signum, frame):
                    (root/'STOP').write_text('Stop requested\n'); os.killpg(proc.pid, signal.SIGTERM)
                signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
                state.update(status='running', controller_pid=os.getpid(), worker_pid=proc.pid)
                atomic_json(root/'state.json', state)
                try:
                    code = proc.wait(timeout=plan['job_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                    code = 124
            verify(root)
            size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
            ok = (code == 0 and (root/'result.json').exists() and size <= plan['max_output_bytes']
                  and shutil.disk_usage(root).free >= plan['min_free_bytes'])
            state.update(status='completed' if ok else 'failed', exit_code=code, output_bytes=size)
            atomic_json(root/'state.json', state); print(json.dumps(state))


if __name__ == '__main__':
    main()
