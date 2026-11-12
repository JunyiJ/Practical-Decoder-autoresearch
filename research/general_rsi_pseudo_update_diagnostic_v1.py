"""Training-only common-state mechanism diagnostic. No persistent model outputs."""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/tokenizer-evidence/general-rsi-pseudo-update-001'
STOP = False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def check_stop():
    if STOP or json.loads((ROOT / 'research_state/general-rsi-state.json').read_text()).get('stop_requested', False):
        raise InterruptedError('Human stop')


def digest(value):
    """Typed recursive CPU tensor digest, including order and complete state."""
    import torch
    h = hashlib.sha256()
    def visit(v):
        h.update(type(v).__name__.encode() + b'\0')
        if isinstance(v, torch.Tensor):
            assert v.device.type == 'cpu'
            h.update(str((v.dtype, tuple(v.shape))).encode())
            h.update(v.contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k, item in v.items(): visit(k); visit(item)
        elif isinstance(v, (tuple, list)):
            for item in v: visit(item)
        else: h.update(repr(v).encode())
        h.update(b'\xff')
    visit(value)
    return h.hexdigest()


def batch_hash(*tensors):
    h = hashlib.sha256()
    for t in tensors: h.update(t.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def ce_batch(values, starts):
    import torch
    x = torch.stack([values[s:s+512] for s in starts]).reshape(8, 128)
    y = torch.stack([values[s+1:s+513] for s in starts]).reshape(8, 128)
    return x, y


def inputs(config, data):
    """Reproducible input inventory. Evaluation excludes every update source range."""
    import torch
    from research.general_rsi_pseudo_spans_v1 import Sampler
    values = data.view.train_data
    seed_rng = torch.Generator().manual_seed(config['roles']['train_sampler'])
    seed_starts = torch.randint(len(values)-512, (2,), generator=seed_rng).tolist()
    ce_rng = torch.Generator().manual_seed(config['ce_seed'])
    pseudo = Sampler(config['pseudo_seed'])
    excluded = [(s, s+513) for s in seed_starts]
    cases, batches = [], []
    for case in range(8):
        starts = torch.randint(len(values)-512, (2,), generator=ce_rng).tolist()
        x, y = ce_batch(values, starts)
        px, py, prev, mask, details = pseudo.batch()
        excluded.extend((s, s+513) for s in starts)
        for detail in details:
            _, s, length, prefix = map(int, pseudo.spans[detail['inventory_index']])
            excluded.append((s-prefix, s+length))
        cases.append(dict(case=case, ce_starts=starts, ce_sha256=batch_hash(x, y),
                          pseudo_sha256=batch_hash(px, py, prev, mask), pseudo_rows=details))
        batches.append((x, y, px, py, prev, mask))
    rng = torch.Generator().manual_seed(config['evaluation_seed'])
    windows = []
    eligible = [i for i, r in enumerate(data.indexes['train']) if r['tokens'] >= 129]
    proposals = 0
    while len(windows) < 64:
        proposals += 1
        assert proposals < 10000
        doc = eligible[int(torch.randint(len(eligible), (), generator=rng))]
        row = data.indexes['train'][doc]
        start = row['offset'] + int(torch.randint(row['tokens']-128, (), generator=rng))
        if any(start < b and a < start+129 for a, b in excluded): continue
        windows.append(dict(document=doc, source=row['source'], start=start, targets=128))
        excluded.append((start, start+129))
    eval_batches = [(values[w['start']:w['start']+128][None, :],
                     values[w['start']+1:w['start']+129][None, :]) for w in windows]
    inventory = dict(seed_ce_starts=seed_starts, seed_ce_sha256=batch_hash(*ce_batch(values, seed_starts)),
                     cases=cases, evaluation_windows=windows, proposals=proposals,
                     evaluation_sha256=batch_hash(*(t for pair in eval_batches for t in pair)),
                     evaluation_targets=8192, disjoint_from_all_update_source_positions=True)
    return inventory, batches, eval_batches


def evaluate(model, batches):
    import torch
    from torch.nn import functional as F
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    prior = model.training
    versions = tuple(p._version for _, p in model.ordered_parameters())
    sums = []
    try:
        model.eval()
        with torch.no_grad(), FP32RMSNormInference():
            for x, y in batches:
                check_stop()
                logits = model.primary_logits(x.to('mps')).values[0]
                losses = F.cross_entropy(logits, y[0].to('mps'), reduction='none')
                assert bool(torch.isfinite(losses).all())
                sums.append(math.fsum(losses.cpu().double().tolist()))
    finally: model.train(prior)
    assert versions == tuple(p._version for _, p in model.ordered_parameters())
    return dict(ce=math.fsum(sums)/8192, window_loss_sums=sums, targets=8192)


def displacement(model, common, gradients):
    import torch
    from research.byte_input_model_v6.optimizer import is_muon
    groups = {k: dict(parameter_sq=0., update_sq=0., gradient_sq=0., gradient_dot_update=0., tensors=0)
              for k in ('Muon', 'AdamW')}
    with torch.no_grad():
        for name, p in model.ordered_parameters():
            before = common['parameters'][name].double()
            delta = p.detach().cpu().double() - before
            assert bool(torch.isfinite(delta).all())
            g = groups['Muon' if is_muon(name, p) else 'AdamW']
            g['parameter_sq'] += float(before.square().sum())
            g['update_sq'] += float(delta.square().sum()); g['tensors'] += 1
            if gradients is not None:
                grad = gradients[name].double()
                g['gradient_sq'] += float(grad.square().sum())
                g['gradient_dot_update'] += float((grad*delta).sum())
    for g in groups.values():
        g['update_norm'] = math.sqrt(g['update_sq'])
        g['relative_update_norm'] = math.sqrt(g['update_sq']/g['parameter_sq'])
        den = math.sqrt(g['gradient_sq']*g['update_sq'])
        g['gradient_update_cosine'] = g['gradient_dot_update']/den if den else None
    return groups


def execute(plan, binding, attempt):
    import torch
    from research.general_rsi_pseudo_training_v1 import Context
    from research.general_rsi_pseudo_spans_v1 import repetition_loss
    from research.byte_input_model_v6 import build_optimizer
    from research.gradient_clip_model_v1.clipping import clip_gradients
    assert torch.__version__ == plan['torch_version'] and torch.backends.mps.is_available()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    torch.set_num_threads(4)
    peak_mps = 0
    def boundary():
        nonlocal peak_mps
        check_stop(); torch.mps.synchronize()
        peak_mps = max(peak_mps, torch.mps.driver_allocated_memory())
        assert peak_mps < plan['mps_limit_bytes']
    records = []
    for config in plan['parents']:
        boundary()
        ctx = Context('native_ce', 1200*10**9, plan['ema_half_life_ns'], config['roles'],
                      config['pseudo_seed'], config['parent'], binding)
        inventory, batches, evaluation = inputs(config, ctx.data)
        assert inventory == config['inputs']
        first = ctx.update(); boundary()
        assert first['batch_sha256'] == inventory['seed_ce_sha256'] and first['lr_ratio'] == 1.
        common = ctx.capture(); common_hash = digest(common)
        baseline = evaluate(ctx.model, evaluation)
        put(attempt / f"parent-{config['parent']['slot']:02d}.json",
            dict(parent=config['parent'], seed_ce=first, state_sha256=common_hash, baseline=baseline))
        for i, (x, y, px, py, prev, mask) in enumerate(batches):
            for branch in plan['branches']:
                boundary(); tick = time.monotonic()
                # Qualified restore requires pristine destination optimizer/history.
                # Model ownership is unchanged. No old child state is reused.
                ctx.optimizer.zero_grad()
                ctx.optimizer = build_optimizer(ctx.model, .00025)
                ctx.history = []
                ctx.restore(common)
                restored_hash = digest(ctx.capture())
                assert restored_hash == common_hash
                ctx.model.train(); gradients = None; clip = None; objective = None; info = None
                if branch == 'zero':
                    for _, p in ctx.model.ordered_parameters(): p.grad = torch.zeros_like(p)
                elif branch in ('pseudo', 'ce'):
                    if branch == 'pseudo':
                        loss, info = repetition_loss(ctx.model.primary_logits(px.to('mps')).values,
                                                     py.to('mps'), prev.to('mps'), mask.to('mps'))
                    else: loss = ctx.model.forward_train(x.to('mps'), y.to('mps')).loss
                    objective = float(loss.detach()); assert math.isfinite(objective)
                    loss.backward()
                    clip = clip_gradients((p for _, p in ctx.model.ordered_parameters()), 1.0)
                    gradients = {n:p.grad.detach().cpu().clone() for n,p in ctx.model.ordered_parameters()}
                    del loss
                if branch != 'identity': ctx.optimizer.step()
                ctx.optimizer.zero_grad(); ctx.optimizer.validate(); boundary()
                movement = displacement(ctx.model, common, gradients)
                del gradients
                if branch == 'identity':
                    assert digest(ctx.capture()) == common_hash
                    assert all(g['update_sq'] == 0. for g in movement.values())
                score = evaluate(ctx.model, evaluation); boundary()
                if branch == 'identity': assert score == baseline
                record = dict(parent_slot=config['parent']['slot'], case=i, branch=branch,
                              restored_state_sha256=restored_hash, input=inventory['cases'][i],
                              objective_before=objective, pseudo_info=info, clip=clip,
                              displacement=movement, evaluation=score,
                              natural_ce_delta=score['ce']-baseline['ce'],
                              optimizer_steps=ctx.optimizer._steps, seconds=time.monotonic()-tick)
                put(attempt / f"p{config['parent']['slot']:02d}-c{i:02d}-{branch}.json", record)
                records.append(record)
                print(json.dumps({k:record[k] for k in ('parent_slot','case','branch','natural_ce_delta','seconds')}), flush=True)
        del ctx, common, batches, evaluation
        gc.collect(); torch.mps.empty_cache()
    assert len(records) == 96
    return dict(kind='common_state_pseudo_update_diagnostic_v1', plan_sha256=binding,
                cases=24, branch_observations=96, seed_ce_updates=3, branch_updates=72,
                peak_mps_boundary_bytes=peak_mps, test_opened=False, validation_scored=False,
                no_saved_model=True, records=[dict(parent_slot=r['parent_slot'],case=r['case'],branch=r['branch'],
                    natural_ce_delta=r['natural_ce_delta']) for r in records])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    os.chdir(ROOT)
    lock = (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_stop()
    attempt = OUT / f'attempt-{args.attempt:02d}'
    attempt.mkdir(exist_ok=False)
    def stop(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    plan = json.loads((OUT / 'plan.json').read_text()); binding = sha(OUT / 'plan.json')
    put(attempt / 'launch.json', dict(pid=os.getpid(), plan_sha256=binding, started_unix=time.time()))
    try:
        with CpuPrerequisiteGuard(attempt / 'guard.json', ROOT, rss_limit_bytes=plan['rss_limit_bytes'],
                                 reserve_bytes=plan['reserve_bytes'], timeout_seconds=plan['timeout_seconds']):
            for path, expected in plan['sources'].items(): assert sha(path) == expected, path
            assert json.loads((OUT / 'admission.json').read_text())['plan_sha256'] == binding
            result = execute(plan, binding, attempt)
            result['total_seconds'] = time.monotonic()-started
            put(attempt / 'result.json', result)
    except BaseException:
        put(attempt / 'failure.json', dict(traceback=traceback.format_exc(), seconds=time.monotonic()-started))
        raise


if __name__ == '__main__': main()
