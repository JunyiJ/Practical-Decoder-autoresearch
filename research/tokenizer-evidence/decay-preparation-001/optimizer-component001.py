"""Generated real optimizer arithmetic/resume prerequisite; no corpus or model."""
import copy
import hashlib
import io
import json
from pathlib import Path
import torch
from research.ffn_structure_v1.runtime.train.optimizers import HybridOptimizer
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
from research.local_mix_quality_v3.support import idle

B = Path(__file__).resolve().parent
ROOT = B.parents[2]
def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
plan = json.loads((B/'optimizer-component001-plan.json').read_text())
def verify():
    for path, digest in plan['sources'].items():
        assert sha(ROOT/path) == digest, path
verify()
assert not any((B/name).exists() for name in ('optimizer-component001-result.json', 'optimizer-component001-guard.json', 'optimizer-component001-failure.json'))
torch.set_num_threads(4)

def setup(decay, dtype):
    matrix = torch.nn.Parameter(torch.arange(32, dtype=dtype).reshape(4, 8)/31-.5)
    vector = torch.nn.Parameter(torch.tensor([.5, -1., 2., -.25], dtype=dtype))
    unused = torch.nn.Parameter(torch.tensor([3., -4.], dtype=dtype))
    muon = torch.optim.Muon([matrix], lr=.000625, weight_decay=0., adjust_lr_fn='match_rms_adamw')
    adam = torch.optim.AdamW([vector, unused], lr=.00125, weight_decay=0.)
    return [matrix, vector, unused], HybridOptimizer([muon, adam], decay)

def gradients(params, step):
    params[0].grad = torch.arange(32, dtype=params[0].dtype).reshape(4, 8)/64 + (step+1)/100
    params[1].grad = torch.tensor([.1, -.2, .3, -.4], dtype=params[1].dtype)*(step+1)
    params[2].grad = None

def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else: assert a == b

with session_lock(GLOBAL_LOCK.parent):
    idle()
    try:
        with CpuPrerequisiteGuard(B/'optimizer-component001-guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
            rows = []
            for dtype in (torch.float64, torch.float32):
                for decay in (.00006, 0.):
                    p, opt = setup(decay, dtype)
                    oracle, manual = setup(0., dtype)
                    saved = None
                    for step in range(6):
                        gradients(p, step); gradients(oracle, step)
                        with torch.no_grad():
                            for param in oracle:
                                if param.grad is not None and decay:
                                    param.mul_(1-decay)
                        # Independent direct child steps, no wrapper shrinkage.
                        for child in manual.optimizers: child.step()
                        opt.step()
                        equal(p, oracle)
                        equal(opt.state_dict()['optimizers'], manual.state_dict()['optimizers'])
                        opt.zero_grad(); manual.zero_grad()
                        if step == 2:
                            buffer = io.BytesIO()
                            torch.save(dict(parameters=[x.detach().clone() for x in p], optimizer=opt.state_dict()), buffer)
                            saved = buffer.getvalue()
                    restored = torch.load(io.BytesIO(saved), weights_only=False)
                    pp, oo = setup(decay, dtype)
                    with torch.no_grad():
                        for param, value in zip(pp, restored['parameters']): param.copy_(value)
                    oo.load_state_dict(restored['optimizer'])
                    assert all(group is child_group for group, child_group in zip(oo.param_groups, [g for child in oo.optimizers for g in child.param_groups]))
                    for step in range(3, 6):
                        gradients(pp, step); oo.step(); oo.zero_grad()
                    equal(pp, p); equal(oo.state_dict(), opt.state_dict())
                    equal(p[2], torch.tensor([3., -4.], dtype=dtype))
                    _, wrong = setup(0. if decay else .00006, dtype)
                    rejected = False
                    try: wrong.load_state_dict(restored['optimizer'])
                    except ValueError: rejected = True
                    assert rejected
                    rows.append(dict(dtype=str(dtype), decay=decay, steps=6, exact_manual_order=True, exact_serialized_resume=True, no_gradient_unchanged=True, cross_arm_rejected=True))
            # Isolate shrinkage over the proposed horizon with zero updates.
            p = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
            opt = HybridOptimizer([torch.optim.SGD([p], lr=.00125)], .00006)
            for _ in range(1200):
                p.grad = torch.zeros_like(p); opt.step()
            factor = (1-.00006)**1200
            assert abs(p.item()-factor) < 1e-13
            verify()
            result = dict(status='passed', cases=rows, shrinkage_only_1200_factor=p.item(), analytical_factor=factor,
                          sources_unchanged=len(plan['sources']), scientific_seeds_used=False, corpus_used=False,
                          full_model_tested=False, quality_result=False, checkpoints_on_disk=False,
                          limitation='Generated CPU optimizer composition only; actual model policy/grouping, MPS, raw/EMA/RNG/H200/cache integration and guarded worker acceptance remain required.')
        save(B/'optimizer-component001-result.json', result)
        print(json.dumps(result))
    except BaseException as exc:
        save(B/'optimizer-component001-failure.json', dict(status='failed_closed', error=repr(exc), no_retry=True))
        raise
