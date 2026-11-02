"""Generated cooldown-onset/group-rate/serialized toy-SGD prerequisite; no corpus."""
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from fractions import Fraction
import torch
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule, multiplier, schedule_contract
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
from research.parallel_residual_quality_v1.support import idle

B = Path(__file__).resolve().parent
ROOT = B.parents[2]
def read(p): return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def save(p, value):
    with p.open('x') as f: json.dump(value, f, indent=2); f.write('\n')
plan = read(B / 'schedule-component001-plan.json')
for path, digest in plan['sources'].items():
    assert sha(Path(path)) == digest, path
assert not (B / 'schedule-component001-result.json').exists()
assert not (B / 'schedule-component001-guard.json').exists()
torch.set_num_threads(4)

def setup(spec):
    params = [torch.nn.Parameter(torch.tensor([1., -2.], dtype=torch.float64)),
              torch.nn.Parameter(torch.tensor([.5, 3.], dtype=torch.float64))]
    opt = torch.optim.SGD([dict(params=[params[0]], lr=.00125, parameter_names=['a'], width_lr_scale=1.),
                           dict(params=[params[1]], lr=.000625, parameter_names=['b'], width_lr_scale=.5)], momentum=.9)
    schedule = LearningRateSchedule(opt, SimpleNamespace(max_iters=1200, lr_schedule=spec))
    return params, opt, schedule

with session_lock(GLOBAL_LOCK.parent):
    idle()
    try:
        with CpuPrerequisiteGuard(B / 'schedule-component001-guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
            rows = []
            for arm, start in (('cooldown800', 800), ('cooldown600', 600)):
                spec = dict(kind='linear_cooldown', total_steps=1200, start_step=start, min_ratio=.1)
                oracle = []
                for step in range(1200):
                    ratio = Fraction(1) if step <= start else Fraction(1)-Fraction(9,10)*(step-start)/(1199-start)
                    oracle.append(float(ratio))
                got = [multiplier(spec, step) for step in range(1200)]
                assert max(abs(x-y) for x,y in zip(got, oracle)) < 3e-16
                params, opt, schedule = setup(spec)
                saved = {}
                def update(params, opt, schedule, step):
                    ratio = schedule.apply(step)
                    assert ratio == got[step]
                    for base, group in zip((.00125, .000625), opt.param_groups): assert group['lr'] == base*got[step]
                    for index, param in enumerate(params): param.grad = param.detach().clone()*(index+1)
                    opt.step(); opt.zero_grad(set_to_none=True)
                for step in range(1200):
                    update(params, opt, schedule, step)
                    if step in (0, 598, 599, 600, 799, 800, 1198):
                        buffer = io.BytesIO()
                        torch.save(dict(params=[p.detach().clone() for p in params], optimizer=opt.state_dict(), step=step), buffer)
                        saved[step] = buffer.getvalue()
                target = [p.detach().clone() for p in params]
                target_state = copy.deepcopy(opt.state_dict())
                for step, blob in saved.items():
                    restored = torch.load(io.BytesIO(blob), weights_only=False)
                    pp, oo, ss = setup(spec)  # Capture original base rates before loading scheduled rates.
                    with torch.no_grad():
                        for p, v in zip(pp, restored['params']): p.copy_(v)
                    oo.load_state_dict(restored['optimizer'])
                    for remaining in range(step+1, 1200): update(pp, oo, ss, remaining)
                    assert all(torch.equal(p, v) for p,v in zip(pp, target))
                    actual = oo.state_dict()
                    assert actual['param_groups'] == target_state['param_groups']
                    for index in actual['state']:
                        assert torch.equal(actual['state'][index]['momentum_buffer'], target_state['state'][index]['momentum_buffer'])
                # Reapplying an absolute update must not compound group multipliers.
                for step in (1199, 0, 599, 600, 601, 799, 800, 801, 1199):
                    schedule.apply(step); rates = [g['lr'] for g in opt.param_groups]
                    schedule.apply(step); assert rates == [g['lr'] for g in opt.param_groups]
                assert abs(sum(got)-(1020. if start==800 else 930.))<1e-10
                rows.append(dict(arm=arm, updates=1200, start_step=start, integral=sum(got), base_lr_sum=.00125*sum(got), resumed_after_steps=list(saved),
                                 exact_parameter_and_momentum_resume=True, group_rates_match=True))
            for bad in [dict(kind='linear_cooldown',total_steps=1200,start_step=1199,min_ratio=.1), dict(kind='linear_cooldown',total_steps=1200,start_step=True,min_ratio=.1), dict(kind='linear_cooldown',total_steps=1200,start_step=600,min_ratio=0.), dict(kind='linear_cooldown',total_steps=1199,start_step=600,min_ratio=.1)]:
                try: schedule_contract(SimpleNamespace(max_iters=1200,lr_schedule=bad))
                except ValueError: pass
                else: raise AssertionError('Invalid schedule accepted')
            for path, digest in plan['sources'].items(): assert sha(Path(path)) == digest
        guard = read(B / 'schedule-component001-guard.json')
        assert guard['status'] == 'finished' and guard['reason'] is None
        save(B / 'schedule-component001-result.json', dict(status='passed', cases=rows,
             plan_sha256=sha(B / 'schedule-component001-plan.json'), guard_sha256=sha(B / 'schedule-component001-guard.json'),
             scope='Generated CPU float64 toy SGD only; actual model/Muon/AdamW/EMA/raw-checkpoint contracts still required.',
             research_corpus_read=False, scientific_seed_consumed=False, training_panel_launched=False))
    except Exception as exc:
        save(B / 'schedule-component001-failure.json', dict(status='failed', error=repr(exc), no_retry=True))
        raise
print(json.dumps(dict(status='passed', cases=rows)))
