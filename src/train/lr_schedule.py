"""Opt-in, absolute-update learning-rate schedules with fixed resume horizons.

Width-scaled group rates are captured before loading optimizer state. The
schedule never compounds a multiplier onto a previously scheduled rate.
"""
from collections.abc import Mapping
import math


def schedule_contract(training):
    raw = getattr(training, 'lr_schedule', None)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError('lr_schedule must be a mapping or null')
    spec = dict(raw)
    kind = spec.get('kind')
    keys = {'constant': {'kind'},
            'cosine': {'kind', 'total_steps', 'min_ratio'},
            'linear_cooldown': {'kind', 'total_steps', 'min_ratio', 'start_step'}}
    if not isinstance(kind, str) or kind not in keys or set(spec) != keys[kind]:
        raise ValueError('Unknown schedule or invalid schedule fields')
    if kind == 'constant':
        return spec
    total = spec['total_steps']
    if type(total) is not int or total < 2 or total != training.max_iters:
        raise ValueError('Scheduled total_steps must equal max_iters and be >= 2')
    floor = spec['min_ratio']
    if (isinstance(floor, bool) or not isinstance(floor, (int, float))
            or not math.isfinite(floor) or not 0 < floor <= 1):
        raise ValueError('Schedule min_ratio must be finite and in (0, 1]')
    if kind == 'linear_cooldown':
        start = spec['start_step']
        if type(start) is not int or not 0 <= start < total - 1:
            raise ValueError('Cooldown start_step must be before the final update')
    return spec


def multiplier(spec, step):
    """Multiplier used for zero-based update ``step``; spec is validated."""
    if type(step) is not int or step < 0:
        raise ValueError('Update index must be a nonnegative integer')
    if spec is None or spec['kind'] == 'constant':
        return 1.0
    if step >= spec['total_steps']:
        raise ValueError('Update index exceeds the frozen schedule horizon')
    last = spec['total_steps'] - 1
    floor = spec['min_ratio']
    # Exact endpoints make persisted final-rate audits independent of cosine
    # endpoint rounding, while all intermediate values retain the definition.
    if step == last:
        return float(floor)
    if spec['kind'] == 'cosine':
        return floor + (1 - floor) * (1 + math.cos(math.pi * step / last)) / 2
    start = spec['start_step']
    return 1.0 if step <= start else 1 - (1 - floor) * (step - start) / (last - start)


class LearningRateSchedule:
    def __init__(self, optimizer, training):
        self.spec = schedule_contract(training)
        self.optimizer = optimizer
        self.base_lrs = tuple(float(g['lr']) for g in optimizer.param_groups)
        if any(not math.isfinite(v) or v < 0 for v in self.base_lrs):
            raise ValueError('Optimizer base learning rates must be finite and nonnegative')
        self.identities = tuple(self._identity(g) for g in optimizer.param_groups)

    @staticmethod
    def _identity(group):
        return (tuple(group.get('parameter_names', ())), group.get('width_lr_scale', 1.),
                tuple(id(p) for p in group['params']))

    def apply(self, step):
        if self.spec is None:
            return None  # Legacy runs: no group changes and no added metrics.
        groups = self.optimizer.param_groups
        if tuple(self._identity(g) for g in groups) != self.identities:
            raise ValueError('Optimizer groups changed after schedule construction')
        ratio = multiplier(self.spec, step)
        for base, group in zip(self.base_lrs, groups):
            group['lr'] = base * ratio
        return ratio
