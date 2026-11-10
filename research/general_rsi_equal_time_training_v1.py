"""Time-budgeted general training. Frozen step-budgeted sources stay unchanged."""
import copy
import hashlib
import json
import math
import os
import time
from research.general_rsi_training_clock_v1 import TrainingClock
from pathlib import Path

import torch
from torch.nn import functional as F
from research.general_rsi_data_v1 import GeneralData, MANIFEST_SHA
from research.byte_input_model_v6 import ByteInputModel, Spec, build_optimizer
from research.block_norm_v1.runtime.train.lr_schedule import multiplier
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
from research.byte_context_cpu_reference_v1 import cpu_cross_entropy

KIND = 'general_rsi_equal_time_training_v1'


from research.general_rsi_training_v2 import clone_tree, sha, write_checkpoint, read_checkpoint


class Context:
    def __init__(self, arm, lr, budget_ns, half_life_ns, roles, book, binding, device='mps'):
        assert arm in ('native', 'raw_bytes') and lr in (.00125, .0025)
        assert device == 'mps'
        assert type(half_life_ns) is int and half_life_ns > 0
        self.clock = TrainingClock(budget_ns)
        self.half_life_ns = half_life_ns
        self.ema_last_ns = None
        self.recovery_charges = []
        assert set(roles) == {'native_initializer', 'composer_initializer', 'train_sampler', 'validation_sampler'}
        assert len(set(roles.values())) == 4
        assert type(binding) is str and len(binding) == 64
        self.config = dict(arm=arm, lr=lr, budget_ns=budget_ns, half_life_ns=half_life_ns, roles=copy.deepcopy(roles),
                           manifest_sha256=MANIFEST_SHA, binding=binding, device=device)
        mps_rng = torch.mps.get_rng_state().clone()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(roles['native_initializer'])
                spec = Spec('native') if arm == 'native' else Spec(arm, roles['composer_initializer'], book.identity)
                self.model = ByteInputModel(spec, None if arm == 'native' else book).to(device)
        finally:
            torch.mps.set_rng_state(mps_rng)
        self.optimizer = build_optimizer(self.model, float(lr))
        self.base_lrs = tuple(g['lr'] for g in self.optimizer.param_groups)
        self.data = GeneralData(roles['train_sampler'], roles['validation_sampler'], device)
        self.schedule = dict(kind='training_time_linear_cooldown', budget_ns=budget_ns, min_ratio=.1)
        self.ema_start = (2*budget_ns+2)//3
        self.ema = {}
        self.snapshots = 0
        self.history = []

    @property
    def step(self): return self.optimizer._steps

    def charge_lost_work(self, duration_ns):
        """Charge audited work since restored state. Never replay for free."""
        assert type(duration_ns) is int and duration_ns > 0
        self.recovery_charges.append(dict(at_step=self.step, before_ns=self.clock.spent_ns,
                                          duration_ns=duration_ns))
        self.clock = TrainingClock.restore(self.clock.state_dict(), uncheckpointed_work_ns=duration_ns)

    def update(self, now_ns=time.perf_counter_ns):
        begin_ns = self.clock.spent_ns
        ratio = self.clock.lr_ratio()
        with self.clock.update(torch.mps.synchronize, now_ns):
            for g, base in zip(self.optimizer.param_groups, self.base_lrs):
                g['lr'] = base * ratio
            self.model.train()
            self.optimizer.zero_grad()
            x, y = self.data.batch()
            output = self.model.forward_train(x, y)
            loss = float(output.loss.detach())
            assert math.isfinite(loss)
            output.loss.backward()
            clip = clip_gradients((p for _, p in self.model.ordered_parameters()), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            # Causal boundary: elapsed time before this update determines EMA.
            # Its own averaging cost is charged by the enclosing clock interval.
            with torch.no_grad():
                if begin_ns >= self.ema_start:
                    if self.ema_last_ns is None:
                        self.ema = {n: p.detach().clone() for n, p in self.model.ordered_parameters()}
                    else:
                        alpha = 1.0-2.0**(-(begin_ns-self.ema_last_ns)/self.half_life_ns)
                        for n, p in self.model.ordered_parameters():
                            self.ema[n].lerp_(p.detach(), alpha)
                    self.ema_last_ns = begin_ns
                    self.snapshots += 1
            batch_hash = hashlib.sha256(x.cpu().numpy().tobytes()+y.cpu().numpy().tobytes()).hexdigest()
        row = dict(step=self.step, loss=loss, lr_ratio=ratio, clip=clip,
                   batch_sha256=batch_hash, begin_ns=begin_ns, end_ns=self.clock.spent_ns)
        self.history.append(row)
        return row

    def capture(self):
        self.optimizer.validate()
        items = self.model.ordered_parameters()
        assert all(p.grad is None for _, p in items) and len(self.history) == self.step
        by_id = {id(p): n for n, p in items}
        children = []
        for child in self.optimizer._optimizers:
            groups = [dict(name=by_id[id(g['params'][0])], options={k:v for k,v in g.items() if k != 'params'})
                      for g in child.param_groups]
            state = {by_id[id(p)]: fields for p, fields in child.state.items()}
            children.append(dict(kind=type(child).__name__, groups=groups, state=state))
        return clone_tree(dict(kind=KIND, config=self.config, model_identity=self.model.identity,
            immutable_buffers=self.model._own_expected_buffer_hashes,
            parameters=dict(items), children=children, step=self.step, schedule=self.schedule,
            base_lrs=self.base_lrs, ema_start=self.ema_start, ema=self.ema, snapshots=self.snapshots,
            clock=self.clock.state_dict(), ema_last_ns=self.ema_last_ns, recovery_charges=self.recovery_charges,
            samplers={n:g.get_state() for n,g in self.data.view.generators.items()},
            cpu_rng=torch.random.get_rng_state(), mps_rng=torch.mps.get_rng_state(), history=self.history))

    def restore(self, payload):
        """Only restore into a pristine context; discard destination on any failure."""
        assert self.step == 0 and not self.history
        assert payload['kind'] == KIND and payload['config'] == self.config
        assert payload['model_identity'] == self.model.identity
        assert payload['immutable_buffers'] == self.model._own_expected_buffer_hashes
        assert payload['schedule'] == self.schedule and payload['base_lrs'] == self.base_lrs
        assert payload['ema_start'] == self.ema_start
        step = payload['step']
        assert type(step) is int and 0 <= step
        assert len(payload['history']) == step
        assert [r['step'] for r in payload['history']] == list(range(1, step+1))
        saved_clock = TrainingClock.restore(payload['clock'])
        assert saved_clock.budget_ns == self.config['budget_ns']
        charges = payload['recovery_charges']
        assert type(charges) is list
        previous_end = 0
        charge_index = 0
        for completed in range(step+1):
            while charge_index < len(charges) and charges[charge_index]['at_step'] == completed:
                charge = charges[charge_index]
                assert charge['before_ns'] == previous_end
                assert type(charge['duration_ns']) is int and charge['duration_ns'] > 0
                previous_end += charge['duration_ns']
                charge_index += 1
            if completed == step: break
            row = payload['history'][completed]
            assert previous_end == row['begin_ns'] < row['end_ns']
            assert row['begin_ns'] < saved_clock.budget_ns
            assert row['lr_ratio'] == TrainingClock(saved_clock.budget_ns, row['begin_ns']).lr_ratio()
            previous_end = row['end_ns']
        assert charge_index == len(charges) and saved_clock.spent_ns == previous_end
        ema_rows = [r for r in payload['history'] if r['begin_ns'] >= self.ema_start]
        expected_snapshots = len(ema_rows)
        assert payload['ema_last_ns'] == (ema_rows[-1]['begin_ns'] if ema_rows else None)
        assert payload['snapshots'] == expected_snapshots
        items = dict(self.model.ordered_parameters())
        assert list(payload['parameters']) == list(items)
        assert list(payload['ema']) == (list(items) if expected_snapshots else [])
        for panel in (payload['parameters'], payload['ema']):
            for n, value in panel.items():
                assert value.device.type == 'cpu' and value.dtype == items[n].dtype
                assert value.shape == items[n].shape and bool(torch.isfinite(value).all())
        assert len(payload['children']) == 2
        staged = []
        for index, (child, saved) in enumerate(zip(self.optimizer._optimizers, payload['children'])):
            assert saved['kind'] == type(child).__name__ and len(saved['groups']) == len(child.param_groups)
            names = []
            for actual, entry in zip(child.param_groups, saved['groups']):
                n = next(n for n,p in items.items() if p is actual['params'][0]); names.append(n)
                assert entry['name'] == n
                assert {k:v for k,v in entry['options'].items() if k != 'lr'} == {k:v for k,v in actual.items() if k not in ('params','lr')}
                expected_lr = actual['lr'] * (payload['history'][-1]['lr_ratio'] if step else 1.)
                assert entry['options']['lr'] == expected_lr
            assert list(saved['state']) == (names if step else [])
            states = {}
            for n, fields in saved['state'].items():
                assert set(fields) == ({'momentum_buffer'} if index == 0 else {'step','exp_avg','exp_avg_sq'})
                states[n] = {}
                for key, value in fields.items():
                    assert value.dtype == torch.float32 and value.device.type == 'cpu' and bool(torch.isfinite(value).all())
                    assert value.shape == (() if key == 'step' else items[n].shape)
                    if key == 'step': assert value.item() == step
                    if key == 'exp_avg_sq': assert bool((value >= 0).all())
                    states[n][key] = value.to('cpu' if key == 'step' else 'mps').clone()
            staged.append(states)
        assert set(payload['samplers']) == {'train','val'}
        for value in [*payload['samplers'].values(), payload['cpu_rng'], payload['mps_rng']]:
            assert value.dtype == torch.uint8 and value.device.type == 'cpu' and value.ndim == 1
        with torch.no_grad():
            for n,p in items.items(): p.copy_(payload['parameters'][n])
        for child, saved, states in zip(self.optimizer._optimizers, payload['children'], staged):
            for group, entry in zip(child.param_groups, saved['groups']): group['lr'] = entry['options']['lr']
            for n, fields in states.items(): child.state[items[n]] = fields
        self.optimizer._steps = step
        self.ema = {n:v.to('mps').clone() for n,v in payload['ema'].items()}
        self.snapshots = expected_snapshots
        self.clock = saved_clock
        self.recovery_charges = copy.deepcopy(charges)
        self.ema_last_ns = payload['ema_last_ns']
        self.history = copy.deepcopy(payload['history'])
        for n,g in self.data.view.generators.items(): g.set_state(payload['samplers'][n])
        torch.random.set_rng_state(payload['cpu_rng']); torch.mps.set_rng_state(payload['mps_rng'])
        self.optimizer.validate()

