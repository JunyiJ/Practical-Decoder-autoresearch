"""General-corpus training state and scoring, isolated from frozen campaigns."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from research.general_rsi_data_v1 import GeneralData, MANIFEST_SHA
from research.byte_input_model_v6 import ByteInputModel, Spec, build_optimizer
from research.block_norm_v1.runtime.train.lr_schedule import multiplier
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
from research.byte_context_cpu_reference_v1 import cpu_cross_entropy

KIND = 'general_rsi_complete_training_v1'


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(v) for v in value)
    return copy.deepcopy(value)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2):
            value.update(block)
    return value.hexdigest()


def write_checkpoint(path, payload, cap=512*1024**2):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    assert not path.exists() and not temporary.exists()
    class Bounded:
        def __init__(self, stream): self.stream = stream
        def write(self, value):
            assert self.stream.tell() + len(value) <= cap
            return self.stream.write(value)
        def tell(self): return self.stream.tell()
        def flush(self): return self.stream.flush()
    with temporary.open('xb') as stream:
        torch.save(payload, Bounded(stream))
        stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    return dict(path=str(path.resolve()), size=path.stat().st_size, sha256=sha(path))


def read_checkpoint(receipt):
    path = Path(receipt['path'])
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size == receipt['size'] and sha(path) == receipt['sha256']
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert sha(path) == receipt['sha256']
    return payload


class Context:
    def __init__(self, arm, lr, horizon, roles, book, binding, device='mps'):
        assert arm in ('native', 'raw_bytes') and lr in (.00125, .0025)
        assert horizon in (40, 2048, 10240) and device == 'mps'
        assert set(roles) == {'native_initializer', 'composer_initializer', 'train_sampler', 'validation_sampler'}
        assert len(set(roles.values())) == 4
        assert type(binding) is str and len(binding) == 64
        self.config = dict(arm=arm, lr=lr, horizon=horizon, roles=copy.deepcopy(roles),
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
        self.schedule = dict(kind='linear_cooldown', total_steps=horizon,
                             min_ratio=.1, start_step=2*horizon//3)
        self.ema_start = math.ceil((2*horizon/3)/10)*10
        self.ema = {}
        self.snapshots = 0
        self.history = []

    @property
    def step(self): return self.optimizer._steps

    def update(self):
        assert self.step < self.config['horizon']
        ratio = multiplier(self.schedule, self.step)
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g['lr'] = base * ratio
        self.model.train()
        self.optimizer.zero_grad()
        x, y = self.data.batch()
        output = self.model.forward_train(x, y)
        loss = float(output.loss.detach())
        output.loss.backward()
        clip = clip_gradients((p for _, p in self.model.ordered_parameters()), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        with torch.no_grad():
            if self.step == self.ema_start:
                self.ema = {n: p.detach().clone() for n, p in self.model.ordered_parameters()}
                self.snapshots = 1
            elif self.step > self.ema_start and (self.step-self.ema_start) % 10 == 0:
                for n, p in self.model.ordered_parameters():
                    self.ema[n].lerp_(p.detach(), 1.0-2.0**(-10/200))
                self.snapshots += 1
        row = dict(step=self.step, loss=loss, lr_ratio=ratio, clip=clip,
                   batch_sha256=hashlib.sha256(x.cpu().numpy().tobytes()+y.cpu().numpy().tobytes()).hexdigest())
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
        assert type(step) is int and 0 <= step <= self.config['horizon']
        assert len(payload['history']) == step
        assert [r['step'] for r in payload['history']] == list(range(1, step+1))
        expected_snapshots = 0 if step < self.ema_start else 1+(step-self.ema_start)//10
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
                expected_lr = actual['lr'] * (multiplier(self.schedule, step-1) if step else 1.)
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
        self.history = copy.deepcopy(payload['history'])
        for n,g in self.data.view.generators.items(): g.set_state(payload['samplers'][n])
        torch.random.set_rng_state(payload['cpu_rng']); torch.mps.set_rng_state(payload['mps_rng'])
        self.optimizer.validate()


def score(model, data, document_ids=None):
    """Full validation or declared document probe; FP64 aggregation of FP32 CE."""
    prior_mode = model.training
    sampler_before = {n:g.get_state().clone() for n,g in data.view.generators.items()}
    versions = tuple(p._version for _,p in model.ordered_parameters())
    selected = set(range(len(data.indexes['val']))) if document_ids is None else set(document_ids)
    assert selected and all(type(i) is int and 0 <= i < len(data.indexes['val']) for i in selected)
    values = {i: [] for i in sorted(selected)}
    device = model.validate()
    try:
        model.eval()
        with torch.no_grad(), FP32RMSNormInference():
            for doc, source, x, y in data.validation_chunks():
                if doc not in selected: continue
                logits = model.primary_logits(x.to(device)).values[0]
                targets = y[0].to(device)
                losses = (cpu_cross_entropy(logits, targets) if device.type == 'cpu'
                          else F.cross_entropy(logits, targets, reduction='none'))
                assert bool(torch.isfinite(losses).all()) and bool((losses >= 0).all())
                values[doc].extend(losses.cpu().double().tolist())
    finally:
        model.train(prior_mode)
    assert versions == tuple(p._version for _,p in model.ordered_parameters())
    assert all(torch.equal(sampler_before[n], g.get_state()) for n,g in data.view.generators.items())
    rows = []
    for doc, losses in values.items():
        row = data.indexes['val'][doc]
        assert len(losses) == row['tokens']-1
        rows.append(dict(document=doc, source=row['source'], count=len(losses), loss_sum=math.fsum(losses)))
    count = sum(r['count'] for r in rows)
    sources = {}
    for source in sorted({r['source'] for r in rows}):
        group = [r for r in rows if r['source'] == source]
        n = sum(r['count'] for r in group)
        sources[source] = dict(count=n, ce=math.fsum(r['loss_sum'] for r in group)/n)
    return dict(count=count, ce=math.fsum(r['loss_sum'] for r in rows)/count, documents=rows,
                sources=sources, test_opened=False, policy='document_chunks128_all_targets_after_first_v1')
