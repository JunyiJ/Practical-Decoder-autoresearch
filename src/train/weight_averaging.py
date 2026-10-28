"""Optional spaced EMA of a training tail, without feedback into online weights.

The state belongs to the online checkpoint. Averaged weights are inference-only;
optimizer continuation must always resume the original online parameters.
"""
from collections.abc import Mapping
import copy
import math
import torch


def averaging_contract(training):
    raw = getattr(training, 'weight_averaging', None)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError('weight_averaging must be a mapping or null')
    spec = dict(raw)
    if set(spec) != {'kind', 'total_steps', 'start_step', 'interval', 'half_lives'}:
        raise ValueError('Unexpected weight averaging fields')
    if spec['kind'] != 'spaced_tail_ema':
        raise ValueError('Unsupported weight averaging kind')
    total, start, interval = (spec[k] for k in ('total_steps', 'start_step', 'interval'))
    if (type(total) is not int or total < 2 or total != training.max_iters
            or type(start) is not int or not 1 <= start < total
            or type(interval) is not int or interval < 1 or (total-start) % interval):
        raise ValueError('Averaging requires an aligned tail within its fixed horizon')
    half_lives = spec['half_lives']
    if not isinstance(half_lives, (list, tuple)):
        # OmegaConf ListConfig implements Sequence but not list/tuple.
        from omegaconf import ListConfig
        if not isinstance(half_lives, ListConfig):
            raise ValueError('half_lives must be an ordered list')
    half_lives = list(half_lives)
    if (not 1 <= len(half_lives) <= 3
            or any(type(h) is not int or h < 1 for h in half_lives)
            or half_lives != sorted(set(half_lives))):
        raise ValueError('Declare one to three unique increasing positive half-lives')
    betas = [2.0 ** (-interval/h) for h in half_lives]
    if any(not math.isfinite(beta) or not 0 < beta < 1 for beta in betas):
        raise ValueError('Averaging half-life is numerically degenerate')
    spec['half_lives'] = half_lives
    return spec


class TailWeightAverages:
    """Update after each completed optimizer step; collect only at fixed snapshots."""
    def __init__(self, model, training):
        self.spec = averaging_contract(training)
        self.model = model
        self.completed_steps = 0
        self.snapshots = 0
        self.averages = {}
        self.parameters = tuple(model.named_parameters()) if self.spec else ()
        if self.spec and (not self.parameters or any(p.dtype != torch.float32 for _, p in self.parameters)):
            raise ValueError('Tail averaging currently requires nonempty float32 parameters')
        self.identities = tuple((name, id(p), tuple(p.shape), p.dtype, p.device) for name, p in self.parameters)
        self.betas = {str(h): 2.0 ** (-self.spec['interval']/h) for h in self.spec['half_lives']} if self.spec else {}

    def _check_parameters(self):
        current = tuple((name, id(p), tuple(p.shape), p.dtype, p.device) for name, p in self.model.named_parameters())
        if current != self.identities:
            raise ValueError('Online parameter identity, order, shape, dtype or device changed')

    @torch.no_grad()
    def update(self, completed_steps):
        if self.spec is None:
            return
        if type(completed_steps) is not int or completed_steps != self.completed_steps + 1 or completed_steps > self.spec['total_steps']:
            raise ValueError('Averaging updates must follow every optimizer step exactly once')
        self._check_parameters()
        start = self.spec['start_step']
        if completed_steps == start:
            self.averages = {str(h): {name: p.detach().clone() for name, p in self.parameters}
                             for h in self.spec['half_lives']}
            self.snapshots = 1
        elif completed_steps > start and (completed_steps-start) % self.spec['interval'] == 0:
            for h, beta in self.betas.items():
                for name, parameter in self.parameters:
                    self.averages[h][name].lerp_(parameter.detach(), 1.0-beta)
            self.snapshots += 1
        self.completed_steps = completed_steps

    def state_dict(self):
        if self.spec is None:
            return None
        self._check_parameters()
        return {'version': 1, 'spec': copy.deepcopy(self.spec),
                'completed_steps': self.completed_steps, 'snapshots': self.snapshots,
                'parameter_names': [name for name, _ in self.parameters],
                'averages': {h: {name: value.detach().clone() for name, value in panel.items()}
                             for h, panel in self.averages.items()}}

    @torch.no_grad()
    def load_state_dict(self, state, completed_steps):
        if self.spec is None:
            if state is not None:
                raise ValueError('Disabled averaging cannot consume averaging state')
            return
        self._check_parameters()
        keys = {'version', 'spec', 'completed_steps', 'snapshots', 'parameter_names', 'averages'}
        if (not isinstance(state, dict) or set(state) != keys or type(state['version']) is not int
                or state['version'] != 1 or state['spec'] != self.spec
                or type(completed_steps) is not int or not 0 <= completed_steps <= self.spec['total_steps']
                or type(state['completed_steps']) is not int or state['completed_steps'] != completed_steps
                or state['parameter_names'] != [name for name, _ in self.parameters]):
            raise ValueError('Averaging resume contract differs from online checkpoint')
        expected = 0 if completed_steps < self.spec['start_step'] else 1 + (completed_steps-self.spec['start_step']) // self.spec['interval']
        panels = state['averages']
        if (type(state['snapshots']) is not int or state['snapshots'] != expected
                or not isinstance(panels, dict) or set(panels) != (set(self.betas) if expected else set())):
            raise ValueError('Averaging snapshot counters or half-lives changed')
        loaded = {}
        for h, panel in panels.items():
            if not isinstance(panel, dict) or set(panel) != {name for name, _ in self.parameters}:
                raise ValueError('Incomplete averaged parameter set')
            loaded[h] = {}
            for name, parameter in self.parameters:
                value = panel[name]
                if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                        or value.dtype != parameter.dtype or not torch.isfinite(value).all()):
                    raise ValueError('Invalid averaged tensor shape, dtype or finite state')
                loaded[h][name] = value.detach().to(parameter.device).clone()
        self.averages = loaded
        self.completed_steps = completed_steps
        self.snapshots = expected

    @torch.no_grad()
    def averaged_parameters(self, half_life):
        if (self.spec is None or type(half_life) is not int or half_life not in self.spec['half_lives']
                or self.completed_steps != self.spec['total_steps']):
            raise ValueError('Only a registered final-horizon average can be exported')
        self._check_parameters()
        panel = self.averages[str(half_life)]
        if any(not torch.isfinite(v).all() for v in panel.values()):
            raise ValueError('Cannot export nonfinite averaged weights')
        return {name: value.detach().clone() for name, value in panel.items()}


def derived_checkpoint_payload(parent_path, half_life):
    """Derive a new inference-only checkpoint; never rewrite the raw parent.

    The online optimizer belongs to the raw trajectory and is deliberately absent
    from the derived payload. Nonlearned persistent buffers are copied unchanged.
    """
    import hashlib
    from pathlib import Path
    from omegaconf import OmegaConf
    from ..models.gpt import GPT
    from ..utils.checkpoint import read_checkpoint, load_model_weights

    parent = Path(parent_path).resolve()
    def sha(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
        return h.hexdigest()
    parent_sha = sha(parent)
    raw = read_checkpoint(parent, map_location='cpu')
    if ('averaged_weights' in raw or 'optimizer_state' not in raw
            or raw.get('parameter_layout_version') != 2):
        raise ValueError('Averaging export requires a raw current-layout online checkpoint')
    cfg = OmegaConf.create(raw['cfg'])
    with torch.random.fork_rng(devices=[]):
        model = GPT(cfg.model)
        load_model_weights(model, raw, for_resume=True)
        tracker = TailWeightAverages(model, cfg.training)
        if tracker.spec is None or raw['training_state']['contract'].get('weight_averaging') != tracker.spec:
            raise ValueError('Parent averaging contract is absent or inconsistent')
        tracker.load_state_dict(raw['training_state'].get('weight_averaging_state'), raw['step'])
        parameters = tracker.averaged_parameters(half_life)
    if any(not torch.isfinite(value).all() for value in raw['model_state'].values()):
        raise ValueError('Parent model contains nonfinite tensors')
    model_state = {name: value.detach().clone() for name, value in raw['model_state'].items()}
    model_state.update(parameters)
    payload = dict(model_state=model_state, model_math_version=2, parameter_layout_version=2,
                   cfg=copy.deepcopy(raw['cfg']), step=raw['step'],
                   averaged_weights=dict(spec=tracker.spec, half_life=half_life,
                       snapshots=tracker.snapshots, completed_steps=tracker.completed_steps,
                       parameter_names=[name for name, _ in tracker.parameters],
                       purpose='inference_only_not_optimizer_resume'),
                   parent_checkpoint=dict(path=str(parent), sha256=parent_sha))
    tokenizer = raw['training_state']['contract'].get('tokenizer')
    if tokenizer is not None:
        payload['tokenizer_contract'] = copy.deepcopy(tokenizer)
    if sha(parent) != parent_sha:
        raise ValueError('Raw parent changed during averaging export')
    return payload


def export_averaged_checkpoint(parent_path, output_path, half_life, *, payload_writer=None):
    """Publish a new inference-only derivation, refusing any existing output."""
    import hashlib
    import os
    from pathlib import Path
    if payload_writer is not None and not callable(payload_writer):
        raise TypeError('Averaging payload writer must be callable')
    parent, output = Path(parent_path).resolve(), Path(output_path).resolve()
    if parent == output or output.exists():
        raise ValueError('Averaged checkpoint must have a new output path')
    payload = derived_checkpoint_payload(parent, half_life)
    if payload_writer is None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + '.tmp')
        with temporary.open('xb') as stream:
            torch.save(payload, stream)
        os.link(temporary, output)
        temporary.unlink()
        output_sha=hashlib.sha256(output.read_bytes()).hexdigest()
    else:
        payload_writer(output,payload)
        # The explicit bounded producer must not be followed by a whole-file
        # read_bytes allocation merely to hash its completed archive.
        hasher=hashlib.sha256()
        with output.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024**2),b''):hasher.update(chunk)
        output_sha=hasher.hexdigest()
    return dict(checkpoint=str(output), checkpoint_sha256=output_sha,
                parent_checkpoint=payload['parent_checkpoint'], averaged_weights=payload['averaged_weights'])
