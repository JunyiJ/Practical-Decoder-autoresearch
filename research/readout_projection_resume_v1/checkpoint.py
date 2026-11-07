"""Complete own RAW preflight before mutation; H200 is inference only."""
import copy
import hashlib
import math
import os
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT, require, plain, canonical, fingerprint
from research.readout_projection_model_v1.model import GPT, load_weights
from research.readout_projection_model_v1.contract import projection_seed
from research.block_norm_v1.runtime.models.rope import precompute_freqs_cis
from research.block_norm_v1.runtime.train.lr_schedule import schedule_contract, multiplier
from research.block_norm_v1.runtime.train.weight_averaging import averaging_contract
from research.embedding_input_dropout_resume_guard_v1.checkpoint import cpu_state
from .contract import sources, runtime
from .data import architecture, View

RAW = 'raw_native_readout_projection_v1'
H200 = 'h200_native_readout_projection_v1'
FILE_CAPS = {RAW: 512*1024**2, H200: 128*1024**2}


def _path(path):
    p = Path(path)
    require(p.is_absolute() and p == p.resolve() and p.is_relative_to(ROOT/'research_runs')
            and p.parent.is_dir() and not any(a.is_symlink() for a in (p, *p.parents)),
            'Canonical own research-run checkpoint path required')
    return p


def publish_payload(path, payload, *, cap_bytes=None):
    """Caller owns worker lock/admission; stream one exclusive bounded CPU archive."""
    require(type(payload) is dict and type(payload.get('kind')) is str and payload['kind'] in FILE_CAPS,
            'Own RAW/H200 payload kind required')
    p = _path(path); temporary = p.with_suffix(p.suffix+'.tmp')
    cap = FILE_CAPS[payload['kind']] if cap_bytes is None else cap_bytes
    require(type(cap) is int and 0 < cap <= FILE_CAPS[payload['kind']], 'Only an equal or stricter archive byte cap required')
    require(not p.exists() and not temporary.exists(), 'One new checkpoint output only')
    class Bounded:
        def __init__(self, stream):
            self.stream = stream
        def write(self, data):
            require(self.stream.tell()+len(data) <= cap, 'Checkpoint serialization exceeds complete byte cap')
            return self.stream.write(data)
        def flush(self):
            return self.stream.flush()
        def tell(self):
            return self.stream.tell()
    with temporary.open('xb') as stream:
        torch.save(payload, Bounded(stream)); stream.flush(); os.fsync(stream.fileno())
    # No overwrite. The temporary and published names briefly share one allocation.
    os.link(temporary, p); temporary.unlink()
    fd = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    require(p.stat().st_nlink == 1 and 0 < p.stat().st_size <= cap, 'Single-link complete archive required')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            h.update(chunk)
    return dict(kind=payload['kind'], path=str(p), sha256=h.hexdigest(), bytes=p.stat().st_size)


def read_payload(reference):
    """Hash-bound own CPU archive; full destination preflight is still mandatory."""
    require(type(reference) is dict and set(reference) == {'kind', 'path', 'sha256', 'bytes'}
            and type(reference['kind']) is str and reference['kind'] in FILE_CAPS
            and type(reference['path']) is str
            and type(reference['sha256']) is str and len(reference['sha256']) == 64
            and all(c in '0123456789abcdef' for c in reference['sha256'])
            and type(reference['bytes']) is int and 0 < reference['bytes'] <= FILE_CAPS[reference['kind']],
            'Exact bounded own archive reference required')
    p = _path(reference['path'])
    def digest():
        require(p.is_file() and p.stat().st_nlink == 1 and p.stat().st_size == reference['bytes'], 'Retained archive path/size/link differs')
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024**2), b''):
                h.update(chunk)
        return h.hexdigest()
    require(digest() == reference['sha256'], 'Retained archive hash differs before read')
    payload = torch.load(p, map_location='cpu', weights_only=True)
    require(type(payload) is dict and payload.get('kind') == reference['kind'], 'Archive payload kind differs')
    require(digest() == reference['sha256'], 'Retained archive changed during read')
    return payload


def cpu_tree(value):
    if type(value) is torch.Tensor:
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def normalized(cfg):
    value = plain(cfg); architecture(cfg)
    require(type(value.get('training')) is dict, 'Complete copied training configuration required')
    for name in ('checkpoint_dir', 'metrics_file', 'summary_file', 'resume_from'):
        value['training'].pop(name, None)
    return value


def initializer(model, cfg):
    expected = dict(backbone=cfg.initializer_seed, private_projection=projection_seed(cfg.initializer_seed))
    require(type(model) is GPT and model.initializer_seed == expected['backbone']
            and model.private_projection_seed == expected['private_projection'], 'Own initializer identity differs')
    return expected


def check_view(view):
    require(type(view) is View and set(view.splits) == set(view.generators) == {'train', 'val'}
            and view.batch_size == 8 and view.block_size == 128 and view.batch_layout == 'paired_blocks512'
            and view.vocab_size == 50257 and type(view.data_identity) is dict and bool(view.data_identity), 'Own bounded private sampler/data view required')
    for n, gen in view.generators.items():
        require(type(gen) is torch.Generator and gen.device.type == 'cpu', 'Private CPU sampler generator required')
        cpu_state(gen.get_state())
    return copy.deepcopy(view.data_identity)


def identity(cfg, view):
    return dict(config=normalized(cfg), data=check_view(view), environment=runtime(),
                source_fingerprint=fingerprint(sources()), schedule=schedule_contract(cfg.training), ema=averaging_contract(cfg.training))


def tensor(value, wanted, label, *, nonnegative=False):
    require(type(value) is torch.Tensor and value.device.type == 'cpu' and value.layout == torch.strided
            and value.is_contiguous() and value.dtype == wanted.dtype and value.shape == wanted.shape
            and bool(torch.isfinite(value).all()), 'Finite exact CPU checkpoint tensor required: '+label)
    if nonnegative:
        require(bool((value >= 0).all()), 'Negative second moment: '+label)


def model_state(model, saved):
    current = model.state_dict()
    require(type(saved) is dict and list(saved) == list(current), 'Exact ordered persisted state inventory required')
    cos, sin = precompute_freqs_cis(32, 128)
    for n, value in saved.items():
        tensor(value, current[n], n)
        if n.endswith(('freq_cos', 'freq_sin')):
            require(torch.equal(value, cos if n.endswith('freq_cos') else sin), 'Persisted RoPE differs')
    expected_mask = torch.tril(torch.ones(128, 128, dtype=torch.bool))
    require(not any('causal_mask' in n for n in saved) and all(torch.equal(b.attn.causal_mask.cpu(), expected_mask)
            for b in model.backbone.transformer.h), 'Reconstructible nonpersistent causal mask differs')


def history(model, rows, step, cfg):
    require(type(step) is int and 0 <= step <= cfg.training.max_iters and type(rows) is list and len(rows) == step, 'Complete bounded clipping history required')
    names = [n for n, _ in model.named_parameters()]
    for i, row in enumerate(rows, 1):
        require(type(row) is dict and set(row) == {'step', 'threshold', 'pre_norm', 'post_norm', 'coefficient', 'gradient_tensors', 'active_parameter_names', 'ce_loss'}
                and type(row['step']) is int and row['step'] == i and type(row['threshold']) is float and row['threshold'] == 1., 'Exact clipping fields required')
        require(all(type(row[k]) is float and math.isfinite(row[k]) for k in ('pre_norm', 'post_norm', 'coefficient', 'ce_loss'))
                and row['pre_norm'] >= 0 and row['post_norm'] >= 0 and row['ce_loss'] >= 0, 'Finite clipping/loss required')
        require(row['active_parameter_names'] == names and type(row['gradient_tensors']) is int and row['gradient_tensors'] == len(names), 'All learned readout parameters must be exposed')
        c = min(1., 1/(row['pre_norm']+1e-6))
        require(0 <= row['coefficient'] <= 1 and row['post_norm'] <= 1.+1e-5 and abs(c-row['coefficient']) < 2e-5
                and abs(row['post_norm']-row['pre_norm']*row['coefficient']) <= max(1e-5, row['post_norm']*2e-5), 'Clipping arithmetic differs')
    return {n: step for n in names}


def optimizer_state(model, opt, saved, cfg, step, counts):
    require(opt.owner() is model and opt.architecture == model.signature, 'Live own optimizer/model identity required')
    require(type(saved) is dict and set(saved) == {'version', 'independent_decay', 'optimizers', 'readout_projection_signature', 'completed_steps', 'gradient_steps'}
            and type(saved['version']) is int and saved['version'] == 1 and type(saved['independent_decay']) is float and saved['independent_decay'] == .00006
            and saved['readout_projection_signature'] == model.signature and type(saved['completed_steps']) is int and saved['completed_steps'] == step
            and type(saved['gradient_steps']) is dict and saved['gradient_steps'] == counts and all(type(v) is int for v in saved['gradient_steps'].values())
            and type(saved['optimizers']) is list and len(saved['optimizers']) == 2, 'Complete own optimizer/exposure identity required')
    ratio = multiplier(schedule_contract(cfg.training), step-1) if step else 1.
    for child, stored in zip(opt.optimizers, saved['optimizers']):
        expected = child.state_dict()['param_groups']
        require(type(stored) is dict and set(stored) == {'state', 'param_groups'} and type(stored['state']) is dict
                and type(stored['param_groups']) is list and len(stored['param_groups']) == len(expected), 'Exact child optimizer structure required')
        slots = set()
        for actual, wanted, live in zip(stored['param_groups'], expected, child.param_groups):
            require(type(actual) is dict and {k: v for k, v in actual.items() if k != 'lr'} == {k: v for k, v in wanted.items() if k != 'lr'}, 'Ordered group identity differs')
            require(type(actual['lr']) is float and math.isfinite(actual['lr']) and actual['lr'] == cfg.training.learning_rate*live['width_lr_scale']*ratio, 'Scheduled group LR differs')
            index = actual['params'][0]; p = live['params'][0]
            require(type(index) is int and index not in slots, 'Unique exact optimizer slot required')
            if not step:
                continue
            slots.add(index); state = stored['state'].get(index)
            keys = {'momentum_buffer'} if type(child) is torch.optim.Muon else {'step', 'exp_avg', 'exp_avg_sq'}
            require(type(state) is dict and set(state) == keys, 'Complete active parameter optimizer state required')
            for key, value in state.items():
                if key == 'step':
                    require(type(value) is torch.Tensor and value.device.type == 'cpu' and value.dtype == torch.float32
                            and value.ndim == 0 and bool(torch.isfinite(value)) and float(value) == step, 'Actual Adam parameter step differs')
                else:
                    tensor(value, p, key, nonnegative=key == 'exp_avg_sq')
        require(set(stored['state']) == slots and all(type(k) is int for k in stored['state']), 'Only exact updated optimizer slots may exist')


def ema_state(model, saved, cfg, step):
    spec = averaging_contract(cfg.training); names = dict(model.named_parameters())
    count = 0 if step < spec['start_step'] else 1+(step-spec['start_step'])//spec['interval']
    require(type(saved) is dict and set(saved) == {'version', 'spec', 'completed_steps', 'snapshots', 'parameter_names', 'averages'}
            and type(saved['version']) is int and saved['version'] == 1 and saved['spec'] == spec
            and type(saved['completed_steps']) is int and saved['completed_steps'] == step and type(saved['snapshots']) is int
            and saved['snapshots'] == count and saved['parameter_names'] == list(names)
            and type(saved['averages']) is dict and set(saved['averages']) == ({'200'} if count else set()), 'Exact learned-parameter H200 state required')
    for panel in saved['averages'].values():
        require(type(panel) is dict and list(panel) == list(names), 'Complete ordered learned-parameter-only EMA required')
        for n, value in panel.items():
            tensor(value, names[n], 'EMA:'+n)


def rng_states(payload, view, cfg):
    cpu_state(payload['cpu_rng'])
    require(type(payload['sampler_state']) is dict and set(payload['sampler_state']) == {'train', 'val'}, 'Exact private sampler state inventory required')
    for n, value in payload['sampler_state'].items():
        cpu_state(value); require(value.shape == view.generators[n].get_state().shape, 'Private sampler encoding differs')
        decoded = torch.Generator(device='cpu').set_state(value)
        require(decoded.initial_seed() == (cfg.training.seed if n == 'train' else cfg.training.eval_seed), 'Private sampler seed identity differs')
    live = torch.mps.get_rng_state().clone(); value = payload['mps_rng']
    require(type(value) is torch.Tensor and value.dtype == torch.uint8 and value.device.type == 'cpu'
            and value.is_contiguous() and value.shape == live.shape, 'Exact MPS RNG layout required')
    try:
        torch.mps.set_rng_state(value)
    except RuntimeError as exc:
        raise ValueError('Invalid MPS RNG encoding') from exc
    finally:
        torch.mps.set_rng_state(live)
    require(torch.equal(torch.mps.get_rng_state(), live), 'MPS decoder preflight changed caller stream')


def raw_payload(model, opt, ema, cfg, step, rows, view):
    require(ema.model is model and ema.completed_steps == step and opt.completed_steps == step, 'Live model/optimizer/EMA steps differ')
    payload = dict(kind=RAW, architecture=copy.deepcopy(model.signature), initializer=initializer(model, cfg), cfg=plain(cfg), step=step,
        model_state=cpu_tree(model.state_dict()), optimizer_state=cpu_tree(opt.state_dict()), ema=cpu_tree(ema.state_dict()),
        training_identity=identity(cfg, view), clipping_history=copy.deepcopy(rows),
        sampler_state={n: g.get_state().clone() for n, g in view.generators.items()}, cpu_rng=torch.get_rng_state().clone(), mps_rng=torch.mps.get_rng_state().clone())
    validate_resume(model, opt, ema, payload, cfg, view)
    return payload


def validate_resume(model, opt, ema, payload, cfg, view):
    model.validate(); require(ema.model is model, 'Own live EMA owner required')
    require(type(payload) is dict and set(payload) == {'kind', 'architecture', 'initializer', 'cfg', 'step', 'model_state', 'optimizer_state', 'ema', 'training_identity', 'clipping_history', 'sampler_state', 'cpu_rng', 'mps_rng'}
            and payload['kind'] == RAW and payload['architecture'] == model.signature == architecture(cfg)
            and payload['initializer'] == initializer(model, cfg) and canonical(normalized(payload['cfg'])) == canonical(normalized(cfg)), 'Only own complete RAW identity can resume')
    require(payload['training_identity'] == identity(cfg, view), 'Complete source/runtime/recipe/data identity differs')
    step = payload['step']; counts = history(model, payload['clipping_history'], step, cfg)
    model_state(model, payload['model_state']); optimizer_state(model, opt, payload['optimizer_state'], cfg, step, counts)
    ema_state(model, payload['ema'], cfg, step); rng_states(payload, view, cfg)
    return step


def resume(model, opt, ema, payload, cfg, view):
    step = validate_resume(model, opt, ema, payload, cfg, view)
    load_weights(model, dict(kind='native_readout_projection_weights_v1', architecture=model.signature, model_state=payload['model_state']))
    opt.load_state_dict(payload['optimizer_state']); ema.load_state_dict(payload['ema'], step)
    for n, value in payload['sampler_state'].items():
        view.generators[n].set_state(value)
    torch.set_rng_state(payload['cpu_rng']); torch.mps.set_rng_state(payload['mps_rng'])
    return step, copy.deepcopy(payload['clipping_history'])


def parent_identity(parent):
    require(type(parent) is dict and parent.get('kind') == RAW, 'Explicit own RAW ancestry required')
    require(set(parent) == {'kind', 'path', 'sha256', 'bytes'} and type(parent['path']) is str and type(parent['sha256']) is str
            and len(parent['sha256']) == 64 and all(c in '0123456789abcdef' for c in parent['sha256'])
            and type(parent['bytes']) is int, 'Exact retained own RAW reference required')
    p = _path(parent['path'])
    require(p.is_file() and p.stat().st_nlink == 1 and p.stat().st_size == parent['bytes']
            and 0 < parent['bytes'] <= FILE_CAPS[RAW], 'Actual retained RAW path/size required')
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    require(h.hexdigest() == parent['sha256'], 'Retained RAW parent hash differs')


def h200_payload(model, ema, cfg, parent):
    model.validate(); initializer(model, cfg); parent_identity(parent)
    require(ema.model is model and ema.completed_steps == cfg.training.max_iters, 'Final-horizon own EMA required')
    state = cpu_tree(model.state_dict()); state.update(cpu_tree(ema.averaged_parameters(200)))
    return dict(kind=H200, architecture=copy.deepcopy(model.signature), initializer=initializer(model, cfg), cfg=plain(cfg),
        step=ema.completed_steps, model_state=state, parent=copy.deepcopy(parent),
        averaged_weights=dict(spec=ema.spec, snapshots=ema.snapshots, parameter_names=[n for n, _ in model.named_parameters()],
                              half_life=200, purpose='inference_only_not_optimizer_resume'))


def load_export(model, payload, cfg):
    model.validate()
    require(type(payload) is dict and set(payload) == {'kind', 'architecture', 'initializer', 'cfg', 'step', 'model_state', 'parent', 'averaged_weights'}
            and payload['kind'] == H200 and payload['architecture'] == model.signature == architecture(cfg)
            and payload['initializer'] == initializer(model, cfg) and canonical(normalized(payload['cfg'])) == canonical(normalized(cfg))
            and type(payload['step']) is int and payload['step'] == cfg.training.max_iters, 'Only own final H200 inference export required')
    parent_identity(payload['parent']); spec = averaging_contract(cfg.training)
    expected = dict(spec=spec, snapshots=1+(cfg.training.max_iters-spec['start_step'])//spec['interval'],
        parameter_names=[n for n, _ in model.named_parameters()], half_life=200, purpose='inference_only_not_optimizer_resume')
    require(payload['averaged_weights'] == expected, 'Complete H200 metadata differs')
    model_state(model, payload['model_state'])
    load_weights(model, dict(kind='native_readout_projection_weights_v1', architecture=model.signature, model_state=payload['model_state']))
