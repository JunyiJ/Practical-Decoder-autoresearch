"""Prospective vocabulary offsets; pure metadata checks do not import torch.

The historical global character alphabet is retained. Only the training prefix
contributes counts. These are prefix frequencies, not sampler-weighted targets.
"""
from collections import Counter
from collections.abc import Mapping
import hashlib
import io
import json
import math

POLICY = 'mha4_train_bound_output_bias'
KINDS = ('none', 'learned_zero', 'learned_train_unigram')
FIELDS = ('output_bias_kind', 'output_bias_width_policy', 'output_bias_prior')
TOKENIZER_POLICY = 'mha4_tokenizer_no_output_bias'


def tokenizer_no_bias_prior(loader, data_path):
    """Compact binding for an explicitly absent bias with a subword vocabulary."""
    from pathlib import Path
    body = dict(version=2, tokenizer=dict(loader.tokenizer_contract),
                corpus_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                train_fraction=loader.train_fraction, val_fraction=loader.val_fraction,
                train_ids_sha256=digest(loader.train_data.tolist()))
    return dict(body, prior_sha256=digest(body))


def _get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def _has(cfg, key):
    return key in cfg if isinstance(cfg, Mapping) else hasattr(cfg, key)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def centered_log_prior(counts):
    if not counts or any(type(c) is not int or c < 0 for c in counts) or sum(counts) <= 0:
        raise ValueError('Prior requires nonnegative integer counts with positive total')
    total = sum(counts) + len(counts)
    values = [math.log((count + 1) / total) for count in counts]
    center = math.fsum(values) / len(values)
    return [v - center for v in values]


def build_training_prior(corpus_bytes, train_fraction, val_fraction):
    """Match TinyDataLoader UTF-8/universal-newline and split rounding exactly."""
    if not isinstance(corpus_bytes, bytes):
        raise ValueError('Corpus must be bytes for a stable raw-file hash')
    if any(type(x) not in (int, float) or not math.isfinite(x)
           for x in (train_fraction, val_fraction)):
        raise ValueError('Split fractions must be finite numbers')
    total = train_fraction + val_fraction
    if train_fraction <= 0 or val_fraction <= 0 or total > 1. + 1e-12:
        raise ValueError('Invalid split fractions')
    with io.TextIOWrapper(io.BytesIO(corpus_bytes), encoding='utf-8', newline=None) as stream:
        text = stream.read()
    alphabet = sorted(set(text))
    end = int(train_fraction * len(text))
    val_end = len(text) if math.isclose(total, 1., abs_tol=1e-12, rel_tol=0) else int(total * len(text))
    counts_by_character = Counter(text[:end])
    counts = [counts_by_character[c] for c in alphabet]
    initial = centered_log_prior(counts)
    body = dict(version=1, corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest(),
        tokenizer='TinyDataLoader.sorted_global_characters.universal_newlines.v1',
        alphabet=alphabet, token_count=len(text), train_fraction=float(train_fraction),
        val_fraction=float(val_fraction), train_end=end, val_end=val_end,
        counts=counts, counts_sha256=digest(counts), pseudocount=1,
        initial_values=initial, initial_values_sha256=digest(initial))
    return dict(body, prior_sha256=digest(body))


def validate_prior(prior, vocab_size):
    if not isinstance(prior, Mapping):
        raise ValueError('Output-bias prior must be a complete mapping')
    # Normalize OmegaConf lists/mappings to ordinary JSON-compatible containers.
    def plain(x):
        if isinstance(x, Mapping): return {k: plain(v) for k, v in x.items()}
        if isinstance(x, (str, int, float, bool)) or x is None: return x
        return [plain(v) for v in x]
    p = plain(prior)
    required = {'version', 'corpus_sha256', 'tokenizer', 'alphabet', 'token_count',
        'train_fraction', 'val_fraction', 'train_end', 'val_end', 'counts',
        'counts_sha256', 'pseudocount', 'initial_values', 'initial_values_sha256', 'prior_sha256'}
    if set(p) != required or p['version'] != 1 or type(p['version']) is not int:
        raise ValueError('Unknown or incomplete prior schema')
    if type(vocab_size) is not int or vocab_size < 1:
        raise ValueError('Prior vocabulary size must be a positive integer')
    alphabet = p['alphabet']
    if (len(alphabet) != vocab_size or any(type(c) is not str or len(c) != 1 for c in alphabet)
            or alphabet != sorted(set(alphabet))):
        raise ValueError('Prior character ordering differs')
    if p['tokenizer'] != 'TinyDataLoader.sorted_global_characters.universal_newlines.v1':
        raise ValueError('Unknown prior tokenizer')
    for key in ('corpus_sha256', 'counts_sha256', 'initial_values_sha256', 'prior_sha256'):
        value = p[key]
        if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Malformed prior digest')
    if any(type(p[k]) is not int or p[k] < 1 for k in ('token_count', 'train_end', 'val_end')):
        raise ValueError('Invalid prior boundaries')
    tr, va = p['train_fraction'], p['val_fraction']
    if (any(type(x) not in (int,float) or not math.isfinite(x) or x <= 0 for x in (tr, va))
            or tr + va > 1. + 1e-12):
        raise ValueError('Invalid prior split')
    expected_val = p['token_count'] if math.isclose(tr+va,1.,abs_tol=1e-12,rel_tol=0) else int((tr+va)*p['token_count'])
    if p['train_end'] != int(tr*p['token_count']) or p['val_end'] != expected_val:
        raise ValueError('Prior split rounding differs')
    if (len(p['counts']) != vocab_size or any(type(x) is not int or x < 0 for x in p['counts'])
            or sum(p['counts']) != p['train_end']):
        raise ValueError('Prior counts do not cover exactly the training prefix')
    expected = centered_log_prior(p['counts'])
    if type(p['pseudocount']) is not int or p['pseudocount'] != 1:
        raise ValueError('Prior smoothing changed')
    if (any(type(x) is not float or not math.isfinite(x) for x in p['initial_values'])
            or p['initial_values'] != expected or digest(expected) != p['initial_values_sha256']):
        raise ValueError('Prior initialization is not the declared centered log counts')
    if digest(p['counts']) != p['counts_sha256']:
        raise ValueError('Prior count hash differs')
    body = {k:v for k,v in p.items() if k != 'prior_sha256'}
    if digest(body) != p['prior_sha256']:
        raise ValueError('Prior lineage hash differs')
    return p


def output_bias_contract(cfg):
    if not any(_has(cfg,k) for k in FIELDS): return None
    from .tokenizer_architecture import BIAS_POLICY, architecture_contract
    tokenizer_policy = _get(cfg, 'output_bias_width_policy')
    if tokenizer_policy in (TOKENIZER_POLICY, BIAS_POLICY):
        from .readout_norm import readout_norm_contract
        backbone = architecture_contract(cfg) if tokenizer_policy == BIAS_POLICY else readout_norm_contract(cfg)
        if _get(cfg, 'output_bias_kind') != 'none' or backbone is None:
            raise ValueError('Tokenizer policy requires absent output bias and inherited readout')
        raw = _get(cfg, 'output_bias_prior')
        # OmegaConf containers must be normalized before hashing.
        from omegaconf import OmegaConf
        p = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else dict(raw)
        body = {k: v for k, v in p.items() if k != 'prior_sha256'}
        if (set(body) != {'version', 'tokenizer', 'corpus_sha256', 'train_fraction', 'val_fraction', 'train_ids_sha256'}
                or p['version'] != 2 or p['prior_sha256'] != digest(body)
                or p['tokenizer']['vocab_size'] != _get(cfg, 'vocab_size')
                or p['tokenizer']['name'] != 'gpt2' or p['tokenizer']['split_unit'] != 'raw_text'):
            raise ValueError('Invalid tokenizer no-bias binding')
        return dict(output_bias_kind='none', output_bias_width_policy=tokenizer_policy, output_bias_prior=p)
    if _get(cfg,'output_bias_kind') not in KINDS or _get(cfg,'output_bias_width_policy') != POLICY:
        raise ValueError('Output bias requires its explicit reviewed kind and policy')
    from .readout_norm import readout_norm_contract
    if readout_norm_contract(cfg) is None:
        raise ValueError('Output bias requires explicit inherited readout normalization')
    p = validate_prior(_get(cfg,'output_bias_prior'), _get(cfg,'vocab_size'))
    return dict(output_bias_kind=_get(cfg,'output_bias_kind'),
        output_bias_width_policy=POLICY, output_bias_prior=p)


def output_bias_signature(cfg):
    c = output_bias_contract(cfg)
    return None if c is None else (c['output_bias_kind'], c['output_bias_width_policy'], c['output_bias_prior']['prior_sha256'])


def build_output_bias(cfg):
    c = output_bias_contract(cfg)
    if c is None or c['output_bias_kind'] == 'none': return None
    import torch
    values = c['output_bias_prior']['initial_values'] if c['output_bias_kind'] == 'learned_train_unigram' else [0.] * _get(cfg,'vocab_size')
    return torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))


def validate_output_bias_load(model, checkpoint):
    cfg = getattr(model,'cfg',{})
    saved = _get(checkpoint.get('cfg',{}),'model',{})
    if output_bias_contract(cfg) != output_bias_contract(saved):
        raise ValueError('Checkpoint output-bias initialization or data semantics differ')
    if getattr(model,'output_bias_signature',None) != output_bias_signature(cfg):
        raise ValueError('Constructed output bias differs from mutable config')


def validate_output_bias_training_data(cfg, loader, data_path):
    """Check binding before any model construction; never sample a split."""
    c = output_bias_contract(cfg)
    if c is None: return
    from .tokenizer_architecture import BIAS_POLICY
    if c['output_bias_width_policy'] in (TOKENIZER_POLICY, BIAS_POLICY):
        if not hasattr(loader, 'tokenizer_contract') or tokenizer_no_bias_prior(loader, data_path) != c['output_bias_prior']:
            raise ValueError('Declared tokenizer prior differs from actual training corpus/split')
        return
    from pathlib import Path
    p = build_training_prior(Path(data_path).read_bytes(), loader.train_fraction, loader.val_fraction)
    if p != c['output_bias_prior']:
        raise ValueError('Declared output prior differs from actual training corpus/split')
    if [loader.itos[i] for i in range(loader.vocab_size)] != p['alphabet']:
        raise ValueError('Actual tokenizer ordering differs from output prior')
    counts = [0] * loader.vocab_size
    for token in loader.train_data.tolist(): counts[token] += 1
    if counts != p['counts']:
        raise ValueError('Actual training-token counts differ from output prior')
