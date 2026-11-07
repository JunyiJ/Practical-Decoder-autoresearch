"""Own configuration and native train/validation view; no holdout encoding."""
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import tiktoken
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT, require, plain, canonical, config as native_config
from research.relu_squared_model_v1.contract import model_config, signature
from research.block_norm_v1.runtime.data.loaders import _get_batch
from research.final_ln_scale_model_v1.runtime.models.output_bias import digest

DATA = ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality002-plan.json'


def config(arm, steps, initializer_seed, train_seed, eval_seed, lr=.00125, *, global_cpu_seed, global_mps_seed):
    require(all(type(s) is int and 0 <= s < 2**63 for s in
                (initializer_seed, train_seed, eval_seed, global_cpu_seed, global_mps_seed)),
            'Explicit audited initializer/sampler/global roles required')
    cfg = plain(native_config('incumbent', steps, train_seed, lr))
    cfg.pop('model'); cfg.update(model_config(arm)); cfg['initializer_seed'] = initializer_seed
    cfg['rng_identity'] = dict(global_cpu=global_cpu_seed, global_mps=global_mps_seed)
    cfg['training'].update(eval_seed=eval_seed, checkpoint_dir=str(ROOT/'research_runs/UNLAUNCHED-relu-squared-fixture/checkpoints'),
        metrics_file=str(ROOT/'research_runs/UNLAUNCHED-relu-squared-fixture/metrics.jsonl'), summary_file=str(ROOT/'research_runs/UNLAUNCHED-relu-squared-fixture/summary.json'))
    return OmegaConf.create(cfg)


def model_spec(cfg):
    value = plain(cfg)
    keys = ('policy', 'native_template', 'ffn', 'readout', 'optimizer', 'cache_identity')
    require(type(value) is dict and not any(k in value for k in
                ('model', 'gate_model', 'moe_model', 'readout_model', 'readout_projection', 'backbone_template'))
            and all(k in value for k in (*keys, 'initializer_seed', 'rng_identity'))
            and type(value['native_template']) is dict and type(value['ffn']) is dict and type(value['readout']) is dict,
            'Own FFN config cannot masquerade as another model')
    spec = {k: value[k] for k in keys}
    signature(spec)
    require(type(value['initializer_seed']) is int and 0 <= value['initializer_seed'] < 2**63, 'Bound initializer identity required')
    require(type(value['rng_identity']) is dict and set(value['rng_identity']) == {'global_cpu', 'global_mps'}
            and all(type(s) is int and 0 <= s < 2**63 for s in value['rng_identity'].values()),
            'Own explicit global RNG role identities required')
    require(type(value.get('training')) is dict
            and all(type(value['training'].get(k)) is int and 0 <= value['training'][k] < 2**63
                    for k in ('seed', 'eval_seed')), 'Own explicit train/eval sampler roles required')
    return spec


def architecture(cfg):
    return signature(model_spec(cfg))


def identity(cfg):
    architecture(cfg)
    require(cfg.data.tokenizer == 'gpt2' and cfg.data.train_fraction == .8 and cfg.data.val_fraction == .1
            and cfg.training.batch_size == 8 and cfg.training.batch_layout == 'paired_blocks512', 'Native codec/splits/paired sampling required')
    expected = json.loads(DATA.read_text())['data']
    require(hashlib.sha256(Path(cfg.data.path).read_bytes()).hexdigest() == expected['corpus_sha256'], 'Original corpus identity required')
    return copy.deepcopy(expected)


class View(SimpleNamespace):
    def get_batch(self, split='train'):
        require(split in ('train', 'val'), 'Only private train/validation streams exist')
        return _get_batch(self, split)


def load(cfg, device='cpu'):
    expected = identity(cfg)
    require(device in ('cpu', 'mps'), 'CPU/MPS native view required')
    text = Path(cfg.data.path).read_bytes().decode('utf8'); a = int(.8*len(text)); b = int((.8+.1)*len(text))
    require((len(text), a, b) == (expected['raw_characters'], expected['raw_train_end'], expected['raw_val_end']), 'Original raw splits required')
    enc = tiktoken.get_encoding('gpt2')
    tokenizer = dict(name='gpt2', implementation='tiktoken', version=importlib.metadata.version('tiktoken'),
        vocabulary_sha256=hashlib.sha256(json.dumps(sorted((t.hex(), r) for t, r in enc._mergeable_ranks.items())).encode()).hexdigest(),
        pattern=enc._pat_str, special_tokens=dict(enc._special_tokens), special_token_policy='ordinary_text', split_unit='raw_text', vocab_size=enc.n_vocab)
    require(tokenizer == expected['tokenizer'], 'Exact original codec/library contract required')
    ids = {n: enc.encode(s, disallowed_special=()) for n, s in [('train', text[:a]), ('val', text[a:b])]}
    require(len(ids['train']) == 267688 and len(ids['val']) == 34279 and digest(ids['train']) == expected['train_ids_sha256']
            and digest(ids['val']) == expected['validation_ids_sha256'] and digest(ids['val'][1:]) == expected['validation_target_ids_sha256'], 'Original physical native IDs required')
    require(hashlib.sha256(Path(cfg.data.path).read_bytes()).hexdigest() == expected['corpus_sha256'], 'Corpus changed during load')
    tensors = {n: torch.tensor(v, dtype=torch.int64) for n, v in ids.items()}
    return View(train_data=tensors['train'], val_data=tensors['val'], splits=tensors,
        generators={n: torch.Generator(device='cpu').manual_seed(s) for n, s in [('train', cfg.training.seed), ('val', cfg.training.eval_seed)]},
        batch_size=8, block_size=128, batch_layout='paired_blocks512', vocab_size=50257, device=device, data_identity=expected, tokenizer_contract=tokenizer)
