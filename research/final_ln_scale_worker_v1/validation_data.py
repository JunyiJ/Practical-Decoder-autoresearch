"""Read-only train/validation view: tokenize only those two frozen raw splits.

The training runtime remains unchanged. This view has no test array, generator,
combined dataset, sampler or text suffix. It cannot be used to resume training.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import tiktoken
import torch
from research.block_norm_v1 import campaign_contract as c
from research.final_ln_scale_model_v1.runtime.data.loaders import _configure_splits
from research.final_ln_scale_model_v1.runtime.models.output_bias import validate_output_bias_training_data


def load(cfg,checkpoint=None):
    c.require(cfg.data.tokenizer=='gpt2' and cfg.model.vocab_size==50257,'Explicit GPT2 validation view required')
    path=Path(cfg.data.path);before=c.file_sha(path,root=c.ROOT,max_bytes=4*1024**2)
    text=path.read_text(encoding='utf-8');parts=SimpleNamespace()
    # Reuse exactly the frozen raw-boundary rounding; never encode its test tail.
    _configure_splits(parts,text,cfg.data.train_fraction,cfg.data.val_fraction,cfg.training.seed,cfg.training.eval_seed)
    enc=tiktoken.get_encoding('gpt2')
    ranks=[(token.hex(),rank) for token,rank in enc._mergeable_ranks.items()]
    tokenizer=dict(name='gpt2',implementation='tiktoken',version=importlib.metadata.version('tiktoken'),
        vocabulary_sha256=hashlib.sha256(json.dumps(sorted(ranks)).encode()).hexdigest(),pattern=enc._pat_str,
        special_tokens=dict(enc._special_tokens),special_token_policy='ordinary_text',split_unit='raw_text',vocab_size=enc.n_vocab)
    splits={name:torch.tensor(enc.encode(parts.splits[name],disallowed_special=()),dtype=torch.int64) for name in ('train','val')}
    view=SimpleNamespace(train_data=splits['train'],val_data=splits['val'],splits=splits,tokenizer_contract=tokenizer,
        vocab_size=enc.n_vocab,train_fraction=parts.train_fraction,val_fraction=parts.val_fraction,
        eval_seed=parts.eval_seed,batch_layout=cfg.training.batch_layout)
    if checkpoint is not None:
        saved=checkpoint.get('tokenizer_contract',checkpoint.get('training_state',{}).get('contract',{}).get('tokenizer'))
        c.require(c.canonical(saved)==c.canonical(tokenizer),'Checkpoint tokenizer differs from read-only view')
    validate_output_bias_training_data(cfg.model,view,str(path))
    c.require(c.file_sha(path,root=c.ROOT,max_bytes=4*1024**2)==before,'Corpus changed during read-only view construction')
    return view
