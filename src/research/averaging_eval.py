"""Strict derived-weight validation with the existing validation-only scoring panel."""
from pathlib import Path
import torch
from omegaconf import OmegaConf
from .coverage_eval import score_validation
from .protocol import file_digest
from ..models.gpt import GPT
from ..train.train import _make_loader, _training_contract, _contracts_match, _resolve_device
from ..train.weight_averaging import derived_checkpoint_payload
from ..utils.checkpoint import read_checkpoint, load_model_weights


def validate_derivation(job):
    path=Path(job['checkpoint']).resolve();parent=Path(job['parent_checkpoint']).resolve()
    if path==parent or file_digest(path)!=job['checkpoint_sha256'] or file_digest(parent)!=job['parent_sha256']:
        raise ValueError('Derived checkpoint or parent changed')
    expected=derived_checkpoint_payload(parent,job['half_life'])
    saved=read_checkpoint(path)
    if set(saved)!=set(expected):raise ValueError('Derived payload field set changed')
    for key in expected:
        if key=='model_state':
            if set(saved[key])!=set(expected[key]):raise ValueError('Derived tensor names changed')
            for name,want in expected[key].items():
                got=saved[key][name]
                if (not isinstance(got,torch.Tensor) or got.dtype!=want.dtype or got.shape!=want.shape
                        or not torch.equal(got,want)):
                    raise ValueError('Derived tensor differs from registered parent average')
        elif saved[key]!=expected[key]:raise ValueError('Derived metadata differs from registered parent')
    if saved['step']!=job['completed_steps'] or saved['parent_checkpoint']['sha256']!=job['parent_sha256']:
        raise ValueError('Derived step or lineage changed')
    if file_digest(parent)!=job['parent_sha256'] or file_digest(path)!=job['checkpoint_sha256']:
        raise ValueError('Evidence changed during derivation verification')
    return saved


def evaluate_averaged_job(job):
    if job['contexts']!=[128]:raise ValueError('Only registered common-context validation is supported')
    saved=validate_derivation(job);parent=read_checkpoint(job['parent_checkpoint'])
    cfg=OmegaConf.create(saved['cfg']);cfg.training.strict_device=True
    if file_digest(Path(cfg.data.path))!=job['data_sha256']:raise ValueError('Corpus changed')
    if cfg.data.train_fraction!=.8 or cfg.data.val_fraction!=.1:raise ValueError('Reserved data split changed')
    torch.set_num_threads(4);device=_resolve_device(cfg);loader=_make_loader(cfg,device)
    if not _contracts_match(parent['training_state']['contract'],_training_contract(cfg,device,loader)):
        raise ValueError('Raw parent training contract changed')
    model=GPT(cfg.model).to(device);load_model_weights(model,saved)
    result=score_validation(model,loader.val_data,128,8,2048)
    if file_digest(Path(job['checkpoint']))!=job['checkpoint_sha256'] or file_digest(Path(job['parent_checkpoint']))!=job['parent_sha256']:
        raise ValueError('Evidence changed during scoring')
    return dict(evaluations={'128':result},checkpoint_sha256=job['checkpoint_sha256'],
        parent_checkpoint_sha256=job['parent_sha256'],half_life=job['half_life'],
        holdout_consumed=False,device=device,torch=str(torch.__version__))
