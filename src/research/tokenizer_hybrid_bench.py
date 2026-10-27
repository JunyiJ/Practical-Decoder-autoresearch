"""Source-frozen hybrid timing with explicit layout labels and all recurrent/history bytes."""
import argparse
import gc
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..models.gpt import GPT
from ..utils.checkpoint import read_checkpoint, load_model_weights, checkpoint_loader
from .memory_guard import start_memory_guard
from .protocol import file_digest
from .runner import atomic_json
from .tokenizer_attention_bench import timed, check_cached
from .tokenizer_hybrid_study import LAYOUT


def benchmark(control, hybrid, output, control_family):
    if control_family not in ('gated_mla', 'kda'):
        raise ValueError('Unsupported tokenizer benchmark candidate')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('MPS fallback forbidden')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required')
    torch.set_num_threads(4);start_memory_guard()
    models={};configs={};metadata={};fixed=None
    for family,path in [(control_family,control),('hybrid',hybrid)]:
        c=read_checkpoint(path);cfg=OmegaConf.create(c['cfg'])
        if tuple(cfg.model.attention_layout) != LAYOUT[family] or cfg.model.dim != 64:
            raise ValueError('Checkpoint layout/width mismatch')
        loader=checkpoint_loader(cfg,c,cfg.data.path,'cpu')
        ids=loader.train_data[:96].clone()
        if fixed is not None and not torch.equal(fixed,ids):raise ValueError('Different timing tokens')
        fixed=ids
        model=GPT(cfg.model).to('mps').eval();load_model_weights(model,c)
        models[family]=model;configs[family]=cfg
        metadata[family]=dict(checkpoint=str(path.resolve()),sha256=file_digest(path),
                              width=cfg.model.dim,parameters=sum(p.numel() for p in model.parameters()),
                              parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()))
        del c,loader;gc.collect()
    if configs[control_family].model.dim!=configs['hybrid'].model.dim:raise ValueError('Different widths')
    rows=[];correctness=[]
    for batch in (1,8):
        ids=fixed.unsqueeze(0).repeat(batch,1).to('mps')
        for family in (control_family,'hybrid'):
            error=check_cached(models[family],configs[family],ids)
            correctness.append(dict(batch=batch,family=family,max_absolute_error=error))
            for _ in range(2):timed(models[family],configs[family],ids)
        for repeat in range(8):
            for family in ((control_family,'hybrid') if repeat%2==0 else ('hybrid',control_family)):
                rows.append(dict(batch=batch,repeat=repeat,family=family,**timed(models[family],configs[family],ids)))
    result=dict(status='completed',device='mps',dtype='float32',metadata=metadata,
                fixed_train_token_ids=fixed.tolist(),prompt_tokens=32,decode_tokens=64,
                warmups=2,repeats=8,correctness=correctness,records=rows,
                scope='Two models resident; model loading/cache allocation outside timing. Memory observations are not single-model deployment peaks. Teacher-forced fixed training text; no test access or quality selection.')
    atomic_json(output,result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--hybrid', type=Path, required=True)
    p.add_argument('--control-family', choices=['gated_mla', 'kda'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    benchmark(a.control, a.hybrid, a.output, a.control_family)


if __name__ == '__main__':
    main()
