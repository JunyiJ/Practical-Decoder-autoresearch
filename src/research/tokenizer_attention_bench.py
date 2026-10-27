"""Paired cached MPS timing for two reviewed tokenizer checkpoints."""
import argparse
import gc
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf

from ..attention.cache import KVCache
from ..models.gpt import GPT
from ..utils.checkpoint import read_checkpoint, load_model_weights, checkpoint_loader
from .memory_guard import start_memory_guard
from .protocol import file_digest
from .runner import atomic_json


def buffer_bytes(caches):
    return sum(b.numel()*b.element_size() for c in caches for b in c.buffers())


@torch.no_grad()
def check_cached(model, cfg, ids):
    ids=ids[:,:12]
    full=model(ids)[0]
    caches=[KVCache(cfg,ids.device,batch_size=len(ids),dtype=torch.float32) for _ in range(4)]
    chunks=[model(ids[:,lo:hi],cache=caches,start_pos=lo)[0] for lo,hi in ((0,3),(3,7),(7,12))]
    torch.testing.assert_close(torch.cat(chunks,dim=1),full,rtol=3e-4,atol=3e-5)
    return float((torch.cat(chunks,dim=1)-full).abs().max())


@torch.no_grad()
def timed(model, cfg, ids):
    cache=[KVCache(cfg,ids.device,batch_size=len(ids),dtype=torch.float32) for _ in range(4)]
    torch.mps.synchronize();started=time.perf_counter()
    output=model(ids[:,:32],cache=cache)[0]
    torch.mps.synchronize();prefill=time.perf_counter()-started
    del output
    started=time.perf_counter()
    for pos in range(32,96):
        output=model(ids[:,pos:pos+1],cache=cache,start_pos=pos)[0]
        del output
    torch.mps.synchronize();decode=time.perf_counter()-started
    return dict(prefill_seconds=prefill,decode_seconds=decode,
                decode_tokens_per_second=64*len(ids)/decode,kv_bytes=buffer_bytes(cache),
                observed_mps_current_bytes=torch.mps.current_allocated_memory(),
                observed_mps_driver_bytes=torch.mps.driver_allocated_memory())


def benchmark(mha, candidate, output, candidate_family='gqa'):
    if candidate_family not in ('gqa', 'mla'):
        raise ValueError('Unsupported tokenizer benchmark candidate')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required')
    torch.set_num_threads(4);start_memory_guard()
    models={};configs={};metadata={};fixed=None
    for family,path in [('mha',mha),(candidate_family,candidate)]:
        c=read_checkpoint(path);cfg=OmegaConf.create(c['cfg'])
        if cfg.model.attn_type!=family:raise ValueError('Checkpoint family mismatch')
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
    if configs['mha'].model.dim!=configs[candidate_family].model.dim:raise ValueError('Different widths')
    rows=[];correctness=[]
    for batch in (1,8):
        ids=fixed.unsqueeze(0).repeat(batch,1).to('mps')
        for family in ('mha',candidate_family):
            error=check_cached(models[family],configs[family],ids)
            correctness.append(dict(batch=batch,family=family,max_absolute_error=error))
            for _ in range(2):timed(models[family],configs[family],ids)
        for repeat in range(8):
            for family in (('mha',candidate_family) if repeat%2==0 else (candidate_family,'mha')):
                rows.append(dict(batch=batch,repeat=repeat,family=family,**timed(models[family],configs[family],ids)))
    result=dict(status='completed',device='mps',dtype='float32',metadata=metadata,
                fixed_train_token_ids=fixed.tolist(),prompt_tokens=32,decode_tokens=64,
                warmups=2,repeats=8,correctness=correctness,records=rows,
                scope='Two models resident; model loading/cache allocation outside timing. Memory observations are not single-model deployment peaks. Teacher-forced fixed training text; no test access or quality selection.')
    atomic_json(output,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--mha',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--gqa',type=Path);group.add_argument('--mla',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    benchmark(a.mha,a.mla if a.mla is not None else a.gqa,a.output,'mla' if a.mla is not None else 'gqa')


if __name__=='__main__':main()
