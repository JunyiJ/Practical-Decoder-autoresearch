"""Validate RNG encoding and copied configuration before touching destination state."""
import torch
from research.body_allocation_model_v1.contract import require,plain,canonical
from research.embedding_input_dropout_resume_v1 import checkpoint as inherited

def normalized(cfg):
    d=plain(cfg);require(type(d) is dict and type(d.get('training')) is dict,'Complete copied configuration required')
    for k in ('checkpoint_dir','metrics_file','summary_file','resume_from'):d['training'].pop(k,None)
    return d

def cpu_state(value):
    require(type(value) is torch.Tensor and value.dtype==torch.uint8 and value.device.type=='cpu'
        and value.is_contiguous() and value.ndim==1 and value.shape==torch.get_rng_state().shape,'CPU/sampler RNG layout differs')
    # A private generator checks the encoding without changing the live stream.
    try:torch.Generator(device='cpu').set_state(value)
    except RuntimeError as exc:raise ValueError('Invalid CPU/sampler RNG encoding') from exc

def metadata_preflight(payload,cfg):
    require(canonical(normalized(payload.get('cfg')))==canonical(normalized(cfg)),'Copied config differs from bound resume recipe')
    cpu_state(payload.get('cpu_rng'));sampler=payload.get('sampler_state')
    require(type(sampler) is dict and set(sampler)=={'train','val'},'Exact sampler inventory required')
    for value in sampler.values():cpu_state(value)
    live=torch.mps.get_rng_state().clone();value=payload.get('mps_rng')
    require(type(value) is torch.Tensor and value.dtype==torch.uint8 and value.device.type=='cpu'
        and value.is_contiguous() and value.ndim==1 and value.shape==live.shape,'MPS RNG layout differs')
    # The backend has no separately constructible MPS Generator. Probe its decoder
    # transactionally under the research lock, restoring the live stream always.
    try:torch.mps.set_rng_state(value)
    except RuntimeError as exc:raise ValueError('Invalid MPS RNG encoding') from exc
    finally:torch.mps.set_rng_state(live)
    require(torch.equal(torch.mps.get_rng_state(),live),'MPS preflight changed live RNG')

def validate_resume(model,opt,ema,payload,cfg):
    metadata_preflight(payload,cfg)
    return inherited.validate_resume(model,opt,ema,payload,cfg)

def resume(model,opt,ema,payload,cfg):
    validate_resume(model,opt,ema,payload,cfg)
    return inherited.resume(model,opt,ema,payload,cfg)
