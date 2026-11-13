"""Depth-bound standalone RAW/H200 publication and preflighted parameter loading."""
import torch
from research.general_rsi_training_v2 import clone_tree,write_checkpoint,read_checkpoint
from research.general_rsi_cross_size_v1.model import DepthModel
from research.general_rsi_cross_size_v1.contract import Spec
KIND='general_rsi_cross_size_inference_v1'

def export(context,path,endpoint):
    assert endpoint in ('raw','h200') and context.step==context.config['horizon']
    context.optimizer.validate()
    parameters=dict(context.model.ordered_parameters()) if endpoint=='raw' else context.ema
    assert list(parameters)==[n for n,_ in context.model.ordered_parameters()]
    assert endpoint=='raw' or context.snapshots>0
    return write_checkpoint(path,clone_tree(dict(kind=KIND,identity=context.model.identity,
        config=context.config,endpoint=endpoint,step=context.step,snapshots=context.snapshots,
        immutable_buffers=context.model._own_expected_buffer_hashes,parameters=parameters)),cap=192*2**20)

def apply(model,payload,config,endpoint):
    # Validate the complete payload before the first destination mutation.
    assert payload['kind']==KIND and payload['config']==config and payload['endpoint']==endpoint
    assert payload['step']==config['horizon'] and payload['identity']==model.identity
    assert payload['immutable_buffers']==model._own_expected_buffer_hashes
    items=dict(model.ordered_parameters());assert list(items)==list(payload['parameters'])
    for n,p in items.items():
        v=payload['parameters'][n]
        assert type(v) is torch.Tensor and v.device.type=='cpu' and v.shape==p.shape
        assert v.dtype==p.dtype and v.layout==torch.strided and bool(torch.isfinite(v).all())
    with torch.no_grad():
        for n,p in items.items():p.copy_(payload['parameters'][n])
    model.validate()

def load(receipt,config,endpoint,book,device='mps'):
    assert device in ('cpu','mps')
    payload=read_checkpoint(receipt)
    assert payload['config']==config and payload['kind']==KIND
    rng=torch.mps.get_rng_state().clone() if device=='mps' else None
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['roles']['native_initializer'])
            arm=config['arm'];spec=Spec(arm,config['depth'],tuple(config['extra_group_seeds']),
                None if arm=='native' else config['roles']['composer_initializer'],None if arm=='native' else book.identity)
            model=DepthModel(spec,None if arm=='native' else book).to(device)
        apply(model,payload,config,endpoint)
    finally:
        if rng is not None:torch.mps.set_rng_state(rng)
    return model
