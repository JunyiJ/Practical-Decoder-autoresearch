"""Configurable time-EMA endpoints on the isolated native context path."""
import torch
from research.general_rsi_context_length_training_v1 import Context as PilotContext
from research.general_rsi_training_clock_v1 import TrainingClock
from research.general_rsi_training_v2 import clone_tree,write_checkpoint,read_checkpoint
from research.general_rsi_context_length_v1.model import ContextModel
from research.general_rsi_context_length_v1.contract import Spec

class Context(PilotContext):
    def __init__(self,context,budget_ns,half_life_ns,roles,binding):
        assert type(budget_ns) is int and budget_ns>0 and type(half_life_ns) is int and half_life_ns>0
        super().__init__(context,roles,binding)
        self.clock=TrainingClock(budget_ns);self.half_life_ns=half_life_ns
        self.config=dict(self.config,budget_ns=budget_ns,half_life_ns=half_life_ns,endpoint_policy='context_length_time_ema_v1')
        self.schedule=dict(kind='training_time_linear_cooldown',budget_ns=budget_ns,min_ratio=.1)
        self.ema_start=(2*budget_ns+2)//3

    def export(self,path,endpoint):
        assert endpoint in ('raw','time_ema');self.optimizer.validate()
        parameters=dict(self.model.ordered_parameters()) if endpoint=='raw' else self.ema
        assert len(parameters)==88 and (endpoint=='raw' or self.snapshots>0)
        return write_checkpoint(path,clone_tree(dict(kind='context_length_inference_v1',identity=self.model.identity,config=self.config,endpoint=endpoint,step=self.step,clock=self.clock.state_dict(),immutable_buffers=self.model._own_expected_buffer_hashes,parameters=parameters)),cap=128*2**20)

def load_inference(receipt,context,binding):
    payload=read_checkpoint(receipt)
    assert payload['kind']=='context_length_inference_v1' and payload['config']['binding']==binding and payload['config']['context']==context
    assert payload['identity']==('general_rsi_context_length_v1','native',256,50257,context,'uncached')
    rng=torch.mps.get_rng_state().clone()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(payload['config']['roles']['native_initializer']);model=ContextModel(Spec(context)).to('mps')
    finally:torch.mps.set_rng_state(rng)
    assert payload['identity']==model.identity and payload['immutable_buffers']==model._own_expected_buffer_hashes
    items=dict(model.ordered_parameters());assert list(items)==list(payload['parameters'])
    for n,p in items.items():
        v=payload['parameters'][n];assert v.device.type=='cpu' and v.shape==p.shape and v.dtype==p.dtype and torch.isfinite(v).all()
    with torch.no_grad():
        for n,p in items.items():p.copy_(payload['parameters'][n])
    model.validate();return model
