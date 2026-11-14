"""Combined native depth/context on the qualified time-clock and restart math."""
import copy
import torch
from research.general_rsi_equal_time_training_v1 import Context as TimeContext, KIND as BASE_KIND
from research.general_rsi_training_clock_v1 import TrainingClock
from research.general_rsi_training_v2 import clone_tree,write_checkpoint,read_checkpoint
from research.general_rsi_data_v1 import MANIFEST_SHA
from research.general_rsi_context_length_v1.data import TrainData
from research.general_rsi_depth_context_v1.contract import Spec,counts
from research.general_rsi_depth_context_v1.model import Model
from research.general_rsi_depth_context_v1.optimizer import build_optimizer
KIND='general_rsi_depth_context_time_training_v1'
ENDPOINT='general_rsi_depth_context_inference_v1'
class Context(TimeContext):
    def __init__(self,depth,context,budget_ns,half_life_ns,roles,extra_group_seeds,binding):
        assert type(depth) is int and depth in (4,16)
        assert type(context) is int and context in (128,256)
        assert type(budget_ns) is int and budget_ns>0 and type(half_life_ns) is int and half_life_ns>0
        assert set(roles)=={'native_initializer','composer_initializer','train_sampler','validation_sampler'}
        assert len(set([*roles.values(),*extra_group_seeds]))==7 and len(binding)==64
        self.clock=TrainingClock(budget_ns);self.half_life_ns=half_life_ns
        self.ema_last_ns=None;self.recovery_charges=[]
        self.config=dict(arm='native',depth=depth,context=context,lr=.0025,budget_ns=budget_ns,half_life_ns=half_life_ns,roles=copy.deepcopy(roles),extra_group_seeds=tuple(extra_group_seeds),manifest_sha256=MANIFEST_SHA,binding=binding,device='mps',model_policy='general_rsi_depth_context_v1')
        rng=torch.mps.get_rng_state().clone()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(roles['native_initializer'])
                self.model=Model(Spec(depth,context,tuple(extra_group_seeds))).to('mps')
        finally:torch.mps.set_rng_state(rng)
        self.optimizer=build_optimizer(self.model,.0025);self.base_lrs=tuple(g['lr'] for g in self.optimizer.param_groups)
        self.data=TrainData(roles['train_sampler'],context)
        self.data.view.generators['val']=torch.Generator().manual_seed(roles['validation_sampler'])
        self.schedule=dict(kind='training_time_linear_cooldown',budget_ns=budget_ns,min_ratio=.1)
        self.ema_start=(2*budget_ns+2)//3;self.ema={};self.snapshots=0;self.history=[]
    def capture(self):
        p=super().capture();p['kind']=KIND;return p
    def restore(self,payload):
        assert payload['kind']==KIND
        p=dict(payload);p['kind']=BASE_KIND;return super().restore(p)
    def export(self,path,endpoint):
        assert endpoint in ('raw','time_ema');self.optimizer.validate()
        parameters=dict(self.model.ordered_parameters()) if endpoint=='raw' else self.ema
        assert len(parameters)==counts(self.model.spec)[0] and (endpoint=='raw' or self.snapshots>0)
        return write_checkpoint(path,clone_tree(dict(kind=ENDPOINT,identity=self.model.identity,config=self.config,endpoint=endpoint,step=self.step,clock=self.clock.state_dict(),immutable_buffers=self.model._own_expected_buffer_hashes,parameters=parameters)),cap=192*2**20)
def load_inference(receipt,depth,context,binding):
    payload=read_checkpoint(receipt)
    assert payload['kind']==ENDPOINT and payload['config']['binding']==binding
    assert payload['config']['depth']==depth and payload['config']['context']==context
    spec=Spec(depth,context,tuple(payload['config']['extra_group_seeds']))
    rng=torch.mps.get_rng_state().clone()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(payload['config']['roles']['native_initializer']);model=Model(spec).to('mps')
    finally:torch.mps.set_rng_state(rng)
    assert payload['identity']==model.identity and payload['immutable_buffers']==model._own_expected_buffer_hashes
    items=dict(model.ordered_parameters());assert list(items)==list(payload['parameters'])
    for n,p in items.items():
        v=payload['parameters'][n];assert v.device.type=='cpu' and v.shape==p.shape and v.dtype==p.dtype and torch.isfinite(v).all()
    with torch.no_grad():
        for n,p in items.items():p.copy_(payload['parameters'][n])
    model.validate();return model
