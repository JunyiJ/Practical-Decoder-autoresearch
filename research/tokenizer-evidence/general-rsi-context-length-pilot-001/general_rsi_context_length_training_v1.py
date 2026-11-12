"""New context identity using the qualified update/clock/full-state implementation."""
import copy
import torch
from research.general_rsi_equal_time_training_v1 import Context as TimeContext
from research.general_rsi_training_clock_v1 import TrainingClock
from research.general_rsi_data_v1 import MANIFEST_SHA
from research.general_rsi_context_length_v1.contract import Spec
from research.general_rsi_context_length_v1.model import ContextModel
from research.general_rsi_context_length_v1.optimizer import ContextOptimizer
from research.general_rsi_context_length_v1.data import TrainData

class Context(TimeContext):
    def __init__(self,context,roles,binding):
        assert context in (128,256) and type(context) is int
        assert set(roles)=={'native_initializer','composer_initializer','train_sampler','validation_sampler'} and len(set(roles.values()))==4
        assert len(binding)==64
        # This pilot ends after24 updates, before cooldown or averaging can begin.
        # The clock/history remain complete restart state; no quality horizon is set here.
        budget_ns=1200*10**9;half_life_ns=60*10**9
        self.clock=TrainingClock(budget_ns);self.half_life_ns=half_life_ns;self.ema_last_ns=None;self.recovery_charges=[]
        self.config=dict(arm='native',context=context,model_policy='general_rsi_context_length_v1',lr=.0025,budget_ns=budget_ns,half_life_ns=half_life_ns,roles=copy.deepcopy(roles),manifest_sha256=MANIFEST_SHA,binding=binding,device='mps')
        rng=torch.mps.get_rng_state().clone()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(roles['native_initializer']);self.model=ContextModel(Spec(context)).to('mps')
        finally:torch.mps.set_rng_state(rng)
        self.optimizer=ContextOptimizer(self.model,.0025);self.base_lrs=tuple(g['lr'] for g in self.optimizer.param_groups)
        self.data=TrainData(roles['train_sampler'],context)
        # The inherited complete-state format retains a private validation RNG.
        # This pilot never reads validation data, samples it or scores it.
        self.data.view.generators['val']=torch.Generator().manual_seed(roles['validation_sampler'])
        self.schedule=dict(kind='training_time_linear_cooldown',budget_ns=budget_ns,min_ratio=.1)
        self.ema_start=(2*budget_ns+2)//3;self.ema={};self.snapshots=0;self.history=[]
