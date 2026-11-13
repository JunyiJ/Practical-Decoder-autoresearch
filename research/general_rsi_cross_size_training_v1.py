"""Complete-state fixed-width depth context using the qualified update/restore path."""
import copy,math
import torch
from research.general_rsi_training_v2 import Context as OriginalContext, KIND as ORIGINAL_KIND
from research.general_rsi_data_v1 import GeneralData,MANIFEST_SHA
from research.general_rsi_cross_size_v1.model import DepthModel
from research.general_rsi_cross_size_v1.contract import Spec
from research.general_rsi_cross_size_v1.optimizer import build_optimizer
KIND='general_rsi_cross_size_complete_training_v1'
class Context(OriginalContext):
    def __init__(self,arm,depth,roles,extra_group_seeds,book,binding,horizon=40):
        assert arm in ('native','raw_bytes') and type(depth) is int and depth in (4,8,16)
        assert horizon in (40,10240) and type(horizon) is int
        assert set(roles)=={'native_initializer','composer_initializer','train_sampler','validation_sampler'}
        assert len(set([*roles.values(),*extra_group_seeds]))==7 and len(binding)==64
        self.config=dict(arm=arm,depth=depth,lr=.0025,horizon=horizon,roles=copy.deepcopy(roles),extra_group_seeds=tuple(extra_group_seeds),manifest_sha256=MANIFEST_SHA,binding=binding,device='mps')
        rng=torch.mps.get_rng_state().clone()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(roles['native_initializer'])
                spec=Spec(arm,depth,tuple(extra_group_seeds),None if arm=='native' else roles['composer_initializer'],None if arm=='native' else book.identity)
                self.model=DepthModel(spec,None if arm=='native' else book).to('mps')
        finally:torch.mps.set_rng_state(rng)
        self.optimizer=build_optimizer(self.model,.0025)
        self.base_lrs=tuple(g['lr'] for g in self.optimizer.param_groups)
        self.data=GeneralData(roles['train_sampler'],roles['validation_sampler'],'mps')
        self.schedule=dict(kind='linear_cooldown',total_steps=horizon,min_ratio=.1,start_step=2*horizon//3)
        self.ema_start=math.ceil((2*horizon/3)/10)*10
        self.ema={};self.snapshots=0;self.history=[]
    def capture(self):
        p=super().capture();p['kind']=KIND;return p
    def restore(self,payload):
        assert payload['kind']==KIND
        p=dict(payload);p['kind']=ORIGINAL_KIND
        return super().restore(p)
