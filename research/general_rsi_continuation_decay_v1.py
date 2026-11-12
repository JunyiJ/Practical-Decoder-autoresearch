"""Separate continuation decay policy; canonical children and frozen sources unchanged."""
import torch
from research.byte_input_model_v6.optimizer import ByteInputOptimizer
from research.general_rsi_pseudo_training_v1 import Context as Parent, KIND as PARENT_KIND

KIND='general_rsi_continuation_group_decay_v1'
POLICIES={'all':(.00006,.00006),'other_only':(0.,.00006),'readout_only':(.00006,0.),'none':(0.,0.)}

class GroupDecayOptimizer(ByteInputOptimizer):
    def __init__(self,model,learning_rate,policy):
        assert policy in POLICIES
        self.policy=policy;self.decay=POLICIES[policy]
        super().__init__(model,learning_rate)

    def validate(self,require_gradients=False):
        assert self.policy in POLICIES and self.decay==POLICIES[self.policy]
        super().validate(require_gradients)
        assert sum(n=='lm_head.weight' for n,_ in self._items)==1

    def apply_decay(self):
        assert self.decay==POLICIES[self.policy]
        with torch.no_grad():
            for name,p in self._items:
                coefficient=self.decay[0 if name=='lm_head.weight' else 1]
                p.mul_(1.-coefficient)

    def step(self):
        self.validate(require_gradients=True);self.apply_decay()
        for child in self._optimizers:child.step()
        self._steps+=1

class Context(Parent):
    def __init__(self,policy,budget_ns,half_life_ns,roles,pseudo_seed,parent,binding):
        assert policy in POLICIES
        super().__init__('native_ce',budget_ns,half_life_ns,roles,pseudo_seed,parent,binding)
        self.optimizer=GroupDecayOptimizer(self.model,.00025,policy)
        self.base_lrs=tuple(g['lr'] for g in self.optimizer.param_groups)
        self.config.update(quality_arm=policy,execution=KIND,decay_policy=POLICIES[policy],
            phase_policy='Natural CE each step; readout/other independent decay before canonical child updates')

    def capture(self):
        assert self.optimizer.policy==self.config['quality_arm'] and self.optimizer.decay==self.config['decay_policy']
        value=super().capture();value['kind']=KIND;value['decay_policy']=self.optimizer.decay
        return value

    def restore(self,payload):
        assert payload['kind']==KIND and payload['config']==self.config
        assert payload['decay_policy']==self.config['decay_policy']==self.optimizer.decay
        assert self.optimizer.policy==self.config['quality_arm']
        value=dict(payload,kind=PARENT_KIND);value.pop('decay_policy')
        super().restore(value)
