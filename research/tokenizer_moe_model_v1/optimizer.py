import copy
import math
import torch
from research.final_ln_scale_model_v1.runtime.train.optimizers import EpsilonOptimizer
from .contract import require


class Optimizer(EpsilonOptimizer):
    def __init__(self,children,architecture):
        super().__init__(children,.00006);self.architecture=copy.deepcopy(architecture)
        self.completed_steps=0
        self.gradient_steps={g['parameter_names'][0]:0 for g in self.param_groups}
    def step(self):
        active=[g['parameter_names'][0] for g in self.param_groups if g['params'][0].grad is not None]
        super().step()
        self.completed_steps+=1
        for name in active:self.gradient_steps[name]+=1
    def state_dict(self):
        return dict(super().state_dict(),tokenizer_moe_signature=copy.deepcopy(self.architecture),
                    completed_steps=self.completed_steps,gradient_steps=copy.deepcopy(self.gradient_steps))
    def load_state_dict(self,state):
        require(state.get('tokenizer_moe_signature')==self.architecture,'Foreign own optimizer signature')
        steps=state.get('completed_steps');counts=state.get('gradient_steps')
        require(type(steps) is int and steps>=0 and type(counts) is dict and counts.keys()==self.gradient_steps.keys()
                and all(type(v) is int and 0<=v<=steps for v in counts.values()),'Invalid per-parameter gradient/decay exposure')
        super().load_state_dict(state)
        self.completed_steps=steps;self.gradient_steps=copy.deepcopy(counts)


def build_optimizer(model,training):
    model.validate();t=training
    require(t.optimizer=='muon' and t.decay_mode=='independent' and type(t.weight_decay) is float and t.weight_decay==.00006,'Adopted optimizer/independent decay required')
    require(type(t.learning_rate) is float and t.learning_rate in (.000625,.00125,.0025),'Finite prospective learning rate required')
    require(t.adamw_beta2==.999 and t.adamw_epsilon==1e-6,'Adopted beta2/epsilon required')
    require(t.adamw_beta2_policy=='mla4_adamw_beta2_fixed_v1' and t.adamw_epsilon_policy=='mla4_adamw_epsilon_fixed_v1','Explicit adopted moment/epsilon policies required')
    muon=[];adam=[]
    for name,p in model.named_parameters():
        router=name.endswith('.mlp.router.weight')
        hidden=name.startswith('transformer.h.') and p.ndim==2 and not router
        scale=1/math.sqrt(model.width_mult) if hidden else 1.
        group=dict(params=[p],lr=t.learning_rate*scale,weight_decay=0.,parameter_names=[name],width_lr_scale=scale)
        (muon if hidden else adam).append(group)
    require(len(muon)==model.signature['muon_groups'] and len(adam)==model.signature['adam_groups'],'Exact own optimizer coverage required')
    children=[torch.optim.Muon(muon,lr=t.learning_rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw'),
              torch.optim.AdamW(adam,lr=t.learning_rate,weight_decay=0.,betas=(.9,.999),eps=1e-6)]
    return Optimizer(children,model.signature)
