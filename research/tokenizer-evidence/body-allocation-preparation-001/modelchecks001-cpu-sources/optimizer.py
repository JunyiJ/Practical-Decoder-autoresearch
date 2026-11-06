"""Exact inherited grouping equations, explicit geometry and adopted epsilon."""
import math
import torch
from research.final_ln_scale_model_v1.runtime.train.optimizers import EpsilonOptimizer
from .contract import require,plain

class Optimizer(EpsilonOptimizer):
    def __init__(self,children,decay,architecture):
        super().__init__(children,decay);self.architecture=plain(architecture)
    def state_dict(self):return dict(super().state_dict(),body_allocation_signature=self.architecture)
    def load_state_dict(self,state):
        require(state.get('body_allocation_signature')==self.architecture,'Foreign optimizer geometry')
        return super().load_state_dict(state)

def build_optimizer(model,cfg):
    model.validate();t=cfg.training
    require(t.optimizer=='muon' and t.decay_mode=='independent' and type(t.weight_decay) is float and t.weight_decay==.00006,'Adopted optimizer/decay required')
    require(type(t.learning_rate) is float and t.learning_rate in (.000625,.00125,.0025),'Finite peak LR required')
    require(t.adamw_beta2==.999 and t.adamw_epsilon==1e-6,'Adopted AdamW beta2/epsilon required')
    matrices=[];adam=[]
    for name,p in model.named_parameters():
        hidden=name.startswith('transformer.h.') and p.ndim==2
        scale=1/math.sqrt(model.width_mult) if hidden else 1.
        g=dict(params=[p],lr=t.learning_rate*scale,weight_decay=0.,parameter_names=[name],width_lr_scale=scale)
        (matrices if hidden else adam).append(g)
    require(len(matrices)==model.signature['muon_groups'] and len(adam)==model.signature['adam_groups'],'Optimizer coverage differs')
    children=[torch.optim.Muon(matrices,lr=t.learning_rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw'),
        torch.optim.AdamW(adam,lr=t.learning_rate,weight_decay=0.,betas=(.9,.999),eps=1e-6)]
    return Optimizer(children,.00006,model.signature)
