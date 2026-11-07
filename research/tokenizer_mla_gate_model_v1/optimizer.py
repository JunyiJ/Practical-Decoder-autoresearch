import copy,math
import torch
from research.final_ln_scale_model_v1.runtime.train.optimizers import EpsilonOptimizer
from .contract import require

class Optimizer(EpsilonOptimizer):
    def __init__(self,children,architecture):
        super().__init__(children,.00006);self.architecture=copy.deepcopy(architecture);self.completed_steps=0;self.gradient_steps={g['parameter_names'][0]:0 for g in self.param_groups}
    def step(self):
        active=[g['parameter_names'][0] for g in self.param_groups if g['params'][0].grad is not None];super().step();self.completed_steps+=1
        for n in active:self.gradient_steps[n]+=1
    def state_dict(self):return dict(super().state_dict(),tokenizer_mla_gate_signature=copy.deepcopy(self.architecture),completed_steps=self.completed_steps,gradient_steps=copy.deepcopy(self.gradient_steps))
    def load_state_dict(self,state):
        require(type(state) is dict and state.get('tokenizer_mla_gate_signature')==self.architecture,'Foreign own gate optimizer signature')
        step=state.get('completed_steps');counts=state.get('gradient_steps');require(type(step) is int and step>=0 and type(counts) is dict and counts.keys()==self.gradient_steps.keys() and all(type(v) is int and 0<=v<=step for v in counts.values()),'Exact perparameter actual update/decay exposure required')
        super().load_state_dict(state);self.completed_steps=step;self.gradient_steps=copy.deepcopy(counts)

def build_optimizer(model,training):
    model.validate();t=training;require(t.optimizer=='muon' and t.decay_mode=='independent' and type(t.weight_decay) is float and t.weight_decay==.00006,'Adopted independent-decay optimizer required')
    require(type(t.learning_rate) is float and t.learning_rate in (.000625,.00125,.0025) and t.adamw_beta2==.999 and t.adamw_epsilon==1e-6,'Finite prospective LR/adopted moments/epsilon required')
    require(t.adamw_beta2_policy=='mla4_adamw_beta2_fixed_v1' and t.adamw_epsilon_policy=='mla4_adamw_epsilon_fixed_v1','Explicit adopted moment policies required')
    muon=[];adam=[]
    for name,p in model.named_parameters():
        hidden=name.startswith('transformer.h.') and p.ndim==2;scale=1/math.sqrt(model.width_mult) if hidden else 1.;group=dict(params=[p],lr=t.learning_rate*scale,weight_decay=0.,parameter_names=[name],width_lr_scale=scale);(muon if hidden else adam).append(group)
    require(len(muon)==model.signature['muon_groups'] and len(adam)==model.signature['adam_groups'],'Complete own optimizer role inventory required')
    children=[torch.optim.Muon(muon,lr=t.learning_rate,weight_decay=0.,adjust_lr_fn='match_rms_adamw'),torch.optim.AdamW(adam,lr=t.learning_rate,weight_decay=0.,betas=(.9,.999),eps=1e-6)]
    return Optimizer(children,model.signature)
