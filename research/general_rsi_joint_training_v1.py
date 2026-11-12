"""One optimizer advance after fixed CE/pseudo gradient accumulation."""
import math
import time
import torch
from research.general_rsi_pseudo_training_v1 import Context as ParentContext
from research.general_rsi_equal_time_training_v1 import Context as Base, KIND as BASE_KIND
from research.general_rsi_training_v2 import clone_tree
from research.general_rsi_pseudo_spans_v1 import repetition_loss
from research.general_rsi_pseudo_update_diagnostic_v1 import batch_hash
from research.gradient_clip_model_v1.clipping import clip_gradients

KIND = 'general_rsi_joint_gradient_training_v1'
WEIGHTS = {'ce_full': (1., 0.), 'ce_half': (.5, 0.), 'ce_pseudo_mean': (.5, .5)}


class Context(ParentContext):
    def __init__(self, arm, budget_ns, half_life_ns, roles, pseudo_seed, parent, binding):
        assert arm in WEIGHTS
        super().__init__('native_ce',budget_ns,half_life_ns,roles,pseudo_seed,parent,binding)
        self.config.update(quality_arm=arm,execution=KIND,loss_weights=WEIGHTS[arm],
            phase_policy='Fresh CE every update; joint arm adds fresh pseudo backward before one clip and optimizer step')

    def backward_components(self):
        """Caller clears gradients. No clipping, optimizer mutation or averaging here."""
        ce_weight,pseudo_weight=self.config['loss_weights']
        assert all(p.grad is None for _,p in self.model.ordered_parameters())
        x,y=self.data.batch()
        ce=self.model.forward_train(x,y).loss
        ce_value=float(ce.detach());assert math.isfinite(ce_value)
        (ce_weight*ce).backward();del ce
        natural_hash=batch_hash(x,y)
        pseudo_value=None;pseudo_hash=None;details=[];info=dict(targets=0,saturated=0)
        if pseudo_weight:
            px,py,previous,mask,details=self.pseudo.batch('mps')
            pseudo,info=repetition_loss(self.model.primary_logits(px).values,py,previous,mask)
            pseudo_value=float(pseudo.detach());assert math.isfinite(pseudo_value)
            # Existing CE gradients stay present. Both graphs use the same weights.
            (pseudo_weight*pseudo).backward();del pseudo
            pseudo_hash=batch_hash(px,py,previous,mask)
        return dict(loss=ce_weight*ce_value+pseudo_weight*(pseudo_value or 0.),
                    ce_loss=ce_value,pseudo_loss=pseudo_value,ce_weight=ce_weight,pseudo_weight=pseudo_weight,
                    batch_sha256=natural_hash,pseudo_sha256=pseudo_hash,pseudo_rows=details,
                    ce_targets=y.numel(),pseudo_targets=info['targets'],saturated_targets=info['saturated'])

    def update(self,now_ns=time.perf_counter_ns):
        begin_ns=self.clock.spent_ns;ratio=self.clock.lr_ratio()
        with self.clock.update(torch.mps.synchronize,now_ns):
            for group,base in zip(self.optimizer.param_groups,self.base_lrs):group['lr']=base*ratio
            self.model.train();self.optimizer.zero_grad()
            components=self.backward_components()
            clip=clip_gradients((p for _,p in self.model.ordered_parameters()),1.0)
            self.optimizer.step();self.optimizer.zero_grad()
            with torch.no_grad():
                if begin_ns>=self.ema_start:
                    if self.ema_last_ns is None:self.ema={n:p.detach().clone() for n,p in self.model.ordered_parameters()}
                    else:
                        alpha=1.-2.**(-(begin_ns-self.ema_last_ns)/self.half_life_ns)
                        for n,p in self.model.ordered_parameters():self.ema[n].lerp_(p.detach(),alpha)
                    self.ema_last_ns=begin_ns;self.snapshots+=1
        row=dict(step=self.step,lr_ratio=ratio,clip=clip,begin_ns=begin_ns,end_ns=self.clock.spent_ns,**components)
        self.history.append(row);return row

    def capture(self):
        state=Base.capture(self);state['kind']=KIND
        state['pseudo_sampler']=self.pseudo.state_dict()
        state['component_counts']=dict(ce=self.step,pseudo=self.step if self.config['loss_weights'][1] else 0)
        assert state['pseudo_sampler']['draws']==8*state['component_counts']['pseudo']
        return clone_tree(state)

    def restore(self,payload):
        assert payload['kind']==KIND and payload['config']==self.config
        joint=bool(self.config['loss_weights'][1]);step=payload['step']
        assert payload['component_counts']==dict(ce=step,pseudo=step if joint else 0)
        assert payload['pseudo_sampler']['draws']==8*payload['component_counts']['pseudo']
        for row in payload['history']:
            assert (row['ce_weight'],row['pseudo_weight'])==self.config['loss_weights']
            assert row['ce_targets']==1024 and (row['pseudo_targets']>0)==joint
            assert (row['pseudo_loss'] is not None)==joint and (row['pseudo_sha256'] is not None)==joint
            assert len(row['pseudo_rows'])==(8 if joint else 0)
        self.pseudo.load_state_dict(payload['pseudo_sampler'])
        base=dict(payload,kind=BASE_KIND);base.pop('pseudo_sampler');base.pop('component_counts')
        Base.restore(self,base)
