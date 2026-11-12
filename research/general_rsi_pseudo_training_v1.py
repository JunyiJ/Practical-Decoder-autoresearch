"""Native continuation with alternating CE and synthetic-span repetition updates."""
import hashlib
import math
import time
import torch
from research.general_rsi_equal_time_training_v1 import Context as Base, KIND as BASE_KIND
from research.general_rsi_training_v2 import clone_tree,sha
from research.byte_input_model_v6 import build_optimizer
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.general_rsi_pseudo_spans_v1 import Sampler,repetition_loss

KIND='general_rsi_pseudo_repetition_training_v1'
ARMS=('native_ce','native_pseudo_repetition')


class Context(Base):
    def __init__(self,arm,budget_ns,half_life_ns,roles,pseudo_seed,parent,binding):
        assert arm in ARMS and pseudo_seed not in roles.values()
        super().__init__('native',.0025,budget_ns,half_life_ns,roles,None,binding)
        assert sha(parent['path'])==parent['sha256']
        payload=torch.load(parent['path'],map_location='cpu',weights_only=True)
        assert payload['kind']=='general_rsi_equal_time_inference_v1' and payload['endpoint']=='time_ema'
        assert payload['slot']['arm']=='native' and payload['slot']['slot']==parent['slot']
        assert payload['binding']==parent['binding'] and payload['model_identity']==self.model.identity
        assert payload['immutable_buffers']==self.model._own_expected_buffer_hashes
        items=dict(self.model.ordered_parameters());assert list(items)==list(payload['parameters'])
        with torch.no_grad():
            for name,p in items.items():
                saved=payload['parameters'][name];assert saved.shape==p.shape and saved.dtype==torch.float32 and bool(torch.isfinite(saved).all())
                p.copy_(saved)
        del payload
        self.optimizer=build_optimizer(self.model,.00025)
        self.base_lrs=tuple(g['lr'] for g in self.optimizer.param_groups)
        self.pseudo=Sampler(pseudo_seed)
        self.config.update(lr=.00025,quality_arm=arm,execution=KIND,pseudo_seed=pseudo_seed,
            pseudo_inventory=self.pseudo.binding,parent=parent,phase_policy='CE on odd1-based steps; pseudo on even steps only in candidate')

    def update(self,now_ns=time.perf_counter_ns):
        begin_ns=self.clock.spent_ns;ratio=self.clock.lr_ratio()
        is_pseudo=self.config['quality_arm']=='native_pseudo_repetition' and self.step%2==1
        with self.clock.update(torch.mps.synchronize,now_ns):
            for g,base in zip(self.optimizer.param_groups,self.base_lrs):g['lr']=base*ratio
            self.model.train();self.optimizer.zero_grad()
            if is_pseudo:
                x,y,previous,mask,details=self.pseudo.batch('mps')
                objective,info=repetition_loss(self.model.primary_logits(x).values,y,previous,mask)
                hash_extra=previous.cpu().numpy().tobytes()+mask.cpu().numpy().tobytes()
            else:
                x,y=self.data.batch();objective=self.model.forward_train(x,y).loss
                info=dict(targets=y.numel(),saturated=0);details=[];hash_extra=b''
            loss=float(objective.detach());assert math.isfinite(loss)
            objective.backward();clip=clip_gradients((p for _,p in self.model.ordered_parameters()),1.0)
            self.optimizer.step();self.optimizer.zero_grad()
            with torch.no_grad():
                if begin_ns>=self.ema_start:
                    if self.ema_last_ns is None:self.ema={n:p.detach().clone() for n,p in self.model.ordered_parameters()}
                    else:
                        alpha=1.-2.**(-(begin_ns-self.ema_last_ns)/self.half_life_ns)
                        for n,p in self.model.ordered_parameters():self.ema[n].lerp_(p.detach(),alpha)
                    self.ema_last_ns=begin_ns;self.snapshots+=1
            batch_hash=hashlib.sha256(x.cpu().numpy().tobytes()+y.cpu().numpy().tobytes()).hexdigest()
            policy_hash=hashlib.sha256(hash_extra).hexdigest() if is_pseudo else None
        row=dict(step=self.step,loss=loss,lr_ratio=ratio,clip=clip,batch_sha256=batch_hash,begin_ns=begin_ns,end_ns=self.clock.spent_ns,
            objective='pseudo' if is_pseudo else 'ce',valid_loss_targets=info['targets'],saturated_targets=info['saturated'],
            alignment_sha256=policy_hash,pseudo_rows=details)
        self.history.append(row);return row

    def capture(self):
        state=super().capture();state['kind']=KIND;state['pseudo_sampler']=self.pseudo.state_dict()
        state['objective_counts']={name:sum(r['objective']==name for r in self.history) for name in ['ce','pseudo']}
        return clone_tree(state)

    def restore(self,payload):
        assert payload['kind']==KIND and payload['config']==self.config
        counts={name:sum(r['objective']==name for r in payload['history']) for name in ['ce','pseudo']}
        assert counts==payload['objective_counts'] and payload['pseudo_sampler']['draws']==8*counts['pseudo']
        for row in payload['history']:
            expected='pseudo' if self.config['quality_arm']=='native_pseudo_repetition' and row['step']%2==0 else 'ce'
            assert row['objective']==expected
        self.pseudo.load_state_dict(payload['pseudo_sampler'])
        base=dict(payload,kind=BASE_KIND);base.pop('pseudo_sampler');base.pop('objective_counts')
        super().restore(base)
