"""Own split-local two-right-target paired512 sampler; SOURCE ONLY, unqualified.

This is independent data-path preparation. It does not import a model, retry a
closed generated proof, encode a corpus, open holdout, or supply a physical scorer.
"""
import copy
import hashlib
import json
import torch

POLICY='future_token_split_local_paired512_v1'


def require(condition,message):
    if not condition:raise ValueError(message)


def positions(length,starts,*,split):
    """Pure integer plan: two train anchors -> eight rows of130 absolute indices.

    A train anchor supplies512 input positions and TWO right targets. Adjacent
    context128 rows overlap only their two target context IDs. No reshaping of
    a514-token chunk, wrap, padding, or cross-split concatenation is used.
    """
    require(type(length) is int and 0<length<=2**24,'Bounded exact split length required')
    require(type(split) is str and split in ('train','val'),'Only train/val layouts exist')
    count=2 if split=='train' else 8
    require(type(starts) in (list,tuple) and len(starts)==count
        and all(type(s) is int for s in starts),'Exact explicit integer start inventory required')
    span=514 if split=='train' else 130
    require(length>=span and all(0<=s<=length-span for s in starts),'Every right-two target must stay in its own split')
    row_starts=[s+offset for s in starts for offset in (0,128,256,384)] if split=='train' else list(starts)
    return tuple(tuple(start+j for j in range(130)) for start in row_starts)


def _tensor_data(value,name):
    require(type(value) is torch.Tensor and value.layout==torch.strided and value.device.type=='cpu'
        and value.dtype==torch.int64 and value.ndim==1 and value.is_contiguous()
        and 0<value.numel()<=2**24 and not value.requires_grad,'Own bounded contiguous CPU int64 split required: '+name)
    require(bool(((value>=0)&(value<50257)).all()),'In-range native vocabulary IDs required: '+name)


def _role(value):
    require(type(value) is int and 0<=value<2**63,'Explicit newly audited private sampler role required')
    return value


def _copy_generator(generator):
    # The probe never draws: its state is replaced before use. Only persistent
    # train/val generators have declared seed identities, supplied by the caller.
    result=torch.Generator(device='cpu')
    result.set_state(generator.get_state().clone())
    return result


class Sampler:
    """CPU-owned train/val data and private streams; no global RNG draws.

    next() draws on a copy and commits only after validating the complete output.
    This is a sampler transaction, not an optimizer/backward rollback guarantee.
    Construction requires future fresh-role admission; no seed is chosen here.
    """
    def __init__(self,train,val,*,train_seed,eval_seed):
        _tensor_data(train,'train');_tensor_data(val,'val')
        require(train.numel()>=514 and val.numel()>=130,'Both split-local right-two layouts must be eligible')
        train_seed,eval_seed=_role(train_seed),_role(eval_seed)
        require(train_seed!=eval_seed,'Independent explicit private train/eval roles required')
        self.data={'train':train.clone(),'val':val.clone()}
        self.roles={'train':train_seed,'val':eval_seed}
        self.generators={name:torch.Generator(device='cpu').manual_seed(seed) for name,seed in self.roles.items()}
        self.completed_calls={'train':0,'val':0}
        self.identity={name:dict(length=value.numel(),sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
                       for name,value in self.data.items()}
        self.signature=dict(policy=POLICY,context=128,batch=8,vocab=50257,train_layout='two_512_input_blocks_with_two_right_targets',
            validation_layout='eight_independent_128_inputs_with_two_right_targets',roles=copy.deepcopy(self.roles),splits=copy.deepcopy(self.identity))
        self._signature_identity=json.dumps(self.signature,sort_keys=True,separators=(',',':'))

    def validate(self):
        require(json.dumps(self.signature,sort_keys=True,separators=(',',':'))==self._signature_identity,'Mutable sampler identity differs')
        require(type(self.roles) is dict and self.roles==self.signature['roles'] and set(self.data)==set(self.generators)=={'train','val'},'Exact own train/val inventory required')
        require(type(self.completed_calls) is dict and set(self.completed_calls)=={'train','val'}
            and all(type(v) is int and v>=0 for v in self.completed_calls.values()),'Exact nonnegative sampler call counters required')
        for name,value in self.data.items():
            _tensor_data(value,name)
            require(value.numel()==self.signature['splits'][name]['length'],'Own split length changed')
            require(hashlib.sha256(value.numpy().tobytes()).hexdigest()==self.signature['splits'][name]['sha256'],'Own split token identity changed')
            generator=self.generators[name]
            require(type(generator) is torch.Generator and generator.device==torch.device('cpu')
                and generator.initial_seed()==self.roles[name],'Own CPU private generator role differs')

    def next(self,split='train'):
        self.validate()
        require(type(split) is str and split in ('train','val'),'No holdout or foreign split stream exists')
        data=self.data[split];span=514 if split=='train' else 130;count=2 if split=='train' else 8
        high=data.numel()-span+1
        require(high>0,'Positive exact eligible start count required')
        working=_copy_generator(self.generators[split])
        anchors=torch.randint(high,(count,),generator=working,dtype=torch.int64,device='cpu')
        layout=positions(data.numel(),anchors.tolist(),split=split)
        indices=torch.tensor(layout,dtype=torch.int64,device='cpu')
        sequence=data[indices].clone()
        require(sequence.shape==(8,130) and indices.shape==(8,130)
            and bool((indices>=0).all()) and bool((indices<data.numel()).all()),'Complete physical right-two batch required')
        # All arms receive this SAME sequence and all1024 input positions.
        # Only the objective decides the auxiliary target offset.
        result=dict(policy=POLICY,split=split,sequence=sequence,positions=indices,anchors=anchors.clone(),
            primary_targets=1024,second_next_targets=1024)
        self.generators[split].set_state(working.get_state())
        self.completed_calls[split]+=1
        return result

    def state_dict(self):
        self.validate()
        return dict(kind='future_token_private_sampler_state_v1',signature=copy.deepcopy(self.signature),
            completed_calls=copy.deepcopy(self.completed_calls),
            generators={name:g.get_state().clone() for name,g in self.generators.items()})

    def load_state_dict(self,state):
        self.validate()
        require(type(state) is dict and set(state)=={'kind','signature','completed_calls','generators'}
            and state['kind']=='future_token_private_sampler_state_v1' and state['signature']==self.signature,'Own complete sampler identity required')
        calls=state['completed_calls'];saved=state['generators']
        require(type(calls) is dict and set(calls)=={'train','val'} and all(type(v) is int and v>=0 for v in calls.values()),'Exact private call counters required')
        require(type(saved) is dict and set(saved)=={'train','val'},'Complete private generator inventory required')
        staged={}
        for name,value in saved.items():
            current=self.generators[name].get_state()
            require(type(value) is torch.Tensor and value.device.type=='cpu' and value.dtype==torch.uint8
                and value.layout==torch.strided and value.shape==current.shape and value.is_contiguous(),'Own CPU generator-state tensor required')
            probe=_copy_generator(self.generators[name]);probe.set_state(value.clone())
            require(probe.initial_seed()==self.roles[name],'Saved private role differs')
            staged[name]=probe
        # Destination generators/counters change only after BOTH states validate.
        self.generators=staged
        self.completed_calls=copy.deepcopy(calls)
