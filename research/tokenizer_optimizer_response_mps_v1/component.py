"""Own CPU-input synthetic-state policy with explicit CPU/MPS response transport."""
import copy,math,os
import torch
from research.final_ln_scale_model_v1.runtime.train.optimizers import EpsilonOptimizer
from research.gradient_clip_model_v1.clipping import clip_gradients

KIND='generated_fixed_state_optimizer_response_mps_v1'
MAX_ELEMENTS=4096

def require(ok,message):
    if not ok:raise ValueError(message)

def validate(state,gradients):
    require(type(state) is dict and set(state)=={'kind','parameters'} and state['kind']==KIND,'Own synthetic response state required;not a RAW/resume consumer')
    entries=state['parameters']
    require(type(entries) is dict and bool(entries) and all(type(n) is str and bool(n) for n in entries),'Complete named synthetic entries required')
    require(type(gradients) is dict and gradients.keys()==entries.keys(),'Complete identical gradient inventory required')
    total=0
    for name,s in entries.items():
        require(type(s) is dict and set(s)=={'algorithm','value','updates','first','second','lr'},'Exact synthetic parameter/state fields required')
        require(s['algorithm'] in ('muon','adamw') and type(s['updates']) is int and 0<=s['updates']<=1000000,'Known algorithm and actual perparameter update count required')
        p=s['value'];m=s['first'];v=s['second'];g=gradients[name]
        require(type(p) is torch.Tensor and p.device.type=='cpu' and p.layout==torch.strided and p.dtype==torch.float32 and 0<p.numel()<=MAX_ELEMENTS and bool(torch.isfinite(p).all()),'Finite tiny CPU FP32 synthetic parameter required')
        require(not p.requires_grad,'Caller state must be detached')
        require(type(m) is torch.Tensor and m.shape==p.shape and m.device==p.device and m.dtype==p.dtype and m.layout==torch.strided and not m.requires_grad and bool(torch.isfinite(m).all()),'Complete finite same-shaped first moment required')
        require(type(s['lr']) is float and s['lr']==(.000625 if s['algorithm']=='muon' else .00125),'Exact declared synthetic nominal group LR required')
        if s['algorithm']=='muon':require(p.ndim==2 and v is None,'Muon needs a matrix and only momentum;no fake Adam step')
        else:require(type(v) is torch.Tensor and v.shape==p.shape and v.device==p.device and v.dtype==p.dtype and v.layout==torch.strided and not v.requires_grad and bool(torch.isfinite(v).all()) and bool((v>=0).all()),'Finite nonnegative same-shaped Adam second moment required')
        if s['updates']==0:require(bool((m==0).all()) and (v is None or bool((v==0).all())),'Fresh synthetic state must have zero moments')
        require(g is None or (type(g) is torch.Tensor and g.device==p.device and g.dtype==p.dtype and g.layout==torch.strided and g.shape==p.shape and not g.requires_grad and bool(torch.isfinite(g).all())),'Complete finite matching detached gradient or explicit None required')
        total+=p.numel()
    require(total<=MAX_ELEMENTS and any(g is not None for g in gradients.values()),'Tiny complete fixture and at least one active gradient required')

def response(state,gradients,device):
    # Validate ALL caller tensors and state before allocating/stepping either child.
    require(type(device) is str and device in ('cpu','mps'),'Declared CPU or strict MPS response device required')
    validate(state,gradients)
    if device=='mps':
        require(torch.backends.mps.is_available() and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0' and os.environ.get('PYTORCH_MPS_FAST_MATH')=='0','Available strict MPS required without fallback')
    out=copy.deepcopy(state);parameters={n:torch.nn.Parameter(s['value'].to(device=device,copy=True)) for n,s in out['parameters'].items()}
    muon=[];adam=[]
    for name,s in out['parameters'].items():
        group=dict(params=[parameters[name]],lr=s['lr'],weight_decay=0.,parameter_names=[name])
        (muon if s['algorithm']=='muon' else adam).append(group)
        parameters[name].grad=None if gradients[name] is None else gradients[name].to(device=device,copy=True)
    children=[]
    if muon:children.append(torch.optim.Muon(muon,weight_decay=0.,momentum=.95,nesterov=True,ns_coefficients=(3.4445,-4.775,2.0315),eps=1e-7,ns_steps=5,adjust_lr_fn='match_rms_adamw'))
    if adam:children.append(torch.optim.AdamW(adam,weight_decay=0.,betas=(.9,.999),eps=1e-6))
    for child in children:
        for group in child.param_groups:
            name=group['parameter_names'][0];p=parameters[name];s=out['parameters'][name]
            if s['algorithm']=='muon':child.state[p]={'momentum_buffer':s['first'].to(device=device,copy=True)}
            else:child.state[p]={'step':torch.tensor(float(s['updates'])), 'exp_avg':s['first'].to(device=device,copy=True),'exp_avg_sq':s['second'].to(device=device,copy=True)}
    observation=clip_gradients(list(parameters.values()),1.)
    # Actual incumbent EpsilonOptimizer ancestor applies independent decay once.
    optimizer=EpsilonOptimizer(children,.00006);optimizer.step()
    with torch.no_grad():
        for child in children:
            for group in child.param_groups:
                name=group['parameter_names'][0];s=out['parameters'][name];p=parameters[name];saved=child.state[p]
                s['value']=p.detach().to(device='cpu',copy=True)
                if gradients[name] is not None:
                    s['updates']+=1
                    s['first']=saved['momentum_buffer' if s['algorithm']=='muon' else 'exp_avg'].to(device='cpu',copy=True)
                    if s['algorithm']=='adamw':
                        s['second']=saved['exp_avg_sq'].to(device='cpu',copy=True)
                        require(float(saved['step'])==s['updates'],'Actual Adam perparameter update count differs')
    require(all(bool(torch.isfinite(s['value']).all()) and bool(torch.isfinite(s['first']).all()) and (s['second'] is None or bool(torch.isfinite(s['second']).all())) for s in out['parameters'].values()),'Finite cloned response required')
    delta={n:s['value']-state['parameters'][n]['value'] for n,s in out['parameters'].items()}
    return dict(state=out,delta=delta,clipping=observation,device=device,scope='Tiny generated CPU/MPS response transport only;no actual-model/checkpoint consumer or batch/quality prediction.')
