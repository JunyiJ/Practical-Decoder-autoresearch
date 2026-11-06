"""Raw/EMA transport with complete prevalidation and geometry-bound resume."""
import copy,hashlib
from pathlib import Path
import math
import torch
from .contract import plain,signature,require,canonical
from .model import load_weights
from research.block_norm_v1.runtime.train.weight_averaging import averaging_contract
from research.block_norm_v1.runtime.train.lr_schedule import schedule_contract

def training_identity(cfg):
    t=plain(cfg.training)
    for k in ('checkpoint_dir','metrics_file','summary_file','resume_from'):t.pop(k,None)
    from .evidence import environment,source_inventory
    from .contract import fingerprint
    return dict(model=signature(cfg.model),training=t,data=plain(cfg.data),
                corpus_sha256=hashlib.sha256(Path(cfg.data.path).read_bytes()).hexdigest(),
                environment=environment(),sources_sha256=fingerprint(source_inventory()),
                torch_version=str(torch.__version__),schedule=schedule_contract(cfg.training),ema=averaging_contract(cfg.training))

def validate_history(rows,step,cfg):
    s=signature(cfg.model)
    require(type(rows) is list and len(rows)==step and 0<=step<=cfg.training.max_iters,'Complete clipping history required')
    for i,r in enumerate(rows,1):
        require(set(r)=={'step','threshold','pre_norm','post_norm','coefficient','gradient_tensors'} and r['step']==i
            and r['threshold']==1. and r['gradient_tensors']==s['muon_groups']+s['adam_groups'],'Clipping identity differs')
        require(all(type(r[k]) is float and math.isfinite(r[k]) for k in ('pre_norm','post_norm','coefficient')),'Finite clipping history required')
        expected=min(1.,1/(r['pre_norm']+1e-6))
        require(r['pre_norm']>=0 and 0<=r['coefficient']<=1 and r['post_norm']<=1.+1e-5
            and abs(r['coefficient']-expected)<2e-5 and abs(r['post_norm']-r['pre_norm']*r['coefficient'])<=max(1e-5,r['post_norm']*2e-5),'Clipping arithmetic differs')

def raw_payload(model,opt,ema,cfg,step,history,sampler_state):
    model.validate();validate_history(history,step,cfg)
    return dict(kind='raw_body_allocation_v1',architecture=copy.deepcopy(model.signature),cfg=plain(cfg),step=step,
        model_state=model.state_dict(),optimizer_state=opt.state_dict(),ema=ema.state_dict(),
        training_identity=training_identity(cfg),clipping_history=copy.deepcopy(history),sampler_state=sampler_state,
        cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state() if model.lm_head.weight.device.type=='mps' else None)

def validate_resume(model,opt,ema,payload,cfg):
    require(payload.get('kind')=='raw_body_allocation_v1' and payload.get('architecture')==model.signature,'Raw matching architecture required for resume')
    require(payload.get('training_identity')==training_identity(cfg),'Resume environment/recipe differs')
    step=payload.get('step');require(type(step) is int and 0<=step<=cfg.training.max_iters,'Resume step differs')
    validate_history(payload.get('clipping_history'),step,cfg)
    names=dict(model.named_parameters());saved=payload.get('model_state');current=model.state_dict()
    require(type(saved) in (dict,type(current)) and saved.keys()==current.keys(),'Model state inventory differs')
    for n,v in saved.items():require(type(v) is torch.Tensor and v.shape==current[n].shape and v.dtype==torch.float32 and bool(torch.isfinite(v).all()),'Invalid model state '+n)
    e=payload.get('ema');spec=averaging_contract(cfg.training);snapshots=0 if step<spec['start_step'] else 1+(step-spec['start_step'])//spec['interval']
    require(type(e) is dict and set(e)=={'version','spec','completed_steps','snapshots','parameter_names','averages'}
        and e['version']==1 and e['spec']==spec and e['completed_steps']==step and e['snapshots']==snapshots
        and e['parameter_names']==list(names),'EMA metadata differs')
    require(set(e['averages'])==({'200'} if snapshots else set()),'EMA half-life inventory differs')
    for panel in e['averages'].values():
        require(panel.keys()==names.keys(),'EMA tensor inventory differs')
        for n,v in panel.items():require(type(v) is torch.Tensor and v.shape==names[n].shape and v.dtype==torch.float32 and bool(torch.isfinite(v).all()),'Invalid EMA tensor '+n)
    state=payload.get('optimizer_state');require(type(state) is dict and state.get('body_allocation_signature')==model.signature
        and state.get('version')==1 and state.get('independent_decay')==.00006 and len(state.get('optimizers',[]))==2,'Optimizer identity differs')
    for child,stored in zip(opt.optimizers,state['optimizers']):
        expected=child.state_dict()['param_groups'];groups=stored.get('param_groups')
        require(type(groups) is list and len(groups)==len(expected),'Optimizer group count differs')
        ids=[]
        for actual,wanted,live in zip(groups,expected,child.param_groups):
            require({k:v for k,v in actual.items() if k!='lr'}=={k:v for k,v in wanted.items() if k!='lr'},'Optimizer groups differ')
            require(type(actual['lr']) is float and math.isfinite(actual['lr']) and actual['lr']>=0,'Optimizer LR invalid')
            index=actual['params'][0];ids.append(index);p=live['params'][0]
            if step:
                s=stored['state'].get(index);keys={'momentum_buffer'} if type(child) is torch.optim.Muon else {'step','exp_avg','exp_avg_sq'}
                require(type(s) is dict and set(s)==keys,'Optimizer tensor inventory differs')
                for n,v in s.items():
                    require(type(v) is torch.Tensor and v.dtype==torch.float32 and bool(torch.isfinite(v).all()),'Optimizer state dtype/finite differs')
                    require(v.numel()==1 and float(v)==step if n=='step' else v.shape==p.shape,'Optimizer state shape/step differs')
        require(set(stored['state'])==(set(ids) if step else set()),'Optimizer state coverage differs')
    require(type(payload.get('cpu_rng')) is torch.Tensor and payload['cpu_rng'].dtype==torch.uint8,'CPU RNG state missing')
    if model.lm_head.weight.device.type=='mps':require(type(payload.get('mps_rng')) is torch.Tensor and payload['mps_rng'].dtype==torch.uint8,'MPS RNG state missing')
    sampler=payload.get('sampler_state');require(type(sampler) is dict and set(sampler)=={'train','val'}
        and all(type(v) is torch.Tensor and v.dtype==torch.uint8 and v.ndim==1 for v in sampler.values()),'Train/validation sampler state differs')
    return step

def resume(model,opt,ema,payload,cfg):
    step=validate_resume(model,opt,ema,payload,cfg)
    load_weights(model,payload);opt.load_state_dict(payload['optimizer_state']);ema.load_state_dict(payload['ema'],step)
    torch.set_rng_state(payload['cpu_rng'].cpu())
    if model.lm_head.weight.device.type=='mps':torch.mps.set_rng_state(payload['mps_rng'].cpu())
    return step,copy.deepcopy(payload['clipping_history']),payload['sampler_state']

def h200_payload(model,ema,cfg,parent):
    require(ema.completed_steps==cfg.training.max_iters,'Only final H200 export allowed')
    state=dict(model.state_dict());state.update(ema.averaged_parameters(200))
    return dict(kind='h200_body_allocation_v1',architecture=copy.deepcopy(model.signature),cfg=plain(cfg),
        step=ema.completed_steps,model_state=state,parent=parent,
        averaged_weights=dict(spec=ema.spec,snapshots=ema.snapshots,parameter_names=[n for n,_ in model.named_parameters()],half_life=200,purpose='inference_only_not_optimizer_resume'))
