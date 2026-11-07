"""Own gate transport; equation provenance: immutable tokenizer_moe_resume_v1, without routing or legacy type dispatch."""
import copy
import hashlib
import math
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import require,plain,fingerprint,canonical
from research.tokenizer_mla_gate_model_v1.model import load_weights
from research.block_norm_v1.runtime.models.rope import precompute_freqs_cis
from research.block_norm_v1.runtime.train.weight_averaging import averaging_contract
from research.block_norm_v1.runtime.train.lr_schedule import schedule_contract,multiplier
from research.embedding_input_dropout_resume_guard_v1.checkpoint import metadata_preflight
from .contract import sources,runtime
from .data import architecture,identity,restore_samplers

RAW='raw_tokenizer_mla_gate_v1'
H200='h200_tokenizer_mla_gate_v1'


def training_identity(cfg):
    training=plain(cfg.training)
    for name in ('checkpoint_dir','metrics_file','summary_file','resume_from'):training.pop(name,None)
    return dict(model=architecture(cfg),training=training,data=plain(cfg.data),data_identity=identity(cfg),
                corpus_sha256=hashlib.sha256(Path(cfg.data.path).read_bytes()).hexdigest(),environment=runtime(),
                sources_sha256=fingerprint(sources()),schedule=schedule_contract(cfg.training),ema=averaging_contract(cfg.training))


def validate_history(model,rows,step,cfg):
    require(type(step) is int and 0<=step<=cfg.training.max_iters and type(rows) is list and len(rows)==step,'Complete bounded clipping history required')
    names=[n for n,_ in model.named_parameters()]
    for i,row in enumerate(rows,1):
        require(type(row) is dict and set(row)=={'step','threshold','pre_norm','post_norm','coefficient','gradient_tensors','active_parameter_names','ce_loss'} and type(row['step']) is int and row['step']==i and type(row['threshold']) is float and row['threshold']==1.,'Exact complete clipping fields required')
        require(all(type(row[k]) is float and math.isfinite(row[k]) for k in ('pre_norm','post_norm','coefficient','ce_loss')),'Finite clipping/loss scalars required')
        require(row['active_parameter_names']==names and type(row['gradient_tensors']) is int and row['gradient_tensors']==len(names),'Complete learned gate/control gradient coverage required')
        coefficient=min(1.,1/(row['pre_norm']+1e-6))
        require(row['pre_norm']>=0 and row['post_norm']>=0 and 0<=row['coefficient']<=1 and row['post_norm']<=1.+1e-5 and abs(row['coefficient']-coefficient)<2e-5 and abs(row['post_norm']-row['pre_norm']*row['coefficient'])<=max(1e-5,row['post_norm']*2e-5),'Complete clipping arithmetic differs')
    return {n:step for n in names}


def validate_model_state(model,saved):
    current=model.state_dict()
    require(type(saved) in (dict,type(current)) and saved.keys()==current.keys(),'Exact own model tensor inventory required')
    for n,value in saved.items():
        require(type(value) is torch.Tensor and value.layout==torch.strided and value.is_contiguous() and value.dtype==current[n].dtype and value.shape==current[n].shape and bool(torch.isfinite(value).all()),'Invalid model tensor: '+n)
        if n.endswith(('freq_cos','freq_sin')):
            cos,sin=precompute_freqs_cis(32,128);require(torch.equal(value.cpu(),cos if n.endswith('freq_cos') else sin),'RoPE values differ')


def raw_payload(model,opt,ema,cfg,step,history,samplers):
    model.validate();counts=validate_history(model,history,step,cfg)
    require(opt.completed_steps==step and opt.gradient_steps==counts,'Live optimizer update exposure differs')
    require(ema.completed_steps==step,'Live EMA step differs from raw parent')
    validate_model_state(model,model.state_dict())
    return dict(kind=RAW,architecture=copy.deepcopy(model.signature),cfg=plain(cfg),step=step,model_state=model.state_dict(),
                optimizer_state=opt.state_dict(),ema=ema.state_dict(),training_identity=training_identity(cfg),
                clipping_history=copy.deepcopy(history),sampler_state=samplers,cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state())


def validate_resume(model,opt,ema,payload,cfg):
    model.validate()
    require(type(payload) is dict and set(payload)=={'kind','architecture','cfg','step','model_state','optimizer_state','ema','training_identity','clipping_history','sampler_state','cpu_rng','mps_rng'}
        and payload['kind']==RAW and payload['architecture']==model.signature==architecture(cfg),'Only own complete raw checkpoint can resume')
    metadata_preflight(payload,cfg)
    require(payload['training_identity']==training_identity(cfg),'Complete source/runtime/recipe/data identity differs')
    step=payload['step'];counts=validate_history(model,payload['clipping_history'],step,cfg)
    validate_model_state(model,payload['model_state'])
    state=payload['optimizer_state']
    require(type(state) is dict and set(state)=={'version','independent_decay','optimizers','tokenizer_mla_gate_signature','completed_steps','gradient_steps'}
        and type(state['version']) is int and state['version']==1 and state['independent_decay']==.00006
        and state['tokenizer_mla_gate_signature']==model.signature and state['completed_steps']==step and type(state['completed_steps']) is int
        and state['gradient_steps']==counts and all(type(v) is int for v in state['gradient_steps'].values())
        and type(state['optimizers']) is list and len(state['optimizers'])==2,'Own optimizer/exposure identity differs')
    ratio=multiplier(schedule_contract(cfg.training),step-1) if step else 1.
    for child,stored in zip(opt.optimizers,state['optimizers']):
        fresh=child.state_dict()['param_groups']
        require(type(stored) is dict and set(stored)=={'state','param_groups'} and type(stored['state']) is dict and type(stored['param_groups']) is list and len(stored['param_groups'])==len(fresh),'Child optimizer structure differs')
        expected_slots=set()
        for actual,wanted,live in zip(stored['param_groups'],fresh,child.param_groups):
            require(type(actual) is dict and {k:v for k,v in actual.items() if k!='lr'}=={k:v for k,v in wanted.items() if k!='lr'},'Optimizer group identity differs')
            base=cfg.training.learning_rate*live['width_lr_scale']
            require(type(actual['lr']) is float and math.isfinite(actual['lr']) and actual['lr']==base*ratio,'Scheduled group LR differs')
            index=actual['params'][0];name=actual['parameter_names'][0];updates=counts[name];parameter=live['params'][0]
            require(type(index) is int,'Exact optimizer slot ID required')
            if not updates:continue
            expected_slots.add(index);slot=stored['state'].get(index)
            keys={'momentum_buffer'} if type(child) is torch.optim.Muon else {'step','exp_avg','exp_avg_sq'}
            require(type(slot) is dict and set(slot)==keys,'Active parameter optimizer state differs')
            for key,value in slot.items():
                require(type(value) is torch.Tensor and value.layout==torch.strided and value.is_contiguous() and value.dtype==torch.float32 and bool(torch.isfinite(value).all()),'Finite contiguous FP32 optimizer tensor required')
                if key=='step':require(value.ndim==0 and value.device.type=='cpu' and float(value)==updates,'Adam step must match actual parameter updates')
                else:require(value.shape==parameter.shape,'Optimizer tensor shape differs')
        require(set(stored['state'])==expected_slots,'Only actually updated parameter states may exist')
    names=dict(model.named_parameters());e=payload['ema'];spec=averaging_contract(cfg.training)
    snapshots=0 if step<spec['start_step'] else 1+(step-spec['start_step'])//spec['interval']
    require(type(e) is dict and set(e)=={'version','spec','completed_steps','snapshots','parameter_names','averages'} and type(e['version']) is int and e['version']==1
        and e['spec']==spec and type(e['completed_steps']) is int and e['completed_steps']==step
        and type(e['snapshots']) is int and e['snapshots']==snapshots and e['parameter_names']==list(names),'Learned-parameter-only EMA metadata differs')
    require(type(e['averages']) is dict and set(e['averages'])==({'200'} if snapshots else set()),'EMA half-life inventory differs')
    for panel in e['averages'].values():
        require(type(panel) is dict and panel.keys()==names.keys(),'EMA must cover only exact learned parameters')
        for n,value in panel.items():require(type(value) is torch.Tensor and value.layout==torch.strided and value.is_contiguous() and value.dtype==torch.float32 and value.shape==names[n].shape and bool(torch.isfinite(value).all()),'EMA tensor invalid: '+n)
    return step


def resume(model,opt,ema,payload,cfg,view):
    step=validate_resume(model,opt,ema,payload,cfg)
    require(view.data_identity==identity(cfg) and set(view.generators)=={'train','val'},'Bound destination data/sampler inventory required before mutation')
    load_weights(model,dict(kind='tokenizer_mla_gate_weights_v1',architecture=payload['architecture'],model_state=payload['model_state']))
    opt.load_state_dict(payload['optimizer_state']);ema.load_state_dict(payload['ema'],step)
    restore_samplers(view,payload['sampler_state'])
    torch.set_rng_state(payload['cpu_rng']);torch.mps.set_rng_state(payload['mps_rng'])
    return step,copy.deepcopy(payload['clipping_history'])


def h200_payload(model,ema,cfg,parent):
    model.validate();require(ema.completed_steps==cfg.training.max_iters,'Only final-horizon H200 can be exported')
    validate_parent(parent)
    state=dict(model.state_dict());state.update(ema.averaged_parameters(200))
    return dict(kind=H200,architecture=copy.deepcopy(model.signature),cfg=plain(cfg),step=ema.completed_steps,model_state=state,parent=parent,
                averaged_weights=dict(spec=ema.spec,snapshots=ema.snapshots,parameter_names=[n for n,_ in model.named_parameters()],half_life=200,purpose='inference_only_not_optimizer_resume'))


def load_export(model,payload,cfg):
    require(type(payload) is dict and set(payload)=={'kind','architecture','cfg','step','model_state','parent','averaged_weights'}
        and payload['kind']==H200 and payload['architecture']==model.signature==architecture(cfg)
        and canonical(payload['cfg'])==canonical(plain(cfg)) and payload['step']==cfg.training.max_iters,'Only own H200 inference export required')
    validate_parent(payload['parent'])
    spec=averaging_contract(cfg.training)
    expected=dict(spec=spec,snapshots=1+(cfg.training.max_iters-spec['start_step'])//spec['interval'],parameter_names=[n for n,_ in model.named_parameters()],half_life=200,purpose='inference_only_not_optimizer_resume')
    require(payload['averaged_weights']==expected,'H200 metadata differs')
    validate_model_state(model,payload['model_state'])
    load_weights(model,dict(kind='tokenizer_mla_gate_weights_v1',architecture=model.signature,model_state=payload['model_state']))


def validate_parent(parent):
    require(type(parent) is dict,'Explicit raw parent binding required')
    if set(parent)=={'fixture_kind','raw_state_sha256'}:
        require(parent['fixture_kind']=='in_memory_no_weight_file' and type(parent['raw_state_sha256']) is str and len(parent['raw_state_sha256'])==64 and all(c in '0123456789abcdef' for c in parent['raw_state_sha256']),'Exact in-memory fixture parent digest required')
        return
    require(set(parent) in ({'path','sha256'},{'path','sha256','bytes'}) and type(parent['path']) is str and type(parent['sha256']) is str,'Exact retained raw parent reference required')
    from research.body_allocation_model_v1.contract import ROOT
    path=Path(parent['path']);require(path.is_absolute() and path.is_relative_to(ROOT/'research_runs') and not path.is_symlink() and path.is_file(),'Actual own retained raw parent required')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    require(h.hexdigest()==parent['sha256'] and ('bytes' not in parent or type(parent['bytes']) is int and parent['bytes']==path.stat().st_size),'Raw parent bytes/hash changed')
