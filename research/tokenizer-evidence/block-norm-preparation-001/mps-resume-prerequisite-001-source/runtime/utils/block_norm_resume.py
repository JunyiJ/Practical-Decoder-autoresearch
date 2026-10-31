"""Fail-closed new-policy raw-resume preflight; legacy behavior stays unchanged."""
import math
import torch
from omegaconf import OmegaConf
from ..models.block_norm import checkpoint_signature, _same_typed, require


def validate_resume(model, checkpoint):
    if checkpoint_signature(model) is None:
        return
    require('averaged_weights' not in checkpoint, 'Resume requires raw online weights')
    cfg=OmegaConf.create(checkpoint.get('cfg', {}))
    step=checkpoint.get('step')
    require(type(step) is int and 0<=step<=cfg.training.max_iters, 'Resume step differs')
    state=checkpoint.get('training_state')
    require(type(state) is dict, 'Resume training state missing')
    contract=state.get('contract')
    require(type(contract) is dict and _same_typed(contract.get('model'),OmegaConf.to_container(cfg.model,resolve=True)),
            'Resume model contract differs')
    device=next(model.parameters()).device
    require(contract.get('device')==str(device) and contract.get('torch_version')==str(torch.__version__),
            'Resume device or PyTorch differs')
    seconds=state.get('training_seconds')
    require(type(seconds) in (int,float) and math.isfinite(seconds) and seconds>=0
            and type(state.get('metrics')) is list and 'best_metric' in state and 'best_checkpoint' in state,
            'Resume progress state missing or invalid')
    def rng(value):
        require(type(value) is torch.Tensor and value.dtype==torch.uint8 and value.ndim==1,
                'Resume RNG state invalid')
        try:torch.Generator().set_state(value.cpu())
        except RuntimeError as exc:raise ValueError('Resume RNG state invalid') from exc
    rng(state.get('torch_rng'))
    loaders=state.get('loader_rng')
    require(type(loaders) is dict and {'train','val'}<=loaders.keys()<= {'train','val','test'},
            'Resume loader RNG states missing or unknown')
    for value in loaders.values():rng(value)
    if device.type=='mps':
        value=state.get('mps_rng')
        require(type(value) is torch.Tensor and value.dtype==torch.uint8 and value.ndim==1
                and value.numel()==torch.mps.get_rng_state().numel(),
                'Resume MPS RNG missing or incompatible size')
    elif device.type!='cpu':
        raise ValueError('Block norm resume supports only reviewed CPU/MPS backends')
    from ..train.optimizers import build_optimizer, optimizer_options
    from ..train.lr_schedule import LearningRateSchedule
    fresh=build_optimizer(model,cfg.training);schedule=LearningRateSchedule(fresh,cfg.training)
    require(_same_typed(contract.get('optimizer_contract'),optimizer_options(cfg.training)), 'Resume optimizer contract differs')
    require(_same_typed(contract.get('lr_schedule'),schedule.spec), 'Resume schedule contract differs')
    if step: schedule.apply(step-1)
    expected=fresh.state_dict();saved=checkpoint.get('optimizer_state')
    require(type(saved) is dict and saved.keys()==expected.keys(), 'Resume optimizer state missing or malformed')
    require(_same_typed(saved['version'],expected['version'])
            and _same_typed(saved['independent_decay'],expected['independent_decay'])
            and type(saved['optimizers']) is list and len(saved['optimizers'])==len(fresh.optimizers),
            'Resume optimizer composition differs')
    for child, reference, panel in zip(fresh.optimizers,expected['optimizers'],saved['optimizers']):
        require(type(panel) is dict and panel.keys()==reference.keys()
                and _same_typed(panel['param_groups'],reference['param_groups']),
                'Resume optimizer parameter order/group/rates differ')
        slots=panel['state'];require(type(slots) is dict,'Resume optimizer slots invalid')
        ids=[i for g in reference['param_groups'] for i in g['params']]
        params=[p for g in child.param_groups for p in g['params']]
        require(set(slots)==(set(ids) if step else set()), 'Resume optimizer slots incomplete')
        for i,p in zip(ids,params):
            if not step:continue
            values=slots[i]
            keys={'momentum_buffer'} if isinstance(child,torch.optim.Muon) else {'step','exp_avg','exp_avg_sq'}
            require(type(values) is dict and set(values)==keys,'Resume optimizer tensor schema differs')
            for name,value in values.items():
                shape=() if name=='step' else tuple(p.shape)
                require(type(value) is torch.Tensor and tuple(value.shape)==shape and value.dtype==torch.float32
                        and bool(torch.isfinite(value).all()), 'Resume optimizer tensor shape/dtype/finite differs')
                if name=='step':require(value.item()==step,'Resume Adam step differs')
    from ..train.weight_averaging import TailWeightAverages
    tracker=TailWeightAverages(model,cfg.training)
    require(_same_typed(contract.get('weight_averaging'),tracker.spec), 'Resume EMA contract differs')
    tracker.load_state_dict(state.get('weight_averaging_state'),step)
