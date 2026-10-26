"""Actual CPU training/resume gates; defer execution until active research exits."""
import copy
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.models.gpt import GPT
from src.research.decay_quality_study import suite_spec, DECAYS
from src.research.averaging_eval import validate_derivation
from src.research.protocol import file_digest
from src.train.train import run_training
from src.train.weight_averaging import export_averaged_checkpoint
from src.utils.checkpoint import read_checkpoint, load_model_weights
from tests.test_decay_quality_study import plan
from tests.test_schedule_training import equal_nested


@pytest.mark.parametrize('decay_family,family',[('anchor','none'),('weak','rms'),('strong','layer')])
@pytest.mark.parametrize('width',[64,256])
def test_decay_training_resume_optimizer_and_averaging_contract(tmp_path,decay_family,family,width):
    torch.set_num_threads(1)
    attention_gate='none'
    cfg=OmegaConf.create(suite_spec(plan(common_norm=family),decay_family,.005)['candidate']['config'])
    corpus=tmp_path/'synthetic.txt';corpus.write_text('abcdefghijklmno\n'*1024)
    cfg.data.path=str(corpus);cfg.model.update(dim=width,n_heads=width//32)
    cfg.training.update(max_iters=12,device='cpu',strict_device=True,seed=61,eval_seed=2026,
        eval_every=4,eval_iters=2,checkpoint_every=4,checkpoint_dir=str(tmp_path/'full'),keep_best_checkpoint=False)
    cfg.training.lr_schedule.update(total_steps=12,start_step=8)
    cfg.training.weight_averaging.update(total_steps=12,start_step=8,interval=2)
    result=run_training(copy.deepcopy(cfg));parent=Path(result['final_checkpoint']);saved=read_checkpoint(parent)
    assert saved['step']==12 and result['metrics'][-1]['tokens_processed']==12288
    model=GPT(OmegaConf.create(saved['cfg']['model']));load_model_weights(model,saved)
    names=dict(model.named_parameters());norms={n for n in names if '.mlp.hidden_norm.' in n}
    assert len(norms)==(0 if family=='none' else 4 if family=='rms' else 8)
    assert all(names[n].shape==(21*width//8,) for n in norms)
    assert not any(n.endswith(('.q_norm.weight','.k_norm.weight')) for n in names)
    gates={n for n in names if n.endswith('output_gate.weight')}
    assert len(gates)==(4 if attention_gate=='sigmoid_headwise_zero' else 0)
    assert all(names[n].shape==(width//32,width) and torch.count_nonzero(names[n])>0 for n in gates)
    assert all(torch.isfinite(t).all() for t in saved['model_state'].values())
    assert saved['optimizer_state']['independent_decay']==DECAYS[decay_family]
    assert saved['cfg']['training']['weight_decay']==DECAYS[decay_family]
    seen=set()
    for opt in saved['optimizer_state']['optimizers']:
        for group in opt['param_groups']:
            assert group['weight_decay']==0.
            name,=group['parameter_names'];seen.add(name);slot=opt['state'][group['params'][0]]
            matrix=name.startswith('transformer.h.') and names[name].ndim==2
            scale=1/math.sqrt(width/64) if matrix else 1.
            assert group['width_lr_scale']==scale and group['lr']==.005*.1*scale
            if matrix:
                assert set(slot)=={'momentum_buffer'}
                if name in gates:
                    assert torch.count_nonzero(slot['momentum_buffer'])>0
            else:
                assert set(slot)=={'step','exp_avg','exp_avg_sq'} and slot['step'].item()==12
            for key,value in slot.items():
                assert torch.isfinite(value).all()
                if key!='step':assert value.shape==names[name].shape
            if name in norms:
                assert group['width_lr_scale']==1 and not matrix
                initial=torch.zeros(21*width//8) if name.endswith('.bias') else torch.ones(21*width//8)
                assert not torch.equal(names[name],initial) and torch.count_nonzero(slot['exp_avg_sq'])>0
    assert seen==set(names)
    average_state=saved['training_state']['weight_averaging_state']
    assert average_state['snapshots']==3 and set(average_state['averages'])=={'200'}
    assert set(average_state['parameter_names'])==set(names)
    # Averaging does not feed back into the online trajectory or private RNG.
    disabled=copy.deepcopy(cfg);del disabled.training.weight_averaging
    disabled.training.checkpoint_dir=str(tmp_path/'no_average')
    plain=read_checkpoint(run_training(disabled)['final_checkpoint'])
    for key in ('model_state','optimizer_state'):equal_nested(saved[key],plain[key])
    for key in ('loader_rng','torch_rng'):equal_nested(saved['training_state'][key],plain['training_state'][key])
    for step in (4,8):
        middle=list((tmp_path/'full').glob(f'step_{step:06d}_*.pt'));assert len(middle)==1
        resumed=copy.deepcopy(cfg);resumed.training.resume_from=str(middle[0]);resumed.training.checkpoint_dir=str(tmp_path/f'resume{step}')
        restored=read_checkpoint(run_training(resumed)['final_checkpoint'])
        for key in ('model_state','optimizer_state'):equal_nested(saved[key],restored[key])
        for key in ('loader_rng','torch_rng','weight_averaging_state'):equal_nested(saved['training_state'][key],restored['training_state'][key])
    # A changed coefficient cannot silently continue the online optimizer.
    wrong=copy.deepcopy(cfg);wrong.training.resume_from=str(parent);wrong.training.checkpoint_dir=str(tmp_path/'reject')
    wrong.training.weight_decay=DECAYS['strong' if decay_family!='strong' else 'weak']
    with pytest.raises(ValueError,match='Resume requires matching'):run_training(wrong)
    gen=torch.Generator().manual_seed(61)
    for _ in range(12):torch.randint(int(.8*len(corpus.read_text()))-512,(2,),generator=gen)
    assert torch.equal(saved['training_state']['loader_rng']['train'],gen.get_state())
    assert torch.equal(saved['training_state']['loader_rng']['test'],torch.Generator().manual_seed(2027).get_state())
    parent_sha=file_digest(parent);derived_path=tmp_path/'average.pt'
    export_averaged_checkpoint(parent,derived_path,200)
    derived=validate_derivation(dict(checkpoint=str(derived_path),checkpoint_sha256=file_digest(derived_path),
        parent_checkpoint=str(parent),parent_sha256=parent_sha,half_life=200,completed_steps=12))
    assert file_digest(parent)==parent_sha
    for name,value in derived['model_state'].items():
        target=average_state['averages']['200'][name] if name in names else saved['model_state'][name]
        assert torch.equal(value,target)
    load_model_weights(model,derived)
    with pytest.raises(ValueError,match='cannot resume'):load_model_weights(model,derived,for_resume=True)
