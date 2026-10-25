"""Actual CPU horizon-contract tests; deferred until current research exits."""
import copy
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from src.train.train import run_training
from src.train.lr_schedule import multiplier
from src.utils.checkpoint import read_checkpoint
from src.train.weight_averaging import export_averaged_checkpoint
from tests.test_schedule_training import equal_nested
from tests.test_horizon_study import plan
from src.research.horizon_study import suite_spec


@pytest.mark.parametrize('width',[64,256])
@pytest.mark.parametrize('family',['control3000','long6000_stretched','long6000_fixedtail'])
def test_horizon_schedule_resume_and_sampler_prefix(tmp_path,width,family):
    torch.set_num_threads(1);p=plan(200);cfg=OmegaConf.create(suite_spec(p,family,.005)['candidate']['config'])
    total,start=(12,8) if family=='control3000' else (24,16 if family=='long6000_stretched' else 20)
    corpus=tmp_path/'text.txt';corpus.write_text('abcdefghijklmno\n'*1024)
    cfg.data.path=str(corpus);cfg.model.update(dim=width,n_heads=width//32)
    cfg.training.update(max_iters=total,device='cpu',strict_device=True,seed=42,eval_seed=2026,
        eval_every=4,eval_iters=2,checkpoint_every=4,checkpoint_dir=str(tmp_path/'full'),keep_best_checkpoint=False)
    cfg.training.lr_schedule.update(total_steps=total,start_step=start)
    cfg.training.weight_averaging.update(total_steps=total,start_step=start,interval=2)
    result=run_training(copy.deepcopy(cfg));saved=read_checkpoint(result['final_checkpoint']);assert saved['step']==total
    assert result['weight_averaging']['snapshots']==1+(total-start)//2
    assert result['metrics'][-1]['tokens_processed']==total*1024
    for m in result['metrics']:assert m['lr_multiplier']==multiplier(dict(cfg.training.lr_schedule),m['step'])
    mid=list((tmp_path/'full').glob('step_000004_*.pt'));assert len(mid)==1
    resumed=copy.deepcopy(cfg);resumed.training.resume_from=str(mid[0]);resumed.training.checkpoint_dir=str(tmp_path/'resume')
    cp=read_checkpoint(run_training(resumed)['final_checkpoint'])
    for key in ['model_state','optimizer_state']:equal_nested(saved[key],cp[key])
    for key in ['weight_averaging_state','loader_rng','torch_rng']:equal_nested(saved['training_state'][key],cp['training_state'][key])
    # The longer horizon rejects the shorter schedule/averaging contract on resume.
    wrong=copy.deepcopy(resumed);wrong.training.max_iters+=12;wrong.training.lr_schedule.total_steps+=12;wrong.training.weight_averaging.total_steps+=12
    with pytest.raises(ValueError,match='Resume requires matching'):run_training(wrong)
    gen=torch.Generator().manual_seed(42)
    for _ in range(total):torch.randint(int(.8*len(corpus.read_text()))-512,(2,),generator=gen)
    assert torch.equal(saved['training_state']['loader_rng']['train'],gen.get_state())
    assert torch.equal(saved['training_state']['loader_rng']['test'],torch.Generator().manual_seed(2027).get_state())
    derived=export_averaged_checkpoint(result['final_checkpoint'],tmp_path/'returned.pt',200)
    assert derived['averaged_weights']['completed_steps']==total and derived['averaged_weights']['snapshots']==1+(total-start)//2
