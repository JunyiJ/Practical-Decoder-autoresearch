"""Actual CPU training/export checks. Run only after the active research worker exits."""
import copy
import hashlib
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.research.averaging_eval import validate_derivation
from src.research.capacity_study import suite_spec
from src.train.train import run_training
from src.train.weight_averaging import export_averaged_checkpoint
from src.utils.checkpoint import read_checkpoint, load_model_weights
from tests.test_capacity_study import plan
from tests.test_schedule_training import equal_nested


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize('family', ['dense4', 'wide4', 'dense6'])
@pytest.mark.parametrize('width', [64, 256])
def test_averaging_online_invariance_resume_and_derived_exports(tmp_path, family, width):
    torch.set_num_threads(1)
    cfg = OmegaConf.create(suite_spec(plan(), family, .005)['candidate']['config'])
    corpus=tmp_path/'synthetic.txt';corpus.write_text('abcdefghijklmno\n'*1024)
    cfg.data.path=str(corpus);cfg.model.update(dim=width,n_heads=width//32)
    cfg.training.update(max_iters=8,device='cpu',strict_device=True,seed=42,
        eval_seed=2026,eval_every=2,eval_iters=2,checkpoint_every=2,
        checkpoint_dir=str(tmp_path/'averaged'),keep_best_checkpoint=False)
    cfg.training.lr_schedule=dict(kind='linear_cooldown',total_steps=8,start_step=4,min_ratio=.1)
    cfg.training.weight_averaging=dict(kind='spaced_tail_ema',total_steps=8,start_step=4,
        interval=2,half_lives=[1,5,20])
    full=run_training(copy.deepcopy(cfg));parent=Path(full['final_checkpoint']);parent_sha=sha(parent)
    original=read_checkpoint(parent)
    assert full['weight_averaging']['snapshots']==3
    assert [r['averaging_snapshots'] for r in full['metrics']]==[0,0,1,2,3]
    # Disabled averaging must leave the actual online trajectory bit-identical.
    baseline=copy.deepcopy(cfg);del baseline.training.weight_averaging
    baseline.training.checkpoint_dir=str(tmp_path/'baseline')
    raw=read_checkpoint(run_training(baseline)['final_checkpoint'])
    for key in ('model_state','optimizer_state'):equal_nested(original[key],raw[key])
    for key in ('loader_rng','torch_rng'):equal_nested(original['training_state'][key],raw['training_state'][key])
    # Resume before initialization, at initialization, and after one tail update.
    for step in (2,4,6):
        middle=list((tmp_path/'averaged').glob(f'step_{step:06d}_*.pt'));assert len(middle)==1
        resumed_cfg=copy.deepcopy(cfg);resumed_cfg.training.resume_from=str(middle[0])
        resumed_cfg.training.checkpoint_dir=str(tmp_path/f'resumed{step}')
        resumed=read_checkpoint(run_training(resumed_cfg)['final_checkpoint'])
        for key in ('model_state','optimizer_state'):equal_nested(original[key],resumed[key])
        for key in ('loader_rng','torch_rng','weight_averaging_state'):
            equal_nested(original['training_state'][key],resumed['training_state'][key])
    model=GPT(OmegaConf.create(original['cfg']['model']))
    params=set(dict(model.named_parameters()))
    for half_life in (1,5,20):
        target=tmp_path/f'ema{half_life}.pt';rng=torch.get_rng_state().clone()
        meta=export_averaged_checkpoint(parent,target,half_life)
        assert torch.equal(rng,torch.get_rng_state()) and sha(parent)==parent_sha
        assert meta['parent_checkpoint']['sha256']==parent_sha and meta['checkpoint_sha256']==sha(target)
        derived=read_checkpoint(target)
        job=dict(checkpoint=str(target),checkpoint_sha256=sha(target),parent_checkpoint=str(parent),parent_sha256=parent_sha,half_life=half_life,completed_steps=8)
        checked=validate_derivation(job);equal_nested(checked,derived)
        assert 'optimizer_state' not in derived and 'training_state' not in derived
        load_model_weights(model,derived)
        panel=original['training_state']['weight_averaging_state']['averages'][str(half_life)]
        assert set(panel)==params
        for name,value in derived['model_state'].items():
            equal_nested(value,panel[name] if name in params else original['model_state'][name])
        with pytest.raises(ValueError,match='cannot resume'):load_model_weights(model,derived,for_resume=True)
        with pytest.raises(ValueError,match='new output'):export_averaged_checkpoint(parent,target,half_life)
        with pytest.raises(ValueError,match='raw current-layout'):export_averaged_checkpoint(target,tmp_path/f'again{half_life}.pt',half_life)
    corrupt=copy.deepcopy(derived);first=next(iter(corrupt['model_state']));corrupt['model_state'][first].flatten()[0]+=1
    corrupt_path=tmp_path/'corrupt.pt';torch.save(corrupt,corrupt_path)
    job.update(checkpoint=str(corrupt_path),checkpoint_sha256=sha(corrupt_path))
    with pytest.raises(ValueError,match='differs from registered'):validate_derivation(job)
    changed=copy.deepcopy(cfg);changed.training.resume_from=str(parent)
    changed.training.weight_averaging.half_lives=[1,5,21]
    with pytest.raises(ValueError,match='Resume requires matching'):run_training(changed)
