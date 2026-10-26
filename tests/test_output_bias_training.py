"""Actual CPU trainer gates. Do not execute while a research model worker is live."""
import copy
import math
from unittest.mock import patch
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.models.gpt import GPT
from src.research import warmup_quality_study as study
from src.research.averaging_eval import validate_derivation
from src.research.protocol import file_digest
from src.train.train import run_training
from src.train.weight_averaging import export_averaged_checkpoint
from src.utils.checkpoint import read_checkpoint, load_model_weights
from tests.test_schedule_training import equal_nested
from src.models.output_bias import build_training_prior, POLICY


@pytest.mark.parametrize('kind', ['none','learned_zero','learned_train_unigram'])
@pytest.mark.parametrize('width', [64, 256])
def test_actual_output_bias_resume_optimizer_rng_and_h200(tmp_path,kind,width):
    attention, scale = 'mha', .5
    torch.set_num_threads(1)
    corpus=tmp_path/'synthetic.txt'; corpus.write_text('aaaaaaaaabbbbccdefghijklmno\n'*1024)
    model,training=study.backbone(dict(attention_family=attention,learning_rate=.005,weight_decay=.00006))
    model.update(dim=width,n_heads=width//32,input_width_policy='mha4_input_scale',input_embedding_scale=scale,
        readout_norm_kind='layer_affine',readout_norm_width_policy='mha4_readout_norm',readout_norm_epsilon=1e-5,
        output_bias_kind=kind,output_bias_width_policy=POLICY,output_bias_prior=build_training_prior(corpus.read_bytes(),.8,.1))
    training.update(max_iters=8,device='cpu',strict_device=True,cpu_threads=1,
        seed=9373,eval_seed=2026,eval_every=1,eval_iters=2,checkpoint_every=1,
        checkpoint_dir=str(tmp_path/'full'),keep_best_checkpoint=False)
    training['lr_schedule']=dict(kind='linear_cooldown',total_steps=8,start_step=4,min_ratio=.1)
    training['weight_averaging'].update(total_steps=8,start_step=4,interval=2)
    cfg=OmegaConf.create(dict(data=dict(path=str(corpus),train_fraction=.8,val_fraction=.1),
                             model=model,training=training))
    # Corpus/prior mismatch must fail before GPT construction, including the none arm.
    wrong_data=copy.deepcopy(cfg)
    wrong_data.model.output_bias_prior=build_training_prior(corpus.read_bytes().replace(b'a',b'b',1),.8,.1)
    with patch('src.train.train.GPT',side_effect=AssertionError('Model must not be constructed')):
        with pytest.raises(ValueError,match='actual training corpus'):
            run_training(wrong_data)
    full=run_training(copy.deepcopy(cfg)); parent=Path(full['final_checkpoint'])
    saved=read_checkpoint(parent)
    assert saved['step']==8 and full['metrics'][-1]['tokens_processed']==8192
    assert [m['step'] for m in full['metrics']]==list(range(8))
    expected=[1. if i<=4 else .1 if i==7 else 1-.9*(i-4)/3 for i in range(8)]
    for metric,ratio in zip(full['metrics'],expected):
        assert metric['lr_multiplier']==pytest.approx(ratio,rel=0,abs=1e-15)
        assert metric['base_learning_rate']==.005
        assert metric['scheduled_learning_rate']==pytest.approx(.005*ratio,rel=0,abs=1e-15)
    assert saved['training_state']['contract']['lr_schedule']==dict(cfg.training.lr_schedule)
    assert saved['optimizer_state']['independent_decay']==.00006
    assert all(torch.isfinite(t).all() for t in saved['model_state'].values())
    loaded=GPT(OmegaConf.create(saved['cfg']['model'])); load_model_weights(loaded,saved,for_resume=True)
    assert loaded.input_embedding_scale == scale
    assert saved['training_state']['contract']['model']['input_embedding_scale'] == scale
    names=dict(loaded.named_parameters()); seen=set()
    assert len(loaded.transformer.h)==4
    if attention=='kda':
        assert all(block.attn.dim_alpha==width//4 for block in loaded.transformer.h)
        assert 'dim_alpha' not in saved['cfg']['model']
    for optimizer in saved['optimizer_state']['optimizers']:
        for group in optimizer['param_groups']:
            name,=group['parameter_names']; seen.add(name)
            parameter=names[name]; slot=optimizer['state'][group['params'][0]]
            matrix=name.startswith('transformer.h.') and parameter.ndim==2
            factor=1/math.sqrt(width/64) if matrix else 1.
            assert group['width_lr_scale']==factor and group['weight_decay']==0.
            assert group['lr']==.005*.1*factor
            if matrix:
                assert set(slot)=={'momentum_buffer'}
                assert torch.count_nonzero(slot['momentum_buffer'])>0
            else:
                assert set(slot)=={'step','exp_avg','exp_avg_sq'} and slot['step'].item()==8
                assert torch.count_nonzero(slot['exp_avg_sq'])>0
            for key,value in slot.items():
                assert torch.isfinite(value).all()
                if key!='step': assert value.shape==parameter.shape
    assert seen==set(names)
    averages=saved['training_state']['weight_averaging_state']
    assert averages['snapshots']==3 and set(averages['averages'])=={'200'}
    assert set(averages['parameter_names'])==set(names)

    # Resume from before cooldown, its boundary, and the average tail.
    for completed in (1,4,6):
        middle,=(tmp_path/'full').glob(f'step_{completed:06d}_*.pt')
        cfg_resume=copy.deepcopy(cfg)
        cfg_resume.training.resume_from=str(middle)
        cfg_resume.training.checkpoint_dir=str(tmp_path/f'resume_{completed}')
        resumed=run_training(cfg_resume); restored=read_checkpoint(resumed['final_checkpoint'])
        for key in ('model_state','optimizer_state'): equal_nested(saved[key],restored[key])
        for key in ('loader_rng','torch_rng','weight_averaging_state'):
            equal_nested(saved['training_state'][key],restored['training_state'][key])
        for a,b in zip(full['metrics'],resumed['metrics']):
            for key in ('step','tokens_processed','train_ce_loss','val_ce_loss',
                        'lr_multiplier','scheduled_learning_rate'):
                assert a[key]==b[key]
        assert len(full['metrics'])==len(resumed['metrics'])

    # Averaging is a passive observer of this exact output-bias trajectory.
    plain_cfg=copy.deepcopy(cfg); del plain_cfg.training.weight_averaging
    plain_cfg.training.checkpoint_dir=str(tmp_path/'plain')
    plain=read_checkpoint(run_training(plain_cfg)['final_checkpoint'])
    for key in ('model_state','optimizer_state'): equal_nested(saved[key],plain[key])
    for key in ('loader_rng','torch_rng'): equal_nested(saved['training_state'][key],plain['training_state'][key])
    gen=torch.Generator().manual_seed(9373)
    for _ in range(8): torch.randint(int(.8*len(corpus.read_text()))-512,(2,),generator=gen)
    assert torch.equal(saved['training_state']['loader_rng']['train'],gen.get_state())
    assert set(saved['training_state']['loader_rng'])=={'train','val','test'}
    val_gen=torch.Generator().manual_seed(2026)
    length=len(corpus.read_text());val_length=int(.9*length)-int(.8*length)
    # Every sampled validation resets its private RNG; the final two batches
    # determine the saved state regardless of earlier validation calls.
    for _ in range(2): torch.randint(val_length-128,(8,),generator=val_gen)
    assert torch.equal(saved['training_state']['loader_rng']['val'],val_gen.get_state())
    assert torch.equal(saved['training_state']['loader_rng']['test'],torch.Generator().manual_seed(2027).get_state())

    # Resume binds schedule and the exact final-normalization semantics.
    for field,value in [('start_step',5),('min_ratio',.2)]:
        wrong=copy.deepcopy(cfg_resume); wrong.training.lr_schedule[field]=value
        with pytest.raises(ValueError,match='Resume requires matching'): run_training(wrong)
    assert saved['cfg']['model']['output_bias_kind']==kind
    assert saved['training_state']['contract']['model']['output_bias_kind']==kind
    assert ('output_bias' in names)==(kind!='none')

    parent_hash=file_digest(parent); derived_path=tmp_path/'H200.pt'
    export_averaged_checkpoint(parent,derived_path,200)
    derived=validate_derivation(dict(checkpoint=str(derived_path),
        checkpoint_sha256=file_digest(derived_path),parent_checkpoint=str(parent),
        parent_sha256=parent_hash,half_life=200,completed_steps=8))
    assert file_digest(parent)==parent_hash
    assert 'optimizer_state' not in derived and 'training_state' not in derived
    for name,value in derived['model_state'].items():
        expected_value=averages['averages']['200'][name] if name in names else saved['model_state'][name]
        assert torch.equal(value,expected_value)
    load_model_weights(loaded,derived)
    with pytest.raises(ValueError,match='cannot resume'): load_model_weights(loaded,derived,for_resume=True)

    # Zero and unigram share shapes but cannot exchange initialization semantics.
    for other in (x for x in ('none','learned_zero','learned_train_unigram') if x != kind):
        wrong=copy.deepcopy(cfg_resume); wrong.model.output_bias_kind=other
        with pytest.raises(ValueError,match='Resume requires matching'): run_training(wrong)
        changed=GPT(wrong.model)
        for ckpt in (saved, derived):
            with pytest.raises(ValueError,match='output-bias initialization'):
                load_model_weights(changed,ckpt)
