import copy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.data.loaders import TinyDataLoader
from src.train.train import _make_loader, _contracts_match, run_training, evaluate_checkpoint
from src.models.gpt import GPT
from src.models.output_bias import validate_output_bias_training_data, output_bias_contract
from src.research.tokenizer_study import make_config, choose_screen, specs, trial_id
from src.utils.checkpoint import read_checkpoint, load_model_weights, checkpoint_loader
from src.utils.generate import generate


@pytest.fixture
def cfg(tmp_path):
    corpus=tmp_path/'corpus.txt'
    corpus.write_text('Hello world! To be or not to be.\n'*100)
    return OmegaConf.create(dict(data=dict(path=str(corpus),tokenizer='gpt2',train_fraction=.8,val_fraction=.1),
        model=dict(dim=8,n_heads=2,n_layers=1,vocab_size=1,block_size=8,dropout=0.,rope=1,
                   norm_type='rmsnorm',attn_type='mha',mlp_type='dense'),
        training=dict(device='cpu',strict_device=True,seed=42,eval_seed=2026,batch_size=2,
                      learning_rate=.001,weight_decay=.01,grad_clip=1.,max_iters=2,
                      eval_every=2,eval_iters=1,checkpoint_dir=str(tmp_path/'checkpoints'),
                      checkpoint_every=0,keep_best_checkpoint=False)))


def test_raw_text_boundaries_and_roundtrip(cfg):
    loader=_make_loader(cfg,'cpu')
    text=Path(cfg.data.path).read_text()
    a,b=int(.8*len(text)),int(.9*len(text))
    assert [loader.decode(loader.splits[k]) for k in ('train','val','test')]==[text[:a],text[a:b],text[b:]]
    assert loader.vocab_size==50257
    assert loader.encode('Hello world!')==[15496,995,0]
    for s in ['你好 🌍 café','<|endoftext|>','\r\n']:
        assert loader.decode(loader.encode(s))==s
    for split in loader.splits:
        x,y=loader.get_batch(split)
        assert torch.equal(x[:,1:],y[:,:-1])


def test_char_default_and_unknown_rejected(cfg):
    del cfg.data.tokenizer
    assert isinstance(_make_loader(cfg,'cpu'),TinyDataLoader)
    cfg.data.tokenizer='typo'
    with pytest.raises(ValueError,match='tokenizer'):_make_loader(cfg,'cpu')


def test_paired_layout_and_private_rng(cfg):
    corpus = Path(cfg.data.path)
    corpus.write_text(corpus.read_text() * 3)
    cfg.model.block_size=128;cfg.training.batch_size=8;cfg.training.batch_layout='paired_blocks512'
    a=_make_loader(cfg,'cpu');b=_make_loader(cfg,'cpu')
    test_rng=a.generators['test'].get_state().clone()
    for _ in range(2):
        x,y=a.get_batch('train');bx,by=b.get_batch('train')
        assert x.shape==(8,128) and torch.equal(x,bx) and torch.equal(y,by)
    a.get_batch('val')
    assert torch.equal(test_rng,a.generators['test'].get_state())
    cfg.training.batch_size=7
    with pytest.raises(ValueError,match='1024'):_make_loader(cfg,'cpu')


def test_actual_training_checkpoint_evaluation_and_generation(cfg):
    loader=_make_loader(cfg,'cpu')
    result=run_training(cfg)
    c=read_checkpoint(result['final_checkpoint'])
    assert c['cfg']['model']['vocab_size']==50257
    assert c['training_state']['contract']['tokenizer']==loader.tokenizer_contract
    checkpoint_loader(OmegaConf.create(c['cfg']), c, cfg.data.path, 'cpu')
    corrupted=copy.deepcopy(c)
    corrupted['training_state']['contract']['tokenizer']['vocabulary_sha256']='0'*64
    with pytest.raises(ValueError,match='tokenizer identity'):
        checkpoint_loader(OmegaConf.create(c['cfg']),corrupted,cfg.data.path,'cpu')
    assert evaluate_checkpoint(cfg,result['final_checkpoint'],'val')['loss']==result['final_metric']['val_loss']
    model=GPT(OmegaConf.create(c['cfg']['model']));load_model_weights(model,c)
    out=generate(model,torch.tensor([loader.encode('Hello')]),3,greedy=True)
    assert out.shape==(1,4) and isinstance(loader.decode(out[0]),str)
    changed=copy.deepcopy(c['training_state']['contract']);changed['tokenizer']['vocabulary_sha256']='0'*64
    assert not _contracts_match(changed,c['training_state']['contract'])
    cfg.data.tokenizer='char'
    with pytest.raises(ValueError,match='matching'):evaluate_checkpoint(cfg,result['final_checkpoint'],'val')


def test_actual_resume_matches_uninterrupted(cfg,tmp_path):
    # Stop after step one while preserving the planned two-step contract.
    full=run_training(cfg)
    cfg.training.checkpoint_dir=str(tmp_path/'partial')
    cfg.training.max_train_seconds=.00000001
    partial=run_training(cfg)
    assert partial['completed_steps']==1
    cfg.training.max_train_seconds=None
    cfg.training.resume_from=partial['final_checkpoint']
    resumed=run_training(cfg)
    a,b=read_checkpoint(full['final_checkpoint']),read_checkpoint(resumed['final_checkpoint'])
    assert resumed['completed_steps']==2
    assert all(torch.equal(a['model_state'][k],b['model_state'][k]) for k in a['model_state'])


def test_averaged_export_preserves_tokenizer(cfg,tmp_path):
    from src.train.weight_averaging import export_averaged_checkpoint
    cfg.training.weight_averaging=dict(kind='spaced_tail_ema',total_steps=2,start_step=1,
                                      interval=1,half_lives=[200])
    result=run_training(cfg)
    export_averaged_checkpoint(result['final_checkpoint'],tmp_path/'h200.pt',200)
    c=read_checkpoint(tmp_path/'h200.pt')
    loader=checkpoint_loader(OmegaConf.create(c['cfg']),c,cfg.data.path,'cpu')
    assert c['tokenizer_contract']==loader.tokenizer_contract
    assert 'optimizer_state' not in c


def test_inherited_model_tokenizer_binding(tmp_path):
    cfg=OmegaConf.create(make_config(tmp_path))
    loader=_make_loader(cfg,'cpu')
    validate_output_bias_training_data(cfg.model,loader,cfg.data.path)
    model=GPT(cfg.model)
    assert model.output_bias is None and model.transformer.h[0].attn.attention_logit_scale==.5
    cfg.model.output_bias_kind='learned_train_unigram'
    with pytest.raises(ValueError,match='absent'):output_bias_contract(cfg.model)
    cfg.model.output_bias_kind='none'
    loader.train_data=loader.train_data.flip(0)
    with pytest.raises(ValueError,match='actual'):validate_output_bias_training_data(cfg.model,loader,cfg.data.path)


def test_frozen_screen_and_confirmation_manifest():
    plan=dict(baseline_lr=.01,learning_rates=[.01,.005,.0025],screen_seeds=[2601,2602],
              confirm_seeds=[2611,2612,2613],widths=[64,256],min_gain=.02)
    records=[]
    for spec in specs(plan):
        loss={.01:5.,.005:4.9,.0025:5.1}[spec['lr']]
        records.append(dict(**spec,h200=dict(mean_ce=loss)))
    selection=choose_screen(records,plan)
    assert selection['selected_lr']==.005 and len(specs(plan,selection))==18
    # Large-width and fresh-seed evidence cannot affect the screen.
    records.append(dict(id=trial_id('confirm',256,2611,.0025),h200=dict(mean_ce=0.)))
    assert choose_screen(records,plan)==selection
    for row in records:
        if row['id']==trial_id('screen',64,2602,.005):row['h200']['mean_ce']=5.01
    assert choose_screen(records,plan)['selected_lr']==.01
    assert len(specs(plan,choose_screen(records,plan)))==12
