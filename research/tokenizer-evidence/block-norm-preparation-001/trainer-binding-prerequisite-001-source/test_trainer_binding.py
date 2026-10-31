"""Real GPT-2 loader/trainer on generated text; no corpus quality experiment."""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from research.block_norm_v1.test_model_policy import config
from research.block_norm_v1.runtime.models.gpt import GPT
from research.block_norm_v1.runtime.models.output_bias import tokenizer_no_bias_prior, digest
from research.block_norm_v1.runtime.models.block_norm import CHECKPOINT_KEY
from research.block_norm_v1.runtime.train.train import run_training, _make_loader, _training_contract
from research.block_norm_v1.runtime.train.weight_averaging import export_averaged_checkpoint
from research.block_norm_v1.runtime.utils.checkpoint import read_checkpoint, load_model_weights, checkpoint_loader


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


class FixtureSink:
    """Exclusive, bounded fixture outputs; not a production campaign controller."""
    def __init__(self,root,cfg):
        self.root=root.resolve();self.cfg=OmegaConf.to_container(cfg,resolve=True)
        self.checkpoints=[];self.metrics=[]

    def validate_config(self,cfg):
        assert OmegaConf.to_container(cfg,resolve=True)==self.cfg

    def path(self,path):
        path=Path(path).resolve()
        assert path.parent==self.root and not path.is_symlink()
        return path

    def checkpoint(self,path,payload):
        path=self.path(path)
        assert not self.checkpoints and payload['step']==2
        assert 'optimizer_state' in payload and 'training_state' in payload
        with path.open('xb') as stream:torch.save(payload,stream)
        assert path.stat().st_size<600*1024**2
        self.checkpoints.append(str(path))

    def metric(self,path,metric):
        path=self.path(path)
        assert metric['step']==len(self.metrics)
        with path.open('x' if not self.metrics else 'a') as stream:
            stream.write(json.dumps(metric,sort_keys=True)+'\n')
        self.metrics.append(metric)

    def summary(self,path,value):
        with self.path(path).open('x') as stream:json.dump(value,stream,indent=2)


@pytest.fixture(scope='module',params=['rms_incumbent','layer_matched_epsilon'])
def trained(request):
    root=Path(os.environ['BLOCK_NORM_FIXTURE_ROOT'])/request.param
    root.mkdir()
    # Unicode and literal special-token markers exercise the actual ordinary-text API.
    text=''.join(f'Row {i}: café Ω, red blue green <|endoftext|> {i%17}.\n' for i in range(400))
    path=root/'generated.txt';path.write_text(text,encoding='utf-8')
    cfg=config(request.param)
    cfg.data.path=str(path);cfg.data.train_fraction=.8;cfg.data.val_fraction=.2
    cfg.training.device='cpu';cfg.training.strict_device=True
    cfg.training.max_iters=2;cfg.training.batch_size=1;cfg.training.batch_layout='independent'
    cfg.training.seed=372501;cfg.training.eval_seed=372502
    cfg.training.eval_every=1;cfg.training.eval_iters=1
    cfg.training.checkpoint_dir=str(root);cfg.training.checkpoint_every=0
    cfg.training.keep_best_checkpoint=False;cfg.training.resume_from=None
    cfg.training.metrics_file=str(root/'metrics.jsonl');cfg.training.summary_file=str(root/'summary.json')
    cfg.training.max_train_seconds=None
    cfg.training.lr_schedule=dict(kind='linear_cooldown',total_steps=2,start_step=1,min_ratio=.1)
    cfg.training.weight_averaging=dict(kind='spaced_tail_ema',total_steps=2,start_step=1,interval=1,half_lives=[200])
    data=_make_loader(cfg,'cpu')
    assert set(data.splits)=={'train','val'} and cfg.model.block_size==128
    assert set(data.generators)=={'train','val'}
    boundary=int(.8*len(text))
    assert data.split_texts==dict(train=text[:boundary],val=text[boundary:])
    assert torch.equal(data.train_data,torch.tensor(data.encode(text[:boundary])))
    cfg.model.output_bias_prior=tokenizer_no_bias_prior(data,path)
    sink=FixtureSink(root,cfg)
    result=run_training(cfg,output_sink=sink)
    assert len(sink.checkpoints)==1 and len(sink.metrics)==2
    raw=Path(result['final_checkpoint']);before=sha(raw)
    derived=root/'h200.pt'
    def writer(path,payload):
        assert path==derived.resolve()
        with path.open('xb') as stream:torch.save(payload,stream)
        assert path.stat().st_size<160*1024**2
    export=export_averaged_checkpoint(raw,derived,200,payload_writer=writer)
    assert sha(raw)==before
    with (root/'config.json').open('x') as stream:json.dump(OmegaConf.to_container(cfg,resolve=True),stream,indent=2)
    with (root/'export.json').open('x') as stream:json.dump(export,stream,indent=2)
    yield dict(cfg=cfg,root=root,data=data,result=result,raw=raw,derived=derived)
    gc.collect()


def test_actual_trainer_outputs_and_bound_raw_h200(trained,record_property):
    cfg=trained['cfg'];result=trained['result'];data=trained['data']
    assert result['completed_steps']==2 and result['stop_reason']=='max_iters'
    assert result['best_checkpoint'] is None and result['weight_averaging']['snapshots']==2
    assert result['final_metric']['tokens_processed']==256
    assert json.loads((trained['root']/'summary.json').read_text())==result
    assert [json.loads(s) for s in (trained['root']/'metrics.jsonl').read_text().splitlines()]==result['metrics']
    raw=read_checkpoint(trained['raw'],map_location='cpu')
    state=raw['training_state']
    assert state['contract']==_training_contract(cfg,'cpu',data)
    assert state['contract']['data_sha256']==sha(cfg.data.path)
    assert state['contract']['tokenizer']==data.tokenizer_contract
    assert set(state['loader_rng'])=={'train','val'}
    assert not torch.equal(state['loader_rng']['train'],data.generators['train'].get_state())
    assert state['metrics']==result['metrics'] and raw['step']==2
    model=GPT(cfg.model);load_model_weights(model,raw,for_resume=True)
    averaged=read_checkpoint(trained['derived'],map_location='cpu')
    assert raw[CHECKPOINT_KEY]==averaged[CHECKPOINT_KEY]
    assert 'optimizer_state' not in averaged and 'training_state' not in averaged
    assert averaged['tokenizer_contract']==data.tokenizer_contract
    assert averaged['parent_checkpoint']['sha256']==sha(trained['raw'])
    for name,value in state['weight_averaging_state']['averages']['200'].items():
        assert torch.equal(value,averaged['model_state'][name])
    load_model_weights(model,averaged)
    with pytest.raises(ValueError,match='cannot resume'):load_model_weights(model,averaged,for_resume=True)
    for payload in (raw,averaged):
        loader=checkpoint_loader(copy.deepcopy(cfg),payload,cfg.data.path,'cpu')
        assert torch.equal(loader.train_data,data.train_data)
        assert torch.equal(loader.val_data,data.val_data) and 'test' not in loader.splits
    record_property('generated_text_raw_bytes',Path(cfg.data.path).stat().st_size)
    record_property('train_tokens',len(data.train_data));record_property('val_tokens',len(data.val_data))
    record_property('synthetic_optimizer_updates',2)
    record_property('raw_bytes',trained['raw'].stat().st_size)
    record_property('h200_bytes',trained['derived'].stat().st_size)


@pytest.mark.parametrize('field',['corpus_sha256','train_ids_sha256','tokenizer'])
def test_trainer_rejects_self_consistent_but_wrong_data_binding(trained,field):
    cfg=copy.deepcopy(trained['cfg'])
    prior=OmegaConf.to_container(cfg.model.output_bias_prior,resolve=True)
    if field=='tokenizer':prior[field]['vocabulary_sha256']='0'*64
    else:prior[field]='0'*64
    prior['prior_sha256']=digest({k:v for k,v in prior.items() if k!='prior_sha256'})
    cfg.model.output_bias_prior=prior
    before={p.name:sha(p) for p in trained['root'].iterdir()}
    with pytest.raises(ValueError):run_training(cfg)
    assert before=={p.name:sha(p) for p in trained['root'].iterdir()}


def test_saved_tokenizer_rejection_and_resume_contract_rejection(trained):
    cfg=trained['cfg'];averaged=read_checkpoint(trained['derived'],map_location='cpu')
    averaged['tokenizer_contract']['vocabulary_sha256']='0'*64
    with pytest.raises(ValueError,match='tokenizer identity'):
        checkpoint_loader(copy.deepcopy(cfg),averaged,cfg.data.path,'cpu')
    wrong=copy.deepcopy(cfg);wrong.training.resume_from=str(trained['raw']);wrong.training.eval_seed+=1
    before={p.name:sha(p) for p in trained['root'].iterdir()}
    with pytest.raises(ValueError,match='Resume requires matching'):run_training(wrong)
    assert before=={p.name:sha(p) for p in trained['root'].iterdir()}


def test_campaign_batch_layout_loader_only(trained):
    cfg=copy.deepcopy(trained['cfg']);cfg.training.batch_size=8;cfg.training.batch_layout='paired_blocks512'
    data=_make_loader(cfg,'cpu');generator=torch.Generator().manual_seed(cfg.training.seed)
    starts=torch.randint(len(data.train_data)-512,(2,),generator=generator)
    x,y=data.get_batch('train')
    assert x.shape==y.shape==(8,128)
    assert torch.equal(x,torch.stack([data.train_data[i:i+512] for i in starts]).reshape(8,128))
    assert torch.equal(y,torch.stack([data.train_data[i+1:i+513] for i in starts]).reshape(8,128))
    assert torch.equal(generator.get_state(),data.generators['train'].get_state())
