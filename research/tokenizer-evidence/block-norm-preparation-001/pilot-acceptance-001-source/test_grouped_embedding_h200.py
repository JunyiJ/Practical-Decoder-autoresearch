"""V2 full-batch generated-text trainer and persisted raw/H200 correctness only."""
import copy
import gc
import io
import json
import os
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from research.block_norm_v1.test_trainer_binding import sha
from research.block_norm_v1.test_grouped_embedding_integration import state_hash
from research.block_norm_v1.runtime.models.gpt import GPT
from research.block_norm_v1.runtime.models.output_bias import tokenizer_no_bias_prior
from research.block_norm_v1.runtime.models.block_norm import CHECKPOINT_KEY
from research.block_norm_v1.runtime.train.train import run_training, _make_loader, _training_contract
from research.block_norm_v1.runtime.train.weight_averaging import export_averaged_checkpoint
from research.block_norm_v1.runtime.utils.checkpoint import read_checkpoint, load_model_weights, checkpoint_loader
from research.block_norm_v1.runtime.attention.cache import build_caches


class CappedWriter:
    def __init__(self,stream,limit):self.stream=stream;self.limit=limit
    def write(self,data):
        if self.tell()+memoryview(data).nbytes>self.limit:raise ValueError('Fixture archive byte limit exceeded')
        return self.stream.write(data)
    def tell(self):return self.stream.tell()
    def flush(self):return self.stream.flush()


def write_json(path,value):
    encoded=(json.dumps(value,indent=2)+'\n').encode()
    assert len(encoded)<2*1024**2
    with path.open('xb') as stream:stream.write(encoded)


def emit(event,**fields):
    path=Path(os.environ['BLOCK_NORM_TRACE_PATH'])
    with path.open('a') as stream:
        stream.write(json.dumps(dict(event=event,**fields),sort_keys=True)+'\n')
        assert stream.tell()<2*1024**2


class FixtureSink:
    def __init__(self,root,cfg,limit):
        self.root=root.resolve();self.cfg=OmegaConf.to_container(cfg,resolve=True);self.limit=limit
        self.checkpoints=[];self.metrics=[]
    def validate_config(self,cfg):assert OmegaConf.to_container(cfg,resolve=True)==self.cfg
    def path(self,path):
        path=Path(path).resolve();assert path.parent==self.root and not path.is_symlink();return path
    def checkpoint(self,path,payload):
        path=self.path(path)
        assert not self.checkpoints and payload['step']==2
        assert 'optimizer_state' in payload and 'training_state' in payload
        with path.open('xb') as stream:torch.save(payload,CappedWriter(stream,self.limit))
        self.checkpoints.append(str(path))
    def metric(self,path,value):
        path=self.path(path);assert value['step']==len(self.metrics)
        encoded=(json.dumps(value,sort_keys=True)+'\n').encode();assert len(encoded)<1024**2
        with path.open('xb' if not self.metrics else 'ab') as stream:stream.write(encoded)
        self.metrics.append(value)
    def summary(self,path,value):write_json(self.path(path),value)


def test_writer_refuses_overflow_before_write():
    stream=io.BytesIO();writer=CappedWriter(stream,4);assert writer.write(b'abc')==3
    with pytest.raises(ValueError,match='limit'):writer.write(b'de')
    assert stream.getvalue()==b'abc' and writer.tell()==3
    assert writer.write(b'd')==1 and stream.getvalue()==b'abcd'


@pytest.fixture(scope='module',params=['rms_incumbent','layer_matched_epsilon'])
def trained(request):
    plan=json.loads(Path(os.environ['BLOCK_NORM_PLAN_PATH']).read_text())
    root=Path(os.environ['BLOCK_NORM_FIXTURE_ROOT'])/request.param;root.mkdir()
    cfg=OmegaConf.create(plan['configs'][request.param]);assert Path(cfg.training.checkpoint_dir)==root
    text=Path(plan['generated_text_source']).read_text();path=root/'generated.txt'
    with path.open('x',encoding='utf-8') as stream:stream.write(text)
    assert sha(path)==plan['generated_text_sha256'] and str(path)==cfg.data.path
    data=_make_loader(cfg,'cpu');assert set(data.splits)==set(data.generators)=={'train','val'}
    assert OmegaConf.to_container(cfg.model.output_bias_prior)==tokenizer_no_bias_prior(data,path)
    # Match the actual paired512 sampler before the trainer creates its fresh loader.
    generator=torch.Generator().manual_seed(cfg.training.seed)
    starts=torch.randint(len(data.train_data)-512,(2,),generator=generator)
    x,y=data.get_batch('train')
    assert x.shape==y.shape==(8,128)
    assert torch.equal(x,torch.stack([data.train_data[i:i+512] for i in starts]).reshape(8,128))
    assert torch.equal(y,torch.stack([data.train_data[i+1:i+513] for i in starts]).reshape(8,128))
    assert torch.equal(generator.get_state(),data.generators['train'].get_state())
    write_json(root/'config.json',OmegaConf.to_container(cfg,resolve=True))
    sink=FixtureSink(root,cfg,plan['storage']['raw_cap_bytes']);result=run_training(cfg,output_sink=sink)
    assert len(sink.checkpoints)==1 and len(sink.metrics)==2
    raw=Path(result['final_checkpoint']);raw_sha=sha(raw);derived=root/'h200.pt'
    emit('trained_raw',arm=request.param,updates=result['completed_steps'],raw_bytes=raw.stat().st_size,raw_sha256=raw_sha,
         final_tokens_processed=result['final_metric']['tokens_processed'],device=result['device'])
    gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
    before_cpu=torch.get_rng_state();before_mps=torch.mps.get_rng_state()
    def writer(output,payload):
        assert output==derived.resolve()
        with output.open('xb') as stream:torch.save(payload,CappedWriter(stream,plan['storage']['h200_cap_bytes']))
    export=export_averaged_checkpoint(raw,derived,200,payload_writer=writer)
    assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_mps,torch.mps.get_rng_state())
    assert sha(raw)==raw_sha;write_json(root/'export.json',export)
    emit('exported_h200',arm=request.param,h200_bytes=derived.stat().st_size,h200_sha256=sha(derived),parent_sha256=raw_sha,rng_preserved=True)
    files={p.name:sha(p) for p in root.iterdir()}
    yield dict(cfg=cfg,root=root,data=data,result=result,raw=raw,derived=derived,files=files,arm=request.param)
    assert files=={p.name:sha(p) for p in root.iterdir()}
    gc.collect();torch.mps.synchronize();torch.mps.empty_cache()


def test_full_batch_actual_trainer_and_exact_h200_transport(trained,record_property):
    cfg=trained['cfg'];data=trained['data'];result=trained['result']
    assert result['device']=='mps' and result['completed_steps']==2 and result['stop_reason']=='max_iters'
    assert result['best_checkpoint'] is None and result['weight_averaging']['snapshots']==2
    assert result['final_metric']['tokens_processed']==2048
    assert json.loads((trained['root']/'summary.json').read_text())==result
    assert [json.loads(s) for s in (trained['root']/'metrics.jsonl').read_text().splitlines()]==result['metrics']
    raw=read_checkpoint(trained['raw']);avg=read_checkpoint(trained['derived']);state=raw['training_state']
    assert state['contract']==_training_contract(cfg,'mps',data)
    assert state['contract']['batch_layout']=='paired_blocks512' and state['contract']['data_sha256']==sha(cfg.data.path)
    assert state['metrics']==result['metrics'] and raw['step']==2
    assert set(state['loader_rng'])=={'train','val'} and state['mps_rng'].numel()==torch.mps.get_rng_state().numel()
    assert raw[CHECKPOINT_KEY]==avg[CHECKPOINT_KEY]
    assert avg['parent_checkpoint']==dict(path=str(trained['raw'].resolve()),sha256=sha(trained['raw']))
    assert avg['tokenizer_contract']==data.tokenizer_contract
    assert 'optimizer_state' not in avg and 'training_state' not in avg
    averages=state['weight_averaging_state']['averages']['200']
    assert set(avg['averaged_weights']['parameter_names'])==set(averages)
    for n,v in avg['model_state'].items():
        assert torch.equal(v,averages[n] if n in averages else raw['model_state'][n])
    gpu=GPT(cfg.model).to('mps');load_model_weights(gpu,raw,for_resume=True)
    load_model_weights(gpu,avg);before=state_hash(gpu)
    with pytest.raises(ValueError,match='cannot resume'):load_model_weights(gpu,avg,for_resume=True)
    assert state_hash(gpu)==before
    for payload in [raw,avg]:
        loader=checkpoint_loader(copy.deepcopy(cfg),payload,cfg.data.path,'cpu')
        assert loader.tokenizer_contract==data.tokenizer_contract
        assert torch.equal(loader.train_data,data.train_data) and torch.equal(loader.val_data,data.val_data)
        assert set(loader.splits)=={'train','val'}
    emit('transport_accepted',arm=trained['arm'],exact_ema_parameters=len(averages),raw_resume_preflight=True,h200_resume_rejected_before_mutation=True)
    record_property('full_batch_updates',2);record_property('tokens_processed',2048)
    record_property('raw_bytes',trained['raw'].stat().st_size);record_property('h200_bytes',trained['derived'].stat().st_size)
    del raw,avg,gpu,state,averages,loader,payload
    gc.collect();torch.mps.synchronize();torch.mps.empty_cache()


@pytest.mark.parametrize('batch',[1,8])
def test_h200_cpu_mps_inference_cached_parity(trained,batch,record_property):
    cfg=trained['cfg'];saved=read_checkpoint(trained['derived'])
    cpu_model=GPT(cfg.model).eval();load_model_weights(cpu_model,saved)
    gpu_model=GPT(cfg.model).to('mps').eval();load_model_weights(gpu_model,saved)
    assert state_hash(cpu_model)==state_hash(gpu_model)
    x=((torch.arange(batch*17)*7)%13).reshape(batch,17)
    cache=build_caches(cfg,'mps',batch_size=batch,dtype=torch.float32)
    with torch.no_grad():
        expected,_=cpu_model(x);actual,_=gpu_model(x.to('mps'));parts=[];start=0
        torch.testing.assert_close(actual.cpu(),expected,atol=3e-4,rtol=3e-4)
        for size in [5,1,4,7]:
            part,_=gpu_model(x[:,start:start+size].to('mps'),cache=cache,start_pos=start)
            parts.append(part);start+=size
        cached=torch.cat(parts,1)
        torch.testing.assert_close(cached,actual,atol=3e-4,rtol=3e-4)
    errors=dict(cpu_mps=float((actual.cpu()-expected).abs().max()),full_cached=float((cached-actual).abs().max()))
    emit('h200_inference',arm=trained['arm'],shape=[batch,17],errors=errors)
    record_property('max_logit_errors',errors)
    del cpu_model,gpu_model,cache,saved,parts,part,cached,expected,actual
    gc.collect();torch.mps.synchronize();torch.mps.empty_cache()


def test_bad_h200_tokenizer_and_existing_export_rejected(trained):
    cfg=trained['cfg'];saved=read_checkpoint(trained['derived'])
    saved['tokenizer_contract']['vocabulary_sha256']='0'*64
    with pytest.raises(ValueError,match='tokenizer identity'):
        checkpoint_loader(copy.deepcopy(cfg),saved,cfg.data.path,'cpu')
    with pytest.raises(ValueError,match='new output path'):
        export_averaged_checkpoint(trained['raw'],trained['derived'],200)
    assert trained['files']=={p.name:sha(p) for p in trained['root'].iterdir()}
