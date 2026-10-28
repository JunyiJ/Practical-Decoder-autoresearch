"""Tiny checkpoint integration with mocked MPS/model/cache APIs; no real GPT."""
import copy
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.research import tokenizer_wide_hybrid_pilot_cache as cache
from src.research.protocol import file_digest


@pytest.fixture
def setup(tmp_path, monkeypatch):
    corpus=tmp_path/'corpus';corpus.write_text('Synthetic data; no research corpus')
    monkeypatch.setattr(cache,'CORPUS_SHA256',file_digest(corpus))
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK',raising=False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    events=[];models=[]
    monkeypatch.setattr(torch.mps,'empty_cache',lambda:events.append('empty'))
    original=cache.config
    def config(out,family):
        cfg=original(out,family);cfg['data']['path']=str(corpus);return cfg
    monkeypatch.setattr(cache,'config',config)
    trial=tmp_path/'trial';trial.mkdir();(trial/'checkpoints').mkdir()
    spec=cache.specs()[0];cfg=config(trial,'hybrid');tokenizer={'synthetic':'identity'}
    request=dict(spec=spec,config=cfg,result_path=str(trial/'result.json'))
    (trial/'request.json').write_text(json.dumps(request))
    result={}
    for kind in ('raw','h200'):
        path=trial/'checkpoints'/(kind+'.pt')
        payload=dict(cfg=copy.deepcopy(cfg),model_state={'weight':torch.ones(2)})
        if kind=='raw':payload['training_state']={'contract':{'tokenizer':tokenizer}}
        else:payload['tokenizer_contract']=tokenizer
        torch.save(payload,path);result[kind]={'checkpoint':str(path)}
    (trial/'result.json').write_text(json.dumps(result))
    files={str(p):file_digest(p) for p in trial.rglob('*') if p.is_file()}
    def values(*args,**kwargs):
        assert kwargs=={'kind':'pilot'};events.append('values');return {'files':files}
    monkeypatch.setattr(cache,'review_values',values)
    def loader(*args):
        events.append('loader');return SimpleNamespace(tokenizer_contract=tokenizer,train_data=torch.arange(96))
    monkeypatch.setattr(cache,'checkpoint_loader',loader)
    class FakeGPT(nn.Module):
        def __init__(self,cfg):
            super().__init__()
            assert all(r() is None for r in models), 'previous model must be released'
            self.weight=nn.Parameter(torch.zeros(2));models.append(weakref.ref(self));events.append('new')
        def to(self,device):assert device=='mps';return self
    monkeypatch.setattr(cache,'GPT',FakeGPT)
    def load(model,payload):events.append('load');model.load_state_dict(payload['model_state'])
    monkeypatch.setattr(cache,'load_model_weights',load)
    def check(model,cfg,family,ids,sha):
        assert not model.training and torch.equal(model.weight,torch.ones(2))
        assert ids==list(range(96));events.append('check')
        return dict(parameters=2,parameter_storage_bytes=8,checks=[dict(batch=b,passed=True) for b in (1,8)])
    monkeypatch.setattr(cache,'check_model',check)
    binding=dict(trial=str(trial),request=request,request_sha256=files[str(trial/'request.json')],
        result_sha256=files[str(trial/'result.json')],training_contract={'tokenizer':tokenizer})
    return binding,tokenizer,cache.token_digest(list(range(96))),events,models,files,corpus


def run(setup):return cache.review_caches(*setup[:3])


def test_serial_raw_export_identity_and_release(setup):
    report=run(setup)
    assert list(report['endpoints'])==['raw','h200'] and not report['adopted']
    assert report['status']=='pilot_cache_checks_complete_pending_resource_review'
    assert report['contract']==cache.CACHE_CONTRACT
    assert setup[3]==['values','loader','new','load','check','empty','loader','new','load','check','empty']
    assert all(r() is None for r in setup[4])
    assert all(file_digest(Path(p))==sha for p,sha in setup[5].items())


@pytest.mark.parametrize('bad',['spec','config','tokenizer','token_hash','corpus','rebuild','nonfinite','payload_config','payload_tokenizer'])
def test_corrupt_binding_or_payload(setup,monkeypatch,bad):
    binding,tokenizer,_,events,_,files,corpus=setup
    if bad=='spec':binding['request']['spec']['seed']=3511
    elif bad=='config':binding['request']['config']['model']['rope']=1
    elif bad=='tokenizer':binding['training_contract']['tokenizer']={}
    elif bad=='token_hash':
        with pytest.raises(ValueError,match='tokens differ'):
            cache.review_caches(binding,tokenizer,'0'*64)
        assert 'new' not in events;return
    elif bad=='corpus':corpus.write_text('changed')
    elif bad=='rebuild':
        monkeypatch.setattr(cache,'checkpoint_loader',lambda *args:SimpleNamespace(tokenizer_contract={},train_data=torch.arange(96)))
    else:
        path=Path(binding['trial'])/'checkpoints/raw.pt'
        payload=torch.load(path,weights_only=True)
        if bad=='nonfinite':payload['model_state']['weight'][0]=float('nan')
        elif bad=='payload_config':payload['cfg']['training']['seed']=0
        elif bad=='payload_tokenizer':payload['training_state']['contract']['tokenizer']={}
        torch.save(payload,path);files[str(path)]=file_digest(path)
    with pytest.raises(ValueError):run(setup)
    assert 'new' not in events


@pytest.mark.parametrize('failure_index',[1,2])
def test_cache_failure_stops_and_cleans(setup,monkeypatch,failure_index):
    events=setup[3];calls=[]
    def fail(*args):
        calls.append(1)
        if len(calls)==failure_index:raise RuntimeError('synthetic cache failure')
        return {'checks':[]}
    monkeypatch.setattr(cache,'check_model',fail)
    with pytest.raises(RuntimeError,match='synthetic'):run(setup)
    assert len(calls)==failure_index and events.count('empty')==failure_index


def test_evidence_changed_during_check(setup,monkeypatch):
    def mutate(*args):setup[6].write_text('changed');return {'checks':[]}
    monkeypatch.setattr(cache,'check_model',mutate)
    with pytest.raises(ValueError,match='inputs changed'):run(setup)
    assert setup[3].count('new')==1


@pytest.mark.parametrize('fallback,available',[('1',True),('0',False)])
def test_strict_mps(setup,monkeypatch,fallback,available):
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK',fallback)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:available)
    with pytest.raises(RuntimeError,match='Strict MPS'):run(setup)
    assert not setup[3]


@pytest.mark.parametrize('bad',[None,'parameter_count','state_bytes'])
def test_single_model_cache_capacity_adapter(monkeypatch,bad):
    model=nn.Linear(2,1,bias=False).eval();events=[]
    monkeypatch.setattr(cache,'validate_mps_model',lambda *args:events.append('validate'))
    projection=dict(parameters=3 if bad=='parameter_count' else 2,parameter_bytes=8,
                    cache_bytes_batch1=8 if bad=='state_bytes' else 4)
    monkeypatch.setattr(cache,'_projection',lambda family:projection)
    original=torch.tensor
    monkeypatch.setattr(torch,'tensor',lambda data,**kwargs:original(data,dtype=kwargs['dtype']))
    def build(cfg,device,batch_size,dtype):
        assert device=='mps';return [SimpleNamespace(buffers=lambda:[torch.zeros(batch_size,1)])]
    monkeypatch.setattr(cache,'build_caches',build)
    def check(model,ids,factory):
        assert ids.shape[1]==96 and len(factory())==1;events.append(ids.shape[0])
        return dict(passed=True,max_absolute_error=0.)
    monkeypatch.setattr(cache,'check_kernel',check)
    monkeypatch.setattr(torch.mps,'synchronize',lambda:None)
    monkeypatch.setattr(torch.mps,'current_allocated_memory',lambda:1024)
    monkeypatch.setattr(torch.mps,'driver_allocated_memory',lambda:2048)
    ids=list(range(96));args=(model,None,'hybrid',ids,cache.token_digest(ids))
    if bad:
        with pytest.raises(ValueError,match='differs'):cache.check_model(*args)
    else:
        result=cache.check_model(*args)
        assert events==['validate',1,8]
        assert [r['state_bytes'] for r in result['checks']]==[4,32]
        assert result['parameters']==2 and result['parameter_storage_bytes']==8
