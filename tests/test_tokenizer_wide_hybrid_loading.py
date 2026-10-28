"""Mocked loading integration, tiny tensors, synthetic corpus; no MPS dispatch."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from src.research import tokenizer_wide_hybrid_loading as l
from src.research.protocol import file_digest
from src.research.tokenizer_wide_hybrid_rules import confirmation_specs
from src.research.tokenizer_wide_hybrid_timing import token_digest


@pytest.fixture
def setup(tmp_path,monkeypatch):
    corpus=tmp_path/'synthetic.txt';corpus.write_text('Synthetic identity, not the research corpus')
    monkeypatch.setattr(l,'CORPUS_SHA256',file_digest(corpus))
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK',raising=False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    monkeypatch.setattr(torch.mps,'synchronize',lambda:None)
    original=l.rank_config
    def config(*args,**kwargs):
        cfg=original(*args,**kwargs);cfg['data']['path']=str(corpus);return cfg
    monkeypatch.setattr(l,'rank_config',config)
    selection={f:dict(lr=.00125,divisor=4) for f in l.FAMILIES}
    tokenizer={'synthetic':'no real corpus'};ids=list(range(96));sha=token_digest(ids)
    bindings={};payloads={};events=[]
    for spec in confirmation_specs(selection):
        if spec['seed']!=3511:continue
        family=spec['family'];trial=tmp_path/family;trial.mkdir()
        cfg=config(trial,family,512,3511,.00125,1200,'mps',bottleneck_divisor=4)
        request=dict(spec=spec,config=cfg,result_path=str(trial/'result.json'))
        (trial/'request.json').write_text(json.dumps(request))
        result={}
        for kind in ('raw','h200'):
            path=trial/(kind+'.synthetic');path.write_text(family+kind)
            result[kind]=dict(checkpoint=str(path),checkpoint_sha256=file_digest(path))
        (trial/'result.json').write_text(json.dumps(result))
        bindings[family]=dict(trial=str(trial),request=request,request_sha256=file_digest(trial/'request.json'),
            result_sha256=file_digest(trial/'result.json'),training_contract={'tokenizer':tokenizer})
        payloads[str(trial/'h200.synthetic')]=dict(cfg=cfg,tokenizer_contract=tokenizer,model_state={'weight':torch.ones(2)})
    def review(trial,request,*args):
        events.append(('review',request['spec']['family']))
        result=json.loads((trial/'result.json').read_text())
        return dict(checkpoint_hashes={k:result[k]['checkpoint_sha256'] for k in ('raw','h200')})
    monkeypatch.setattr(l,'review_trial',review)
    def read(path,**kwargs):
        assert kwargs=={'map_location':'cpu'};events.append(('read',str(path)));return payloads[str(path)]
    monkeypatch.setattr(l,'read_checkpoint',read)
    def loader(*args):
        events.append(('loader',None));return SimpleNamespace(tokenizer_contract=tokenizer,train_data=torch.tensor(ids))
    monkeypatch.setattr(l,'checkpoint_loader',loader)
    class FakeGPT(nn.Module):
        def __init__(self,cfg):
            super().__init__();self.weight=nn.Parameter(torch.zeros(2));events.append(('model',None))
        def to(self,device):assert device=='mps';return self
    monkeypatch.setattr(l,'GPT',FakeGPT)
    monkeypatch.setattr(l,'load_model_weights',lambda model,payload:model.load_state_dict(payload['model_state']))
    return bindings,selection,tokenizer,sha,payloads,events,corpus


def run(s):return l.load_selected_exports(s[0],3511,s[1],s[2],s[3])


def test_mocked_three_family_binding_and_loading(setup):
    result=run(setup)
    assert set(result['models'])==set(l.FAMILIES) and result['token_ids']==list(range(96))
    for f in l.FAMILIES:
        assert not result['models'][f].training
        assert torch.equal(result['models'][f].weight,torch.ones(2))
        assert result['metadata'][f]['parameter_storage_bytes']==8
    assert [e[0] for e in setup[5]]==['review','read','loader','model']*3


@pytest.mark.parametrize('bad',['selection','seed','tokenizer','config','corpus','families'])
def test_preflight_rejects_before_tensor_load(setup,bad):
    bindings,selection,tokenizer,sha,payloads,events,corpus=setup
    if bad=='selection':selection['kda']['divisor']=8
    elif bad=='seed':bindings['kda']['request']['spec']['seed']=3512
    elif bad=='tokenizer':bindings['kda']['training_contract']={'tokenizer':{'wrong':True}}
    elif bad=='config':bindings['kda']['request']['config']['training']['max_iters']=1199
    elif bad=='corpus':corpus.write_text('changed')
    else:bindings.pop('kda')
    with pytest.raises(ValueError):run(setup)
    assert not events


@pytest.mark.parametrize('bad',['nan','meta','empty','config','tokenizer'])
def test_loaded_payload_rejected_before_model_creation(setup,bad):
    payload=next(iter(setup[4].values()))
    if bad=='nan':payload['model_state']['weight'][0]=float('nan')
    elif bad=='meta':payload['model_state']['weight']=torch.empty(2,device='meta')
    elif bad=='empty':payload['model_state']={}
    elif bad=='config':payload['cfg']=copy.deepcopy(payload['cfg']);payload['cfg']['training']['seed']=3512
    else:payload['tokenizer_contract']={'wrong':True}
    with pytest.raises(ValueError):run(setup)
    assert all(e[0]!='model' for e in setup[5])


@pytest.mark.parametrize('bad',['tokenizer','ids'])
def test_rebuilt_inputs_verified_before_model_creation(setup,monkeypatch,bad):
    def loader(*args):
        return SimpleNamespace(tokenizer_contract=setup[2] if bad=='ids' else {'wrong':True},
                               train_data=torch.tensor([1]*96))
    monkeypatch.setattr(l,'checkpoint_loader',loader)
    with pytest.raises(ValueError):run(setup)
    assert all(e[0]!='model' for e in setup[5])


def test_export_mutation_during_read_rejected(setup,monkeypatch):
    original=l.read_checkpoint
    def changed(path,**kwargs):
        result=original(path,**kwargs);Path(path).write_text('changed');return result
    monkeypatch.setattr(l,'read_checkpoint',changed)
    with pytest.raises(ValueError,match='while loading'):run(setup)
    assert all(e[0]!='model' for e in setup[5])


@pytest.mark.parametrize('kind',['raw','request','corpus'])
def test_earlier_artifacts_rechecked_after_all_loads(setup,monkeypatch,kind):
    original=l.load_model_weights;calls=[]
    def load(model,payload):
        original(model,payload);calls.append(True)
        if len(calls)==3:
            trial=Path(setup[0]['hybrid']['trial'])
            path=setup[6] if kind=='corpus' else trial/('raw.synthetic' if kind=='raw' else 'request.json')
            path.write_text('changed')
    monkeypatch.setattr(l,'load_model_weights',load)
    with pytest.raises(ValueError,match='changed during triple'):run(setup)
    assert len(calls)==3


def test_metadata_audit_failure_prevents_checkpoint_loading(setup,monkeypatch):
    def failed(*args):raise ValueError('synthetic score audit failure')
    monkeypatch.setattr(l,'review_trial',failed)
    with pytest.raises(ValueError,match='score audit'):run(setup)
    assert not setup[5]


def test_strict_mps_guard(setup,monkeypatch):
    monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','1')
    with pytest.raises(RuntimeError,match='Strict MPS'):run(setup)
    assert not setup[5]
