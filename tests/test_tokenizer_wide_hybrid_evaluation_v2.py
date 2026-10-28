"""Synthetic endpoint files, fake MPS models and validation; no corpus training."""
import copy
import json
import time
import weakref
from types import SimpleNamespace

import pytest
import torch

from src.research import tokenizer_wide_hybrid_evaluation_v2 as evaluation
from src.research.protocol import digest,file_digest
from src.research.tokenizer_wide_hybrid_timing import token_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer


@pytest.fixture
def case(writer,tmp_path,monkeypatch):
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK',raising=False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    released=[];monkeypatch.setattr(torch.mps,'empty_cache',lambda:released.append(True))
    monkeypatch.setattr(torch.mps,'synchronize',lambda:None)
    corpus=tmp_path/'synthetic.txt';corpus.write_text('synthetic only')
    monkeypatch.setattr(evaluation,'CORPUS_SHA256',file_digest(corpus))
    trial=writer.root/'evidence/pilots/00';(trial/'checkpoints').mkdir(parents=True)
    cfg=dict(model={'fixture':True},data={'path':str(corpus)},
        training={'max_iters':20,'device':'mps','weight_averaging':{'fixture':True}})
    contract=dict(model=cfg['model'],weight_averaging=cfg['training']['weight_averaging'],tokenizer={'synthetic':True})
    request=dict(config=cfg,spec={'stage':'pilot','family':'hybrid','steps':20},result_path=str(trial/'result.json'))
    monkeypatch.setattr(evaluation,'request_options',lambda root:dict(pilots=[request]))
    rq=writer.publish_json('evidence/pilots/00/request.json',request)
    outputs=dict(kind='wide_hybrid_training_outputs_v2',status='completed',config_sha256=digest(cfg))
    for kind,name in [('raw','final_000020_1.pt'),('h200','h200.pt')]:
        payload=dict(cfg=cfg,step=20,parameter_layout_version=2,model_math_version=2,model_state={'weight':torch.ones(1)})
        if kind=='raw':payload.update(training_state={'contract':contract},optimizer_state={})
        else:payload.update(tokenizer_contract=contract['tokenizer'],averaged_weights={'half_life':200},
            parent_checkpoint={k:outputs['raw'][k] for k in ('path','sha256')})
        path=trial/'checkpoints'/name;torch.save(payload,path)
        outputs[kind]=dict(path=str(path),size=path.stat().st_size,sha256=file_digest(path))
    op=writer.publish_json('evidence/pilots/00/training-outputs.json',outputs)
    writer.publish_json('evidence/pilots/00/training-summary.json',dict(completed_steps=20,final_checkpoint=outputs['raw']['path']))
    val=torch.ones(34279,dtype=torch.int64);val[-8702:]=2
    class Loader:
        tokenizer_contract=contract['tokenizer']
        train_data=torch.arange(96);val_data=val
        enc=SimpleNamespace(decode_single_token_bytes=lambda token:b'abc' if token==1 else b'abcd')
        def encode(self,prompt):assert prompt=='ROMEO:\n';return [1,2]
        def decode(self,ids):return 'synthetic sample'
        @property
        def test_data(self):pytest.fail('closed holdout accessed')
    monkeypatch.setattr(evaluation,'checkpoint_loader',lambda *a:Loader())
    refs=[]
    class Model(torch.nn.Module):
        def __init__(self,cfg):
            assert all(r() is None for r in refs),'two models resident'
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));refs.append(weakref.ref(self))
        def to(self,device):assert device=='mps';return self
    monkeypatch.setattr(evaluation,'GPT',Model)
    monkeypatch.setattr(evaluation,'load_model_weights',lambda m,p: m.load_state_dict(p['model_state']))
    monkeypatch.setattr(evaluation,'validate_mps_model',lambda *a:None)
    calls=[]
    def score(model,data,context,batch,block):
        assert data is val and (context,batch,block)==(128,8,2048)
        calls.append('validation')
        counts=[min(2048,34278-i) for i in range(0,34278,2048)]
        return dict(context=128,eval_batch_size=8,block_tokens=2048,target_count=34278,
                    window_count=268,block_target_counts=counts,block_loss_sums=list(map(float,counts)),mean_ce=1.)
    monkeypatch.setattr(evaluation,'score_validation',score)
    original_tensor=torch.tensor
    def tensor(*a,**kw):
        if kw.get('device')=='mps':kw['device']='cpu'
        return original_tensor(*a,**kw)
    monkeypatch.setattr(torch,'tensor',tensor)
    def generate(model,ids,n,**kw):
        assert n==96 and kw==dict(temperature=.8,top_k=40)
        calls.append('sample');return ids
    monkeypatch.setattr(evaluation,'generate',generate)
    args=(writer,'pilots',0,copy.deepcopy(request),file_digest(rq),file_digest(op),contract,token_digest(list(range(96))))
    return SimpleNamespace(args=args,trial=trial,outputs=outputs,refs=refs,calls=calls,released=released,
                           loader=Loader,corpus=corpus)


def run(case):return evaluation.evaluate_and_publish(*case.args,started=time.monotonic()-1)


def test_serial_pair_publishes_bounded_result_and_preserves_weights(case):
    ref=run(case);result=json.loads((case.trial/'result.json').read_text())
    assert ref['sha256']==file_digest(case.trial/'result.json')
    assert result['holdout_consumed'] is False and result['status']=='completed'
    assert case.calls==['validation','validation','sample'] and len(case.released)==2
    assert all(r() is None for r in case.refs)
    for k in ('raw','h200'):
        assert file_digest(__import__('pathlib').Path(case.outputs[k]['path']))==case.outputs[k]['sha256']
        assert result[k]['mean_ce']==1. and result[k]['target_bytes']==111536
    assert 'sample' not in result['raw'] and result['h200']['sample_settings']==evaluation.SAMPLE
    with pytest.raises(ValueError,match='retried'):run(case)


@pytest.mark.parametrize('bad',['fallback','device','request','outputs','summary','checkpoint','corpus','stop','prior_failure','training_failure'])
def test_bad_provenance_rejects_before_model(case,monkeypatch,bad):
    if bad=='fallback':monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','1')
    elif bad=='device':monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    elif bad=='request':(case.trial/'request.json').write_text('{}')
    elif bad=='outputs':(case.trial/'training-outputs.json').write_text('{}')
    elif bad=='summary':(case.trial/'training-summary.json').write_text('{}')
    elif bad=='checkpoint':__import__('pathlib').Path(case.outputs['raw']['path']).write_bytes(b'changed')
    elif bad=='corpus':case.corpus.write_text('changed')
    elif bad=='stop':(case.args[0].root/'STOP').touch()
    elif bad=='prior_failure':(case.trial/'evaluation-failure.json').write_text('{}')
    else:(case.trial/'training-failure.json').write_text('{}')
    with pytest.raises((ValueError,KeyError)):run(case)
    assert not case.refs and not (case.trial/'result.json').exists()


@pytest.mark.parametrize('bad',['tokenizer','tokens','coverage','bytes','score','nonfinite','generation','changed_input','parent'])
def test_runtime_failure_keeps_attempt_and_rejects_retry(case,monkeypatch,bad):
    if bad=='tokenizer':case.loader.tokenizer_contract={'wrong':True}
    elif bad=='tokens':case.loader.train_data=torch.arange(97)[1:]
    elif bad=='coverage':case.loader.val_data=torch.ones(5,dtype=torch.int64)
    elif bad=='bytes':case.loader.enc=SimpleNamespace(decode_single_token_bytes=lambda i:b'x')
    elif bad=='score':monkeypatch.setattr(evaluation,'score_validation',lambda *a:{'block_loss_sums':[float('nan')],'mean_ce':float('nan')})
    elif bad=='nonfinite':
        old=evaluation.read_checkpoint
        def read(*a,**kw):
            p=old(*a,**kw);p['model_state']['weight'].fill_(float('nan'));return p
        monkeypatch.setattr(evaluation,'read_checkpoint',read)
    elif bad=='parent':
        old=evaluation.read_checkpoint
        def read(*a,**kw):
            p=old(*a,**kw)
            if 'parent_checkpoint' in p:p['parent_checkpoint']['sha256']='0'*64
            return p
        monkeypatch.setattr(evaluation,'read_checkpoint',read)
    elif bad=='generation':
        def generate(*a,**kw):raise RuntimeError('synthetic generation failure')
        monkeypatch.setattr(evaluation,'generate',generate)
    else:
        old=evaluation.score_validation
        def score(*a,**kw):
            v=old(*a,**kw);case.corpus.write_text('changed during evaluation');return v
        monkeypatch.setattr(evaluation,'score_validation',score)
    with pytest.raises((ValueError,RuntimeError)):run(case)
    assert (case.trial/'evaluation-attempt.json').exists()
    assert json.loads((case.trial/'evaluation-failure.json').read_text())['retry_allowed'] is False
    assert not (case.trial/'result.json').exists()
    assert all(r() is None for r in case.refs)
    with pytest.raises(ValueError):run(case)


def test_result_writer_cap_failure_retains_partial(case,monkeypatch):
    monkeypatch.setattr(case.loader,'decode',lambda *a:'x'*300000)
    with pytest.raises(ValueError,match='cap'):run(case)
    assert (case.trial/'result.json.tmp').exists() and not (case.trial/'result.json').exists()
    with pytest.raises(ValueError):run(case)
