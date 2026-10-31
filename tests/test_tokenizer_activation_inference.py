"""Real bounded inference artifacts/guards with explicit GPU/process doubles.

No real model, corpus, MPS kernel or research checkpoint is loaded. The shared
OS process primitive was tested separately; these tests connect entry, worker,
independent read-back and the complete finite three-seed receipt chain.
"""
import copy
import json
import os
from pathlib import Path

import pytest

from src.research import tokenizer_activation_inference as i
from src.research import tokenizer_activation_inference_entry as e
from src.research import tokenizer_activation_entry as training_entry
from src.research.tokenizer_activation_config import config
from src.research.tokenizer_activation_session import reference
from src.research.tokenizer_activation_timing import CONTRACT,FRESH_SEEDS,slot
from src.research.tokenizer_activation_rules import confirmation_specs
from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
from tests.test_tokenizer_activation_session import ledger,publish
from tests.test_tokenizer_activation_timing import IDS,SHA,SELECTION,measurement


@pytest.fixture
def ready(ledger,monkeypatch):
    inputs=dict(tokenizer={'synthetic':True},training96_sha256=SHA,validation_sha256='b'*64,
                validation_targets=34278,validation_target_bytes=111536)
    by_seed={s:{} for s in FRESH_SEEDS}
    for n,spec in enumerate(confirmation_specs(SELECTION),6):
        stem=f'evidence/quality/{n:02d}'
        rq=publish(ledger,stem+'/request.json',dict(spec=spec,config=config(ledger.root/stem,spec)))
        weights={};writer=CheckpointWriter(ledger.writer,'quality',n,1024**2)
        for role,name in [('raw','final_001200_1.pt'),('h200','h200.pt')]:
            stream=writer.open(name);stream.write(b'tiny synthetic placeholder');stream.publish()
            weights[role]=reference(ledger.root,stem+'/checkpoints/'+name)
        by_seed[spec['seed']][spec['family']]=dict(request=rq,acceptance={'synthetic':True},weights=weights)
    def binding(led,seed):
        for refs in by_seed[seed].values():
            led.verify(refs['request'])
            for ref in refs['weights'].values():led.verify(ref)
        return dict(plan=led.plan,seed=seed,selection=SELECTION,quality_reviews=[],dispatches=[],inputs=inputs,
                    bindings=by_seed[seed],contract=CONTRACT)
    monkeypatch.setattr(e,'bind',lambda *a,**k:None)
    monkeypatch.setattr(e,'idle',lambda *a:{'checked':True})
    monkeypatch.setattr(e,'quality_binding',binding);monkeypatch.setattr(i,'quality_binding',binding)
    monkeypatch.setattr(i,'strict_mps',lambda:None);monkeypatch.setattr(i,'start_memory_guard',lambda:None)
    monkeypatch.setattr(i.torch.backends.mps,'is_available',lambda:False)
    monkeypatch.setattr(i,'load_pair',lambda led,b,recheck:dict(metadata=i.expected_models(led,b),token_ids=IDS))
    monkeypatch.setattr(i,'callbacks',lambda loaded,sha,recheck:(lambda f,b:dict(passed=True,max_absolute_error=0.),measurement))
    monkeypatch.setattr(e,'_group_exists',lambda pid:False)
    launches=[]
    def process(writer,stage,physical,argv,cwd,**kwargs):
        seed=FRESH_SEEDS[[slot(s) for s in FRESH_SEEDS].index(physical)]
        request=reference(ledger.root,f'evidence/inference/{physical}/request.json')
        assert argv==e.argv_for(ledger,seed,request)
        parent=os.getpid();child=parent+100000;launches.append(seed);kwargs['parent_check']()
        with monkeypatch.context() as child_context:
            child_context.setattr(e.os,'getpid',lambda:child)
            child_context.setattr(e.os,'getppid',lambda:parent)
            child_context.setattr(e.os,'getpgid',lambda pid:child)
            e.worker(ledger,seed,request)
        result=dict(status='completed',returncode=0,error=None,pid=child,parent_pid=parent,argv=argv,
            capture_kind='bounded_pipe_v2',log_truncated=False,reserve_bytes=CONTRACT['reserve_bytes'],
            required_free_bytes=kwargs['required_free_bytes'],output_limit_bytes=kwargs['output_limit_bytes'],
            log_limit_bytes=kwargs['log_limit_bytes'],timeout_seconds=kwargs['timeout_seconds'],elapsed_seconds=.1,
            observation=dict(free_bytes=CONTRACT['reserve_bytes']+10000000,output_bytes=10000,log_bytes=0),
            synthetic_process_fixture=True)
        writer.publish_json(f'evidence/inference/{physical}/process/process.json',result)
        return result
    monkeypatch.setattr(e,'run_process',process)
    return ledger,launches,binding


def test_all_three_sealed_children_and_independent_parent_reviews(ready):
    ledger,launches,_=ready
    refs=[e.run_once(ledger) for _ in range(3)]
    assert launches==list(FRESH_SEEDS) and e.accepted_prefix(ledger)==refs
    summary=e.run_once(ledger);value=ledger.document(summary)
    assert value['accepted']==refs and value['adopted'] is False
    assert len(value['report']['cells'])==12
    with pytest.raises(ValueError,match='cannot be repeated'):e.run_once(ledger)
    assert launches==list(FRESH_SEEDS)


def test_historical_process_uses_saved_descriptors_not_new_parent_fds(ready,monkeypatch):
    ledger,_,_=ready;ref=e.run_once(ledger);done=ledger.document(ref)
    with monkeypatch.context() as changed:
        changed.setattr(ledger.writer,'lock_fds',(20001,20002))
        process,_=e.process_review(ledger,3611,done['request'])
        assert process==done['process']


@pytest.mark.parametrize('case',['request','timing','model','guard','extra','weights'])
def test_changed_provenance_or_saved_values_blocks_next_seed(ready,case):
    ledger,launches,binding=ready;e.run_once(ledger)
    base=ledger.root/'evidence/inference/3511'
    if case=='extra':(base/'unknown.json').write_text('{}')
    elif case=='weights':
        ref=binding(ledger,3611)['bindings']['silu']['weights']['h200']
        path=ledger.root/ref['path'];path.write_bytes(path.read_bytes()+b'changed')
    else:
        path=(ledger.root/'control/activation-worker-inference-3511.json' if case=='guard' else
              base/({'request':'request','timing':'result','model':'models'}[case]+'.json'))
        path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):e.run_once(ledger)
    assert launches==[3611]


def test_child_failure_is_permanent_and_cannot_advance(ready,monkeypatch):
    ledger,launches,_=ready
    def fail(*a,**kw):raise RuntimeError('injected inference process failure')
    monkeypatch.setattr(e,'run_process',fail)
    with pytest.raises(RuntimeError,match='injected'):e.run_once(ledger)
    path=ledger.root/'evidence/inference/3511/dispatch-failure.json'
    assert json.loads(path.read_text())['retry_allowed'] is False
    with pytest.raises(ValueError,match='incomplete'):e.run_once(ledger)
    assert not launches


def test_incomplete_quality_cannot_enter_inference(ledger,monkeypatch):
    monkeypatch.setattr(training_entry,'bind',lambda *a,**k:{'inputs':{}})
    with pytest.raises(ValueError,match='Complete accepted quality'):i.quality_binding(ledger,3611)
    assert not (ledger.root/'evidence/inference').exists()


def test_future_or_stop_artifact_prevents_dispatch(ready):
    ledger,launches,_=ready
    publish(ledger,'evidence/inference/3513/attempt.json',{'orphan':True})
    with pytest.raises(ValueError,match='future'):e.run_once(ledger)
    publish(ledger,'STOP',{'reason':'fixture'})
    with pytest.raises(ValueError,match='STOP'):e.run_once(ledger)
    assert not launches


@pytest.mark.parametrize('case',['valid','parent','purpose','half_life','optimizer','config','tokenizer','nan','step'])
def test_actual_cpu_export_values_and_lineage_gate(ready,case):
    import torch
    ledger,_,binding=ready;b=binding(ledger,3611);refs=b['bindings']['silu'];rq=ledger.document(refs['request'])
    payload=dict(cfg=copy.deepcopy(rq['config']),tokenizer_contract=b['inputs']['tokenizer'],step=1200,
        model_math_version=2,parameter_layout_version=2,model_state={'tiny_fixture':torch.ones(4)},
        averaged_weights=dict(purpose='inference_only_not_optimizer_resume',half_life=200),
        parent_checkpoint=dict(path=str(ledger.root/refs['weights']['raw']['path']),sha256=refs['weights']['raw']['sha256']))
    if case=='valid':
        i.check_export(payload,rq,refs,b['inputs'],ledger.root);return
    if case=='parent':payload['parent_checkpoint']['sha256']='0'*64
    elif case=='purpose':payload['averaged_weights']['purpose']='resume'
    elif case=='half_life':payload['averaged_weights']['half_life']=100
    elif case=='optimizer':payload['optimizer_state']={}
    elif case=='config':payload['cfg']['training']['seed']=3612
    elif case=='tokenizer':payload['tokenizer_contract']={'different':True}
    elif case=='nan':payload['model_state']['tiny_fixture'][0]=float('nan')
    elif case=='step':payload['step']=20
    with pytest.raises(ValueError):i.check_export(payload,rq,refs,b['inputs'],ledger.root)
