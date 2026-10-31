"""Worker orchestration faults use explicit test doubles; no MPS or corpus work."""
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.research import tokenizer_activation_worker as w
from src.research.tokenizer_activation_session import reference
from tests.test_tokenizer_activation_session import ledger


def test_activation_resource_adapter_exact_profiles(ledger):
    ref=ledger.reserve()
    for stage,count,seconds in [('pilots',2,600),('quality',12,7200)]:
        resource=w.ActivationResources(ledger.writer,stage,count-1,ledger.plan['sha256'],ref['sha256'])
        assert resource.KIND=='activation_trial_resources_v1' and resource.LIMITS['timeout_seconds']==seconds
        assert resource.sample_cap==seconds+3
        with pytest.raises(ValueError):w.ActivationResources(ledger.writer,stage,count,ledger.plan['sha256'],ref['sha256'])


def test_worker_requires_strict_mps_before_output(ledger,monkeypatch):
    ref=ledger.reserve();monkeypatch.setenv('PYTORCH_ENABLE_MPS_FALLBACK','1')
    with pytest.raises(ValueError,match='Strict MPS'):w.produce(ledger,ref,{},lambda:None)
    assert not (ledger.root/'evidence/pilots/00/worker-attempt.json').exists()


@pytest.mark.parametrize('fail',[False,True])
def test_worker_body_bounded_publication_and_no_automatic_acceptance(ledger,monkeypatch,fail):
    ref=ledger.reserve();pending=ledger.inspect()['pending'];trial=ledger.root/pending['stem'];cfg=pending['request']['config']
    calls=[]
    class Resources:
        def __init__(self,*args):calls.append('resource_init')
        def __enter__(self):calls.append('resource_enter');return self
        def mark_success(self,sha):calls.append('resource_success');assert sha==w.file_digest(trial/'result.json')
        def __exit__(self,*args):calls.append('resource_exit');return False
    monkeypatch.delenv('PYTORCH_ENABLE_MPS_FALLBACK',raising=False)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:True)
    monkeypatch.setattr(torch.mps,'empty_cache',lambda:None)
    monkeypatch.setattr(w,'start_memory_guard',lambda:calls.append('memory_guard'))
    monkeypatch.setattr(w,'ActivationResources',Resources)
    monkeypatch.setattr(w,'input_contract',lambda *a:{'synthetic':True})
    def train(actual,output_sink):
        calls.append('train');output_sink.validate_config(actual)
        if fail:raise RuntimeError('synthetic injected training failure')
        raw=trial/'checkpoints/final_000020_1.pt'
        metric=dict(step=0,loss=5.)
        output_sink.metric(trial/'metrics.jsonl',metric)
        output_sink.checkpoint(raw,dict(cfg=cfg,step=20,model_state={'tiny':torch.arange(3.)},optimizer_state={},training_state={}))
        summary=dict(completed_steps=20,final_checkpoint=str(raw),metrics=[metric])
        output_sink.summary(trial/'training-summary.json',summary);return summary
    def export(raw,path,half,payload_writer):
        calls.append('export')
        payload_writer(path,dict(cfg=cfg,model_state={'tiny':torch.arange(3.)},
            parent_checkpoint=dict(path=str(raw),sha256=w.file_digest(raw)),averaged_weights=dict(half_life=200)))
    monkeypatch.setattr(w,'run_training',train);monkeypatch.setattr(w,'export_averaged_checkpoint',export)
    monkeypatch.setattr(w,'endpoint',lambda role,*args:(calls.append(role) or dict(mean_ce=5.),[]))
    monkeypatch.setattr(w,'review_pair',lambda *a:(calls.append('tensor_review') or dict(synthetic=True),{}))
    monkeypatch.setattr(w,'check_measurements',lambda *a:calls.append('metadata_review'))
    if fail:
        with pytest.raises(RuntimeError,match='synthetic injected'):w.produce(ledger,ref,{},lambda:None)
        failure=json.loads((trial/'worker-failure.json').read_text())
        assert failure['retry_allowed'] is False
        assert 'export' not in calls and 'resource_success' not in calls
        with pytest.raises(ValueError,match='Existing producer'):w.produce(ledger,ref,{},lambda:None)
    else:
        result=w.produce(ledger,ref,{},lambda:None)
        assert result==reference(ledger.root,'evidence/pilots/00/worker-outputs.json')
        assert calls==['memory_guard','resource_init','resource_enter','train','export','raw','h200',
                       'tensor_review','metadata_review','resource_success','resource_exit']
        saved=json.loads((trial/'result.json').read_text())
        assert saved['status']=='completed' and 'h200_ce' not in saved['record']
        assert saved['holdout_consumed'] is False
    assert ledger.inspect()['next']['kind']=='pending'
