"""Two-update width8 CPU fixtures and synthetic payloads, never research data."""
import copy
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.research.tokenizer_wide_hybrid_training_outputs import TrainingOutputs
from src.research.protocol import file_digest
from src.train import train as training
from src.train.weight_averaging import export_averaged_checkpoint
from src.utils.checkpoint import read_checkpoint,save_checkpoint
from tests.test_research_training import research_cfg
from tests.test_schedule_training import equal_nested
from test_tokenizer_wide_hybrid_metadata_writer import writer


@pytest.fixture
def cfg(writer,research_cfg):
    trial=writer.root/'evidence/pilots/00'
    research_cfg.training.update(checkpoint_dir=str(trial/'checkpoints'),
        metrics_file=str(trial/'metrics.jsonl'),summary_file=str(trial/'training-summary.json'),
        weight_averaging=dict(kind='spaced_tail_ema',total_steps=2,start_step=1,interval=1,half_lives=[200]))
    return research_cfg


def sink(writer,cfg,cap=2*1024**2):
    return TrainingOutputs(writer,'pilots',0,cap,OmegaConf.to_container(cfg,resolve=True))


def test_actual_tiny_training_and_export_match_default_path(writer,cfg,tmp_path,monkeypatch):
    torch.set_num_threads(1)
    baseline=copy.deepcopy(cfg);base=tmp_path/'default'
    baseline.training.update(checkpoint_dir=str(base/'checkpoints'),metrics_file=str(base/'metrics.jsonl'),
                             summary_file=str(base/'training-summary.json'))
    old=training.run_training(baseline)
    old_export=export_averaged_checkpoint(old['final_checkpoint'],base/'h200.pt',200)
    batches=[]
    from src.data.loaders import TinyDataLoader
    original=TinyDataLoader.get_batch
    def batch(loader,split='train'):
        batches.append(split);return original(loader,split)
    monkeypatch.setattr(TinyDataLoader,'get_batch',batch)
    monkeypatch.setattr(training,'_append_metric',lambda *a:pytest.fail('legacy metric writer used'))
    with sink(writer,cfg) as outputs:
        result=training.run_training(copy.deepcopy(cfg),output_sink=outputs)
        raw=Path(result['final_checkpoint']);before=file_digest(raw)
        exported=export_averaged_checkpoint(raw,raw.parent/'h200.pt',200,payload_writer=outputs.checkpoint)
    assert set(batches)=={'train','val'}
    assert file_digest(raw)==before==outputs.raw['sha256']
    assert exported['checkpoint_sha256']==outputs.export['sha256']
    a,b=read_checkpoint(old['final_checkpoint']),read_checkpoint(raw)
    for key in ('model_state','optimizer_state'):equal_nested(a[key],b[key])
    for key in ('torch_rng','loader_rng','weight_averaging_state'):
        equal_nested(a['training_state'][key],b['training_state'][key])
    equal_nested(read_checkpoint(old_export['checkpoint'])['model_state'],read_checkpoint(exported['checkpoint'])['model_state'])
    trial=raw.parent.parent
    assert json.loads((trial/'training-summary.json').read_text())==result
    assert [json.loads(row) for row in (trial/'metrics.jsonl').read_text().splitlines()]==result['metrics']
    receipt=json.loads((trial/'training-outputs.json').read_text())
    assert receipt['status']=='completed' and receipt['numerical_audits_passed'] is False
    assert len(list(raw.parent.glob('*.pt')))==2 and not list(raw.parent.glob('*.tmp'))
    with pytest.raises(ValueError):
        with sink(writer,cfg) as again:training.run_training(cfg,output_sink=again)


@pytest.mark.parametrize('bad',['config','resume','periodic','best','time','metrics_path','summary_path','checkpoint_path'])
def test_invalid_training_sink_rejects_before_loader_or_seed(writer,cfg,monkeypatch,bad):
    expected=copy.deepcopy(cfg)
    if bad=='config':cfg.training.seed+=1
    else:
        if bad=='resume':cfg.training.resume_from='old.pt'
        elif bad=='periodic':cfg.training.checkpoint_every=1
        elif bad=='best':cfg.training.keep_best_checkpoint=True
        elif bad=='time':cfg.training.max_train_seconds=1
        elif bad=='metrics_path':cfg.training.metrics_file='/tmp/escape.jsonl'
        elif bad=='summary_path':cfg.training.summary_file='/tmp/escape.json'
        else:cfg.training.checkpoint_dir='/tmp/escape-checkpoints'
        expected=copy.deepcopy(cfg)
    monkeypatch.setattr(training,'_make_loader',lambda *a:pytest.fail('loader must not run'))
    monkeypatch.setattr(torch,'manual_seed',lambda *a:pytest.fail('seed must not change'))
    with pytest.raises(ValueError):
        with sink(writer,expected) as outputs:training.run_training(cfg,output_sink=outputs)
    assert not (writer.root/'evidence/pilots/00/training-attempt.json').exists()


def raw_payload(cfg):
    return dict(cfg=OmegaConf.to_container(cfg,resolve=True),step=2,
        model_state={'weight':torch.ones(2)},optimizer_state={},training_state={})


@pytest.mark.parametrize('bad',['outside','periodic','wrong_step','config','export_first'])
def test_checkpoint_policy_failure_is_terminal(writer,cfg,bad):
    outputs=sink(writer,cfg)
    with pytest.raises(ValueError):
        with outputs:
            outputs.validate_config(cfg)
            path=outputs.trial/'checkpoints/final_000002_123.pt';payload=raw_payload(cfg)
            if bad=='outside':path=outputs.trial/'other.pt'
            elif bad=='periodic':path=path.with_name('step_000002_123.pt')
            elif bad=='wrong_step':payload['step']=1
            elif bad=='config':payload['cfg']['training']['seed']=123
            else:path=path.with_name('h200.pt')
            outputs.checkpoint(path,payload)
    assert outputs.failed and outputs.closed
    failure=json.loads((outputs.trial/'training-failure.json').read_text())
    assert failure['status']=='failed' and not failure['retry_allowed']


def test_checkpoint_guard_failure_preserves_partial_and_does_not_fallback(writer,cfg):
    outputs=sink(writer,cfg,100)
    with pytest.raises((ValueError,RuntimeError)):
        with outputs:
            outputs.validate_config(cfg)
            outputs.checkpoint(outputs.trial/'checkpoints/final_000002_1.pt',raw_payload(cfg))
    partial=outputs.trial/'checkpoints/final_000002_1.pt.tmp'
    assert partial.exists() and partial.stat().st_size<=100
    assert not partial.with_suffix('').exists()
    with pytest.raises(ValueError):
        with sink(writer,cfg) as again:again.validate_config(cfg)


def test_missing_h200_cannot_complete_output_sequence(writer,cfg):
    outputs=sink(writer,cfg)
    with pytest.raises(ValueError,match='Incomplete'):
        with outputs:
            outputs.validate_config(cfg)
            outputs.checkpoint(outputs.trial/'checkpoints/final_000002_1.pt',raw_payload(cfg))
            outputs.summary(outputs.trial/'training-summary.json',dict(completed_steps=2,
                final_checkpoint=outputs.raw['path'],metrics=[]))
    assert (outputs.trial/'training-failure.json').exists()
    assert not (outputs.trial/'training-outputs.json').exists()


def test_nan_metric_and_wrong_summary_are_not_silent(writer,cfg):
    outputs=sink(writer,cfg)
    with pytest.raises(ValueError):
        with outputs:
            outputs.validate_config(cfg)
            outputs.metric(outputs.trial/'metrics.jsonl',dict(step=0,loss=float('nan')))
    assert outputs.failed


def test_direct_checkpoint_payload_hook_preserves_schema_without_default_write(tmp_path):
    model=torch.nn.Linear(2,1);optimizer=torch.optim.AdamW(model.parameters());seen=[]
    path=tmp_path/'absent-dir/endpoint.pt'
    save_checkpoint(path,model,optimizer=optimizer,step=2,cfg={'test':True},training_state={'rng':'synthetic'},
                    payload_writer=lambda p,v:seen.append((p,v)))
    assert not path.parent.exists()
    assert seen[0][0]==path and seen[0][1]['model_math_version']==2
    assert set(seen[0][1])=={'model_state','optimizer_state','step','cfg','training_state',
                            'model_math_version','parameter_layout_version'}


def test_payload_hook_error_does_not_use_default_saver(tmp_path):
    path=tmp_path/'absent/endpoint.pt'
    def fail(*args):raise RuntimeError('synthetic sink failure')
    with pytest.raises(RuntimeError,match='sink failure'):
        save_checkpoint(path,torch.nn.Linear(1,1),payload_writer=fail)
    assert not path.parent.exists()


@pytest.mark.parametrize('bad',['parent_hash','parent_path','half_life','optimizer_state','training_state','raw_changed'])
def test_export_lineage_failure_retains_raw_without_export(writer,cfg,bad):
    outputs=sink(writer,cfg)
    with pytest.raises(ValueError):
        with outputs:
            outputs.validate_config(cfg)
            raw=outputs.trial/'checkpoints/final_000002_1.pt'
            outputs.checkpoint(raw,raw_payload(cfg))
            outputs.summary(outputs.trial/'training-summary.json',dict(completed_steps=2,
                final_checkpoint=outputs.raw['path'],metrics=[]))
            payload=dict(cfg=OmegaConf.to_container(cfg,resolve=True),model_state={},
                parent_checkpoint=dict(path=str(raw),sha256=outputs.raw['sha256']),
                averaged_weights=dict(half_life=200))
            if bad=='parent_hash':payload['parent_checkpoint']['sha256']='0'*64
            elif bad=='parent_path':payload['parent_checkpoint']['path']=str(raw.parent/'other.pt')
            elif bad=='half_life':payload['averaged_weights']['half_life']=100
            elif bad=='raw_changed':raw.write_bytes(b'changed synthetic raw')
            else:payload[bad]={}
            outputs.checkpoint(raw.parent/'h200.pt',payload)
    assert raw.exists() and not (raw.parent/'h200.pt').exists()
    assert not (outputs.trial/'training-outputs.json').exists()


def test_caught_writer_error_cannot_be_converted_to_completion(writer,cfg):
    outputs=sink(writer,cfg)
    with pytest.raises(ValueError,match='Incomplete'):
        with outputs:
            outputs.validate_config(cfg)
            with pytest.raises(ValueError):
                outputs.metric(outputs.trial/'metrics.jsonl',dict(step=-1))
    assert (outputs.trial/'training-failure.json').exists()


def test_inconsistent_summary_is_terminal(writer,cfg):
    outputs=sink(writer,cfg)
    with pytest.raises(ValueError,match='summary'):
        with outputs:
            outputs.validate_config(cfg)
            outputs.metric(outputs.trial/'metrics.jsonl',dict(step=0,loss=2.0))
            outputs.checkpoint(outputs.trial/'checkpoints/final_000002_1.pt',raw_payload(cfg))
            outputs.summary(outputs.trial/'training-summary.json',dict(completed_steps=2,
                final_checkpoint=outputs.raw['path'],metrics=[]))
    assert not (outputs.trial/'training-summary.json').exists()


@pytest.mark.parametrize('fail',[False,True])
def test_export_hook_streams_hash_and_never_falls_back(tmp_path,monkeypatch,fail):
    from src.train import weight_averaging as averaging
    parent=tmp_path/'parent.pt';parent.write_bytes(b'synthetic parent')
    output=tmp_path/'absent/export.pt'
    payload=dict(parent_checkpoint={'path':str(parent)},averaged_weights={'half_life':200})
    monkeypatch.setattr(averaging,'derived_checkpoint_payload',lambda *a:payload)
    monkeypatch.setattr(Path,'read_bytes',lambda *a:pytest.fail('whole archive hash allocation'))
    def write(path,value):
        assert path==output and value is payload
        if fail:raise RuntimeError('synthetic exporter failure')
        path.parent.mkdir();path.write_bytes(b'synthetic export')
    if fail:
        with pytest.raises(RuntimeError,match='exporter failure'):
            export_averaged_checkpoint(parent,output,200,payload_writer=write)
        assert not output.parent.exists()
    else:
        result=export_averaged_checkpoint(parent,output,200,payload_writer=write)
        assert result['checkpoint_sha256']==file_digest(output)
