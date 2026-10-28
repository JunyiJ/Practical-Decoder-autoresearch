"""Opt-in bounded outputs for one fresh v2 training and H200 export sequence.

No launcher or numerical authorization. The caller binds the frozen configuration,
allowance and complete campaign/resource prerequisites before entering this sink.
"""
import copy
from functools import wraps
import json
from pathlib import Path
import re

from omegaconf import OmegaConf

from .protocol import digest, file_digest
from .tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
from .tokenizer_wide_hybrid_scores import _require


def _guarded(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        try:
            _require(self.entered and not self.closed and not self.failed,'Inactive or failed training output sink')
            return method(self,*args,**kwargs)
        except BaseException:
            self.failed=True
            raise
    return call


class TrainingOutputs:
    def __init__(self,writer,stage,slot,allowance_bytes,expected_config):
        self.checkpoints=CheckpointWriter(writer,stage,slot,allowance_bytes)
        self.writer=writer;self.stem=f'evidence/{stage}/{slot:02d}'
        self.trial=writer.root/self.stem
        self.expected=copy.deepcopy(expected_config)
        self.entered=self.closed=self.failed=self.configured=False
        self.raw=self.export=self.metrics_stream=None
        self.metrics=[];self.summary_done=False

    def __enter__(self):
        _require(not self.entered and not self.closed,'Training output sink cannot be reentered')
        self.entered=True
        return self

    @_guarded
    def validate_config(self,cfg):
        _require(not self.configured,'Training output sink cannot restart training')
        actual=OmegaConf.to_container(cfg,resolve=True)
        _require(digest(actual)==digest(self.expected),'Training configuration differs from frozen sink binding')
        t=actual['training'];steps=t['max_iters']
        _require(type(steps) is int and steps>0 and type(t['checkpoint_every']) is int
                 and t['checkpoint_every']==0 and t.get('keep_best_checkpoint') is False
                 and t.get('resume_from') is None and t.get('max_train_seconds') is None,
                 'Only fresh final-only fixed-budget training outputs supported')
        for key,relative in (('checkpoint_dir','checkpoints'),('metrics_file','metrics.jsonl'),
                             ('summary_file','training-summary.json')):
            _require(t.get(key)==str(self.trial/relative),'Training output path differs from declared slot')
        with self.writer._serialized():
            _require(not (self.writer.root/'STOP').exists(),'Campaign STOP prevents training outputs')
            for name in ('training-attempt.json','training-failure.json','training-outputs.json',
                         'metrics.jsonl','training-summary.json'):
                _require(not any((self.trial/(name+suffix)).exists() for suffix in ('','.tmp')),
                         'Existing training output attempt cannot be retried')
            _require(not (self.trial/'checkpoints').exists() or not any((self.trial/'checkpoints').iterdir()),
                     'Fresh training cannot reuse checkpoint output')
        self.writer.publish_json(self.stem+'/training-attempt.json',dict(kind='wide_hybrid_training_outputs_v2',
            status='started',config_sha256=digest(actual),checkpoint_allowance_bytes=self.checkpoints.allowance))
        self.configured=True

    def _path(self,path,expected):
        _require(Path(path)==self.trial/expected and Path(path)==Path(path).resolve(),
                 'Output escaped declared canonical slot')

    @_guarded
    def checkpoint(self,path,payload):
        _require(self.configured,'Training config must be bound before checkpoint output')
        path=Path(path);self._path(path,'checkpoints/'+path.name)
        _require(digest(payload.get('cfg'))==digest(self.expected),'Checkpoint config differs from binding')
        if path.name=='h200.pt':
            _require(self.raw is not None and self.summary_done and self.export is None,'H200 requires completed raw training')
            _require(payload.get('parent_checkpoint')==dict(path=self.raw['path'],sha256=self.raw['sha256'])
                     and file_digest(Path(self.raw['path']))==self.raw['sha256']
                     and payload.get('averaged_weights',{}).get('half_life')==200
                     and 'optimizer_state' not in payload and 'training_state' not in payload,
                     'H200 output must retain exact raw parent lineage and be inference-only')
            self.export=self.checkpoints.save(path.name,payload)
        else:
            steps=self.expected['training']['max_iters']
            _require(self.raw is None and not self.summary_done and
                     re.fullmatch(rf'final_{steps:06d}_[0-9]+\.pt',path.name) is not None
                     and payload.get('step')==steps and 'optimizer_state' in payload
                     and 'training_state' in payload and 'averaged_weights' not in payload,
                     'Only one complete raw endpoint may be written')
            self.raw=self.checkpoints.save(path.name,payload)

    @_guarded
    def metric(self,path,value):
        _require(self.configured and not self.summary_done and self.raw is None,'Metrics must precede final checkpoint')
        self._path(path,'metrics.jsonl')
        step=value.get('step')
        _require(type(step) is int and 0<=step<self.expected['training']['max_iters']
                 and (not self.metrics or step>self.metrics[-1]['step']),'Metric step order differs')
        data=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
        _require(len(data)<=65536,'Individual metric exceeds bounded record size')
        if self.metrics_stream is None:self.metrics_stream=self.writer.open_append(self.stem+'/metrics.jsonl')
        self.metrics_stream.write(data);self.metrics.append(copy.deepcopy(value))

    @_guarded
    def summary(self,path,value):
        _require(self.configured and self.raw is not None and not self.summary_done,'Summary requires completed raw endpoint')
        self._path(path,'training-summary.json')
        _require(value.get('completed_steps')==self.expected['training']['max_iters']
                 and value.get('final_checkpoint')==self.raw['path']
                 and digest(value.get('metrics'))==digest(self.metrics),'Training summary/metrics/endpoint differs')
        if self.metrics_stream is not None:self.metrics_stream.close();self.metrics_stream=None
        self.writer.publish_json(self.stem+'/training-summary.json',value)
        self.summary_done=True

    def __exit__(self,exc_type,exc,tb):
        problem=exc
        try:
            if self.metrics_stream is not None:self.metrics_stream.close()
            if problem is None:
                _require(not self.failed and self.configured and self.summary_done and self.raw is not None
                         and self.export is not None,'Incomplete training/export output sequence')
                self.writer.publish_json(self.stem+'/training-outputs.json',dict(
                    kind='wide_hybrid_training_outputs_v2',status='completed',config_sha256=digest(self.expected),
                    raw=self.raw,h200=self.export,numerical_audits_passed=False,dispatch_authorized=False))
        except BaseException as error:
            problem=problem or error
        finally:self.closed=True
        if problem is not None:
            self.failed=True
            if self.configured:
                try:self.writer.publish_json(self.stem+'/training-failure.json',dict(
                    kind='wide_hybrid_training_outputs_v2',status='failed',error=repr(problem)[:1000],retry_allowed=False))
                except BaseException as report_error:problem.add_note('Training failure publication failed: '+repr(report_error)[:1000])
            if exc is None:raise problem
        return False
