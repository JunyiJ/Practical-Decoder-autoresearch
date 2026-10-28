"""Prospective serial inference integration. No prepare/freeze/adoption command.

No real complete quality/tensor prerequisites currently exist for this module.
Only mocked integration tests may run during preparation.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys

import torch

from .protocol import ROOT, digest, file_digest
from .cpu_prerequisite_guard import CpuPrerequisiteGuard, violation
from .memory_guard import start_memory_guard
from .tokenizer_serial_process import run_process, snapshot
from .tokenizer_wide_hybrid_driver import session_lock, PROJECTION
from .tokenizer_wide_hybrid_inference_gate import bind_inference, recheck, CONTRACT
from .tokenizer_wide_hybrid_loading import load_selected_exports
from .tokenizer_wide_hybrid_measurements import mps_callbacks
from .tokenizer_wide_hybrid_timing import measure_triple, SEEDS, FAMILIES
from .tokenizer_wide_hybrid_timing_review import review_file, review_result, summarize
from .tokenizer_wide_hybrid_scores import _json
from .tokenizer_wide_hybrid_simulation import _publish

LIMIT = CONTRACT['output_limit_bytes']
RESERVE = CONTRACT['reserve_bytes']


def _budget(output, initial=False):
    if shutil.disk_usage(output).free < RESERVE+(LIMIT if initial else 0):
        raise RuntimeError('Inference disk reserve/allocation unavailable')
    if snapshot(output)['output_bytes'] > LIMIT:
        raise RuntimeError('Inference panel output cap exceeded')


def _locks(fds, study):
    if len(fds)!=2 or len(set(fds))!=2 or any(type(fd) is not int or fd<=2 for fd in fds):
        raise ValueError('Two inherited worker locks required')
    paths=(ROOT/'research_state/tokenizer-worker-lock/.lock',study/'.lock')
    for fd,path in zip(fds,paths):
        actual,expected=os.fstat(fd),path.stat()
        if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
            raise ValueError('Inherited lock file identity differs')
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)


def _models(metadata,bound,seed):
    rows=json.loads(PROJECTION.read_text())['records']
    dimensions={(r['family'],r['bottleneck_divisor']):r for r in rows if r['width']==512}
    expected={}
    for family in FAMILIES:
        b=bound['bindings'][family];cfg=b['request']['config'];spec=b['request']['spec']
        result=_json(Path(b['trial'])/'result.json',b['result_sha256'])
        row=dimensions[family,spec['divisor']]
        expected[family]=dict(checkpoint=str(Path(result['h200']['checkpoint']).resolve()),
            sha256=result['h200']['checkpoint_sha256'],seed=seed,family=family,
            divisor=spec['divisor'],lr=spec['lr'],config_sha256=digest(cfg),
            tokenizer_sha256=digest(bound['tokenizer']),parameters=row['parameters'],
            parameter_storage_bytes=row['parameter_bytes'])
    if digest(metadata)!=digest(expected):
        raise ValueError('Loaded-model metadata does not match selected artifacts/geometry')


def worker(request_path,request_sha,seed,output,lock_fds):
    request_path,output=Path(request_path).resolve(),Path(output).resolve()
    request=_json(request_path,request_sha);study=Path(request['quality_run']).resolve()
    if (type(seed) is not int or seed not in SEEDS or output!=study/'inference'/f'seed-{seed}'
            or request_path!=study/'inference'/'request.json' or not output.is_dir()):
        raise ValueError('Worker path/seed differs from finite inference layout')
    _locks(lock_fds,study)
    if (output/'worker-state.json').exists():
        raise FileExistsError('Existing worker attempt cannot be retried')
    _budget(output.parent)
    state=dict(status='verifying',seed=seed,request_sha256=request_sha,files={},adopted=False)
    _publish(output/'worker-state.json',state,exclusive=True)
    try:
        with CpuPrerequisiteGuard(output/'guard.json',output,
                rss_limit_bytes=CONTRACT['rss_limit_bytes'],reserve_bytes=RESERVE,
                timeout_seconds=CONTRACT['job_seconds']):
            start_memory_guard();torch.set_num_threads(4)
            bound=bind_inference(request,seed)
            state['status']='loading';_publish(output/'worker-state.json',state)
            loaded=load_selected_exports(bound['bindings'],seed,bound['selection'],
                                         bound['tokenizer'],bound['token_sha256'])
            _models(loaded['metadata'],bound,seed)
            check,measure=mps_callbacks(loaded['models'],loaded['configs'],loaded['token_ids'],bound['token_sha256'])
            state['status']='measuring';_publish(output/'worker-state.json',state)
            result=measure_triple(seed,bound['selection'],loaded['token_ids'],bound['token_sha256'],check=check,measure=measure)
            review_result(result,seed,bound['selection'],bound['token_sha256'])
            again=bind_inference(_json(request_path,request_sha),seed)
            if digest(again['files'])!=digest(bound['files']):
                raise ValueError('Inference provenance changed during measurement')
            _budget(output.parent)
            for name,value in (('result.json',result),('models.json',loaded['metadata'])):
                _publish(output/name,value,exclusive=True);state['files'][str(output/name)]=file_digest(output/name)
            state['status']='completed';_publish(output/'worker-state.json',state)
            _budget(output.parent)
        return state
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));_publish(output/'worker-state.json',state)
        raise


def run(request_path):
    request_path=Path(request_path).resolve();request_sha=file_digest(request_path)
    request=_json(request_path,request_sha);study=Path(request['quality_run']).resolve()
    output=study/'inference'
    global_lock=ROOT/'research_state/tokenizer-worker-lock';global_lock.mkdir(exist_ok=True)
    with session_lock(global_lock) as global_fd,session_lock(study) as study_fd:
        if output.exists():raise FileExistsError('Existing inference attempt requires review; no automatic resume')
        first=bind_inference(request,SEEDS[0])
        if shutil.disk_usage(study).free<RESERVE+LIMIT:raise RuntimeError('Full inference allocation does not fit')
        output.mkdir();_publish(output/'request.json',request,exclusive=True)
        local_sha=file_digest(output/'request.json')
        state=dict(status='ready',active=None,request_sha256=local_sha,reviews=[],adopted=False)
        _publish(output/'state.json',state,exclusive=True)
        results=[]
        try:
            for seed in SEEDS:
                if (study/'STOP').exists() or (output/'STOP').exists():
                    state['status']='paused';_publish(output/'state.json',state);return state
                for receipt in state['reviews']:recheck(receipt['files'])
                bound=bind_inference(_json(output/'request.json',local_sha),seed)
                if digest(bound['files'])!=digest(first['files']):raise ValueError('Inference panel provenance changed')
                _budget(output,initial=True)
                child=output/f'seed-{seed}';child.mkdir()
                state.update(status='running',active=seed);_publish(output/'state.json',state)
                argv=[sys.executable,'-m','src.research.tokenizer_wide_hybrid_inference','worker',
                    '--request',str(output/'request.json'),'--request-sha256',local_sha,
                    '--seed',str(seed),'--output',str(child),'--global-lock-fd',str(global_fd),
                    '--study-lock-fd',str(study_fd)]
                run_process(argv,ROOT,child/'process',(global_fd,study_fd),required_free_bytes=RESERVE+LIMIT,
                    output_limit_bytes=LIMIT,log_limit_bytes=LIMIT,timeout_seconds=CONTRACT['job_seconds'],watch_root=output)
                terminal=_json(child/'worker-state.json',file_digest(child/'worker-state.json'))
                if (terminal.get('status')!='completed' or terminal.get('seed')!=seed
                        or terminal.get('request_sha256')!=local_sha or terminal.get('adopted') is not False):
                    raise ValueError('Worker did not complete reviewed inference')
                expected_paths={str(child/'result.json'),str(child/'models.json')}
                if set(terminal['files'])!=expected_paths:raise ValueError('Missing worker artifact receipts')
                recheck(terminal['files'])
                guard=_json(child/'guard.json',file_digest(child/'guard.json'))
                if (guard.get('status')!='finished' or guard.get('reason') is not None
                        or guard.get('limits')!=dict(rss_limit_bytes=CONTRACT['rss_limit_bytes'],
                            reserve_bytes=RESERVE,timeout_seconds=CONTRACT['job_seconds'])):
                    raise ValueError('Worker guard did not finish successfully')
                sample=guard.get('observation',{})
                values=[guard.get('elapsed_seconds'),sample.get('peak_rss_bytes'),sample.get('free_bytes')]
                if (any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values)
                        or violation(sample,guard['elapsed_seconds'],**guard['limits']) is not None):
                    raise ValueError('Worker guard observations exceed declared limits')
                review=review_file(child/'result.json',terminal['files'][str(child/'result.json')],
                                   seed,bound['selection'],bound['token_sha256'])
                _models(_json(child/'models.json',terminal['files'][str(child/'models.json')]),bound,seed)
                recheck(bound['files']);_json(output/'request.json',local_sha)
                result=_json(child/'result.json',terminal['files'][str(child/'result.json')]);results.append(result)
                files=dict(terminal['files']);files.update({str(child/n):file_digest(child/n)
                    for n in ('worker-state.json','guard.json','process/process.json','process/worker.log')})
                state['reviews'].append(dict(seed=seed,files=files,review=review))
                state.update(status='ready',active=None);_publish(output/'state.json',state);_budget(output)
            for receipt in state['reviews']:recheck(receipt['files'])
            if (study/'STOP').exists() or (output/'STOP').exists():
                state['status']='paused';_publish(output/'state.json',state);return state
            bind_inference(_json(output/'request.json',local_sha),SEEDS[-1])
            _publish(output/'summary.json',summarize(results,first['selection'],first['token_sha256']),exclusive=True)
            state['status']='inference_complete_pending_final_review';_publish(output/'state.json',state)
            _budget(output);return state
        except BaseException as exc:
            state.update(status='failed',error=repr(exc));_publish(output/'state.json',state);raise


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker'])
    p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--global-lock-fd',type=int,required=True);p.add_argument('--study-lock-fd',type=int,required=True)
    a=p.parse_args();worker(a.request,a.request_sha256,a.seed,a.output,(a.global_lock_fd,a.study_lock_fd))


if __name__=='__main__':main()
