"""Prospective guarded pilots. No prepare/freeze/automatic-resume command.

Only mocked integration is currently validated. No complete numerical review or
fitted frozen panel exists; the real prerequisite gate must reject launch.
"""
import argparse
import fcntl
import math
import os
from pathlib import Path
import shutil
import sys

import torch
from omegaconf import OmegaConf

from .protocol import ROOT, digest, file_digest
from .cpu_prerequisite_guard import CpuPrerequisiteGuard, violation
from .tokenizer_serial_process import run_process, snapshot
from .tokenizer_study import worker as train_worker
from .tokenizer_wide_hybrid_driver import session_lock
from .tokenizer_wide_hybrid_pilot_config import CONTRACT, FAMILIES, PROJECTION, storage_contract, require_storage
from .tokenizer_wide_hybrid_pilot_gate import bind_pilot, recheck
from .tokenizer_wide_hybrid_pilot_cache import review_caches, CACHE_CONTRACT, _projection
from .tokenizer_wide_hybrid_scores import review_pilot, _json, _require
from .tokenizer_wide_hybrid_simulation import _publish
from .tokenizer_wide_hybrid_timing import token_digest


def _inputs(config):
    from ..train.train import _make_loader, _training_contract
    cfg = OmegaConf.create(config)
    loader = _make_loader(cfg, 'cpu')
    return _training_contract(cfg, 'mps', loader), token_digest(loader.train_data[:96].tolist())


def _input_contract(bound):
    contract, sha = _inputs(bound['request']['config'])
    _require(digest(contract.get('tokenizer')) == digest(bound['tokenizer'])
             and sha == bound['training_token_sha256'], 'Frozen tokenizer/training input differs')
    return contract


def _locks(fds, root):
    _require(len(fds) == 2 and len(set(fds)) == 2
             and all(type(fd) is int and fd > 2 for fd in fds), 'Two inherited worker locks required')
    for fd, path in zip(fds, (ROOT/'research_state/tokenizer-worker-lock/.lock', root/'.lock')):
        a, b = os.fstat(fd), path.stat()
        _require((a.st_dev, a.st_ino) == (b.st_dev, b.st_ino), 'Inherited lock identity differs')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _budget(root, completed):
    budget = require_storage(shutil.disk_usage(root).free, completed)
    cap = sum(budget['pilot_allowances'].values()) + budget['metadata_allowance_bytes']
    _require(snapshot(root)['output_bytes'] + budget['remaining_pilot_allowance_bytes'] <= cap,
             'Retained pilot output exceeds complete allocation')
    return budget


def _trial_budget(trial, family):
    sample = snapshot(trial)
    _require(sample['free_bytes'] >= CONTRACT['reserve_bytes']
             and sample['output_bytes'] <= storage_contract()['pilot_allowances'][family],
             'Pilot output/reserve limit exceeded')


def _prefix(state, index, plan_sha, family):
    _require(state.get('plan_sha256') == plan_sha and state.get('status') == 'running'
             and state.get('active') == family and state.get('adopted') is False
             and [r['family'] for r in state['reviews']] == list(FAMILIES[:index]),
             'Parent did not authorize this exact pilot prefix')
    for row in state['reviews']:
        recheck(row['files'])


def worker(plan_path, plan_sha, family, request_sha, lock_fds):
    plan_path = Path(plan_path).resolve(); root = plan_path.parent
    _locks(lock_fds, root)
    _require(family in FAMILIES, 'Undeclared pilot family')
    index = FAMILIES.index(family)
    state_path = root/'state.json'
    _prefix(_json(state_path, file_digest(state_path)), index, plan_sha, family)
    bound = bind_pilot(plan_path, plan_sha, family)
    trial = Path(bound['request']['result_path']).resolve().parent
    request_path = trial/'request.json'
    _require(digest(_json(request_path, request_sha)) == digest(bound['request']), 'Worker request differs')
    if (trial/'worker-state.json').exists():
        raise FileExistsError('Existing pilot attempt requires review; no retry')
    _budget(root, FAMILIES[:index])
    state = dict(status='verifying', family=family, plan_sha256=plan_sha,
                 request_sha256=request_sha, files={}, adopted=False)
    _publish(trial/'worker-state.json', state, exclusive=True)
    try:
        with CpuPrerequisiteGuard(trial/'guard.json', root,
                rss_limit_bytes=CONTRACT['rss_limit_bytes'], reserve_bytes=CONTRACT['reserve_bytes'],
                timeout_seconds=CONTRACT['job_seconds']):
            torch.set_num_threads(4)
            contract = _input_contract(bound)
            state['status'] = 'training'; _publish(trial/'worker-state.json', state)
            # The common worker starts the 4GiB RSS/6GiB MPS-driver watchdog and
            # performs the exact frozen20-step train/export/validation request.
            train_worker(request_path)
            state['status'] = 'auditing'; _publish(trial/'worker-state.json', state)
            report = review_caches(dict(trial=str(trial), request=bound['request'],
                request_sha256=request_sha, result_sha256=file_digest(trial/'result.json'),
                training_contract=contract), bound['tokenizer'], bound['training_token_sha256'])
            again = bind_pilot(plan_path, plan_sha, family)
            _require(digest(again['files']) == digest(bound['files']), 'Pilot provenance changed')
            _trial_budget(trial, family)
            _publish(trial/'audit.json', report, exclusive=True)
            state['files'] = {str(trial/'audit.json'): file_digest(trial/'audit.json')}
            state['status'] = 'completed'; _publish(trial/'worker-state.json', state)
            _trial_budget(trial, family)
        return state
    except BaseException as exc:
        state.update(status='failed', error=repr(exc)); _publish(trial/'worker-state.json', state)
        raise


def _guard(trial):
    path = trial/'guard.json'; guard = _json(path, file_digest(path))
    expected = dict(rss_limit_bytes=CONTRACT['rss_limit_bytes'], reserve_bytes=CONTRACT['reserve_bytes'],
                    timeout_seconds=CONTRACT['job_seconds'])
    _require(guard.get('status') == 'finished' and guard.get('reason') is None
             and digest(guard.get('limits')) == digest(expected), 'Pilot guard did not complete')
    sample = guard.get('observation', {})
    values = (guard.get('elapsed_seconds'), sample.get('peak_rss_bytes'), sample.get('free_bytes'))
    _require(all(type(v) in (int,float) and math.isfinite(v) and v >= 0 for v in values)
             and violation(sample, guard['elapsed_seconds'], **expected) is None, 'Pilot guard limits violated')
    return guard


def _report(report, trial, bound, metadata):
    family = bound['request']['spec']['family']; row = _projection(family)
    _require(report.get('status') == 'pilot_cache_checks_complete_pending_resource_review'
             and digest(report.get('spec')) == digest(bound['request']['spec'])
             and report.get('adopted') is False and digest(report.get('contract')) == digest(CACHE_CONTRACT)
             and report.get('token_sha256') == bound['training_token_sha256']
             and report.get('tokenizer_sha256') == digest(bound['tokenizer']), 'Pilot cache report identity differs')
    tensor = report['tensor_review']
    _require(tensor.get('kind') == 'pilot' and tensor.get('adopted') is False
             and digest(tensor.get('metadata')) == digest(metadata), 'Pilot tensor metadata receipt differs')
    values = tensor['values']
    _require(all(values.get(k) is True for k in ('finite_model_ema_optimizer_export_values',
        'exact_export_matches_stored_ema', 'persistent_buffers_unchanged'))
        and all(values.get(k) is False for k in ('optimizer_structure_audited','optimizer_resume_audited',
                                                'ema_recurrence_audited','adopted')), 'Pilot tensor value receipt incomplete')
    result = _json(trial/'result.json', metadata['result_sha256'])
    files = {str(trial/'request.json'): metadata['request_sha256'], str(trial/'result.json'): metadata['result_sha256']}
    _require(set(report['endpoints']) == {'raw','h200'}, 'Missing pilot cache endpoint')
    for kind in ('raw','h200'):
        endpoint = report['endpoints'][kind]; path = Path(result[kind]['checkpoint']).resolve()
        files[str(path)] = metadata['checkpoint_hashes'][kind]
        _require(endpoint.get('checkpoint') == str(path) and endpoint.get('sha256') == files[str(path)]
                 and endpoint.get('parameters') == row['parameters']
                 and endpoint.get('parameter_storage_bytes') == row['parameter_bytes']
                 and [c['batch'] for c in endpoint['checks']] == [1,8], 'Pilot endpoint/geometry differs')
        for check in endpoint['checks']:
            err = check.get('max_absolute_error')
            _require(check.get('passed') is True and type(err) in (int,float) and math.isfinite(err) and err >= 0
                     and type(check.get('state_bytes')) is int and type(check['batch']) is int
                     and check['state_bytes'] == check['batch']*row['cache_bytes_batch1'], 'Pilot cache check invalid')
            for key in ('observed_mps_current_bytes','observed_mps_driver_bytes'):
                _require(type(check.get(key)) is int and check[key] > 0, 'Missing allocator observation')
            _require(check['observed_mps_driver_bytes'] < CONTRACT['driver_limit_bytes'], 'Pilot driver limit exceeded')
    _require(digest(tensor['files']) == digest(files), 'Pilot tensor file bindings differ')
    corpus = str(Path(bound['request']['config']['data']['path']).resolve())
    files[corpus] = bound['files'][corpus]
    files[str(PROJECTION.resolve())] = bound['files'][str(PROJECTION.resolve())]
    _require(digest(report['files']) == digest(files), 'Pilot cache file bindings differ')
    recheck(files)
    return files


def run(root):
    root = Path(root).resolve(); plan_path = root/'plan.json'; plan_sha = file_digest(plan_path)
    global_lock = ROOT/'research_state/tokenizer-worker-lock'; global_lock.mkdir(exist_ok=True)
    with session_lock(global_lock) as global_fd, session_lock(root) as run_fd:
        if (root/'state.json').exists():
            raise FileExistsError('Existing pilot panel requires review; no automatic resume')
        first = bind_pilot(plan_path, plan_sha, FAMILIES[0])
        _budget(root, ())
        state = dict(status='ready', active=None, reviews=[], plan_sha256=plan_sha, adopted=False)
        _publish(root/'state.json', state, exclusive=True)
        try:
            for index, family in enumerate(FAMILIES):
                if (root/'STOP').exists():
                    state['status']='paused'; _publish(root/'state.json',state); return state
                for previous in state['reviews']: recheck(previous['files'])
                bound = bind_pilot(plan_path,plan_sha,family)
                _require(digest(bound['files']) == digest(first['files']), 'Panel provenance changed')
                budget = _budget(root,FAMILIES[:index])
                request = bound['request']; trial = Path(request['result_path']).resolve().parent
                trial.mkdir(exist_ok=False)
                _publish(trial/'request.json',request,exclusive=True); request_sha=file_digest(trial/'request.json')
                state.update(status='running',active=family); _publish(root/'state.json',state)
                argv=[sys.executable,'-m','src.research.tokenizer_wide_hybrid_pilot','worker',
                    '--plan',str(plan_path),'--plan-sha256',plan_sha,'--family',family,
                    '--request-sha256',request_sha,'--global-lock-fd',str(global_fd),'--run-lock-fd',str(run_fd)]
                run_process(argv,ROOT,trial/'process',(global_fd,run_fd),
                    required_free_bytes=budget['required_free_bytes'],
                    output_limit_bytes=budget['pilot_allowances'][family],
                    log_limit_bytes=CONTRACT['log_limit_bytes'],timeout_seconds=CONTRACT['job_seconds'],watch_root=trial)
                terminal=_json(trial/'worker-state.json',file_digest(trial/'worker-state.json'))
                _require(terminal.get('status')=='completed' and terminal.get('family')==family
                    and terminal.get('plan_sha256')==plan_sha and terminal.get('request_sha256')==request_sha
                    and terminal.get('adopted') is False
                    and set(terminal['files'])=={str(trial/'audit.json')}, 'Pilot worker did not complete')
                recheck(terminal['files']); guard=_guard(trial)
                metadata=review_pilot(trial,request,request_sha,file_digest(trial/'result.json'),_input_contract(bound))
                report=_json(trial/'audit.json',terminal['files'][str(trial/'audit.json')])
                files=_report(report,trial,bound,metadata)
                files.update({str(trial/n):file_digest(trial/n) for n in
                    ('audit.json','worker-state.json','guard.json','process/process.json','process/worker.log')})
                recheck(bound['files']);_trial_budget(trial,family)
                state['reviews'].append(dict(family=family,files=files,guard=guard,metadata=metadata))
                state.update(status='ready',active=None);_publish(root/'state.json',state)
                _budget(root,FAMILIES[:index+1])
            for previous in state['reviews']: recheck(previous['files'])
            if (root/'STOP').exists():
                state['status']='paused';_publish(root/'state.json',state);return state
            bind_pilot(plan_path,plan_sha,FAMILIES[-1])
            state['status']='pilots_complete_pending_resource_and_numerical_review'
            _publish(root/'state.json',state);_budget(root,FAMILIES);return state
        except BaseException as exc:
            state.update(status='failed',error=repr(exc));_publish(root/'state.json',state);raise


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker'])
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True)
    p.add_argument('--family',choices=FAMILIES,required=True);p.add_argument('--request-sha256',required=True)
    p.add_argument('--global-lock-fd',type=int,required=True);p.add_argument('--run-lock-fd',type=int,required=True)
    a=p.parse_args();worker(a.plan,a.plan_sha256,a.family,a.request_sha256,(a.global_lock_fd,a.run_lock_fd))


if __name__=='__main__':main()
