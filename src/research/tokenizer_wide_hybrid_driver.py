"""Prospective quality controller integration; no prepare/freeze command.

No valid pilot/numerical review currently exists for this campaign. run() must
reject drafts and incomplete prerequisites. Pilots and inference need separate
controllers. Completion here is metadata-reviewed quality, never adoption.
"""
import json
import contextlib
import fcntl
import os
from pathlib import Path
import shutil
import sys

from .protocol import digest, file_digest, ROOT
from .runner import source_fingerprint
from .tokenizer_study import runtime_identity
from .tokenizer_serial_process import run_process, RESERVE, snapshot
from .tokenizer_wide_hybrid_config import rank_config
from .tokenizer_wide_config import RECIPE
from .tokenizer_wide_hybrid_scores import review_trial
from .tokenizer_wide_hybrid_simulation import next_action, _publish
from . import tokenizer_wide_hybrid_rules as rules

METADATA = 4*1024**2
PROJECTION = ROOT/'research/tokenizer-evidence/wide-hybrid-preparation-001/rank-storage-projection.json'
PROTOCOL = ROOT/'research/TOKENIZER_WIDE_HYBRID_DESIGN.md'
CORPUS_SHA256 = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'


@contextlib.contextmanager
def session_lock(path):
    with (path/'.lock').open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream.fileno()
        # Close only: explicit LOCK_UN would also unlock an inherited child
        # descriptor if cleanup were uncertain. Last close releases the lock.


def allocations():
    rows = json.loads(PROJECTION.read_text())['records']
    return {(r['family'], r['bottleneck_divisor']): r['per_trajectory_allowance_bytes']
            for r in rows if r['width'] == 512}


def remaining_bytes(records, rates, selection):
    next_action(records, rates, selection)  # validates prefix and selection
    rates = rates or dict.fromkeys(rules.FAMILIES, rules.RATES[0])
    worst = selection or {f: dict(lr=rates[f], divisor=2) for f in rules.FAMILIES}
    queue = rules.lr_specs()+rules.rank_specs(rates)+rules.confirmation_specs(worst)
    budgets = allocations()
    return sum(budgets[s['family'], s['divisor']] for s in queue[len(records):])


def verify_plan(plan):
    if plan.get('status') != 'frozen' or plan.get('kind') != 'wide_hybrid_quality_v1':
        raise ValueError('Only a fully frozen quality plan may dispatch')
    if (plan.get('source') != source_fingerprint() or plan.get('runtime') != runtime_identity()
            or plan.get('protocol_sha256') != file_digest(PROTOCOL)
            or plan.get('projection_sha256') != file_digest(PROJECTION)
            or plan.get('reserve_bytes') != RESERVE or plan.get('job_seconds') != 7200
            or plan.get('max_trials') != 24 or plan.get('log_limit_bytes') != METADATA):
        raise ValueError('Frozen source/runtime/protocol/resource contract changed')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('Strict MPS required')
    for item in plan['files']:
        if file_digest(Path(item['path'])) != item['sha256']:
            raise ValueError('Frozen prerequisite/data file changed')
    # Required bindings cannot be omitted from a caller-created files list.
    cfg = rank_config(Path(plan['output'])/'template', 'hybrid', device='mps')
    corpus = Path(cfg['data']['path']).resolve()
    file_map = {str(Path(i['path']).resolve()): i['sha256'] for i in plan['files']}
    if (len(file_map) != len(plan['files']) or file_map.get(str(corpus)) != CORPUS_SHA256
            or file_map.get(str(RECIPE.resolve())) != file_digest(RECIPE)):
        raise ValueError('Missing/duplicate frozen corpus binding')
    required = {
        'numerical': {'all_declared_ranks', 'geometry', 'cache', 'optimizer', 'resume', 'ema'},
        'pilots': {'three_families', 'raw_h200', 'cache', 'resource_limits', 'full_panel_fit'},
    }
    if set(plan['prerequisites']) != set(required):
        raise ValueError('Missing required prerequisite reviews')
    for kind, checks in required.items():
        path = Path(plan['prerequisites'][kind]).resolve()
        if file_map.get(str(path)) != file_digest(path):
            raise ValueError('Prerequisite review not frozen')
        review = json.loads(path.read_text())
        if (review.get('kind') != kind or review.get('status') != 'passed'
                or review.get('source_sha256') != digest(plan['source'])
                or review.get('runtime_sha256') != digest(plan['runtime'])
                or review.get('corpus_sha256') != file_map[str(corpus)]
                or set(review.get('checks', {})) != checks
                or any(v is not True for v in review['checks'].values())):
            raise ValueError('Prerequisite checks or provenance incomplete')
        if not review.get('artifacts') or any(file_map.get(str(Path(p).resolve())) != sha
                                             for p, sha in review['artifacts'].items()):
            raise ValueError('Prerequisite artifacts not frozen')
    if plan.get('max_output_bytes') != remaining_bytes([], None, None)+METADATA:
        raise ValueError('Incomplete full quality output allowance')


def _contract(config):
    from omegaconf import OmegaConf
    from ..train.train import _make_loader, _training_contract
    cfg = OmegaConf.create(config)
    return _training_contract(cfg, 'mps', _make_loader(cfg, 'cpu'))


def _trial_id(index, spec):
    return f"{index:02d}-{spec['stage']}-{spec['family']}-s{spec['seed']}-d{spec['divisor']}"


def run(output):
    output = Path(output).resolve()
    plan_path = output/'plan.json'
    plan = json.loads(plan_path.read_text())
    verify_plan(plan)  # before creating run state or dispatching anything
    if Path(plan['output']).resolve() != output:
        raise ValueError('Frozen output differs')
    plan_hash = file_digest(plan_path)
    global_lock = ROOT/'research_state/tokenizer-worker-lock'; global_lock.mkdir(exist_ok=True)
    with session_lock(global_lock) as global_fd, session_lock(output) as run_fd:
        if (output/'state.json').exists():
            raise FileExistsError('Existing quality attempt requires review; no automatic resume')
        state = dict(status='ready', records=[], rates=None, selection=None, active=None,
                     reviews=[], plan_sha256=plan_hash, adopted=False)
        _publish(output/'state.json', state, exclusive=True)
        def verify():
            verify_plan(plan)
            if file_digest(plan_path) != plan_hash:
                raise ValueError('Plan changed during run')
            for key, name in (('rates', 'lr-selection.json'), ('selection', 'rank-selection.json')):
                path = output/name
                if (state[key] is None and path.exists()) or (state[key] is not None and
                        digest(json.loads(path.read_text())) != digest(state[key])):
                    raise ValueError('Persisted selection changed')
            # Recheck previously reviewed immutable evidence before every dispatch.
            for receipt in state['reviews']:
                for path, expected in receipt['files'].items():
                    if file_digest(Path(path)) != expected:
                        raise ValueError('Previously reviewed artifact changed')
        try:
            while True:
                verify()
                if (output/'STOP').exists():
                    state['status'] = 'paused'; _publish(output/'state.json', state); return state
                remaining = remaining_bytes(state['records'], state['rates'], state['selection'])
                used = snapshot(output)['output_bytes']
                if (shutil.disk_usage(output).free < RESERVE+remaining+METADATA
                        or used+remaining > plan['max_output_bytes']):
                    raise RuntimeError('Full remaining panel does not fit')
                action = next_action(state['records'], state['rates'], state['selection'])
                if action['kind'] == 'complete':
                    state['status'] = 'quality_completed_pending_tensor_and_serving_audits'
                    _publish(output/'state.json', state); return state
                if action['kind'].startswith('save_'):
                    key, name = (('rates', 'lr-selection.json') if action['kind'] == 'save_rates'
                                 else ('selection', 'rank-selection.json'))
                    _publish(output/name, action['value'], exclusive=True)
                    state[key] = action['value']; _publish(output/'state.json', state); continue
                spec = action['spec']; trial = output/_trial_id(len(state['records']), spec)
                trial.mkdir(exist_ok=False)
                cfg = rank_config(trial, spec['family'], 512, spec['seed'], spec['lr'], 1200, 'mps',
                                  bottleneck_divisor=spec['divisor'])
                request = dict(spec=spec, config=cfg, result_path=str(trial/'result.json'))
                _publish(trial/'request.json', request, exclusive=True)
                request_hash = file_digest(trial/'request.json')
                contract = _contract(cfg)
                state.update(status='running', active=spec); _publish(output/'state.json', state)
                run_process([sys.executable, '-m', 'src.research.tokenizer_study', 'worker',
                             '--output', str(trial/'request.json')], ROOT, trial/'process', (global_fd, run_fd),
                            required_free_bytes=RESERVE+remaining+METADATA,
                            output_limit_bytes=allocations()[spec['family'], spec['divisor']],
                            log_limit_bytes=METADATA, timeout_seconds=7200, watch_root=trial)
                verify()
                receipt = review_trial(trial, request, request_hash, file_digest(trial/'result.json'), contract)
                files = {str(trial/'request.json'): request_hash,
                         str(trial/'result.json'): receipt['result_sha256']}
                result = json.loads((trial/'result.json').read_text())
                files.update({result[k]['checkpoint']: receipt['checkpoint_hashes'][k] for k in ('raw', 'h200')})
                state['reviews'].append(dict(files=files, review=receipt))
                state['records'].append(receipt['selection_row'])
                state.update(status='ready', active=None); _publish(output/'state.json', state)
        except BaseException as exc:
            state.update(status='failed', error=repr(exc)); _publish(output/'state.json', state)
            raise
