"""Finite, source-frozen tokenizer MHA/MLA comparison; quality is only an adoption gate."""
import argparse
import copy
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from ..models.tokenizer_architecture import MLA_POLICY, POLICY, architecture_contract
from .protocol import digest, file_digest
from .runner import atomic_json, session_lock, source_fingerprint
from .tokenizer_study import ROOT, runtime_identity, worker

RECIPE = ROOT/'research/tokenizer-evidence/horizon-001/adopted-width256-gqa-recipe.json'
PROTOCOL = ROOT/'research/TOKENIZER_MLA_PROTOCOL.md'
RESOURCE = ROOT/'research/tokenizer-evidence/mla-pilot-001/resource-review.json'
PILOT = ROOT/'research/tokenizer-evidence/mla-pilot-001'
RATES = (.00125, .0025)
SEEDS = (2911, 2912, 2913)


def config(output, family, seed, lr):
    if family not in ('mha', 'mla') or lr not in RATES:
        raise ValueError('Undeclared MLA comparison configuration')
    cfg = copy.deepcopy(json.loads(RECIPE.read_text())['config'])
    cfg['model'].update(attn_type=family, depth_width_policy='depth4_unscaled',
                        attention_width_policy='mha_head32' if family == 'mha' else 'mla_quarter_head32',
                        tokenizer_architecture_policy=POLICY if family == 'mha' else MLA_POLICY)
    cfg['training'].update(seed=seed, learning_rate=lr, checkpoint_dir=str(output/'checkpoints'),
                           metrics_file=str(output/'metrics.jsonl'), summary_file=str(output/'training-summary.json'))
    architecture_contract(cfg['model'])
    return cfg


def specs(selection=None):
    if selection is not None and (set(selection) != {'mha', 'mla'} or any(v not in RATES for v in selection.values())):
        raise ValueError('Invalid frozen learning-rate selection')
    rows = [('screen', 2901, 'mha', lr) for lr in RATES]
    rows += [('screen', 2901, 'mla', lr) for lr in reversed(RATES)]
    if selection is not None:
        rows += [('confirm', seed, family, selection[family]) for i, seed in enumerate(SEEDS)
                 for family in (('mha', 'mla') if i % 2 == 0 else ('mla', 'mha'))]
    return [{'id': f'{stage}-{family}-w256-s{seed}-lr{lr:g}', 'stage': stage, 'family': family,
                 'seed': seed, 'lr': lr, 'width': 256, 'steps': 1200} for stage, seed, family, lr in rows]


def checked_panel(records, expected):
    if sorted(r['id'] for r in records) != sorted(r['id'] for r in expected):
        raise ValueError('Incomplete, duplicate or unexpected trial panel')
    by_id = {r['id']: r for r in records}
    for spec in expected:
        r = by_id[spec['id']]
        if (any(r.get(k) != v for k, v in spec.items()) or r.get('status') != 'completed'
                or not math.isfinite(r['h200']['mean_ce'])):
            raise ValueError('Trial identity/status/metric differs from frozen panel')


def choose(records):
    rows = [r for r in records if r['stage'] == 'screen']
    checked_panel(rows, specs())
    return {f: min((r for r in rows if r['family'] == f), key=lambda r: (r['h200']['mean_ce'], r['lr']))['lr']
            for f in ('mha', 'mla')}


def conclusion(records, selection):
    from .coverage_eval import paired_blocks
    checked_panel(records, specs(selection))
    if choose(records) != selection:
        raise ValueError('Confirmation selection differs from complete screening')
    arms = {f: {r['seed']: r['h200'] for r in records if r['stage'] == 'confirm' and r['family'] == f}
            for f in ('mha', 'mla')}
    intervals = [paired_blocks(arms['mla'], arms['mha'], n) for n in (2048, 4096, 8192)]
    eligible = (intervals[0]['mean_delta'] <= .01
                and all(d <= .01 for d in intervals[0]['paired_seed_deltas'].values())
                and all(i['conditional_text_bootstrap_95pct'][1] <= .01 for i in intervals))
    return {'adopted': False, 'quality_eligible': eligible, 'intervals': intervals,
                'reason': 'Quality gate only. All serial cache/timing measurements and artifact review required; test closed.'}


def check_budget(plan, completed, used, free):
    if type(completed) is not int or not 0 <= completed <= 10:
        raise ValueError('Invalid completed trial count')
    remaining = (10-completed)*plan['per_trial_budget_bytes']
    if free < plan['reserve_bytes']+remaining:
        raise RuntimeError('Insufficient free disk for all remaining trials plus reserve')
    if used+remaining > plan['max_output_bytes']:
        raise RuntimeError('Round output budget would be exceeded')


def verify_pilot():
    plan = json.loads((PILOT/'plan.json').read_text())
    current = source_fingerprint()
    for name, value in plan['source'].items():
        bound = (name.startswith(('src/attention/', 'src/models/', 'src/train/', 'src/data/', 'src/utils/'))
                 or name in ('requirements.txt', 'src/research/tokenizer_study.py', 'src/research/coverage_eval.py'))
        if bound and current.get(name) != value:
            raise ValueError('Pilot model/training/evaluation source changed')
    review = json.loads((PILOT/'review.json').read_text())
    if review['status'] != 'passed' or review['result_sha256'] != file_digest(PILOT/'result.json'):
        raise ValueError('Missing or changed pilot review')
    if json.loads((PILOT/'cache-check.json').read_text())['status'] != 'passed':
        raise ValueError('Pilot cache check failed')
    if plan['runtime'] != runtime_identity():
        raise ValueError('Pilot runtime changed')


def prepare(output):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit clean source before preparing')
    verify_pilot()
    resource = json.loads(RESOURCE.read_text())
    if resource['status'] != 'passed':
        raise ValueError('Resource review failed')
    pair = max(resource['mla_pilot_pair_bytes'], resource['largest_preserved_width256_mha_pair_bytes'])
    cfg = config(output/'template', 'mha', 2901, RATES[0])
    plan = {'version': 1, 'source': source_fingerprint(), 'runtime': runtime_identity(), 'config': cfg,
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'recipe_sha256': file_digest(RECIPE), 'protocol_sha256': file_digest(PROTOCOL),
                'corpus_sha256': file_digest(Path(cfg['data']['path'])), 'resource_review_sha256': file_digest(RESOURCE),
                'pilot_review_sha256': file_digest(PILOT/'review.json'),
                'max_trials': 10, 'steps': 1200, 'screen_seed': 2901, 'confirm_seeds': list(SEEDS),
                'learning_rates': list(RATES), 'neutrality_margin': .01, 'min_cache_reduction': .6,
                'job_seconds': 7200, 'reserve_bytes': 8*1024**3, 'max_output_bytes': 7*1024**3,
                'per_trial_budget_bytes': math.ceil(pair*1.15)}
    check_budget(plan, 0, 0, shutil.disk_usage(ROOT).free)
    output.mkdir(exist_ok=False)
    atomic_json(output/'plan.json', plan)
    atomic_json(output/'state.json', {'status': 'prepared', 'selection': None, 'records': [], 'plan_sha256': digest(plan)})
    return plan


def run(output):
    plan = json.loads((output/'plan.json').read_text())
    state = json.loads((output/'state.json').read_text())
    if state['status'] in ('completed', 'failed', 'paused'):
        return state

    def verify():
        if state['plan_sha256'] != digest(plan) or source_fingerprint() != plan['source'] or runtime_identity() != plan['runtime']:
            raise ValueError('Frozen source/runtime/plan changed')
        for path, key in [(RECIPE, 'recipe_sha256'), (PROTOCOL, 'protocol_sha256'),
                          (Path(plan['config']['data']['path']), 'corpus_sha256'),
                          (RESOURCE, 'resource_review_sha256'), (PILOT/'review.json', 'pilot_review_sha256')]:
            if file_digest(path) != plan[key]:
                raise ValueError('Frozen protocol/recipe/corpus/resource/pilot changed')

    verify()
    def stop(signum, frame):
        (output/'STOP').write_text(f'Stop signal {signum}\n')
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    lock = ROOT/'research_state/tokenizer-worker-lock'; lock.mkdir(exist_ok=True)
    with session_lock(output) as study_fd, session_lock(lock) as worker_fd:
        while True:
            verify()
            if (output/'STOP').exists():
                state['status'] = 'paused'; atomic_json(output/'state.json', state); return state
            done = {r['id'] for r in state['records']}
            if len(done) != len(state['records']):
                raise ValueError('Duplicate completed records')
            used = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            check_budget(plan, len(done), used, shutil.disk_usage(output).free)
            queue = [s for s in specs(state['selection']) if s['id'] not in done]
            if not queue:
                if state['selection'] is None:
                    state['selection'] = choose(state['records'])
                    atomic_json(output/'selection.json', state['selection'])
                    atomic_json(output/'state.json', state); continue
                state.update(status='completed', active=None, conclusion=conclusion(state['records'], state['selection']))
                atomic_json(output/'state.json', state); return state
            spec = queue[0]; trial = output/spec['id']; trial.mkdir(exist_ok=False)
            cfg = config(trial, spec['family'], spec['seed'], spec['lr'])
            atomic_json(trial/'request.json', {'spec': spec, 'config': cfg, 'result_path': str(trial/'result.json')})
            state.update(status='running', active=spec, controller_pid=os.getpid()); atomic_json(output/'state.json', state)
            print('START '+spec['id'], flush=True)
            with (trial/'worker.log').open('x') as log:
                proc = subprocess.Popen([sys.executable, '-m', 'src.research.tokenizer_mla_study', 'worker', '--output', str(trial/'request.json')], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(study_fd, worker_fd))
                state['worker_pid'] = proc.pid; atomic_json(output/'state.json', state)
                try:
                    code = proc.wait(timeout=plan['job_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                    code = 124
            if code or not (trial/'result.json').exists():
                state['records'].append(dict(**spec, status='failed', exit_code=code)); state['status'] = 'failed'
                atomic_json(output/'state.json', state); return state
            result = json.loads((trial/'result.json').read_text()); checked_panel([result], [spec])
            state['records'].append(result); state['active'] = None
            atomic_json(output/'state.json', state)
            print(f'DONE {spec["id"]}: H200 CE={result["h200"]["mean_ce"]:.6f}', flush=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['prepare', 'run', 'worker'])
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); output = a.output.resolve()
    if a.command == 'prepare':
        prepare(output)
    elif a.command == 'worker':
        worker(output)
    else:
        try:
            run(output)
        except Exception as error:
            atomic_json(output/'controller-error.json', {'error': type(error).__name__, 'message': str(error)})
            raise


if __name__ == '__main__':
    main()
