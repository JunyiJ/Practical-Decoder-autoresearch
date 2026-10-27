"""Finite width64 NoPE hybrid versus selected homogeneous control; no automatic adoption."""
import argparse
import copy
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from ..models.tokenizer_architecture import architecture_contract
from ..models.tokenizer_hybrid import POLICY, GEOMETRY, LAYOUTS
from .protocol import digest, file_digest
from .runner import atomic_json, session_lock, source_fingerprint
from .tokenizer_study import ROOT, runtime_identity, worker

RECIPE = ROOT/'research/tokenizer-evidence/horizon-001/adopted-width256-gqa-recipe.json'
PROTOCOL = ROOT/'research/TOKENIZER_HYBRID_PROTOCOL.md'
RESOURCE = ROOT/'research/tokenizer-evidence/hybrid-preparation-001/resource-review.json'
PILOT = ROOT/'research/tokenizer-evidence/hybrid-pilot-001'
RATES = (.00125, .0025)
SEEDS = (3011, 3012, 3013)
FAMILIES = ('hybrid', 'gated_mla', 'kda')
LAYOUT = dict(zip(FAMILIES, LAYOUTS))
PREFLIGHT = ROOT/'research/tokenizer-evidence/hybrid-preparation-001/control-preflight.json'


def config(output, family, seed, lr):
    if family not in FAMILIES or lr not in RATES:
        raise ValueError('Undeclared hybrid comparison configuration')
    cfg = copy.deepcopy(json.loads(RECIPE.read_text())['config'])
    cfg['model'].update(attn_type='hybrid', dim=64, n_heads=2, rope=0,
                        attention_layout=list(LAYOUT[family]),
                        depth_width_policy='depth4_hybrid_unscaled',
                        attention_width_policy=GEOMETRY, tokenizer_architecture_policy=POLICY)
    cfg['training'].update(seed=seed, learning_rate=lr, checkpoint_dir=str(output/'checkpoints'),
                           metrics_file=str(output/'metrics.jsonl'), summary_file=str(output/'training-summary.json'))
    architecture_contract(cfg['model'])
    return cfg


def specs(selection=None):
    if selection is not None:
        if (set(selection) != {'rates', 'control'} or set(selection['rates']) != set(FAMILIES)
                or selection['control'] not in ('gated_mla', 'kda')
                or any(v not in RATES for v in selection['rates'].values())):
            raise ValueError('Invalid frozen family/LR selection')
    rows = [('screen', 3001, f, lr) for i, f in enumerate(FAMILIES)
            for lr in (RATES if i % 2 == 0 else RATES[::-1])]
    if selection is not None:
        control = selection['control']
        rows += [('confirm', seed, family, selection['rates'][family]) for i, seed in enumerate(SEEDS)
                 for family in (('hybrid', control) if i % 2 == 0 else (control, 'hybrid'))]
    return [dict(id=f'{stage}-{family}-w64-s{seed}-lr{lr:g}', stage=stage, family=family,
                 seed=seed, lr=lr, width=64, steps=1200) for stage, seed, family, lr in rows]


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
    winners = {f: min((r for r in rows if r['family'] == f),
                     key=lambda r: (r['h200']['mean_ce'], r['lr'])) for f in FAMILIES}
    control = min(('gated_mla', 'kda'), key=lambda f: (winners[f]['h200']['mean_ce'], f))
    return dict(rates={f: winners[f]['lr'] for f in FAMILIES}, control=control)


def conclusion(records, selection):
    from .coverage_eval import paired_blocks
    checked_panel(records, specs(selection))
    if choose(records) != selection:
        raise ValueError('Confirmation selection differs from complete screening')
    control = selection['control']
    arms = {f: {r['seed']: r['h200'] for r in records if r['stage'] == 'confirm' and r['family'] == f}
            for f in ('hybrid', control)}
    intervals = [paired_blocks(arms['hybrid'], arms[control], n) for n in (2048, 4096, 8192)]
    strict = (intervals[0]['mean_delta'] < -.02
              and all(d < 0 for d in intervals[0]['paired_seed_deltas'].values())
              and all(i['conditional_text_bootstrap_95pct'][1] < 0 for i in intervals))
    neutral = (intervals[0]['mean_delta'] <= .01
               and all(d <= .01 for d in intervals[0]['paired_seed_deltas'].values())
               and all(i['conditional_text_bootstrap_95pct'][1] <= .01 for i in intervals))
    return dict(adopted=False, strict_quality_gate=strict, neutral_quality_gate=neutral,
                memory_route_eligible=neutral and control == 'gated_mla', intervals=intervals,
                reason='Fresh width64 comparison only against selected control; saved-artifact and serial cost review required. Test closed.')


def check_budget(plan, completed, used, free):
    if type(completed) is not int or not 0 <= completed <= 12:
        raise ValueError('Invalid completed trial count')
    remaining = (12-completed)*plan['per_trial_budget_bytes']
    if free < plan['reserve_bytes']+remaining:
        raise RuntimeError('Insufficient free disk for all remaining trials plus reserve')
    if used+remaining > plan['max_output_bytes']:
        raise RuntimeError('Round output budget would be exceeded')


def verify_pilot():
    plan = json.loads((PILOT/'plan.json').read_text())
    current = source_fingerprint()
    for name, value in plan['source'].items():
        bound = (name.startswith(('src/attention/', 'src/models/', 'src/train/', 'src/data/', 'src/utils/'))
                 or name in ('requirements.txt', 'src/research/tokenizer_study.py',
                             'src/research/coverage_eval.py', 'src/research/tokenizer_attention_bench.py'))
        if bound and current.get(name) != value:
            raise ValueError('Pilot model/training/evaluation source changed')
    review = json.loads((PILOT/'review.json').read_text())
    if review['status'] != 'passed' or review['result_sha256'] != file_digest(PILOT/'result.json'):
        raise ValueError('Missing or changed pilot review')
    preflight = json.loads(PREFLIGHT.read_text())
    if (preflight['status'] != 'passed' or preflight['source'] != current
            or preflight['protocol_sha256'] != file_digest(PROTOCOL)
            or preflight['helper_sha256'] != file_digest(PREFLIGHT.with_name('control_preflight.py'))):
        raise ValueError('Missing or changed MPS control preflight')
    if plan['runtime'] != runtime_identity() or preflight['runtime'] != runtime_identity():
        raise ValueError('Pilot runtime changed')


def prepare(output):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit clean source before preparing')
    verify_pilot()
    resource = json.loads(RESOURCE.read_text())
    if resource['status'] != 'passed':
        raise ValueError('Resource review failed')
    pair = resource['per_trial_budget_bytes']
    cfg = config(output/'template', 'hybrid', 3001, RATES[0])
    plan = {'version': 1, 'source': source_fingerprint(), 'runtime': runtime_identity(), 'config': cfg,
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'recipe_sha256': file_digest(RECIPE), 'protocol_sha256': file_digest(PROTOCOL),
                'corpus_sha256': file_digest(Path(cfg['data']['path'])), 'resource_review_sha256': file_digest(RESOURCE),
                'pilot_review_sha256': file_digest(PILOT/'review.json'),
                'preflight_sha256': file_digest(PREFLIGHT),
                'max_trials': 12, 'steps': 1200, 'screen_seed': 3001, 'confirm_seeds': list(SEEDS),
                'learning_rates': list(RATES), 'neutrality_margin': .01, 'strict_min_gain': .02, 'min_cache_reduction': .4,
                'job_seconds': 7200, 'reserve_bytes': 8*1024**3, 'max_output_bytes': 12*pair+4*1024**2,
                'per_trial_budget_bytes': pair}
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
                          (RESOURCE, 'resource_review_sha256'), (PILOT/'review.json', 'pilot_review_sha256'),
                          (PREFLIGHT, 'preflight_sha256')]:
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
                proc = subprocess.Popen([sys.executable, '-m', 'src.research.tokenizer_hybrid_study', 'worker', '--output', str(trial/'request.json')], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(study_fd, worker_fd))
                state['worker_pid'] = proc.pid; atomic_json(output/'state.json', state)
                started = time.monotonic()
                while True:
                    try:
                        code = proc.wait(timeout=2)
                        break
                    except subprocess.TimeoutExpired:
                        used = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
                        reason = ('stop requested' if (output/'STOP').exists() else
                                  'timeout' if time.monotonic()-started >= plan['job_seconds'] else
                                  'disk guard' if used > plan['max_output_bytes'] or
                                  shutil.disk_usage(output).free < plan['reserve_bytes'] else None)
                        if reason is None:
                            continue
                        atomic_json(trial/'guard-failure.json', dict(reason=reason, output_bytes=used))
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                        code = 124
                        break
            verify()
            if code or not (trial/'result.json').exists():
                state['records'].append(dict(**spec, status='failed', exit_code=code)); state['status'] = 'failed'
                atomic_json(output/'state.json', state); return state
            result = json.loads((trial/'result.json').read_text()); checked_panel([result], [spec])
            state['records'].append(result); state['active'] = None
            state['worker_pid'] = None
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
            state_path = output/'state.json'
            if state_path.exists():
                state = json.loads(state_path.read_text())
                state['status'] = 'failed'
                atomic_json(state_path, state)
            atomic_json(output/'controller-error.json', {'error': type(error).__name__, 'message': str(error)})
            raise


if __name__ == '__main__':
    main()
