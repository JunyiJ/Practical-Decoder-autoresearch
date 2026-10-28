"""Finite width256 table sharing with both controls and independent LR selection."""
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
from ..models.tokenizer_tables import POLICY, KINDS
from .tokenizer_table_pilot import ALLOWANCE
from .protocol import digest, file_digest
from .runner import atomic_json, session_lock, source_fingerprint
from .tokenizer_study import ROOT, runtime_identity, worker

RECIPE = ROOT/'research/tokenizer-evidence/horizon-001/adopted-width256-gqa-recipe.json'
PROTOCOL = ROOT/'research/TOKENIZER_TABLE_PROTOCOL.md'
PILOT = ROOT/'research/tokenizer-evidence/table-pilot-001'
RESOURCE = PILOT/'review.json'
RATES = (.00125, .0025)
SEEDS = (3311, 3312, 3313)
FAMILIES = KINDS


def config(output, family, seed, lr):
    if family not in FAMILIES or lr not in RATES or seed not in (3301, *SEEDS):
        raise ValueError('Undeclared table comparison configuration')
    cfg = copy.deepcopy(json.loads(RECIPE.read_text())['config'])
    cfg['model'].update(tokenizer_architecture_policy=POLICY, weight_tying_kind=family,
                        weight_tying_width_policy=POLICY)
    cfg['training'].update(seed=seed, learning_rate=lr, checkpoint_dir=str(output/'checkpoints'),
                           metrics_file=str(output/'metrics.jsonl'), summary_file=str(output/'training-summary.json'))
    architecture_contract(cfg['model'])
    return cfg


def specs(selection=None):
    if selection is not None and (set(selection) != set(FAMILIES) or any(v not in RATES for v in selection.values())):
        raise ValueError('Invalid frozen independent LR selection')
    rows = [('screen', 3301, f, lr) for i, f in enumerate(FAMILIES)
            for lr in (RATES if i % 2 == 0 else RATES[::-1])]
    if selection is not None:
        rows += [('confirm', seed, f, selection[f]) for i, seed in enumerate(SEEDS)
                 for f in FAMILIES[i:]+FAMILIES[:i]]
    return [dict(id=f'{stage}-{family}-w256-s{seed}-lr{lr:g}', stage=stage, family=family,
                 seed=seed, lr=lr, width=256, steps=1200) for stage, seed, family, lr in rows]


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
            for f in FAMILIES}


def conclusion(records, selection):
    from .coverage_eval import paired_blocks
    checked_panel(records, specs(selection))
    if choose(records) != selection:
        raise ValueError('Confirmation selection differs from screening')
    arms = {f: {r['seed']: r['h200'] for r in records if r['stage'] == 'confirm' and r['family'] == f}
            for f in FAMILIES}
    comparisons = {}
    for candidate, control in [('scaled_tied','legacy_untied'), ('scaled_tied','matched_untied'),
                               ('matched_untied','legacy_untied')]:
        intervals = [paired_blocks(arms[candidate], arms[control], n) for n in (2048, 4096, 8192)]
        strict = (intervals[0]['mean_delta'] < -.02
                  and all(d < 0 for d in intervals[0]['paired_seed_deltas'].values())
                  and all(i['conditional_text_bootstrap_95pct'][1] < 0 for i in intervals))
        neutral = (intervals[0]['mean_delta'] <= .01
                   and all(d <= .01 for d in intervals[0]['paired_seed_deltas'].values())
                   and all(i['conditional_text_bootstrap_95pct'][1] <= .01 for i in intervals))
        comparisons[candidate+'_vs_'+control] = dict(strict=strict, neutral=neutral, intervals=intervals)
    tied = [comparisons['scaled_tied_vs_'+f] for f in FAMILIES[:2]]
    return dict(adopted=False, comparisons=comparisons,
                tied_strict_quality_gate=all(c['strict'] for c in tied),
                tied_neutral_quality_gate=all(c['neutral'] for c in tied),
                matched_strict_quality_gate=comparisons['matched_untied_vs_legacy_untied']['strict'],
                reason='All fresh quality gates precede saved-artifact and measured parameter-memory/timing review. Test closed.')


def check_budget(plan, records, used, free):
    counts = {f: 0 for f in FAMILIES}
    for r in records:
        if r['family'] not in counts or r.get('status') != 'completed':
            raise ValueError('Invalid completed record')
        counts[r['family']] += 1
    if any(n > 5 for n in counts.values()) or len({r['id'] for r in records}) != len(records):
        raise ValueError('Invalid completed trial count/identity')
    remaining = sum((5-counts[f])*plan['per_trial_budget_bytes'][f] for f in FAMILIES)
    if free < plan['reserve_bytes']+remaining+4*1024**2:
        raise RuntimeError('Insufficient free disk for all remaining trials plus reserve')
    if used+remaining > plan['max_output_bytes']:
        raise RuntimeError('Round output budget would be exceeded')


def verify_pilot():
    plan = json.loads((PILOT/'plan.json').read_text()); current = source_fingerprint()
    # Additional study/benchmark source may be added; every source present during pilots must stay unchanged.
    if any(current.get(name) != value for name, value in plan['source'].items()):
        raise ValueError('Pilot source changed')
    review = json.loads(RESOURCE.read_text())
    if review['status'] != 'passed' or review['plan_sha256'] != digest(plan):
        raise ValueError('Missing or changed pilot review')
    for record in review['records']:
        result_path = PILOT/record['kind']/'result.json'
        if record['result_sha256'] != file_digest(result_path):
            raise ValueError('Changed pilot result')
        for c in json.loads(result_path.read_text())['checkpoints']:
            if file_digest(Path(c['checkpoint'])) != c['sha256']:
                raise ValueError('Pilot checkpoint changed')
    if plan['runtime'] != runtime_identity() or review['quality_allowances'] != ALLOWANCE:
        raise ValueError('Pilot runtime or reviewed allowance changed')


def prepare(output):
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise ValueError('Commit clean source before preparing')
    verify_pilot()
    cfg = config(output/'template', FAMILIES[0], 3301, RATES[0])
    plan = dict(version=1, source=source_fingerprint(), runtime=runtime_identity(), config=cfg,
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                recipe_sha256=file_digest(RECIPE), protocol_sha256=file_digest(PROTOCOL),
                corpus_sha256=file_digest(Path(cfg['data']['path'])), resource_review_sha256=file_digest(RESOURCE),
                pilot_plan_sha256=file_digest(PILOT/'plan.json'),
                max_trials=15, steps=1200, screen_seed=3301, confirm_seeds=list(SEEDS),
                learning_rates=list(RATES), neutrality_margin=.01, strict_min_gain=.02, min_parameter_reduction=.30,
                screen_order=specs(), confirmation_rotation=[list(FAMILIES[i:]+FAMILIES[:i]) for i in range(3)],
                job_seconds=7200, reserve_bytes=8*1024**3, max_output_bytes=5*sum(ALLOWANCE.values())+4*1024**2,
                per_trial_budget_bytes=ALLOWANCE)
    check_budget(plan, [], 0, shutil.disk_usage(ROOT).free)
    output.mkdir(exist_ok=False)
    atomic_json(output/'plan.json', plan)
    atomic_json(output/'state.json', {'status': 'prepared', 'selection': None, 'records': [], 'plan_sha256': digest(plan)})
    return plan


def run(output):
    plan = json.loads((output/'plan.json').read_text())
    state = json.loads((output/'state.json').read_text())
    if state['status'] in ('completed', 'failed', 'paused'):
        return state
    if state['status'] != 'prepared':
        raise ValueError('Interrupted controller requires explicit review; no automatic restart')

    def verify():
        if state['plan_sha256'] != digest(plan) or source_fingerprint() != plan['source'] or runtime_identity() != plan['runtime']:
            raise ValueError('Frozen source/runtime/plan changed')
        for path, key in [(RECIPE, 'recipe_sha256'), (PROTOCOL, 'protocol_sha256'),
                          (Path(plan['config']['data']['path']), 'corpus_sha256'),
                          (RESOURCE, 'resource_review_sha256'), (PILOT/'plan.json', 'pilot_plan_sha256')]:
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
            check_budget(plan, state['records'], used, shutil.disk_usage(output).free)
            queue = [s for s in specs(state['selection']) if s['id'] not in done]
            if not queue:
                if state['selection'] is None:
                    state['selection'] = choose(state['records'])
                    atomic_json(output/'selection.json', state['selection'])
                    atomic_json(output/'state.json', state); continue
                state.update(status='completed', active=None, worker_pid=None, controller_pid=None, conclusion=conclusion(state['records'], state['selection']))
                atomic_json(output/'state.json', state); return state
            spec = queue[0]; trial = output/spec['id']; trial.mkdir(exist_ok=False)
            cfg = config(trial, spec['family'], spec['seed'], spec['lr'])
            atomic_json(trial/'request.json', {'spec': spec, 'config': cfg, 'result_path': str(trial/'result.json')})
            state.update(status='running', active=spec, controller_pid=os.getpid()); atomic_json(output/'state.json', state)
            print('START '+spec['id'], flush=True)
            with (trial/'worker.log').open('x') as log:
                proc = subprocess.Popen([sys.executable, '-m', 'src.research.tokenizer_table_study', 'worker', '--output', str(trial/'request.json')], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(study_fd, worker_fd))
                state['worker_pid'] = proc.pid; atomic_json(output/'state.json', state)
                started = time.monotonic()
                try:
                    while True:
                        try:
                            code = proc.wait(timeout=2)
                            break
                        except subprocess.TimeoutExpired:
                            used = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
                            reason = ('stop requested' if (output/'STOP').exists() else
                                      'timeout' if time.monotonic()-started >= plan['job_seconds'] else
                                      'disk guard' if used > plan['max_output_bytes'] or
                                      shutil.disk_usage(output).free < plan['reserve_bytes'] or
                                      sum(p.stat().st_size for p in trial.rglob('*') if p.is_file()) > plan['per_trial_budget_bytes'][spec['family']] else None)
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
                except BaseException:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try: proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                    raise
            verify()
            if code or not (trial/'result.json').exists():
                state['records'].append(dict(**spec, status='failed', exit_code=code)); state['status'] = 'failed'
                atomic_json(output/'state.json', state); return state
            result = json.loads((trial/'result.json').read_text()); checked_panel([result], [spec])
            if result.get('holdout_consumed') is not False or result.get('request_sha256') != file_digest(trial/'request.json'):
                raise ValueError('Result request or holdout mismatch')
            if sum(p.stat().st_size for p in trial.rglob('*') if p.is_file()) > plan['per_trial_budget_bytes'][spec['family']]:
                raise RuntimeError('Completed trial exceeded frozen allocation')
            for name in ('raw', 'h200'):
                if file_digest(Path(result[name]['checkpoint'])) != result[name]['checkpoint_sha256']:
                    raise ValueError('Changed completed checkpoint')
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
