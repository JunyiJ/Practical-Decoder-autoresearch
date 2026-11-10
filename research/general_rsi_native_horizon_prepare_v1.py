"""Check the narrow worker adaptation and admit a fresh native horizon panel."""
import ast
import copy
import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

from research.general_rsi_native_horizon_worker_v1 import put, remaining_guard_seconds
from research.general_rsi_training_clock_v1 import TrainingClock

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/tokenizer-evidence/general-rsi-native-horizon-001'
OLD = ROOT/'research/tokenizer-evidence/general-rsi-equal-time-001'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def functions(path):
    return {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}


def check_adaptation():
    old = functions(ROOT/'research/general_rsi_equal_time_worker_v2.py')
    new = functions(ROOT/'research/general_rsi_native_horizon_worker_v1.py')
    expected = copy.deepcopy(old['execute'])
    selections = [n for n in expected.body if isinstance(n, ast.If) and "selection.json" in ast.unparse(n)]
    assert len(selections) == 1
    expected.body.remove(selections[0])
    assert ast.dump(expected) == ast.dump(new['execute'])
    exact = ['put', 'replace', 'stop', 'stopped', 'boot', 'retire', 'remaining_guard_seconds', 'main']
    for name in exact: assert ast.dump(old[name]) == ast.dump(new[name]), name
    old_review = functions(ROOT/'research/general_rsi_equal_time_review_v1.py')['review']
    new_review = functions(ROOT/'research/general_rsi_native_horizon_review_v1.py')['review']
    expected = copy.deepcopy(old_review)
    selections = [n for n in expected.body if isinstance(n, ast.If) and 'selection.json' in ast.unparse(n)]
    assert len(selections) == 1
    expected.body.remove(selections[0])
    assert ast.dump(expected) == ast.dump(new_review)
    for seconds in (7200, 14400):
        clock = TrainingClock(seconds*10**9)
        for fraction, expected_ratio in ((0, 1), (2/3, 1), (5/6, .55), (1, .1)):
            clock.spent_ns = round(seconds*10**9*fraction)
            assert math.isclose(clock.lr_ratio(), expected_ratio, abs_tol=1e-12)
    fixture = dict(slots=[dict(timeout_seconds=10800), dict(timeout_seconds=22800)],
                   max_attempts_per_slot=2, tranche_seconds=100800)
    assert remaining_guard_seconds(fixture, 0, 200, 200, 2) == 10600
    assert remaining_guard_seconds(fixture, 1, 10, 100700, 1) == 100
    rejected = 0
    for slot, slot_spent, total, attempt in ((0, 10800, 10800, 2), (1, 0, 100800, 1), (0, 0, 0, 3)):
        try: remaining_guard_seconds(fixture, slot, slot_spent, total, attempt)
        except AssertionError: rejected += 1
    assert rejected == 3
    return dict(qualified=True, execute_ast_equal_after_removing_lr_selection=True,
        review_ast_equal_after_removing_lr_selection=True, identical_worker_functions=exact,
        new_horizon_schedule_checks=8, cumulative_guard_checks=5,
        scope='No new numerical qualification. Original training, restart, scoring, generation and retirement paths retained. Changes only campaign paths, fixed LR binding, six-slot range, module dispatch and state labels.')


def main():
    assert not OUT.exists()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        state = read(ROOT/'research_state/general-rsi-state.json')
        assert not state.get('stop_requested', False) and state['active_worker_pid'] is None
        assert sha(OLD/'plan.json') == '4f51d54569d8b649c9609d25d42ba5bbaefdfd390db99b3283d5efc1c2c1165d'
        old = read(OLD/'plan.json')
        sources = dict(old['sources'])
        for name, digest in sources.items(): assert sha(ROOT/name) == digest, name
        for name, digest in old['qualification_evidence'].items(): assert sha(ROOT/name) == digest, name
        for asset in old['codebook_assets']: assert sha(asset['path']) == asset['sha256']
        # No training/validation/test payload scan. Search prior recorded seed roles only.
        scan = subprocess.run(['rg', '-l', '62111022', 'research', 'research_state', 'research_runs',
                               '--glob', '*.json', '--glob', '*.jsonl'], cwd=ROOT, text=True, capture_output=True)
        assert scan.returncode == 1 and not scan.stdout and not scan.stderr
        adaptation = check_adaptation()
        slots = []
        names = ('native_initializer', 'composer_initializer', 'train_sampler', 'validation_sampler')
        for bundle in (1, 2, 3):
            roles = dict(zip(names, range(62111022001+100*bundle, 62111022005+100*bundle)))
            for horizon, seconds, timeout in [('short', 7200, 10800), ('long', 14400, 22800)]:
                slots.append(dict(slot=len(slots), phase='confirmation', arm='native', horizon=horizon,
                    bundle=bundle, lr=.0025, budget_ns=seconds*10**9, roles=roles, timeout_seconds=timeout))
        for name in ('general_rsi_native_horizon_worker_v1.py', 'general_rsi_native_horizon_review_v1.py',
                     'general_rsi_native_horizon_prepare_v1.py', 'GENERAL_RSI_NATIVE_EXPOSURE_BOUNDARY.md',
                     'GENERAL_RSI_NATIVE_HORIZON_PROTOCOL.md'):
            sources['research/'+name] = sha(ROOT/'research'/name)
        curves = ROOT/'research/tokenizer-evidence/general-rsi-next-exposure-analysis-001/curves.json'
        native_results = [read(OLD/f'slot-{i:02d}/result.json') for i in (4, 6, 8)]
        observed_weight = max(e['weights']['size'] for r in native_results for e in r['endpoints'])
        observed_update = max(r['training_ns']/1e9/r['steps'] for r in native_results)
        # Cap-based allocation covers new slot metadata and headers, not just historical byte counts.
        outputs = 12*128*1024**2
        overlap = 2*512*1024**2
        metadata = 256*1024**2
        failure = 5*1024**3
        incremental = outputs+overlap+metadata+failure
        reserve = 10*1024**3
        free = shutil.disk_usage(ROOT).free
        assert free >= reserve+incremental
        historical = old['forecast']
        score = historical['measured_full_validation_seconds']['native']
        generation = historical['measured_generation_seconds']['native']
        updates = 64800/observed_update
        checkpoint_count = sum(math.ceil(s['budget_ns']/old['checkpoint_interval_ns'])+2 for s in slots)
        costs = dict(training_seconds=64800, endpoint_and_conservative_probe_seconds=24*score,
            generation_seconds=6*generation, checkpoint_seconds=checkpoint_count*3,
            initialization_seconds=6*30, outside_clock_logging_seconds=updates*.01)
        estimate = sum(costs.values())
        assert estimate < 100800
        forecast = dict(measured_weight_bytes=observed_weight, final_weight_count=12,
            per_weight_cap_bytes=128*1024**2, final_output_cap_bytes=outputs,
            active_restart_serialization_overlap_bytes=overlap, metadata_bytes=metadata,
            failed_write_allowance_bytes=failure, incremental_peak_bytes=incremental,
            reserve_bytes=reserve, required_free_bytes=reserve+incremental, actual_free_bytes=free,
            predicted_updates=updates, checkpoint_count_allowance=checkpoint_count,
            cost_components_seconds=costs, estimated_seconds=estimate, tranche_seconds=100800,
            failure_and_variability_allowance_seconds=100800-estimate,
            assumptions='All existing retained files excluded from free space. Final weights allocated at enforced128MiB caps. Restart pair512MiB each. Metadata includes preparation, all generations, logs and reviews. Checkpoint3s versus prior native mean about1.5s; initialization30s and logging0.01s/update are engineering allowances. Probes charged as full validation. Failures consume the same cumulative bounds.')
        evidence = dict(old['qualification_evidence'])
        for p in [OLD/'plan.json', OLD/'decision.json', curves]+[OLD/f'slot-{i:02d}/result.json' for i in (4, 6, 8)]:
            evidence[str(p.relative_to(ROOT))] = sha(p)
        plan = dict(kind='general_rsi_native_horizon_panel_v1', sources=sources,
            qualification_evidence=evidence, torch_version=old['torch_version'], codebook_assets=old['codebook_assets'],
            slots=slots, seed_role_audit='No prior62111022 matches in research/research_state/research_runs JSON or JSONL; three new paired bundles; bootstrap62111022999.',
            ema_half_life_ns=old['ema_half_life_ns'], ema_rule=old['ema_rule'],
            checkpoint_interval_ns=old['checkpoint_interval_ns'], probe_documents=old['probe_documents'],
            generation_protocol=copy.deepcopy(old['generation_protocol']),
            acceptance=dict(endpoint='time_ema', candidate='long', baseline='short',
                all_three_deltas_negative=True, mean_delta_at_most=-.02, source_regression_at_most=.05,
                bootstrap=dict(seed=62111022999, replicates=10000,
                    method='PCG64 resample108documentIDs; average paired seed loss sums before resampling; sampled target-count denominator; linear99th percentile upper<0'),
                otherwise='retain short recipe; no best seed; no historical pooling or generation rescue'),
            max_attempts_per_slot=2, tranche_seconds=100800, rss_limit_bytes=4*1024**3,
            mps_limit_bytes=6*1024**3, reserve_bytes=reserve, incremental_peak_bytes=incremental,
            forecast=forecast, test_opened=False)
        plan['generation_protocol']['scope'] = 'native_horizon_confirmation_only; inherited unchanged prompts, decoders and rubric'
        OUT.mkdir()
        put(OUT/'adaptation-review.json', dict(**adaptation, sources={k:v for k,v in sources.items() if 'native_horizon' in k}))
        plan['qualification_evidence'][str((OUT/'adaptation-review.json').relative_to(ROOT))] = sha(OUT/'adaptation-review.json')
        put(OUT/'plan.json', plan)
        put(OUT/'admission.json', dict(plan_sha256=sha(OUT/'plan.json'),
            admitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), shared_lock_held=True,
            new_binding_qualified=True, prior_numerical_qualifications_reused=True,
            scientific_scores_seen=False, forecast=forecast))
        print(json.dumps(dict(binding=sha(OUT/'plan.json'), forecast=forecast), indent=2))


if __name__ == '__main__': main()
