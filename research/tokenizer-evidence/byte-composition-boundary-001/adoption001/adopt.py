"""Publish the scoped byte-input recipe after complete review and actual loading.

No model execution, new selection, training, or rewriting of frozen evidence.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def record(path):
    assert path.is_file() and not path.is_symlink()
    return dict(path=str(path), size=path.stat().st_size, sha256=sha(path))


def read(path):
    return json.loads(path.read_text())


def publish(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT/'recipe.json').exists() and not (OUT/'guard.json').exists()
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with CpuPrerequisiteGuard(OUT/'guard.json', ROOT, timeout_seconds=120,
                             rss_limit_bytes=1024**3, reserve_bytes=8*1024**3):
        review_path = BASE/'recovery-final-review001/review.json'
        runtime_path = BASE/'checkpoint-consumer-runtime001/result.json'
        assert sha(review_path) == '84c027105e908d5e9415dcfadedd030b00cadb4943cee246aaf7dfa5f85401cc'
        assert sha(runtime_path) == 'd296c3b75ee1a91d50d9794933eb5658dd413f5722339af36412204fa84b0549'
        review, runtime = read(review_path), read(runtime_path)
        assert review['accepted'] and review['practical_eligible'] and review['adopted'] is False
        assert runtime['accepted'] and runtime['runtime_loading_qualified'] and runtime['adopted'] is False
        assert runtime['final_review_sha256'] == sha(review_path)
        assert runtime['representative_slot'] == 10 and runtime['training_updates'] == runtime['quality_evaluations'] == 0
        assert runtime['cpu_rng_unchanged'] and runtime['mps_rng_unchanged']
        for directory in ('recovery-final-review001', 'checkpoint-consumer-runtime001'):
            path = BASE/directory/('review.guard.json' if directory.startswith('recovery-') else 'guard.json')
            guard = read(path); assert guard['status'] == 'finished' and guard['reason'] is None
        assert sha(BASE/'checkpoint-consumer-runtime001/guard.json') == runtime['guard_sha256']
        assert sha(BASE/'quality-recovery001/final.json') == review['final_sha256']
        final = read(BASE/'quality-recovery001/final.json')
        assert final['accepted_complete_panel'] and final['trajectories'] == 18 and final['updates'] == 21600
        assert read(BASE/'quality-recovery001/STOP')['status'] == 'complete_recovered_quality_panel_pending_independent_review'
        plan_path = BASE/'quality001/plan.json'; plan = read(plan_path)
        assert sha(plan_path) == review['original_plan_sha256']
        assert sha(BASE/'quality-recovery001/plan.json') == review['recovery_plan_sha256']
        assert sha(ROOT/'research/byte_input_checkpoint_consumer_v1.py') == runtime['consumer_sha256']
        for key in ('h200', 'parent'):
            r = runtime[key]; path = Path(r['path'])
            assert path.stat().st_nlink == 1 and path.stat().st_size == r['size'] and sha(path) == r['sha256']
        recipe = dict(kind='adopted_byte_input_recipe_v1', status='adopted_under_original_three_bundle_gates',
            candidate='Real-byte input composition, D256 four-block MLA, AdamW epsilon1e-6',
            previous_recipe=record(ROOT/'research/tokenizer-evidence/adamw-epsilon-preparation-001/adopted-width256-mla-epsilon-recipe.json'),
            scope='Equal training exposure on the frozen tokenized corpus, D256 four-block MLA and 1200 updates; improved full-validation quality, not equal-time or general deployment superiority.',
            plan=record(plan_path), recovery_plan=record(BASE/'quality-recovery001/plan.json'),
            source_inventory=dict(record=record(plan_path), field='sources'),
            final_review=record(review_path), runtime_loading=record(runtime_path),
            consumer=record(ROOT/'research/byte_input_checkpoint_consumer_v1.py'),
            loader=dict(module='research.byte_input_checkpoint_consumer_v1', function='load',
                plan_record=runtime['plan_record'], acceptance_record=runtime['acceptance_record'], qualified_backend='mps',
                caller_requirements='Admit the bound sources, hold the shared research lock, and apply resource guards. Reconstruct typed identity through this helper; do not pass JSON identity lists directly to inference.load.'),
            representative=dict(slot=10, reason='First fresh raw-byte candidate in frozen order; chosen before final outcomes, never loss-selected.',
                h200=runtime['h200'], parent=runtime['parent']),
            training_recipe=runtime['expected_identity_json']['recipe'],
            model_signature_json=runtime['expected_identity_json']['model'],
            selected_learning_rates=review['selection'],
            training_budget=dict(updates=1200, targets=1228800, screen_rates_per_arm=3, fresh_bundles=3),
            contrasts=review['contrasts'], lexical_content_supported=review['lexical_content_supported'],
            uncertainty=review['interpretation'], recovery_lineage=review['recovery_lineage'],
            accepted_slots=final['all_slot_acceptances'],
            cost_evidence=record(BASE/'cost-analysis002/result.json'),
            measured_costs=dict(scope='First fresh bundle, accepted saved timing protocol; not universal serving latency.',
                parameter_count=16351859, native_parameter_count=28492416,
                training_update_ratio=4.1538, cached_decode32_ratio_batch1=1.9010,
                cached_decode32_ratio_batch8=1.9921, prefill96_ratio_batch8=3.1369,
                speed_or_memory_win_claimed=False),
            holdout_scored=False, population_seed_confidence_claimed=False,
            future_baseline='Use this qualified model and recipe in genuinely new campaigns with fresh roles and explicit source applicability. Historical controllers and failed fixtures remain closed. Optimize the measured overhead without changing this completed comparison.')
    assert read(OUT/'guard.json')['status'] == 'finished'
    publish('recipe.json', recipe)
    publish('closure.json', dict(status='closed_adopted', at=datetime.now(timezone.utc).isoformat(),
        runner=record(Path(__file__).resolve()), guard=record(OUT/'guard.json'), recipe=record(OUT/'recipe.json'),
        quality_adopted=True, final_review=record(review_path), runtime_loading=record(runtime_path),
        original_failure_preserved=True, original_scientific_rules_unchanged=True,
        training_launched=False, quality_evaluations=0, selected_representative_slot=10,
        next_action='Integrate the demonstrated direct-byte hashing optimization outside frozen sources, then run the bounded equivalence/runtime check after lock release; no new quality campaign yet.'))
    publish('STOP', dict(status='completed_closed_adopted', recipe=record(OUT/'recipe.json'),
        closure=record(OUT/'closure.json'), no_replay=True))
print(json.dumps(dict(status='closed_adopted', recipe=str(OUT/'recipe.json'))))
