"""Read-only source/evidence review and synthetic cache-gate arithmetic.

No model, corpus, checkpoint, training worker, freeze or retry is constructed.
JSON is printed to stdout; callers preserve it under a new evidence filename.
"""
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.research.tokenizer_wide_hybrid_rules import FAMILIES, SEEDS, memory_gate

EVIDENCE = ROOT / 'research/tokenizer-evidence/wide-hybrid-preparation-001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review():
    projection = json.loads((EVIDENCE/'rank-storage-projection.json').read_text())
    historical = json.loads((EVIDENCE/'rank-review.json').read_text())
    # Do not silently use different geometry or test sources than the saved run.
    for name, expected in historical['hashes'].items():
        assert sha(ROOT/name) == expected, f'Historical rank evidence changed: {name}'
    rows = [r for r in projection['records'] if r['width'] == 512]
    sizes = {(r['family'], r['bottleneck_divisor']): r['cache_bytes_batch1'] for r in rows}
    assert len(rows) == len(sizes) == 9
    assert set(sizes) == set(itertools.product(FAMILIES, (8, 4, 2)))
    # Independent layer-state formula at context128, FP32, head32, conv4.
    for (family, divisor), size in sizes.items():
        kda_layer = 4*41*512
        mla_layer = 4*128*(512//divisor+32)
        expected = {'hybrid': 3*kda_layer+mla_layer,
                    'gated_mla': 4*mla_layer, 'kda': 4*kda_layer}[family]
        assert type(size) is int and size == expected
    combinations = []
    for divisors in itertools.product((8, 4, 2), repeat=3):
        choices = dict(zip(FAMILIES, divisors))
        state = {f: sizes[f, choices[f]] for f in FAMILIES}
        savings = {f: Fraction(state[f]-state['hybrid'], state[f]) for f in FAMILIES[1:]}
        eligible = all(v >= Fraction(1, 10) for v in savings.values())
        # Synthetic projections only, never actual fresh-seed measurements.
        synthetic = {(s,b): {f:b*n for f,n in state.items()} for s in SEEDS for b in (1,8)}
        assert memory_gate(synthetic) == eligible
        combinations.append(dict(divisors=choices, projected_batch1_state_bytes=state,
            projected_fraction_saved={f: str(v) for f,v in savings.items()},
            projected_memory_threshold_eligible=eligible))
    eligible = [r['divisors'] for r in combinations if r['projected_memory_threshold_eligible']]
    assert eligible == [dict(hybrid=8, gated_mla=2, kda=d) for d in (8,4,2)]
    cases = [dict(width=512, family=f, divisor=d,
        meta_geometry_and_optimizer_grouping='passed_retained_rank_review',
        cpu_full_cached_reset_causality_lineage='passed_7_token_batches_1_and_8',
        optimizer_update_resume_ema='not_established_under_rank_policy',
        mps_96_token_cache_workload='not_run') for f in FAMILIES for d in (8,4,2)]
    inputs = [EVIDENCE/n for n in ('rank-review.json','rank-storage-projection.json',
        'cpu-resource-failure-001.json','cpu-failure-review-001.md',
        'pilot-integration-review.json','cpu-pilot-integration-final-tests.log')]
    inputs += [ROOT/n for n in ('tests/test_tokenizer_wide_hybrid_rank.py',
        'tests/test_tokenizer_wide_hybrid.py','src/research/tokenizer_wide_hybrid_config.py',
        'src/research/tokenizer_wide_hybrid_rules.py','src/research/tokenizer_wide_hybrid_pilot.py',
        'src/research/memory_guard.py','src/research/tokenizer_study.py')]
    inputs.append(Path(__file__).resolve())
    return dict(status='preparation_review_not_launch_ready', at=datetime.now(timezone.utc).isoformat(),
        method='manual_test_source_coverage_review_plus_27_combination_exact_arithmetic',
        files={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        coverage=cases, projected_cache_combinations=combinations,
        projected_memory_eligible_selections=eligible,
        required_free_bytes=57224879357, observed_free_bytes=shutil.disk_usage(ROOT).free,
        qualifications=[
            'Geometry and seven-token CPU cache checks are retained prior evidence, not rerun here.',
            'Meta optimizer construction/grouping does not exercise optimizer updates or state reload.',
            'The one completed old quarter-rank hybrid continuation used WIDE_POLICY, not RANK_POLICY.',
            'No complete nine-case rank-policy optimizer/resume/EMA review exists; do not promote the old single case.',
            'Mocked integration receipts do not establish real model correctness or resource feasibility.',
            'Projected cache threshold eligibility is not measured memory or quality adoption evidence.',
            'Same-rank hybrid/control selections cannot pass the current 10% both-control memory gate.',
            'LR/rank selection remains minimum validation CE; do not select ranks to manufacture a memory pass.',
            'The strict-quality route remains available for every selected geometry if its full gates pass.',
            'The memory watchdog is a daemon started by the common worker and continues into the cache audit.',
            'The failed combined CPU fixture remains closed; no retry or launch is authorized by this review.'
        ], numerical_tests_run=False, model_or_corpus_loaded=False,
        real_training_or_mps_worker_dispatched=False, plan_frozen=False, adopted=False)


if __name__ == '__main__':
    print(json.dumps(review(), indent=2, allow_nan=False))
