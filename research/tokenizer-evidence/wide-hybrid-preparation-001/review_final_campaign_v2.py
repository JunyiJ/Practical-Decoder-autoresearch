"""Frozen final quality/cost composition; no model, launch, cleanup or adoption.

Caller owns both locks, has confirmed idle processes and supplies an outer
resource guard. This source file must be included in the campaign's frozen
files. Return the review for exclusive publication OUTSIDE the sealed campaign.
"""
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.research.protocol import digest, file_digest
from src.research import tokenizer_wide_hybrid_rules as rules
from src.research import tokenizer_wide_hybrid_timing_review as timing
from src.research import tokenizer_wide_hybrid_inference_acceptance_v2 as acceptance
from src.research.tokenizer_wide_hybrid_retention_receipts import _document
from src.research.tokenizer_wide_hybrid_metadata_writer import json_encoder
from src.research.tokenizer_wide_hybrid_metadata import snapshot

LIMIT = 128 * 1024


def combine(results, measurements, selection, token_sha256):
    """Exact fresh H200 panel only; provenance is supplied by the wrapper below.

    Raw endpoints remain available for disclosure, never substituted to obtain
    a passing gate. Descriptive text/sample content does not affect selection.
    """
    specs = rules.confirmation_specs(selection)
    if len(results) != 9:
        raise ValueError('All nine ordered fresh confirmation results required')
    by_seed = {s: {} for s in rules.SEEDS}
    counts = None
    raw = []
    for result, spec in zip(results, specs):
        if (digest({k: result.get(k) for k in spec}) != digest(spec)
                or result.get('status') != 'completed' or result.get('holdout_consumed') is not False):
            raise ValueError('Fresh result order/spec/status/holdout differs')
        score = result['h200']; current = score['block_target_counts']
        if (current != [2048]*16+[1510] or score.get('target_count') != 34278
                or score.get('target_bytes') != 111536 or score.get('context') != 128):
            raise ValueError('Exact common validation coverage required')
        if counts is not None and current != counts:
            raise ValueError('Unaligned paired validation blocks')
        if not math.isclose(float(score['mean_ce']),sum(score['block_loss_sums'])/34278,
                            rel_tol=0,abs_tol=1e-10):
            raise ValueError('H200 mean differs from paired loss sums')
        counts = current
        by_seed[spec['seed']][spec['family']] = score['block_loss_sums']
        raw.append(dict(spec=spec, mean_ce=result['raw']['mean_ce'],
                        h200_mean_ce=score['mean_ce'], wall_seconds=result.get('wall_seconds')))
    losses = [[by_seed[s][f] for f in rules.FAMILIES] for s in rules.SEEDS]
    statistics = rules.paired_statistics(losses, counts)
    quality = rules.quality_gates(statistics)
    costs = timing.summarize(measurements, selection, token_sha256)
    strict = quality['strict_quality']
    neutral_memory = quality['neutral_quality'] and costs['memory_gate']
    return dict(kind='wide_hybrid_final_quality_cost_review_v2',
        status='arithmetic_complete_pending_provenance', selection=selection,
        statistics=statistics, quality=quality, costs=costs,
        adoption_eligible=bool(strict or neutral_memory),
        passing_routes=(["strict_quality"] if strict else []) +
                       (["neutral_quality_and_memory"] if neutral_memory else []),
        raw_endpoint_disclosure=raw, adopted=False, holdout_consumed=False,
        interpretation='Conditional on the finite adaptive selection and reused validation text; '
            'not an unconditional confidence claim, test-holdout result or evidence of larger-scale optimality.',
        decision_rule='strict_quality OR (neutral_quality AND measured_memory); latency is disclosed without a speed gate')


def review_completed_campaign(writer, reference):
    if reference.get('path') != 'inference/3513/review.json':
        raise ValueError('Final accepted inference boundary required')
    before = snapshot(writer.root)
    checked = acceptance.verify_boundary(writer, reference)
    bound, scope = checked['bound'], checked['scope']
    if (checked.get('quality_history_reaudited') is not True or scope['state'] != 'accepted'
            or [r['path'] for r in scope['previous_refs']] !=
            [f'inference/{s}/review.json' for s in rules.SEEDS]):
        raise ValueError('Complete independent quality and inference acceptance required')
    script = str(Path(__file__).resolve())
    if bound['files'].get(script) != file_digest(Path(script)):
        raise ValueError('Final decision procedure was not frozen before the campaign')
    root = writer.root/'evidence'
    result_refs, measurement_refs, controllers, results, measurements = [], [], [], [], []
    for seed in rules.SEEDS:
        controller = f'inference/{seed}/controller.json'
        if controller not in scope['evidence']:
            raise ValueError('All three final serial controller receipts required')
        controllers.append(dict(path=controller, sha256=scope['evidence'][controller]['sha256']))
        name = f'inference/{seed}/result.json'
        ref = dict(path=name, sha256=scope['evidence'][name]['sha256'])
        measurement_refs.append(ref); measurements.append(_document(root, ref))
    for spec in rules.confirmation_specs(bound['selection']):
        binding = bound['triples'][spec['seed']][spec['family']]
        path = Path(binding['trial'])/'result.json'
        ref = dict(path=str(path.relative_to(root)), sha256=binding['result_sha256'])
        result_refs.append(ref); results.append(_document(root, ref))
    report = combine(results, measurements, bound['selection'], bound['token_sha256'])
    report.update(status='complete_independent_quality_cost_review',
        plan_sha256=bound['plan_sha256'], final_boundary=reference,
        procedure_sha256=file_digest(Path(script)), quality_history_reaudited=True,
        selection_sha256=digest(bound['selection']),token_sha256=bound['token_sha256'],
        result_refs=result_refs, measurement_refs=measurement_refs, controller_refs=controllers,
        acceptance_sha256=digest(checked), protected_files_sha256=digest(bound['files']),
        external_publication_required=True)
    # Whole campaign provenance includes retained raw/H200 pairs and controls.
    acceptance.outputs.recheck(bound['files'])
    if snapshot(writer.root) != before or (writer.root/'STOP').exists():
        raise ValueError('Final campaign changed during review')
    if len((json_encoder().encode(report)+'\n').encode('ascii')) > LIMIT:
        raise ValueError('Final external review exceeds128KiB reserved allowance')
    return report
