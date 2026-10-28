"""Read-only frozen pilot prerequisite gate; no prepare, dispatch or retry API."""
import os
from pathlib import Path

from .protocol import digest, file_digest
from .runner import source_fingerprint
from .tokenizer_study import runtime_identity
from .tokenizer_wide_config import RECIPE
from .tokenizer_wide_hybrid_driver import CORPUS_SHA256, PROTOCOL
from .tokenizer_wide_hybrid_pilot_config import CONTRACT, PROJECTION, config, specs, storage_contract
from .tokenizer_wide_hybrid_scores import _json, _require
from .tokenizer_wide_hybrid_pilot_cache import CACHE_CONTRACT
from . import tokenizer_wide_hybrid_rules as rules

CHECKS = ('geometry', 'cache', 'optimizer', 'resume', 'ema')


def panel_design():
    """Prospective conditional finite plan, not outcome-dependent choices."""
    ranks = rules.rank_specs(dict.fromkeys(rules.FAMILIES, rules.RATES[0]))
    for spec in ranks:
        spec['lr'] = 'selected_family_lr'
    fresh = rules.confirmation_specs({f: dict(lr=rules.RATES[0], divisor=4) for f in rules.FAMILIES})
    for spec in fresh:
        spec.update(lr='selected_family_lr', divisor='selected_family_divisor')
    return dict(width=512, quality_steps=1200, lr_screens=rules.lr_specs(),
                rank_probes=ranks, confirmations=fresh, trajectories=27,
                lr_selection='minimum_final_h200_ce_then_lower_lr',
                rank_selection='minimum_final_h200_ce_then_larger_divisor',
                controls=list(rules.CONTROLS), strict_mean_delta=-.02,
                strict_each_seed_and_upper_delta=0., neutral_delta=.01,
                minimum_state_reduction=.10, latency_gate=False,
                bootstrap_blocks=list(rules.BLOCKS), bootstrap_draws=2000,
                bootstrap_seed=20261028, upper_quantile=rules.GATE_QUANTILE,
                test_holdout='closed', retain_all_raw_h200=True)


def numerical_cases():
    return [dict(width=512, family=f, divisor=d, checks=dict.fromkeys(CHECKS, True))
            for f in rules.FAMILIES for d in (8, 4, 2)]


def recheck(files):
    _require(all(file_digest(Path(p)) == sha for p, sha in files.items()), 'Frozen pilot evidence changed')


def bind_pilot(plan_path, plan_sha256, family):
    """Bind one of three pilots only to a complete frozen panel and numerical review.

    This gate does not establish current disk fit or hold locks. The future
    controller must validate retained prefix, enforce remaining full-panel space,
    acquire locks, use watchdogs and reject existing attempts before dispatch.
    """
    _require(family in rules.FAMILIES, 'Undeclared pilot family')
    path = Path(plan_path).resolve()
    plan = _json(path, plan_sha256)
    root = Path(plan['output']).resolve()
    _require(path == root/'plan.json' and not (root/'STOP').exists(), 'Pilot plan location or STOP')
    _require(plan.get('kind') == 'wide_hybrid_pilot_v1' and plan.get('status') == 'frozen',
             'Frozen pilot plan required')
    _require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '1', 'Strict MPS required')
    source, runtime = source_fingerprint(), runtime_identity()
    _require(digest(plan.get('source')) == digest(source) and digest(plan.get('runtime')) == digest(runtime),
             'Frozen source/runtime differs')
    _require(plan.get('protocol_sha256') == file_digest(PROTOCOL)
             and plan.get('projection_sha256') == file_digest(PROJECTION)
             and digest(plan.get('pilot_contract')) == digest(CONTRACT)
             and digest(plan.get('panel_design')) == digest(panel_design())
             and digest(plan.get('storage_contract')) == digest(storage_contract()),
             'Frozen design/resource contract differs')
    token_sha = plan.get('training_token_sha256')
    _require(isinstance(plan.get('tokenizer'), dict) and bool(plan['tokenizer'])
             and isinstance(token_sha, str) and len(token_sha) == 64
             and all(c in '0123456789abcdef' for c in token_sha)
             and digest(plan.get('cache_contract')) == digest(CACHE_CONTRACT),
             'Frozen cache/tokenizer/input contract missing or changed')
    requests = []
    for index, spec in enumerate(specs()):
        trial = root/f"{index:02d}-pilot-{spec['family']}-s3491-d2"
        requests.append(dict(spec=spec, config=config(trial, spec['family']), result_path=str(trial/'result.json')))
    _require(digest(plan.get('requests')) == digest(requests), 'Frozen pilot requests differ')
    bound = {str(path): plan_sha256}
    for item in plan['files']:
        p = str(Path(item['path']).resolve())
        _require(p not in bound, 'Duplicate or self-referential frozen file')
        bound[p] = item['sha256']
    corpus = Path(requests[0]['config']['data']['path']).resolve()
    review_path = Path(plan['numerical_review']).resolve()
    required = {str(corpus): CORPUS_SHA256, str(RECIPE.resolve()): file_digest(RECIPE),
                str(PROTOCOL.resolve()): file_digest(PROTOCOL), str(PROJECTION.resolve()): file_digest(PROJECTION)}
    _require(all(bound.get(p) == sha for p, sha in required.items()) and str(review_path) in bound,
             'Missing frozen corpus/recipe/protocol/projection/numerical binding')
    recheck(bound)
    review = _json(review_path, bound[str(review_path)])
    expected_checks = dict.fromkeys(('all_declared_ranks',)+CHECKS, True)
    _require(review.get('kind') == 'numerical' and review.get('status') == 'passed'
             and review.get('source_sha256') == digest(source)
             and review.get('runtime_sha256') == digest(runtime)
             and review.get('corpus_sha256') == CORPUS_SHA256
             and digest(review.get('checks')) == digest(expected_checks)
             and digest(review.get('cases')) == digest(numerical_cases()),
             'Numerical review incomplete or from different geometry/source')
    traces = review.get('artifacts')
    _require(isinstance(traces, dict) and bool(traces), 'Numerical review lacks underlying evidence')
    trace_paths = [str(Path(p).resolve()) for p in traces]
    _require(len(set(trace_paths)) == len(trace_paths), 'Duplicate numerical evidence aliases')
    for (p, sha), resolved in zip(traces.items(), trace_paths):
        _require(resolved not in (str(review_path), str(path)) and bound.get(resolved) == sha,
                 'Numerical evidence missing from frozen files')
    _require(digest(source_fingerprint()) == digest(source) and digest(runtime_identity()) == digest(runtime)
             and not (root/'STOP').exists(), 'Source/runtime/STOP changed during prerequisite review')
    recheck(bound)
    request = next(r for r in requests if r['spec']['family'] == family)
    return dict(request=request, files=bound, pilot_contract=CONTRACT.copy(),
                storage_contract=storage_contract(), panel_design=panel_design(),
                tokenizer=plan['tokenizer'], training_token_sha256=token_sha,
                cache_contract=plan['cache_contract'], adopted=False)
