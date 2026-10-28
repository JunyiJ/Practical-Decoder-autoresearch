"""Draft v2 quality-stage audit binding; read-only, never deletes or dispatches.

The future v2 frozen-plan gate supplies independent expected requests and context.
This gate binds saved numerical, tensor and resource reviews before compiling a
cleanup manifest. It cannot replace execution of those reviews. Pilot cache
review, campaign-wide inventory, lock ownership and deletion remain separate.
"""
import math
import json
from pathlib import Path

from .protocol import digest
from . import tokenizer_wide_hybrid_rules as rules
from .tokenizer_wide_hybrid_retention import allocations, boundary_candidates, RESERVE
from .tokenizer_wide_hybrid_retention_receipts import (
    _check_file, _document, _path, _sha, planned_manifest,
)

NUMERICAL_CHECKS = ('geometry', 'cache', 'optimizer', 'resume', 'ema')
VALUE_CHECKS = ('finite_model_ema_optimizer_export_values',
                'exact_export_matches_stored_ema', 'persistent_buffers_unchanged')
NONCLAIMS = ('optimizer_structure_audited', 'optimizer_resume_audited',
             'ema_recurrence_audited', 'adopted')
RESOURCE_LIMITS = dict(rss_limit_bytes=4*1024**3, driver_limit_bytes=6*1024**3,
                       reserve_bytes=RESERVE, timeout_seconds=7200,
                       log_limit_bytes=4*1024**2)

PILOT_RESOURCE_LIMITS = dict(RESOURCE_LIMITS, timeout_seconds=600)


OUTPUT_FAILURES = tuple(name+suffix for name in
    ('training-failure.json','evaluation-failure.json','worker-failure.json','tensor-failure.json') for suffix in ('','.tmp')) + (
    'training-attempt.json.tmp','training-outputs.json.tmp','training-summary.json.tmp',
    'metrics.jsonl.tmp','evaluation-attempt.json.tmp','result.json.tmp',
    'worker-attempt.json.tmp','worker-outputs.json.tmp','tensor-attempt.json.tmp','tensor-audit.json.tmp')


def reject_output_failures(trial):
    _require(not any((trial/name).exists() or (trial/name).is_symlink() for name in OUTPUT_FAILURES),
             'Failed or partial training/evaluation output cannot pass stage review')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _load(root, name, evidence):
    _require(name in evidence, 'Required evidence file is not bound')
    return _document(root, dict(path=name, sha256=evidence[name]))


def numerical_review(root, reference, context):
    """Nine actual rank-policy cases are mandatory; geometry alone cannot pass."""
    report = _document(root, reference)
    cases = [dict(width=512, family=f, divisor=d, checks=dict.fromkeys(NUMERICAL_CHECKS, True))
             for f in rules.FAMILIES for d in (8, 4, 2)]
    _require(report.get('kind') == 'numerical' and report.get('status') == 'passed'
             and digest(report.get('checks')) == digest(dict.fromkeys(('all_declared_ranks',)+NUMERICAL_CHECKS, True))
             and digest(report.get('cases')) == digest(cases), 'Nine-case numerical review incomplete')
    for key in ('source_sha256', 'runtime_sha256', 'corpus_sha256'):
        _require(report.get(key) == context[key], 'Numerical provenance changed')
    traces = report.get('artifacts')
    _require(isinstance(traces, dict) and bool(traces) and reference['path'] not in traces,
             'Numerical review requires independent trace artifacts')
    for path, sha in traces.items():
        _check_file(root, path, sha)
    return {reference['path']: reference['sha256'], **traces}


def _resource_review(report, process, context, request_sha, result_sha, allowance, *, kind, limits):
    """Validate the prospective v2 wrapper's terminal observations.

    These are sampled observations, not instantaneous memory guarantees. Current
    v1 quality workers do not emit this report; their results cannot satisfy it.
    """
    _require(report.get('kind') == kind
             and report.get('status') == 'completed'
             and report.get('plan_sha256') == context['plan_sha256']
             and report.get('request_sha256') == request_sha
             and report.get('result_sha256') == result_sha
             and digest(report.get('limits')) == digest(limits)
             and report.get('violations') == [], 'Resource review identity/limits incomplete')
    samples = report.get('samples')
    _require(isinstance(samples, list) and 2 <= len(samples) <= limits['timeout_seconds']+3,
             'Resource observations missing or exceed bound')
    previous = -1
    for row in samples:
        _require(set(row) == {'elapsed_seconds', 'peak_rss_bytes', 'mps_driver_bytes', 'free_bytes'},
                 'Resource observation schema changed')
        t = row['elapsed_seconds']
        _require(_finite(t) and previous <= t < limits['timeout_seconds'], 'Resource elapsed time invalid')
        previous = t
        for key in ('peak_rss_bytes', 'mps_driver_bytes', 'free_bytes'):
            _require(type(row[key]) is int and row[key] >= 0, 'Resource byte observation invalid')
        _require(0 < row['peak_rss_bytes'] < limits['rss_limit_bytes']
                 and row['mps_driver_bytes'] < limits['driver_limit_bytes']
                 and row['free_bytes'] >= RESERVE, 'Observed resource limit violated')
    _require(any(r['mps_driver_bytes'] > 0 for r in samples), 'No MPS driver observation')
    _require(process.get('status') == 'completed' and type(process.get('returncode')) is int
             and process['returncode'] == 0 and process.get('error') is None
             and process.get('reserve_bytes') == RESERVE and process.get('timeout_seconds') == limits['timeout_seconds']
             and process.get('output_limit_bytes') == allowance
             and process.get('log_limit_bytes') == limits['log_limit_bytes'],
             'Serial process did not finish within declared limits')
    _require(_finite(process.get('elapsed_seconds')) and process['elapsed_seconds'] < limits['timeout_seconds']
             and type(process.get('required_free_bytes')) is int
             and process['required_free_bytes'] >= RESERVE, 'Process time/storage contract invalid')
    observation = process.get('observation', {})
    for key in ('free_bytes', 'output_bytes', 'log_bytes'):
        _require(type(observation.get(key)) is int and observation[key] >= 0, 'Process observation missing')
    _require(observation['free_bytes'] >= RESERVE and observation['output_bytes'] <= allowance
             and observation['log_bytes'] <= limits['log_limit_bytes'], 'Process resource cap violated')


def resource_review(report, process, context, request_sha, result_sha, allowance):
    """Quality-only acceptance: the 7,200-second contract cannot be substituted."""
    return _resource_review(report, process, context, request_sha, result_sha, allowance,
                            kind='wide_hybrid_trial_resources_v2', limits=RESOURCE_LIMITS)


def pilot_resource_review(report, process, context, request_sha, result_sha, allowance):
    """Pilot-only acceptance: all training/export/cache work must fit 600 seconds."""
    return _resource_review(report, process, context, request_sha, result_sha, allowance,
                            kind='wide_hybrid_pilot_resources_v2', limits=PILOT_RESOURCE_LIMITS)


def bind_quality_boundary(root, boundary, records, rates, selection, *, context,
                          requests, trials, evidence, numerical_ref, selection_refs,
                          projection_ref):
    """Bind ALL stage members, including winners, to immutable reviews.

    trials maps exact stage indices to relative trial directories. requests is
    independently derived from the future frozen plan, not the saved results.
    evidence must contain the full immutable non-checkpoint inventory supplied
    by that controller; mandatory request/result/tensor/resource/process/log and
    selection files are checked here. This function neither walks all outputs
    beyond these stage trial directories or claim the frozen plan was verified.
    """
    root = Path(root).resolve()
    _require(boundary in ('lr', 'rank'), 'Only quality screening boundaries supported')
    decision = boundary_candidates(boundary, records, rates, selection)
    indices = set(decision['audit_required_indices'])
    _require(set(trials) == indices == set(requests) and all(type(i) is int for i in trials),
             'Complete exact stage trial/request bindings required')
    _require(set(context) == {'plan_sha256','source_sha256','runtime_sha256','corpus_sha256'},
             'Independent frozen provenance required')
    for sha in context.values(): _sha(sha)
    resolved_trials = [_path(root, name) for name in trials.values()]
    _require(len(set(resolved_trials)) == len(resolved_trials) and not any(
        a != b and a.is_relative_to(b) for a in resolved_trials for b in resolved_trials),
        'Stage trial directories must be distinct and disjoint')
    bound = dict(evidence)
    def add(name, sha):
        _require(name not in bound or bound[name] == sha, 'Conflicting evidence hashes')
        bound[name] = sha
    for name, sha in numerical_review(root, numerical_ref, context).items(): add(name, sha)
    projection = _document(root, projection_ref)
    budgets = allocations(projection)
    add(projection_ref['path'], projection_ref['sha256'])
    expected_choices = {'rates': rates} if boundary == 'lr' else {'rates': rates, 'selection': selection}
    _require(set(selection_refs) == set(expected_choices), 'Persisted selection references incomplete')
    for key, value in expected_choices.items():
        ref = selection_refs[key]
        _require(digest(_document(root, ref)) == digest(value), 'Persisted scientific selection differs')
        add(ref['path'], ref['sha256'])
    pairs = {}
    # Import uses torch definitions but neither creates a model nor loads data.
    from .tokenizer_wide_hybrid_scores import _score
    for index in sorted(indices):
        trial = _path(root, trials[index])
        stem = trials[index]
        reject_output_failures(trial)
        names = {k: stem+'/'+v for k, v in dict(request='request.json', result='result.json',
            tensor='tensor-audit.json', resource='resource-audit.json',
            process='process/process.json', log='process/worker.log').items()}
        docs = {k: _load(root, name, bound) for k, name in names.items() if k != 'log'}
        _require(names['log'] in bound, 'Process log must be retained and bound')
        log = _check_file(root, names['log'], bound[names['log']])
        _require(log.stat().st_size <= RESOURCE_LIMITS['log_limit_bytes'], 'Retained process log exceeds cap')
        rqsha, rssha = bound[names['request']], bound[names['result']]
        request, result, tensor = docs['request'], docs['result'], docs['tensor']
        _require(digest(request) == digest(requests[index]), 'Saved request differs from frozen expected request')
        spec = {k: records[index][k] for k in ('stage','family','seed','lr','divisor','width','steps')}
        _require(digest(request.get('spec')) == digest(spec)
                 and Path(request['result_path']).resolve() == trial/'result.json', 'Request spec/result path differs')
        _require(all(digest(result.get(k)) == digest(v) for k,v in spec.items())
                 and result.get('status') == 'completed' and result.get('holdout_consumed') is False
                 and result.get('request_sha256') == rqsha, 'Result identity/status/holdout invalid')
        training = result.get('training', {})
        _require(type(training.get('completed_steps')) is int and training['completed_steps'] == 1200
                 and type(training.get('weight_averaging', {}).get('snapshots')) is int
                 and training['weight_averaging']['snapshots'] == 41, 'Training or EMA incomplete')
        geometry = next(r for r in projection['records'] if r['width'] == 512 and
            r['family'] == spec['family'] and r['bottleneck_divisor'] == spec['divisor'])
        pair, original_files = {}, {str(root/names['request']): rqsha, str(root/names['result']): rssha}
        for role in ('raw', 'h200'):
            score = result[role]
            _score(score, geometry['parameters'])
            path = Path(score['checkpoint'])
            _require(path.is_absolute() and path.is_relative_to(trial/'checkpoints'), 'Checkpoint outside trial')
            name = str(path.relative_to(root))
            checked = _check_file(root, name, score['checkpoint_sha256'])
            pair[role] = dict(path=name, size=checked.stat().st_size, sha256=score['checkpoint_sha256'])
            original_files[str(path)] = score['checkpoint_sha256']
        _require(not (root/pair['raw']['path']).samefile(root/pair['h200']['path']), 'Raw/export inode alias')
        _require(digest(records[index]['ce']) == digest(result['h200']['mean_ce']), 'Selection CE differs from result')
        metadata = tensor.get('metadata', {})
        _require(tensor.get('kind') == 'quality' and tensor.get('adopted') is False
                 and tensor.get('status') == 'export_values_reviewed_pending_numerical_and_resource_audits'
                 and digest(tensor.get('files')) == digest(original_files), 'Tensor audit identity/files incomplete')
        _require(metadata.get('metadata_reviewed') is True and metadata.get('tensor_values_audited') is False
                 and metadata.get('adopted') is False and metadata.get('request_sha256') == rqsha
                 and metadata.get('result_sha256') == rssha
                 and digest(metadata.get('selection_row')) == digest(records[index])
                 and digest(metadata.get('checkpoint_hashes')) == digest({k:pair[k]['sha256'] for k in pair})
                 and digest(metadata.get('h200_loss_sums')) == digest(result['h200']['block_loss_sums'])
                 and digest(metadata.get('target_counts')) == digest(result['h200']['block_target_counts']),
                 'Metadata audit does not match selection/result')
        values = tensor.get('values', {})
        _require(all(values.get(k) is True for k in VALUE_CHECKS)
                 and all(values.get(k) is False for k in NONCLAIMS), 'Tensor value audit incomplete or overstated')
        counts = values.get('tensor_counts', {})
        _require(set(counts) == {'raw_model','stored_ema','optimizer','exported_model'}
                 and all(type(n) is int and n > 0 for n in counts.values())
                 and type(values.get('parameters_compared')) is int and values['parameters_compared'] > 0
                 and type(values.get('buffers_compared')) is int and values['buffers_compared'] >= 0,
                 'Tensor audit counts incomplete')
        resource_review(docs['resource'], docs['process'], context, rqsha, rssha,
                        budgets[spec['family'], spec['divisor']])
        from .tokenizer_wide_hybrid_resources import EVENT_LIMIT, verify_events
        event_ref = docs['resource'].get('events', {})
        _require(set(event_ref) == {'path','sha256'} and event_ref['path'] == 'resource-events.jsonl',
                 'Resource event trace reference missing')
        event_name = stem+'/'+event_ref['path']
        _require(bound.get(event_name) == event_ref['sha256'], 'Resource trace not in immutable inventory')
        event_path = _check_file(root, event_name, event_ref['sha256'])
        _require(event_path.stat().st_size <= EVENT_LIMIT, 'Resource trace exceeds cap')
        verify_events(docs['resource'], [json.loads(line) for line in event_path.read_text().splitlines()])
        pairs[index] = pair
    manifest = planned_manifest(boundary, records, rates, selection, pairs, bound)
    expected_weights = {entry['path']: entry for entry in manifest['delete']+manifest['retain']}
    stage_files = set()
    for trial in resolved_trials:
        _require(trial.is_dir(), 'Stage trial directory missing')
        for path in trial.rglob('*'):
            _require(not path.is_symlink(), 'Stage inventory contains a symlink')
            if path.is_dir(): continue
            _require(path.is_file(), 'Non-regular stage artifact')
            name = str(path.relative_to(root)); stage_files.add(name)
            _require(name in bound or name in expected_weights, 'Unbound stage artifact')
    _require(set(expected_weights) <= stage_files, 'Checkpoint omitted from stage inventory')
    for name, sha in bound.items(): _check_file(root, name, sha)
    for name, entry in expected_weights.items():
        _check_file(root, name, entry['sha256'], entry['size'])
    return dict(manifest=manifest, stage_audits_bound=True, stage_inventory_complete=True, inventory_completeness_verified=False,
                frozen_plan_verified=False, deletion_authorized=False, dispatch_authorized=False)
