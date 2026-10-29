"""Read-only byte envelopes, never fabricated campaign receipts or launch authority.

Reuse the hash-pinned completed prefix only as a schema/topology template. Real
prerequisite payloads stay byte exact. All substitutions exist only in memory.
The resulting ledger is deliberately NOT valid scientific or provenance data.
"""
import copy
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.research import tokenizer_wide_hybrid_plan_v2 as plan
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.protocol import digest, file_digest
from src.research.tokenizer_wide_hybrid_metadata import budgets, classify, MAX_FILES
from src.research.tokenizer_wide_hybrid_metadata_writer import json_encoder
from src.research.tokenizer_wide_hybrid_resources import MAX_SAMPLES, PILOT_MAX_SAMPLES

HERE = Path(__file__).resolve().parent
DESTINATION = ROOT / 'research_runs/tokenizer-wide-hybrid-001'
IDENTITY = Path('/private/tmp/wide-hybrid-real-prerequisite-bind-001')
FLOAT = -1.7976931348623157e308  # 24 ASCII bytes, maximal finite-double spelling.


def size(value):
    return len(json_encoder().encode(value).encode('ascii')) + 1


def padding(value, key='', *, meta=False):
    """Extra numeric bytes, without changing identity constants in real inputs.

    Inventory sizes are <=2GiB per checkpoint and <=5MiB per metadata group;
    other byte observations are signed-63-bit. Other live integers use ten
    digits (PID, counts <=2048, finite-panel indices). Fixed bigger ints retain
    their original length. Every finite float gets 24 bytes, including fixed LR.
    """
    if type(value) is float:
        return max(0, 24 - len(json_encoder().encode(value)))
    if type(value) is int:
        width = 19 if key.endswith('_bytes') else 10
        return max(0, width - len(str(value)))
    if isinstance(value, dict):
        return sum(padding(v, k, meta=meta) for k, v in value.items())
    if isinstance(value, list):
        return sum(padding(v, key, meta=meta) for v in value)
    return 0


def audit():
    manifest_path = HERE / 'inference-acceptance-reusable-completed-prefix.json'
    manifest = json.loads(manifest_path.read_text()); old = Path(manifest['root'])
    process_path = HERE / 'inference-dispatch-v2-process-review-final.json'
    process = json.loads(process_path.read_text())
    transport = next(x for x in process['fixtures']
                     if '/test_three_actual_cli_children0/' in x['root'])
    transport_root = Path(transport['root'])
    source, runtime = plan.source_fingerprint(), plan.runtime_identity()
    originals = {}

    def verify_prefix():
        assert {str(p.relative_to(old)): file_digest(p) for p in old.rglob('*')
                if p.is_file()} == manifest['files']
    verify_prefix()

    def relocate(value):
        if isinstance(value, str):
            value = value.replace(str(old), str(DESTINATION)).replace(str(IDENTITY), str(DESTINATION))
            value = value.replace(str(transport_root), str(DESTINATION))
            # Production time_ns in this finite 2026 campaign fits signed 63-bit.
            return re.sub(r'final_(\d{6})_\d+\.pt', r'final_\1_9999999999999999999.pt', value)
        if isinstance(value, dict):
            return {relocate(k): relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        return value

    identity_review = json.loads((HERE / 'actual-prerequisite-binding-v2-review.json').read_text())
    identity_path = IDENTITY / 'evidence/meta/plan.json'
    assert file_digest(identity_path) == identity_review['plan_sha256']
    actual_plan = relocate(json.loads(identity_path.read_text()))
    options = plan.request_options(DESTINATION)
    actual_plan['requests'] = plan.request_identity(options)
    assert digest(actual_plan['source']) == digest(source)
    assert digest(actual_plan['runtime']) == digest(runtime)
    bindings = {x['path']: x['sha256'] for x in actual_plan['files']}
    bindings[str(DESTINATION / 'evidence/meta/plan.json')] = '0' * 64
    initial, immutable_sizes = {}, {}
    for path in (IDENTITY / 'evidence/meta').iterdir():
        if path.name == 'plan.json':
            count = size(actual_plan)
        else:
            assert bindings[str(DESTINATION / 'evidence/meta' / path.name)] == file_digest(path)
            count = path.stat().st_size
        originals[str(path)] = file_digest(path)
        initial['meta/' + path.name] = dict(size=count, sha256='0' * 64)
        immutable_sizes['evidence/meta/' + path.name] = count
    for path, sha in identity_review['original_files'].items():
        assert file_digest(Path(path)) == sha
        originals[path] = sha

    controllers = {}
    for seed in (3511, 3512, 3513):
        name = f'evidence/inference/{seed}/controller.json'
        path = transport_root / name
        assert file_digest(path) == transport['files'][name]
        originals[str(path)] = file_digest(path)
        controllers[name] = relocate(json.loads(path.read_text()))
    added_rows = {n.removeprefix('evidence/'): dict(size=size(d)+padding(d), sha256='0'*64)
                  for n, d in controllers.items()}
    removed = {'meta/source.txt', 'meta/corpus.txt', 'meta/numerical.trace'}
    occurrences = dict(frozen_maps=0, full_inventories=0, delta_inventories=0)

    def substitute(value):
        if isinstance(value, dict):
            if 'frozen_files' in value:
                value['frozen_files'] = dict(bindings)
                occurrences['frozen_maps'] += 1
            if 'meta/plan.json' in value and isinstance(value['meta/plan.json'], dict):
                assert set(value['meta/plan.json']) == {'size', 'sha256'}
                for key in removed:
                    value.pop(key, None)
                value.update(copy.deepcopy(initial))
                # Conservative: count all three final controller rows even in
                # inventories from phases before any controller can exist.
                value.update(copy.deepcopy(added_rows))
                occurrences['full_inventories'] += 1
            if set(value) >= {'base', 'removed', 'added', 'full_count', 'full_sha256'}:
                occurrences['delta_inventories'] += 1
            for key, item in value.items():
                if key != 'frozen_files':
                    substitute(item)
        elif isinstance(value, list):
            for item in value:
                substitute(item)

    docs, byte_files = {}, {}
    for name in manifest['files']:
        if name.endswith('.pt') or name in immutable_sizes or name.removeprefix('evidence/') in removed:
            continue
        path = old / name
        if name.endswith('.json'):
            value = relocate(json.loads(path.read_text())); substitute(value)
            docs[relocate(name)] = value
        else:
            byte_files[relocate(name)] = path.stat().st_size
    docs.update(controllers)

    # Dense MPS metrics have this fixed finite schema. Optional logit-z floats
    # are included conservatively although these recipes leave them disabled.
    metric = {k: FLOAT for k in ('lr_multiplier', 'base_learning_rate', 'scheduled_learning_rate',
        'train_loss', 'train_total_loss', 'train_ce_loss', 'val_loss', 'val_ce_loss', 'ppl',
        'iter_time_sec', 'elapsed_time_sec', 'training_time_sec', 'rss_mb', 'accel_mem_mb',
        'train_logit_z_coefficient', 'train_logit_z_squared_mean', 'train_logit_z_penalty', 'train_logit_z_mean')}
    metric.update(averaging_snapshots=9999999999, step=9999999999, tokens_processed=9999999999,
        train_aux_loss=None, val_aux_loss=None, accel_mem_kind='mps_current_allocated',
        rss_kind='process_lifetime_peak', device='mps', cuda=None, moe_usage=None)
    os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT / 'research_state/tiktoken-cache')
    import tiktoken
    encoding = tiktoken.get_encoding('gpt2')
    token_bytes = max(len(x) for x in encoding._mergeable_ranks)
    sample_bytes = 6 * (len('ROMEO:\n'.encode()) + 96 * token_bytes)
    resource_bounds = {}
    for stage, count, steps, samples, metrics in (
        ('pilots', 3, 20, PILOT_MAX_SAMPLES, 2), ('quality', 24, 1200, MAX_SAMPLES, 5)):
        for slot in range(count):
            stem = f'evidence/{stage}/{slot:02d}'
            candidates = [options['pilots'][slot]] if stage == 'pilots' else options['quality'][str(slot)]
            request = max(candidates, key=size)
            docs[stem + '/request.json'] = request
            result = docs[stem + '/result.json']
            summary = dict(device='mps', metrics=[metric]*metrics, best_metric=metric, final_metric=metric,
                final_checkpoint=result['training']['final_checkpoint'], max_iters=steps,
                completed_steps=steps, training_time_sec=FLOAT, stop_reason='max_iters',
                best_checkpoint=None, eval_every=20 if stage=='pilots' else 300, eval_iters=10,
                weight_averaging=dict(spec=request['config']['training']['weight_averaging'],
                                      completed_steps=steps, snapshots=9999999999))
            docs[stem+'/training-summary.json'] = summary
            result['training'] = summary
            result['wall_seconds'] = FLOAT
            # NUL needs six ASCII escape bytes; the payload is sizing only.
            result['h200']['sample'] = '\0' * (sample_bytes // 6)
            result['h200']['sample_settings'] = dict(prompt='ROMEO:\n', seed=12345,
                temperature=.8, top_k=40, new_tokens=96)
            byte_files[stem+'/metrics.jsonl'] = metrics * (len(json.dumps(metric,sort_keys=True))+1)
            report = docs[stem+'/resource-audit.json']
            row = dict(elapsed_seconds=FLOAT, peak_rss_bytes=4294967295,
                       mps_driver_bytes=6442450943, free_bytes=2**63-1)
            report['samples'] = [row]*samples
            pretty = len(json.dumps(report, sort_keys=True, indent=2)) + 1
            # Header+terminal+pilot bindings allowance, independently of rows.
            events = samples * size(dict(event='sample', **row)) + 4096
            byte_files[stem+'/resource-events.jsonl'] = events
            byte_files[stem+'/process/worker.log'] = 4*1024**2
            resource_bounds[stage] = dict(samples=samples, pretty_report_bytes=pretty,
                events_bytes=events, cap_bytes=2*1024**2)

    limits = budgets(); groups = dict.fromkeys(limits, 0); ledger = dict(immutable_sizes)
    for name, value in docs.items():
        ledger[name] = size(value) + padding(value, meta=name.startswith('evidence/meta/'))
    ledger.update(byte_files)
    # Reserve a further 4KiB per non-meta JSON for success-schema strings and
    # receipt details absent from tiny fixtures. Excludes large resource rows,
    # which already have explicit full-duration envelopes above.
    for name in docs:
        if not name.startswith('evidence/meta/') and not name.endswith('resource-audit.json'):
            ledger[name] += 4096
    for name, count in ledger.items():
        groups[classify(name)] += count
    # One serialized control replacement can coexist with its old payload.
    control_overlap = max([v for n,v in ledger.items() if classify(n)=='control'] or [0])
    groups['control'] += control_overlap
    # Inference enforces a shared4MiB panel cap, stricter than per-seed logs.
    # The audit reserves enough for all modeled payloads plus the remaining
    # shared log allowance, rather than silently granting three4MiB logs.
    inference_payload = sum(v for n,v in ledger.items() if n.startswith('evidence/inference/'))
    inference_remaining = 4*1024**2 - inference_payload
    file_count = len(ledger) + 54 + 2  # all27 historical pairs, two atomic names
    for name, sha in originals.items():
        assert file_digest(Path(name)) == sha
    verify_prefix()
    assert plan.source_fingerprint()==source and plan.runtime_identity()==runtime
    failures = {g: dict(bound_bytes=n, cap_bytes=limits[g], excess=n-limits[g])
                for g,n in groups.items() if n>limits[g]}
    return dict(kind='wide_hybrid_full_metadata_size_review_v2',
        status='size_envelopes_fit_pending_schema_and_decision_review' if not failures else 'size_envelope_exceeds_cap',
        script_sha256=file_digest(Path(__file__)), source_sha256=digest(source),runtime_sha256=digest(runtime),
        schema_manifest_sha256=file_digest(manifest_path),controller_manifest_sha256=file_digest(process_path),
        original_files_unchanged=True, retained_prefix_files=len(manifest['files']),originals=originals,
        proposed_root=str(DESTINATION),initial_exact_bytes=immutable_sizes,occurrences=occurrences,
        ledger=ledger,groups={g:dict(bound_bytes=n,cap_bytes=limits[g],remaining=limits[g]-n) for g,n in groups.items()},
        violations=failures,resource_bounds=resource_bounds,gpt2_max_token_bytes=token_bytes,
        sample_json_content_bound_bytes=sample_bytes,control_replacement_overlap_bytes=control_overlap,
        conservative_file_count=file_count,file_cap=MAX_FILES,
        inference_payload_bytes=inference_payload,inference_shared_log_headroom=inference_remaining,
        total_metadata_bound_before_remaining_inference_log=sum(groups.values()),
        full_panel_fit_proven=False,plan_frozen=False,training_launched=False,
        qualifications=[
            'Virtual byte accounting only: no transformed receipt is saved in a campaign or claimed valid.',
            'Initial actual files are byte exact; dynamic schemas reuse a complete immutable synthetic history.',
            'All inventories include extra final controller rows; all historical54weight files count toward file peak.',
            'Only control replacement doubles payload; immutable publication hardlinks one inode and adds a name.',
            'Logs remain bounded runtime streams; unexpected failure/string growth stops the run rather than expanding caps.',
            'Other-group4KiB-per-document allowance is conservative padding, not a formal proof of every missing synthetic field.',
            'Final quality/cost composition and real-vs-template schema review remain prerequisites; no launch authority.'])


if __name__ == '__main__':
    with CpuPrerequisiteGuard(HERE/'full-metadata-v2-guard.json',ROOT,timeout_seconds=180):
        report = audit()
        with (HERE/'full-metadata-v2-review.json').open('x') as stream:
            json.dump(report,stream,indent=2); stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','violations','resource_bounds',
        'conservative_file_count','inference_shared_log_headroom')}))
