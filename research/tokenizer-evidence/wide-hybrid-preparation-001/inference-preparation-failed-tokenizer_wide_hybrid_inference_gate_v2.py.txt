"""Read-only, retention-aware binding of all three fresh inference triples.

Requires an idle completed slot23 and both locks under the caller's resource
and process guards. No inference/control append, launch, adoption or cleanup is
allowed by this gate. Production history is replayed on every call.
"""
from pathlib import Path
import shutil

from .protocol import digest, file_digest
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_cleanup import _locks
from .tokenizer_wide_hybrid_retention_receipts import _document, _path, _sha
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_plan_v2 import bind_campaign
from .tokenizer_wide_hybrid_inference_gate import CONTRACT, recheck
from .tokenizer_wide_hybrid_driver import _contract
from .tokenizer_wide_hybrid_rules import FAMILIES, SEEDS, confirmation_specs
from . import tokenizer_wide_hybrid_final_confirmation_history_v2 as history

KIND = 'wide_hybrid_inference_binding_v2'


def _triples(root, report, frozen, inventory, tokenizer):
    """Adapt already audited fresh pairs to the existing strict export loader."""
    specs = confirmation_specs(report['selection'])
    triples = {seed: {} for seed in SEEDS}
    for slot, spec in enumerate(specs, 15):
        record = report['records'][slot]
        _require(all(type(record.get(k)) is type(v) and record[k] == v for k, v in spec.items()),
                 'Exact ordered fresh seed triples required')
        stem = f'quality/{slot:02d}'
        request_ref = dict(path=stem+'/request.json', sha256=inventory[stem+'/request.json']['sha256'])
        result_ref = dict(path=stem+'/result.json', sha256=inventory[stem+'/result.json']['sha256'])
        request = _document(root, request_ref)
        result = _document(root, result_ref)
        options = frozen['requests']['quality'][str(slot)]
        _require(sum(digest(option) == digest(request) for option in options) == 1
                 and digest(request['spec']) == digest(spec)
                 and request['result_path'] == str(root/stem/'result.json'),
                 'Fresh request is not uniquely frozen for its slot')
        _require(result['request_sha256'] == request_ref['sha256']
                 and result['status'] == 'completed' and result['holdout_consumed'] is False,
                 'Fresh result identity differs')
        pair = report['checkpoint_pairs'][stem]
        for role in ('raw', 'h200'):
            item = pair[role]
            path = _path(root, item['path'])
            _require(path.parent == root/stem/'checkpoints'
                     and result[role]['checkpoint'] == str(path)
                     and result[role]['checkpoint_sha256'] == item['sha256']
                     and inventory[item['path']] == dict(size=item['size'], sha256=item['sha256']),
                     'Fresh export differs from retained audited pair')
        contract = _contract(request['config'])
        _require(digest(contract.get('tokenizer')) == digest(tokenizer),
                 'Current tokenizer differs from frozen inference input')
        triples[spec['seed']][spec['family']] = dict(trial=str(root/stem), request=request,
            request_sha256=request_ref['sha256'], result_sha256=result_ref['sha256'], training_contract=contract)
    _require(all(set(triple) == set(FAMILIES) for triple in triples.values()), 'Incomplete fresh triples')
    return triples


def bind_inference(writer, *, plan_sha256, completion_ref, projection_ref, rank_completion_ref):
    """Bind all seeds once; this result is evidence, never dispatch permission."""
    return _bind(writer,plan_sha256=plan_sha256,completion_ref=completion_ref,
        projection_ref=projection_ref,rank_completion_ref=rank_completion_ref)


def _bind(writer, *, plan_sha256, completion_ref, projection_ref, rank_completion_ref, inference_request_ref=None):
    _require(isinstance(writer, MetadataWriter), 'Locked metadata writer required')
    _sha(plan_sha256)
    _require(completion_ref.get('path') == 'meta/quality-dispatch-23.json', 'Final slot23 required')
    root = writer.root/'evidence'
    _locks(root, writer.lock_fds)
    census = snapshot(writer.root)
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(), 'STOP prevents inference binding')
    # Frozen campaign verifies source/runtime, numerical/input traces and exact
    # serving workload before any scientific history or corpus reconstruction.
    frozen = bind_campaign(root/'meta/plan.json', plan_sha256)
    plan = _document(root, dict(path='meta/plan.json', sha256=plan_sha256))
    tokenizer = plan['tokenizer']; token_sha = _sha(plan['training_token_sha256'])
    inventory = history.acceptance._inventory(root)
    controls = history.acceptance.readiness._controls(writer, 'not-a-control')
    options = {} if inference_request_ref is None else {'inference_request_ref':inference_request_ref}
    report = history._verify(writer, 23, plan_sha256, completion_ref,
        projection_ref=projection_ref, rank_completion_ref=rank_completion_ref, **options)
    _require(len(report['records']) == 24 and len(report['checkpoint_pairs']) == 12
             and len(report['stages']) == 3 and len(report['audited_indices']) == 12
             and report['historical_transport_verified'] is True and report['confirmation_audits_bound'] is True
             and report['storage']['projection']['phase'] == 'inference', 'Complete retained quality panel required')
    triples = _triples(root, report, frozen, inventory, tokenizer)
    files = dict(frozen['files'])
    for prefix, entries in ((root, inventory), (writer.root, controls)):
        for name, binding in entries.items():
            path = str(_path(prefix, name)); sha = binding['sha256']
            _require(path not in files or files[path] == sha, 'Conflicting inference file binding')
            files[path] = sha
    # Include all live evidence and controls, including original transport and
    # deletion journals. Intentionally deleted losers are proven by history;
    # they are not mistakenly demanded by the old v1 loading gate.
    recheck(files)
    again = bind_campaign(root/'meta/plan.json', plan_sha256)
    _require(digest(again) == digest(frozen), 'Frozen campaign changed during inference binding')
    _require(history.acceptance._inventory(root) == inventory
             and history.acceptance.readiness._controls(writer, 'not-a-control') == controls
             and snapshot(writer.root) == census, 'Inference evidence or control census changed')
    _locks(root, writer.lock_fds)
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink()
             and shutil.disk_usage(writer.root).free >= report['storage']['required_free_bytes'],
             'STOP or remaining capacity changed during inference binding')
    quality_inventory = inventory
    quality_controls = controls
    if inference_request_ref is not None:
        from .tokenizer_wide_hybrid_inference_preparation_v2 import _scope
        scope = _scope(writer,inference_request_ref,plan_sha256,completion_ref,projection_ref,rank_completion_ref)
        quality_inventory = {n:v for n,v in inventory.items() if n not in scope['evidence']}
        quality_controls = {n:v for n,v in controls.items() if n not in scope['controls']}
    return dict(kind=KIND, triples=triples, selection=report['selection'], tokenizer=tokenizer,
        token_sha256=token_sha, contract=dict(CONTRACT, version=2, tokenizer_sha256=digest(tokenizer), token_sha256=token_sha),
        files=files, history=report, plan_sha256=plan_sha256, completion_ref=completion_ref,
        records_sha256=digest(report['records']), checkpoint_pairs_sha256=digest(report['checkpoint_pairs']),
        retention_sha256=digest(report['stages']), inventory_sha256=digest(quality_inventory),
        controls_sha256=digest(quality_controls), inference_provenance_bound=True,
        adopted=False, dispatch_authorized=False, deletion_authorized=False, next_stage_authorized=False)
