"""Read-only cumulative v2 cleanup provenance; no dispatcher or deletion API.

The future controller supplies independently reconstructed manifests, bound
execution/context references and its complete current evidence inventory. This
does not establish stage-audit semantics or frozen-plan readiness itself.
"""
from pathlib import Path
import time

from .protocol import digest
from .tokenizer_wide_hybrid_cleanup import (
    _execution_record, _check, _scan, TIMEOUT, MAX_FILES, DOCUMENT_CAP, ATTEMPT_CAP,
)
from .tokenizer_wide_hybrid_retention import VERSION
from .tokenizer_wide_hybrid_retention_receipts import _document, _path


def recheck_chain(root, stages, current_inventory):
    """Validate the exact pilot/LR/rank prefix and all surviving evidence.

    Each stage has exactly manifest, execution_ref and context_sha256. Only an
    LR-selected pair may be retired by the later rank boundary, with identical
    index/role/path/size/hash. Other earlier protected files stay protected.
    Caller holds worker locks and an outer CPU/resource guard throughout.
    current_inventory is controller-bound, never inferred to excuse missing data.
    """
    root = Path(root).resolve(); deadline = time.monotonic()+TIMEOUT
    if (not isinstance(stages,list) or not 1 <= len(stages) <= 3
            or any(set(s) != {'manifest','execution_ref','context_sha256'} for s in stages)):
        raise ValueError('Exact nonempty cleanup chain required')
    if [s['manifest']['boundary'] for s in stages] != ['pilot','lr','rank'][:len(stages)]:
        raise ValueError('Cleanup chain must be an ordered complete boundary prefix')
    if not isinstance(current_inventory,dict) or not 0 < len(current_inventory) <= MAX_FILES:
        raise ValueError('Complete current evidence inventory required')
    attempts = _path(root,'.retention')
    if (not attempts.is_dir() or {p.name for p in attempts.iterdir()}
            != {s['manifest']['boundary'] for s in stages}):
        raise ValueError('Every existing cleanup attempt must belong to the complete chain')
    live, deleted, lr_anchors, execution_refs = {}, {}, {}, []
    for stage in stages:
        if time.monotonic() >= deadline: raise TimeoutError('Cleanup chain time limit')
        manifest = stage['manifest']; boundary = manifest['boundary']
        inventory, candidates = _execution_record(root,manifest,stage['execution_ref'],stage['context_sha256'])
        if set(inventory) & set(deleted): raise ValueError('Previously deleted path reused in a later inventory')
        for name, binding in live.items():
            if inventory.get(name) != binding:
                raise ValueError('Later inventory omitted or changed earlier protected evidence')
        entries = manifest['delete']+manifest['retain']
        if boundary == 'rank':
            earlier = stages[1]['manifest']
            if manifest['rates_sha256'] != earlier['rates_sha256']:
                raise ValueError('LR selection changed across rank boundary')
            rank_anchors = {e['path']:e for e in entries if e['index'] < 9}
            if digest(rank_anchors) != digest(lr_anchors):
                raise ValueError('Rank stage must bind the same complete LR anchor pairs')
        for entry in entries:
            path = _path(root,entry['path'])
            if path.suffix != '.pt' or path.parent.name != 'checkpoints':
                raise ValueError('Only checkpoint pairs may appear in a cleanup manifest')
        for name, entry in candidates.items():
            if name in live and (boundary != 'rank' or lr_anchors.get(name) != entry):
                raise ValueError('Earlier protected evidence is not an eligible LR anchor')
        # Every immutable output from this execution becomes protected as well.
        review = _document(root,stage['execution_ref'])
        base = '.retention/'+boundary
        own = {base+'/'+name:sha for name,sha in review['files'].items()}
        own[stage['execution_ref']['path']] = stage['execution_ref']['sha256']
        if set(own) & set(inventory): raise ValueError('Execution output aliases its prior inventory')
        actual = _scan(root/base,None)
        if actual != {Path(name).name for name in own}:
            raise ValueError('Cleanup attempt contains unexpected or failed output')
        own_bytes = 0
        for name, sha in own.items():
            size = _path(root,name).stat().st_size
            if size > DOCUMENT_CAP: raise ValueError('Cleanup output document cap exceeded')
            own_bytes += size
        if own_bytes > ATTEMPT_CAP: raise ValueError('Cleanup attempt metadata cap exceeded')
        live = {name:binding for name,binding in inventory.items() if name not in candidates}
        live.update({name:dict(size=_path(root,name).stat().st_size,sha256=sha) for name,sha in own.items()})
        deleted.update(candidates)
        if boundary == 'lr': lr_anchors = {e['path']:e for e in manifest['retain']}
        execution_refs.append(stage['execution_ref'])
    if set(current_inventory) & set(deleted): raise ValueError('Deleted checkpoint included in current inventory')
    for name, binding in live.items():
        if current_inventory.get(name) != binding:
            raise ValueError('Current inventory omitted or changed protected evidence')
    if _scan(root,None) != set(current_inventory): raise ValueError('Current evidence inventory incomplete')
    for name, binding in current_inventory.items(): _check(root,name,binding,deadline)
    for name in deleted:
        if _path(root,name).exists(): raise ValueError('Deleted checkpoint reappeared')
    for ref in execution_refs: _document(root,ref)
    return dict(version=VERSION,status='verified_cleanup_chain',boundaries=len(stages),
                deleted_files=deleted,protected_files_verified=len(current_inventory),
                current_inventory_sha256=digest(current_inventory),
                audit_semantics_verified=False,dispatch_authorized=False)
