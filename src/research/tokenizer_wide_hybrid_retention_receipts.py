"""Draft v2 read-only deletion provenance primitive, not an audit or dispatcher.

Inputs must come from independently bound, complete stage audits and saved
selections. This module does not establish that those audits passed. It verifies
that an exact planned removal finished, retained weights still match and bound
non-checkpoint evidence is unchanged. No deletion API exists here.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath

from .protocol import digest
from .tokenizer_wide_hybrid_retention import VERSION, boundary_candidates


def _sha(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected lowercase SHA256')
    return value


def _relative(name):
    if (type(name) is not str or not name or PurePosixPath(name).is_absolute()
            or str(PurePosixPath(name)) != name or '..' in PurePosixPath(name).parts):
        raise ValueError('Expected canonical relative artifact path')
    return name


def _path(root, name):
    name = _relative(name)
    path = root
    for part in PurePosixPath(name).parts:
        path = path/part
        if path.is_symlink():
            raise ValueError('Symlink artifact or ancestor refused')
    if path == root or not path.resolve().is_relative_to(root):
        raise ValueError('Artifact escaped its evidence root')
    return path


def _check_file(root, name, sha, size=None):
    path = _path(root, name)
    if not path.is_file() or size is not None and path.stat().st_size != size:
        raise ValueError('Retained artifact missing or size changed')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    if h.hexdigest() != _sha(sha):
        raise ValueError('Retained artifact hash changed')
    return path


def planned_manifest(boundary, records, rates, selection, pairs, evidence):
    """Compile exact allowlist from caller-bound audits, never authorize deletion.

    pairs maps every audit_required index to raw/H200 path,size,sha256 bindings.
    evidence maps every bound stage request/result/log/audit/selection to its hash.
    Completeness of evidence is enforced by the future stage controller, not by
    this low-level primitive. Callers must not treat a returned plan as an audit.
    """
    decision = boundary_candidates(boundary, records, rates, selection)
    if (set(pairs) != set(decision['audit_required_indices'])
            or any(type(i) is not int for i in pairs)):
        raise ValueError('All stage pairs, including selected screens, required')
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError('Bound non-checkpoint evidence required')
    for name, sha in evidence.items():
        _relative(name); _sha(sha)
    entries, seen = [], set(evidence)
    for index in sorted(pairs):
        if set(pairs[index]) != {'raw', 'h200'}:
            raise ValueError('Distinct raw/H200 pair required')
        for role in ('raw', 'h200'):
            item = pairs[index][role]
            if set(item) != {'path', 'size', 'sha256'}:
                raise ValueError('Exact checkpoint path/size/hash required')
            name = _relative(item['path']); _sha(item['sha256'])
            if name in seen or type(item['size']) is not int or item['size'] <= 0:
                raise ValueError('Aliased artifact or invalid checkpoint size')
            seen.add(name)
            entries.append(dict(index=index, role=role, **item))
    return dict(version=VERSION, status='planned', boundary=boundary,
                decision_sha256=digest(decision), records_sha256=digest(records),
                rates_sha256=digest(rates), selection_sha256=digest(selection),
                evidence=dict(evidence),
                delete=[e for e in entries if e['index'] in decision['candidate_indices']],
                retain=[e for e in entries if e['index'] in decision['retained_indices']])


def _document(root, ref):
    if set(ref) != {'path', 'sha256'}:
        raise ValueError('Document requires independently bound path and hash')
    path = _path(root, ref['path'])
    if not path.is_file() or path.stat().st_size > 4*1024**2:
        raise ValueError('Receipt document missing or exceeds 4MiB')
    _check_file(root, ref['path'], ref['sha256'])
    value = json.loads(path.read_text())
    _check_file(root, ref['path'], ref['sha256'])
    return value


def recheck_boundary(root, expected_manifest, manifest_ref, receipt_ref):
    """Explain missing candidate weights only using exact completed receipts.

    expected_manifest must be freshly reconstructed with planned_manifest from
    independent saved decisions/audits, not copied from the on-disk manifest.
    Caller owns locks and resource guards. All other inventory is still checked
    normally by the controller. This does not approve training or adoption.
    """
    root = Path(root).resolve()
    if (expected_manifest.get('version') != VERSION
            or expected_manifest.get('status') != 'planned'):
        raise ValueError('Only a v2 planned manifest is accepted')
    manifest = _document(root, manifest_ref)
    if digest(manifest) != digest(expected_manifest):
        raise ValueError('Manifest differs from independently derived candidates')
    entries = manifest['delete']+manifest['retain']
    bound_names = set(manifest['evidence']) | {e['path'] for e in entries}
    if (manifest_ref['path'] == receipt_ref['path']
            or manifest_ref['path'] in bound_names or receipt_ref['path'] in bound_names):
        raise ValueError('Manifest/receipt cannot alias bound artifacts')
    receipt = _document(root, receipt_ref)
    expected_receipt = dict(version=VERSION, status='completed',
                            manifest_sha256=manifest_ref['sha256'],
                            deleted=manifest['delete'],
                            protected_evidence_sha256=digest(manifest['evidence']))
    if digest(receipt) != digest(expected_receipt):
        raise ValueError('Deletion receipt incomplete or differs from allowlist')
    for name, sha in manifest['evidence'].items():
        _check_file(root, name, sha)
    for entry in manifest['retain']:
        _check_file(root, entry['path'], entry['sha256'], entry['size'])
    for entry in manifest['delete']:
        path = _path(root, entry['path'])
        if path.exists():
            raise ValueError('Deleted checkpoint reappeared or deletion did not finish')
    _check_file(root, manifest_ref['path'], manifest_ref['sha256'])
    _check_file(root, receipt_ref['path'], receipt_ref['sha256'])
    return dict(version=VERSION, deleted_files={e['path']: e['sha256'] for e in manifest['delete']},
                retained_pairs_rechecked=len(manifest['retain'])//2,
                non_checkpoint_files_rechecked=len(manifest['evidence']),
                audit_semantics_verified=False, dispatch_authorized=False)
