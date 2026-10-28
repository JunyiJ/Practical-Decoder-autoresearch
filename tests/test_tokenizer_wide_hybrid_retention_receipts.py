"""Tiny synthetic files only; these are not scientific audit or pilot evidence."""
import copy
import hashlib
import json

import pytest

from src.research import tokenizer_wide_hybrid_rules as rules
from src.research import tokenizer_wide_hybrid_retention_receipts as receipts
from src.research.protocol import digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(root, name, doc):
    path = root/name
    path.write_text(json.dumps(doc))
    return dict(path=name, sha256=sha(path))


@pytest.fixture
def fixture(tmp_path):
    records = [dict(s, status='completed', ce=5.) for s in rules.lr_specs()]
    rates = rules.choose_rates(records)
    pairs = {}
    for i in range(9):
        pairs[i] = {}
        for role in ('raw', 'h200'):
            name = f'{i}-{role}.pt'
            path = tmp_path/name
            path.write_bytes(f'fixture {i} {role}'.encode())
            pairs[i][role] = dict(path=name, size=path.stat().st_size, sha256=sha(path))
    evidence = {}
    for name in ('synthetic-audit.json', 'synthetic-selection.json'):
        path = tmp_path/name
        path.write_text('synthetic fixture only, no passing scientific review')
        evidence[name] = sha(path)
    manifest = receipts.planned_manifest('lr', records, rates, None, pairs, evidence)
    manifest_ref = save(tmp_path, 'manifest.json', manifest)
    for e in manifest['delete']:
        (tmp_path/e['path']).unlink()  # tiny disposable test fixtures only
    receipt = dict(version=receipts.VERSION, status='completed',
                   manifest_sha256=manifest_ref['sha256'], deleted=manifest['delete'],
                   protected_evidence_sha256=digest(evidence))
    receipt_ref = save(tmp_path, 'receipt.json', receipt)
    return tmp_path, records, rates, pairs, evidence, manifest, manifest_ref, receipt, receipt_ref


def test_bound_completed_receipt_explains_only_candidate_absence(fixture):
    root, records, rates, pairs, evidence, manifest, mref, receipt, rref = fixture
    # Reconstruct from independent bindings; never trust an on-disk allowlist.
    expected = receipts.planned_manifest('lr', records, rates, None, pairs, evidence)
    result = receipts.recheck_boundary(root, expected, mref, rref)
    assert len(result['deleted_files']) == 12
    assert result['retained_pairs_rechecked'] == 3
    assert result['non_checkpoint_files_rechecked'] == 2
    assert result['audit_semantics_verified'] is False
    assert result['dispatch_authorized'] is False


@pytest.mark.parametrize('mutation', ['partial', 'failed', 'wrong_manifest', 'extra', 'wrong_evidence'])
def test_receipt_cannot_relax_deletion_journal(fixture, mutation):
    root, _, _, _, _, manifest, mref, receipt, rref = fixture
    if mutation == 'partial': receipt['deleted'].pop()
    elif mutation == 'failed': receipt['status'] = 'failed'
    elif mutation == 'wrong_manifest': receipt['manifest_sha256'] = '0'*64
    elif mutation == 'extra': receipt['deleted'].append(copy.deepcopy(manifest['retain'][0]))
    else: receipt['protected_evidence_sha256'] = '0'*64
    rref = save(root, 'receipt.json', receipt)
    with pytest.raises(ValueError): receipts.recheck_boundary(root, manifest, mref, rref)


@pytest.mark.parametrize('mutation', ['metadata', 'missing_metadata', 'protected_weight', 'missing_weight', 'reappeared', 'symlink', 'manifest_hash', 'receipt_hash'])
def test_live_evidence_is_rechecked_even_with_completed_receipt(fixture, mutation):
    root, _, _, _, evidence, manifest, mref, _, rref = fixture
    if mutation == 'metadata': (root/next(iter(evidence))).write_text('changed')
    elif mutation == 'missing_metadata': (root/next(iter(evidence))).unlink()
    elif mutation == 'protected_weight': (root/manifest['retain'][0]['path']).write_bytes(b'changed')
    elif mutation == 'missing_weight': (root/manifest['retain'][0]['path']).unlink()
    elif mutation == 'reappeared': (root/manifest['delete'][0]['path']).write_bytes(b'not allowed')
    elif mutation == 'symlink': (root/manifest['delete'][0]['path']).symlink_to('nonexistent')
    elif mutation == 'manifest_hash': mref['sha256'] = '0'*64
    else: rref['sha256'] = '0'*64
    with pytest.raises(ValueError): receipts.recheck_boundary(root, manifest, mref, rref)


def test_changed_allowlist_cannot_delete_selected_pair(fixture):
    root, _, _, _, _, manifest, _, receipt, _ = fixture
    malicious = copy.deepcopy(manifest)
    malicious['delete'].append(malicious['retain'].pop())
    mref = save(root, 'manifest.json', malicious)
    receipt.update(manifest_sha256=mref['sha256'], deleted=malicious['delete'])
    rref = save(root, 'receipt.json', receipt)
    with pytest.raises(ValueError, match='independently derived'):
        receipts.recheck_boundary(root, manifest, mref, rref)


@pytest.mark.parametrize('mutation', ['missing_pair', 'missing_export', 'alias', 'metadata_alias', 'traversal', 'absolute', 'bool_size', 'bad_sha'])
def test_plan_requires_complete_unaliased_bindings(fixture, mutation):
    _, records, rates, pairs, evidence, _, _, _, _ = fixture
    item = pairs[0]['raw']
    if mutation == 'missing_pair': pairs.pop(0)
    elif mutation == 'missing_export': pairs[0].pop('h200')
    elif mutation == 'alias': pairs[0]['h200'] = copy.deepcopy(item)
    elif mutation == 'metadata_alias': item['path'] = next(iter(evidence))
    elif mutation == 'traversal': item['path'] = '../weight.pt'
    elif mutation == 'absolute': item['path'] = '/weight.pt'
    elif mutation == 'bool_size': item['size'] = True
    else: item['sha256'] = 'x'*64
    with pytest.raises(ValueError): receipts.planned_manifest('lr', records, rates, None, pairs, evidence)


def test_symlink_ancestor_refused(tmp_path):
    (tmp_path/'real').mkdir()
    (tmp_path/'link').symlink_to(tmp_path/'real', target_is_directory=True)
    with pytest.raises(ValueError, match='Symlink'):
        receipts._path(tmp_path, 'link/absent.pt')
