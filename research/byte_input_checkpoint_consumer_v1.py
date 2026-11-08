"""Load a bound accepted byte-input checkpoint using plan-derived typed identity.

This is not scientific selection or adoption. The caller supplies trusted plan
and acceptance file receipts, owns source admission, the shared research lock,
and resource guards for model loading. `prepare` builds metadata/codebook only;
`load` delegates all model reconstruction and tensor checks to qualified v1 APIs.
"""
import copy
import hashlib
import json
from pathlib import Path

PLAN_SHA = 'cd0dcc19ae94e09e3d14aa071043f6c89b498e482e942bc0bfeb65bda15d8ddd'
EVIDENCE = {'attempt.json', 'result.json', 'artifacts.json', 'validation.json',
            'inference.json', 'review.json', 'guard.json', 'STOP'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(record):
    path = Path(record['path'])
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size == record['size'] and sha(path) == record['sha256']
    return path


def read_bound(record):
    assert set(record) == {'path', 'size', 'sha256'} and 0 < record['size'] <= 8*1024**2
    path = verify(record); value = json.loads(path.read_text()); verify(record)
    return value


def prepare(plan_record, acceptance_record):
    """Prepare an original accepted slot; recovery-layout adoption is external."""
    assert plan_record['sha256'] == PLAN_SHA
    plan = read_bound(plan_record); receipt = read_bound(acceptance_record)
    assert receipt['accepted'] and receipt['no_adoption'] and set(receipt['evidence']) == EVIDENCE
    index = receipt['slot']; assert type(index) is int and 0 <= index < 14
    out = Path(acceptance_record['path']).parent
    for name, digest in receipt['evidence'].items():
        assert not (out/name).is_symlink() and sha(out/name) == digest
    read = lambda name: json.loads((out/name).read_text())
    assert read('STOP')['status'] == 'complete_original_scientific_slot'
    guard = read('guard.json'); assert guard['status'] == 'finished' and guard['reason'] is None
    row, attempt, artifacts = read('review.json'), read('attempt.json'), read('artifacts.json')
    assert row['accepted'] and row['slot'] == index and row['completed_steps'] == 1200
    assert row['averaging_snapshots'] == 41 and row['validation_targets'] == 34278
    assert attempt['plan_sha256'] == PLAN_SHA
    spec = plan['slots'][index]
    assert all(row[k] == spec[k] for k in ('slot', 'stage', 'bundle', 'variant'))
    assert attempt['spec'] == {k: row[k] for k in spec}
    for record in plan['sources']+plan['assets']+plan['prerequisites']:
        verify(record)
    assets = [verify(record).read_bytes() for record in plan['codebook_assets']]
    # Imports follow source admission; no model is constructed in prepare().
    from research.byte_input_model_v3 import Spec, build_codebook
    from research.byte_input_model_v3.contract import signature
    from research import byte_input_inference_v1 as inference
    roles = copy.deepcopy(plan['bundles'][row['bundle']]); variant = row['variant']
    book = None if variant == 'native' else build_codebook(
        *assets, variant, roles['nonlexical_key'] if variant == 'nonlexical_codes' else None)
    model_spec = Spec(variant) if book is None else Spec(variant, roles['composer_initializer'], book.identity)
    expected = dict(kind='byte_input_training_context_v2',
        recipe=copy.deepcopy(plan['recipes'][str(row['learning_rate'])]), roles=roles,
        binding=copy.deepcopy(plan['binding']), data=copy.deepcopy(plan['data']), model=signature(model_spec))
    inference.identity(expected)
    # Compare with recorded JSON only after independent reconstruction. Never
    # pass its tuple-erasing JSON representation as the expected typed identity.
    assert json.loads(json.dumps(expected)) == artifacts['identity']
    assert artifacts['parent'] == row['parent'] and artifacts['h200'] == row['h200']
    return dict(identity=expected, codebook=book, parent=row['parent'], h200=row['h200'], spec=attempt['spec'])


def load(plan_record, acceptance_record, backend):
    """Caller must hold the research lock and an appropriate CPU/MPS guard."""
    prepared = prepare(plan_record, acceptance_record)
    from research import byte_input_inference_v1 as inference
    model = inference.load(prepared['h200'], prepared['identity'], prepared['parent'], prepared['codebook'], backend)
    return model, prepared['identity'], prepared['h200']
