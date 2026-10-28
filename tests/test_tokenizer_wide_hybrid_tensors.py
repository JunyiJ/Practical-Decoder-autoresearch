"""Tiny synthetic tensor pairs; no GPT, corpus, optimizer step or MPS work."""
import copy
import json

import pytest
import torch

from src.research import tokenizer_wide_hybrid_tensors as audit
from src.research.protocol import file_digest


def pair():
    raw = dict(model_state={'weight': torch.tensor([1., 2.]), 'buffer': torch.tensor([7])},
               optimizer_state={'optimizers': [{'state': {0: {'step': torch.tensor(20.),
                   'momentum_buffer': torch.tensor([.1, .2])}}, 'lr': .0025}]},
               training_state={'weight_averaging_state': {'parameter_names': ['weight'],
                   'averages': {'200': {'weight': torch.tensor([3., 4.])}}}})
    exported = dict(model_state={'weight': torch.tensor([3., 4.]), 'buffer': torch.tensor([7])},
                    averaged_weights={'parameter_names': ['weight']})
    return raw, exported


def test_values_and_scope():
    raw, exported = pair()
    report = audit.audit_values(raw, exported)
    assert report['exact_export_matches_stored_ema']
    assert report['finite_model_ema_optimizer_export_values']
    assert report['persistent_buffers_unchanged']
    assert report['tensor_counts'] == dict(raw_model=2, stored_ema=1, optimizer=2, exported_model=2)
    assert report['parameters_compared'] == report['buffers_compared'] == 1
    for name in ('optimizer_structure_audited', 'optimizer_resume_audited', 'ema_recurrence_audited', 'adopted'):
        assert report[name] is False


@pytest.mark.parametrize('target', ['raw', 'ema', 'optimizer', 'export'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite(target, value):
    raw, exported = pair()
    tensors = {'raw': raw['model_state']['weight'],
               'ema': raw['training_state']['weight_averaging_state']['averages']['200']['weight'],
               'optimizer': raw['optimizer_state']['optimizers'][0]['state'][0]['momentum_buffer'],
               'export': exported['model_state']['weight']}
    tensors[target][1] = value
    with pytest.raises(ValueError, match='Nonfinite'): audit.audit_values(raw, exported)


@pytest.mark.parametrize('bad', ['parameter', 'buffer', 'shape', 'dtype', 'names', 'duplicate_names',
    'missing_ema', 'extra_ema', 'extra_export', 'extra_half_life', 'export_order', 'optimizer_empty',
    'optimizer_scalar', 'optimizer_object', 'export_optimizer', 'export_training', 'raw_export',
    'noncontiguous', 'meta', 'sparse', 'raw_scalar', 'ema_shape'])
def test_invalid_values_or_structure(bad):
    raw, exported = pair();state = raw['training_state']['weight_averaging_state']
    if bad == 'parameter': exported['model_state']['weight'][0] += 1e-6
    elif bad == 'buffer': exported['model_state']['buffer'][0] += 1
    elif bad == 'shape': exported['model_state']['weight'] = torch.ones(3)
    elif bad == 'dtype': exported['model_state']['weight'] = exported['model_state']['weight'].double()
    elif bad == 'names': state['parameter_names'] = ['absent']
    elif bad == 'duplicate_names': state['parameter_names'] *= 2
    elif bad == 'missing_ema': state['averages']['200'] = {}
    elif bad == 'extra_ema': state['averages']['200']['extra'] = torch.ones(1)
    elif bad == 'extra_export': exported['model_state']['extra'] = torch.ones(1)
    elif bad == 'extra_half_life': state['averages']['400'] = state['averages']['200']
    elif bad == 'export_order': exported['averaged_weights']['parameter_names'] = []
    elif bad == 'optimizer_empty': raw['optimizer_state'] = {}
    elif bad == 'optimizer_scalar': raw['optimizer_state']['bad'] = float('nan')
    elif bad == 'optimizer_object': raw['optimizer_state']['bad'] = object()
    elif bad == 'export_optimizer': exported['optimizer_state'] = {}
    elif bad == 'export_training': exported['training_state'] = {}
    elif bad == 'raw_export': raw['averaged_weights'] = {}
    elif bad == 'noncontiguous': raw['model_state']['weight'] = torch.ones(4)[::2]
    elif bad == 'meta': raw['model_state']['weight'] = torch.ones(2, device='meta')
    elif bad == 'sparse': raw['model_state']['weight'] = torch.ones(2).to_sparse()
    elif bad == 'raw_scalar': raw['model_state']['buffer'] = 7
    elif bad == 'ema_shape': state['averages']['200']['weight'] = torch.ones(3)
    with pytest.raises(ValueError): audit.audit_values(raw, exported)


def test_chunk_boundary_and_tail(monkeypatch):
    monkeypatch.setattr(audit, 'CHUNK_ELEMENTS', 2)
    assert [len(t) for t in audit._chunks(torch.arange(5))] == [2, 2, 1]
    raw, exported = pair()
    raw['model_state']['weight'] = torch.arange(5.)
    raw['training_state']['weight_averaging_state']['averages']['200']['weight'] = torch.arange(5.)
    exported['model_state']['weight'] = torch.arange(5.)
    assert audit.audit_values(raw, exported)['parameters_compared'] == 1
    exported['model_state']['weight'][-1] += .01
    with pytest.raises(ValueError, match='Export differs'): audit.audit_values(raw, exported)


@pytest.fixture
def files(tmp_path, monkeypatch):
    root = tmp_path; cp = root/'checkpoints';cp.mkdir()
    raw, exported = pair();torch.save(raw, cp/'raw.pt');torch.save(exported, cp/'h200.pt')
    request = {'fixture': 'synthetic_no_corpus'}
    (root/'request.json').write_text(json.dumps(request))
    result = {k: {'checkpoint': str(cp/(k+'.pt'))} for k in ('raw', 'h200')}
    (root/'result.json').write_text(json.dumps(result))
    receipt = {'checkpoint_hashes': {k: file_digest(cp/(k+'.pt')) for k in ('raw', 'h200')}}
    calls = []
    def review(*args): calls.append(args);return copy.deepcopy(receipt)
    monkeypatch.setattr(audit, 'review_pilot', review)
    monkeypatch.setattr(audit, 'review_trial', review)
    args = (root, request, file_digest(root/'request.json'), file_digest(root/'result.json'), {'synthetic': True})
    return root, args, calls, receipt


@pytest.mark.parametrize('kind', ['pilot', 'quality'])
def test_mapped_file_review(files, kind):
    root, args, calls, receipt = files
    hashes = {str(p): file_digest(p) for p in root.rglob('*') if p.is_file()}
    report = audit.review_values(*args, kind=kind)
    assert calls == [args]
    assert report['files'] == hashes
    assert report['values']['exact_export_matches_stored_ema']
    assert report['status'] == 'export_values_reviewed_pending_numerical_and_resource_audits'
    assert hashes == {str(p): file_digest(p) for p in root.rglob('*') if p.is_file()}
    assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) < 16384


@pytest.mark.parametrize('target', ['request.json', 'result.json', 'checkpoints/raw.pt', 'checkpoints/h200.pt'])
def test_changes_during_audit(files, monkeypatch, target):
    root, args, _, _ = files
    original = audit.audit_values
    def mutate(raw, exported):
        report = original(raw, exported)
        with (root/target).open('ab') as stream: stream.write(b'changed')
        return report
    monkeypatch.setattr(audit, 'audit_values', mutate)
    with pytest.raises(ValueError, match='inputs changed'): audit.review_values(*args, kind='pilot')


def test_missing_prerequisite_prevents_loading(files, monkeypatch):
    _, args, _, _ = files
    def reject(*args): raise ValueError('metadata prerequisite absent')
    def no_load(*args, **kwargs): raise AssertionError('should never load')
    monkeypatch.setattr(audit, 'review_pilot', reject)
    monkeypatch.setattr(torch, 'load', no_load)
    with pytest.raises(ValueError, match='prerequisite'): audit.review_values(*args, kind='pilot')


def test_preload_hash_mismatch(files, monkeypatch):
    root, args, _, _ = files
    with (root/'checkpoints/raw.pt').open('ab') as stream: stream.write(b'changed')
    def no_load(*args, **kwargs): raise AssertionError('should never load')
    monkeypatch.setattr(torch, 'load', no_load)
    with pytest.raises(ValueError, match='inputs changed'): audit.review_values(*args, kind='pilot')


def test_explicit_kind(files):
    with pytest.raises(ValueError, match='audit kind'): audit.review_values(*files[1], kind='other')
