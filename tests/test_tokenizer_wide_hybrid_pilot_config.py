"""Pure configuration/accounting and two-scalar EMA tests; no model training."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from src.research import tokenizer_wide_hybrid_pilot_config as pilot
from src.research.tokenizer_wide_hybrid_config import FAMILIES, rank_config
from src.research.tokenizer_wide_hybrid_driver import remaining_bytes
from src.train.weight_averaging import TailWeightAverages, averaging_contract


@pytest.mark.parametrize('family', FAMILIES)
def test_config_and_identity(tmp_path, family):
    cfg = pilot.config(tmp_path, family)
    expected = rank_config(tmp_path, family, width=512, seed=3491, lr=.0025,
                           steps=20, device='mps', bottleneck_divisor=2)
    expected['training']['eval_every'] = 20
    assert cfg == expected
    t = cfg['training']
    assert t['lr_schedule']['start_step'] == 13
    assert t['lr_schedule']['total_steps'] == 20
    spec = averaging_contract(SimpleNamespace(**t))
    assert spec == dict(kind='spaced_tail_ema', total_steps=20, start_step=13,
                        interval=1, half_lives=[200])
    assert cfg['model']['rope'] == 0
    assert cfg['model']['dim'] == 512
    assert cfg['model']['attention_bottleneck_divisor'] == 2
    assert t['device'] == 'mps'


def test_order_and_no_aliasing(tmp_path):
    a = pilot.specs()
    assert [s['family'] for s in a] == list(FAMILIES)
    assert all(s == dict(stage='pilot', family=f, width=512, seed=3491,
                         divisor=2, lr=.0025, steps=20) for s, f in zip(a, FAMILIES))
    a[0]['seed'] = 0
    assert pilot.specs()[0]['seed'] == 3491
    a = pilot.config(tmp_path, 'hybrid')
    a['training']['weight_averaging']['half_lives'].append(1)
    assert pilot.config(tmp_path, 'hybrid')['training']['weight_averaging']['half_lives'] == [200]
    assert pilot.CONTRACT['selection_eligible'] is False


def test_short_ema_independent_scalar_reference(tmp_path):
    # This does not instantiate GPT, execute an optimizer or resume a model.
    model = torch.nn.Linear(2, 1, bias=False)
    training = SimpleNamespace(**pilot.config(tmp_path, 'hybrid')['training'])
    averages = TailWeightAverages(model, training)
    reference = None
    beta = 2**(-1/200)
    for step in range(1, 21):
        values = [float(step), float(-2*step)]
        with torch.no_grad():
            model.weight.copy_(torch.tensor([values]))
        averages.update(step)
        if step == 13:
            reference = values
        elif step > 13:
            reference = [beta*a + (1-beta)*b for a, b in zip(reference, values)]
        assert averages.snapshots == max(0, step-12)
    state = averages.state_dict()
    assert state['completed_steps'] == 20 and state['snapshots'] == 8
    torch.testing.assert_close(state['averages']['200']['weight'],
                               torch.tensor([reference]), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('n', range(4))
def test_full_retained_reservation(n):
    contract = pilot.storage_contract(FAMILIES[:n])
    assert contract['full_panel_required_free_bytes'] == 57_224_879_357
    assert contract['quality_allowance_bytes'] == remaining_bytes([], None, None)
    completed = sum(contract['pilot_allowances'][f] for f in FAMILIES[:n])
    assert contract['required_free_bytes'] == 57_224_879_357 - completed
    assert contract['required_free_bytes'] >= pilot.RESERVE + contract['quality_allowance_bytes']
    assert pilot.require_storage(contract['required_free_bytes'], FAMILIES[:n]) == contract
    with pytest.raises(ValueError, match='Complete retained panel'):
        pilot.require_storage(contract['required_free_bytes']-1, FAMILIES[:n])


@pytest.mark.parametrize('prefix', ['hybrid', ['kda'], ['hybrid', 'hybrid'],
                                   list(FAMILIES)+['hybrid'], [True], None])
def test_invalid_prefix(prefix):
    with pytest.raises(ValueError, match='ordered family prefix'):
        pilot.storage_contract(prefix)


@pytest.mark.parametrize('value', [True, -1, 1.0, None])
def test_invalid_free_observation(value):
    with pytest.raises(ValueError, match='integer byte observation'):
        pilot.require_storage(value)


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'negative', 'float',
                                     'rank_bound', 'reserve', 'total', 'trajectories'])
def test_corrupt_projection(tmp_path, monkeypatch, mutation):
    doc = copy.deepcopy(json.loads(pilot.PROJECTION.read_text()))
    row = next(r for r in doc['records'] if r['width'] == 512 and r['bottleneck_divisor'] == 2)
    if mutation == 'missing': doc['records'].remove(row)
    elif mutation == 'duplicate': doc['records'].append(copy.deepcopy(row))
    elif mutation == 'negative': row['per_trajectory_allowance_bytes'] = -1
    elif mutation == 'float': row['per_trajectory_allowance_bytes'] = 1.0
    elif mutation == 'rank_bound': row['per_trajectory_allowance_bytes'] = 1
    elif mutation == 'reserve': doc['reserve_bytes'] -= 1
    elif mutation == 'total': doc['panels']['512']['required_free_bytes'] -= 1
    elif mutation == 'trajectories': doc['panels']['512']['trajectories'] = 26
    path = tmp_path/'projection.json'
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(pilot, 'PROJECTION', path)
    with pytest.raises(ValueError): pilot.storage_contract()


def test_no_partial_pilot_budget():
    contract = pilot.storage_contract()
    just_pilots = pilot.RESERVE + pilot.METADATA + sum(contract['pilot_allowances'].values())
    with pytest.raises(ValueError, match='no partial pilot launch'):
        pilot.require_storage(just_pilots)
