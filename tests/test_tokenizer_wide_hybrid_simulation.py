"""Controller-stage simulations only, without subprocess/model/checkpoint work."""
import copy
import fcntl
import json

import pytest

from src.research import tokenizer_wide_hybrid_rules as rules
from src.research.tokenizer_wide_hybrid_simulation import next_action, simulate


def row(spec):
    return dict(spec, status='completed', ce=5.)


def test_full_serial_stages_persist_before_callback(tmp_path):
    output = tmp_path/'run'; shared = tmp_path/'global.lock'; seen = []
    def score(spec):
        state = json.loads((output/'state.json').read_text())
        assert state['status'] == 'running' and state['active'] == spec
        assert len(state['records']) == len(seen)
        for path in (shared, output/'.lock'):
            with path.open('a+b') as stream:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if spec['stage'] != 'lr':
            assert json.loads((output/'lr-selection.json').read_text()) == state['rates']
        if spec['stage'] == 'confirm':
            assert json.loads((output/'rank-selection.json').read_text()) == state['selection']
        seen.append(spec)
        return dict(synthetic=True, ce=5.)
    state = simulate(output, shared, score, lambda: {'source': 'fixture'})
    assert state['status'] == 'completed' and len(seen) == 24 and not state['adopted']
    assert state['mode'] == 'synthetic_simulation_only'
    assert state['rates'] == dict.fromkeys(rules.FAMILIES, .000625)
    assert all(v == dict(lr=.000625, divisor=8) for v in state['selection'].values())
    assert seen == rules.lr_specs()+rules.rank_specs(state['rates'])+rules.confirmation_specs(state['selection'])
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError): simulate(output, shared, score, lambda: {})
    assert before == {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    assert len(seen) == 24


@pytest.mark.parametrize('fail_at', [0, 8, 9, 14, 15, 23])
def test_failure_halts_preserves_active_and_refuses_retry(tmp_path, fail_at):
    output = tmp_path/'run'; calls = []
    def score(spec):
        calls.append(spec)
        if len(calls) == fail_at+1:
            raise TimeoutError('simulated worker timeout')
        return dict(synthetic=True, ce=5.)
    with pytest.raises(TimeoutError): simulate(output, tmp_path/'lock', score, lambda: {})
    state = json.loads((output/'state.json').read_text())
    assert state['status'] == 'failed' and len(state['records']) == fail_at
    assert state['active'] == calls[-1] and 'simulated worker timeout' in state['error']
    with pytest.raises(FileExistsError): simulate(output, tmp_path/'lock', score, lambda: {})
    assert len(calls) == fail_at+1


def test_stop_and_interrupt_are_not_auto_resumed(tmp_path):
    output = tmp_path/'stop'
    def score(spec):
        (output/'STOP').write_text('human stop fixture')
        return dict(synthetic=True, ce=5.)
    state = simulate(output, tmp_path/'lock', score, lambda: {})
    assert state['status'] == 'paused' and len(state['records']) == 1
    with pytest.raises(FileExistsError): simulate(output, tmp_path/'lock', score, lambda: {})
    output = tmp_path/'interrupt'
    def interrupt(spec): raise KeyboardInterrupt('fixture interruption')
    with pytest.raises(KeyboardInterrupt): simulate(output, tmp_path/'lock', interrupt, lambda: {})
    state = json.loads((output/'state.json').read_text())
    assert state['status'] == 'failed' and state['active'] is not None


def test_shared_lock_blocks_before_run_creation(tmp_path):
    shared = tmp_path/'lock'; output = tmp_path/'run'
    with shared.open('a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): simulate(output, shared, lambda s: None, lambda: {})
    assert not output.exists()


@pytest.mark.parametrize('change', ['provenance', 'selection', 'early_selection', 'nonsynthetic', 'nan'])
def test_changed_provenance_or_selection_is_terminal(tmp_path, change):
    output = tmp_path/'run'; identity = {}; calls = []
    def score(spec):
        calls.append(spec)
        if change == 'provenance': identity['changed'] = True
        elif change == 'early_selection': (output/'rank-selection.json').write_text('{}')
        elif change == 'selection' and spec['stage'] == 'rank': (output/'lr-selection.json').write_text('{}')
        elif change == 'nonsynthetic': return dict(synthetic=False, ce=5.)
        elif change == 'nan': return dict(synthetic=True, ce=float('nan'))
        return dict(synthetic=True, ce=5.)
    with pytest.raises(ValueError): simulate(output, tmp_path/'lock', score, lambda: identity)
    assert len(calls) == (10 if change == 'selection' else 1)
    assert json.loads((output/'state.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('n', [2, 9, 12, 15, 20, 24])
def test_exact_prefix_order_duplicates_and_choices(n):
    lr = [row(s) for s in rules.lr_specs()]; rates = rules.choose_rates(lr)
    ranks = [row(s) for s in rules.rank_specs(rates)]; selected = rules.choose_ranks(lr, ranks)
    all_rows = lr+ranks+[row(s) for s in rules.confirmation_specs(selected)]
    actual_rates = rates if n > 9 else None
    actual_selected = selected if n > 15 else None
    original = all_rows[:n]
    next_action(original, actual_rates, actual_selected)
    for bad in (list(reversed(original)), original[:-1]+[original[0]]):
        with pytest.raises(ValueError): next_action(bad, actual_rates, actual_selected)
    if actual_rates:
        changed = dict(actual_rates, hybrid=.0025)
        with pytest.raises(ValueError): next_action(original, changed, actual_selected)
    if actual_selected:
        changed = copy.deepcopy(actual_selected); changed['hybrid']['divisor'] = 2
        with pytest.raises(ValueError): next_action(original, actual_rates, changed)


def test_no_skipped_selection_barriers():
    lr = [row(s) for s in rules.lr_specs()]; rates = rules.choose_rates(lr)
    rank = [row(s) for s in rules.rank_specs(rates)]; selected = rules.choose_ranks(lr, rank)
    with pytest.raises(ValueError): next_action(lr[:8], rates)
    with pytest.raises(ValueError): next_action(lr+rank[:1])
    with pytest.raises(ValueError): next_action(lr+rank[:5], rates, selected)
    with pytest.raises(ValueError): next_action(lr+rank+[row(rules.confirmation_specs(selected)[0])], rates)
