"""Prospective serial stage/persistence simulation, deliberately no worker CLI.

Synthetic scalar scores only. This does not implement the production resource
watchdog, pilot controller, frozen provenance audit or checkpoint reviewer.
"""
import copy
import fcntl
import json
import os
from pathlib import Path

from .protocol import digest
from . import tokenizer_wide_hybrid_rules as rules


def next_action(records, rates=None, selection=None):
    """Validate the exact completed prefix before proposing one serial action."""
    screen = rules.lr_specs()
    if len(records) < 9:
        if rates is not None or selection is not None:
            raise ValueError('Selection before complete LR stage')
        queue = screen
    else:
        for record, spec in zip(records[:9], screen):
            rules._checked([record], [spec])
        chosen = rules.choose_rates(records[:9])
        if rates is None:
            if len(records) != 9 or selection is not None:
                raise ValueError('Missing persisted LR choice')
            return dict(kind='save_rates', value=chosen)
        if digest(chosen) != digest(rates):
            raise ValueError('Changed LR choice')
        queue = screen + rules.rank_specs(rates)
        if len(records) < 15:
            if selection is not None:
                raise ValueError('Rank selection before complete rank stage')
        else:
            for record, spec in zip(records[9:15], queue[9:15]):
                rules._checked([record], [spec])
            final = rules.choose_ranks(records[:9], records[9:15])
            if selection is None:
                if len(records) != 15:
                    raise ValueError('Missing persisted rank choice')
                return dict(kind='save_selection', value=final)
            if digest(final) != digest(selection):
                raise ValueError('Changed rank choice')
            queue += rules.confirmation_specs(selection)
    if len(records) > len(queue):
        raise ValueError('Extra completed trials')
    for record, spec in zip(records, queue):
        rules._checked([record], [spec])
    return (dict(kind='trial', spec=queue[len(records)]) if len(records) < len(queue)
            else dict(kind='complete'))


def _publish(path, value, exclusive=False):
    """Durable state replacement; immutable stage selections are exclusive."""
    data = (json.dumps(value, sort_keys=True, allow_nan=False, indent=2)+'\n').encode()
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    if exclusive:
        os.link(temporary, path)
        temporary.unlink()
    else:
        os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def simulate(output, global_lock, score, provenance):
    """Fresh-only synchronous synthetic callback, with two exclusive locks.

    score(spec) must return exactly {'synthetic': True, 'ce': finite_number}.
    provenance() supplies a small identity object checked before each action.
    A callback is test infrastructure, not a production worker launcher. There
    is no resume path: any existing output, including completed, is refused.
    """
    output, global_lock = Path(output), Path(global_lock)
    # Acquire the shared lock before creating run evidence, as a production
    # integration must do. Tests supply isolated temporary locks only.
    with global_lock.open('a+b') as shared:
        fcntl.flock(shared.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(exist_ok=False)
        with (output/'.lock').open('a+b') as local:
            fcntl.flock(local.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity = digest(provenance())
            state = dict(mode='synthetic_simulation_only', status='ready', records=[], rates=None,
                         selection=None, active=None, provenance_sha256=identity, adopted=False)
            _publish(output/'state.json', state)
            try:
                while True:
                    if (output/'STOP').exists():
                        state['status'] = 'paused'
                        _publish(output/'state.json', state)
                        return state
                    if digest(provenance()) != identity:
                        raise ValueError('Changed simulation provenance')
                    for key, filename in (('rates', 'lr-selection.json'), ('selection', 'rank-selection.json')):
                        path = output/filename
                        if state[key] is None:
                            if path.exists():
                                raise ValueError('Unexpected early selection file')
                        elif digest(json.loads(path.read_text())) != digest(state[key]):
                            raise ValueError('Changed persisted selection')
                    action = next_action(state['records'], state['rates'], state['selection'])
                    if action['kind'] == 'complete':
                        state['status'] = 'completed'
                        _publish(output/'state.json', state)
                        return state
                    if action['kind'] in ('save_rates', 'save_selection'):
                        key, filename = (('rates', 'lr-selection.json') if action['kind'] == 'save_rates'
                                         else ('selection', 'rank-selection.json'))
                        _publish(output/filename, action['value'], exclusive=True)
                        state[key] = action['value']
                        _publish(output/'state.json', state)
                        continue
                    spec = action['spec']
                    state.update(status='running', active=spec)
                    _publish(output/'state.json', state)
                    value = score(copy.deepcopy(spec))
                    if set(value) != {'synthetic', 'ce'} or value['synthetic'] is not True:
                        raise ValueError('Only synthetic scalar scores accepted')
                    row = dict(spec, status='completed', ce=value['ce'])
                    rules._checked([row], [spec])
                    state['records'].append(row)
                    state.update(status='ready', active=None)
                    _publish(output/'state.json', state)
            except BaseException as exc:
                state.update(status='failed', error=repr(exc))
                _publish(output/'state.json', state)
                raise
