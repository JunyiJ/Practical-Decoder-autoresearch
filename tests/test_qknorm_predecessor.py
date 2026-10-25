"""Synthetic lineage/selection checks only; no training or quality measurements."""
import json

import pytest

from src.research import horizon_study as horizon
from src.research.qknorm_predecessor import reviewed_horizon
from src.research.protocol import digest, file_digest
from src.research.runner import atomic_json
from tests.test_horizon_study import plan, row


def predecessor(tmp_path, adopt=True, winner='long6000_stretched'):
    p = plan()
    p['git_commit'] = 'a' * 40
    screen = [row(p, j, ce=1.96 if j[0] == winner else None) for j in horizon.local_jobs(p)]
    selection = horizon.freeze_selection(p, screen)
    assert selection['winner'] == winner
    confirmation = [row(p, j, 'confirmation', 1.95 if j[0] == winner else None)
                    for j in horizon.confirmation_jobs(p)]
    if not adopt:
        for r in confirmation:
            if r['family'] == winner and r['width'] == 256 and r['seed'] == 2173:
                r['averaged_outputs']['200']['coverage']['evaluations']['128'] = row(
                    p, (winner, .005, 256, 2173), 'confirmation', 2.01)['averaged_outputs']['200']['coverage']['evaluations']['128']
    report = horizon.summarize(p, dict(records=screen + confirmation, selection=selection))
    for name, value in [('plan', p), ('selection', selection), ('report', report)]:
        atomic_json(tmp_path / f'{name}.json', value)
    review = dict(status='verified', distinct_trajectories=24, distinct_checkpoints=48,
                  plan_sha256=digest(p), source_commit=p['git_commit'], source_sha256=p['source_sha256'],
                  report_sha256=file_digest(tmp_path / 'report.json'),
                  selection_sha256=file_digest(tmp_path / 'selection.json'),
                  quality_decision=report['quality_decision'])
    atomic_json(tmp_path / 'review.json', review)
    return tmp_path / 'review.json'


@pytest.mark.parametrize('adopt', [True, False])
@pytest.mark.parametrize('winner', ['long6000_stretched', 'long6000_fixedtail'])
def test_only_final_adopted_horizon_propagates(tmp_path, adopt, winner):
    resolved = reviewed_horizon(predecessor(tmp_path, adopt, winner))
    expected = winner if adopt else 'control3000'
    assert resolved['family'] == expected
    assert (resolved['total_steps'], resolved['cooldown_start']) == horizon.CELLS[expected]
    t = resolved['config']['training']
    assert t['lr_schedule'] == horizon.schedule(expected)
    assert t['max_iters'] == resolved['total_steps'] and t['weight_averaging']['half_lives'] == [200]


@pytest.mark.parametrize('field,value', [('status', 'partial_verified'),
    ('distinct_trajectories', 6), ('distinct_checkpoints', 12), ('plan_sha256', '0' * 64),
    ('report_sha256', '0' * 64), ('source_commit', 'b' * 40)])
def test_partial_or_changed_review_rejected(tmp_path, field, value):
    path = predecessor(tmp_path)
    data = json.loads(path.read_text())
    data[field] = value
    atomic_json(path, data)
    with pytest.raises(ValueError):
        reviewed_horizon(path)


def test_rehashed_forged_adoption_cannot_promote_failed_confirmation(tmp_path):
    path = predecessor(tmp_path, adopt=False)
    report_path = tmp_path / 'report.json'
    report = json.loads(report_path.read_text())
    report['quality_decision']['qualifies_for_incumbent_replacement'] = True
    atomic_json(report_path, report)
    review = json.loads(path.read_text())
    review.update(quality_decision=report['quality_decision'], report_sha256=file_digest(report_path))
    atomic_json(path, review)
    with pytest.raises(ValueError, match='complete evidence'):
        reviewed_horizon(path)


def test_rehashed_missing_confirmation_rejected(tmp_path):
    path = predecessor(tmp_path)
    report_path = tmp_path / 'report.json'
    report = json.loads(report_path.read_text())
    report['records'].pop()
    atomic_json(report_path, report)
    review = json.loads(path.read_text())
    review['report_sha256'] = file_digest(report_path)
    atomic_json(path, review)
    with pytest.raises(ValueError):
        reviewed_horizon(path)
