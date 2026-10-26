"""Synthetic predecessor decisions, not measured KDA quality."""
import json

import pytest

from src.research import decay_quality_study as decay
from src.research.kda_quality_predecessor import reviewed_decay
from src.research.protocol import digest, file_digest
from tests.test_decay_quality_study import plan, complete
from tests.test_schedule_study import coverage


def evidence(tmp_path, best='strong', qualifies=True):
    p = plan()
    p.update(git_commit='1'*40, source_sha256='2'*64)
    state = complete(p, anchor_lr=.0025, best=best)
    if not qualifies:
        for row in state['records']:
            if row['stage'] == 'confirmation' and row['family'] == best:
                value = decay.endpoint(p, row)
                # Preserve valid coverage structure while making the winner worse.
                changed = coverage(value['mean_ce'] + .5)
                value.clear(); value.update(changed)
    report = decay.summarize(p, state)
    for name, obj in [('plan', p), ('selection', state['selection']), ('report', report)]:
        (tmp_path/f'{name}.json').write_text(json.dumps(obj))
    review = dict(status='verified', distinct_trajectories=len(state['records']),
                  distinct_checkpoints=2*len(state['records']), plan_sha256=digest(p),
                  source_commit=p['git_commit'], source_sha256=p['source_sha256'],
                  report_sha256=file_digest(tmp_path/'report.json'),
                  selection_sha256=file_digest(tmp_path/'selection.json'),
                  quality_decision=report['quality_decision'])
    path = tmp_path/'review.json'; path.write_text(json.dumps(review))
    return path, p, state, report


@pytest.mark.parametrize('family', decay.FAMILIES)
def test_final_adopted_coefficient_and_lr_follow_frozen_decision(tmp_path, family):
    path, _, state, report = evidence(tmp_path, best=family)
    assert report['quality_decision']['qualifies_for_incumbent_replacement']
    resolved = reviewed_decay(path)
    assert resolved['weight_decay'] == decay.DECAYS[family]
    assert resolved['learning_rate'] == state['selection']['choices'][family]['lr']
    assert resolved['config']['training']['weight_decay'] == decay.DECAYS[family]


@pytest.mark.parametrize('family', decay.FAMILIES)
def test_failed_confirmation_retains_fixed_incumbent_not_tuned_anchor(tmp_path, family):
    path, p, _, report = evidence(tmp_path, best=family, qualifies=False)
    assert not report['quality_decision']['qualifies_for_incumbent_replacement']
    resolved = reviewed_decay(path)
    assert resolved['weight_decay'] == .00006
    assert resolved['learning_rate'] == p['recipe']['learning_rate'] == .005


@pytest.mark.parametrize('key,value', [('status', 'partial'), ('distinct_trajectories', 18),
                                      ('distinct_checkpoints', 71), ('source_commit', 'changed'),
                                      ('source_sha256', 'changed'), ('plan_sha256', 'changed'),
                                      ('report_sha256', 'changed'), ('selection_sha256', 'changed')])
def test_partial_or_altered_review_rejected(tmp_path, key, value):
    path, _, _, _ = evidence(tmp_path)
    review = json.loads(path.read_text()); review[key] = value
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        reviewed_decay(path)
