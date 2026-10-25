"""Resolve a QK-normalization backbone only from a final audited horizon round.

This is preparation code, not a launcher. It does not train or score models.
"""
import copy
import json
from pathlib import Path

from . import horizon_study as horizon
from .protocol import digest, file_digest


def reviewed_horizon(path):
    path = Path(path)
    review = json.loads(path.read_text())
    if (review.get('status') != 'verified'
            or review.get('distinct_trajectories') != 24
            or review.get('distinct_checkpoints') != 48):
        raise ValueError('Final independently verified 24-trajectory horizon review required')
    run = path.parent
    plan = json.loads((run / 'plan.json').read_text())
    horizon.validate_plan(plan)
    if plan['returned_half_life'] != 200:
        raise ValueError('This QK RMS protocol requires the reviewed H200 incumbent')
    report = json.loads((run / 'report.json').read_text())
    selection = json.loads((run / 'selection.json').read_text())
    if (digest(plan) != review['plan_sha256']
            or plan['git_commit'] != review['source_commit']
            or plan['source_sha256'] != review['source_sha256']
            or file_digest(run / 'report.json') != review['report_sha256']
            or file_digest(run / 'selection.json') != review['selection_sha256']
            or report['selection'] != selection):
        raise ValueError('Horizon predecessor lineage changed')
    rebuilt = horizon.summarize(plan, dict(records=report['records'], selection=selection))
    if report != rebuilt or review['quality_decision'] != rebuilt['quality_decision']:
        raise ValueError('Horizon final report/adoption disagrees with complete evidence')
    decision = rebuilt['quality_decision']
    family = decision['frozen_screen_winner'] if decision['qualifies_for_incumbent_replacement'] else 'control3000'
    config = copy.deepcopy(horizon.suite_spec(plan, family, .005)['candidate']['config'])
    return dict(family=family, total_steps=horizon.CELLS[family][0],
                cooldown_start=horizon.CELLS[family][1], returned_half_life=200,
                learning_rate=.005, config=config,
                review=dict(path=str(path.resolve()), sha256=file_digest(path)),
                plan_sha256=digest(plan), source_commit=plan['git_commit'])
