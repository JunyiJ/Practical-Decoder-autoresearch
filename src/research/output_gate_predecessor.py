"""Resolve the next backbone only from a completed, independently audited QK round."""
import copy
import json
from pathlib import Path

from . import qknorm_study as qk
from .protocol import digest, file_digest


def reviewed_qknorm(path):
    path = Path(path)
    review = json.loads(path.read_text())
    n = review.get('distinct_trajectories')
    if (review.get('status') != 'verified' or n not in (24, 30)
            or review.get('distinct_checkpoints') != 2 * n):
        raise ValueError('Final independently verified QK review required')
    run = path.parent
    plan = json.loads((run / 'plan.json').read_text())
    qk.validate_plan(plan)
    if (plan['returned_half_life'] != 200
            or plan['recipe'] != dict(family='control3000', total_steps=3000, cooldown_start=2000)):
        raise ValueError('Output-gate protocol requires retained3000/start2000/H200')
    report = json.loads((run / 'report.json').read_text())
    selection = json.loads((run / 'selection.json').read_text())
    if (digest(plan) != review['plan_sha256']
            or plan['git_commit'] != review['source_commit']
            or plan['source_sha256'] != review['source_sha256']
            or file_digest(run / 'report.json') != review['report_sha256']
            or file_digest(run / 'selection.json') != review['selection_sha256']
            or report['selection'] != selection or len(report['records']) != n):
        raise ValueError('QK predecessor lineage changed')
    rebuilt = qk.summarize(plan, dict(records=report['records'], selection=selection))
    if report != rebuilt or review['quality_decision'] != rebuilt['quality_decision']:
        raise ValueError('QK final report/adoption disagrees with complete evidence')
    decision = rebuilt['quality_decision']
    if decision['qualifies_for_incumbent_replacement']:
        family = decision['frozen_screen_winner']
        lr = selection['choices'][family]['lr']
    else:
        family, lr = 'standard', .005
    cfg = copy.deepcopy(qk.suite_spec(plan, family, lr)['candidate']['config'])
    return dict(qk_family=family, learning_rate=lr, config=cfg,
                review=dict(path=str(path.resolve()), sha256=file_digest(path)),
                plan_sha256=digest(plan), source_commit=plan['git_commit'])
