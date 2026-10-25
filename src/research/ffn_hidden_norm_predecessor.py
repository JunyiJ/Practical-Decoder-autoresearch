"""Resolve common attention only from complete, reviewed output-gate evidence."""
import copy
import json
from pathlib import Path

from . import output_gate_study as gate
from .protocol import digest, file_digest


def reviewed_output_gate(path):
    path = Path(path)
    review = json.loads(path.read_text())
    n = review.get('distinct_trajectories')
    if (review.get('status') != 'verified' or n not in (36, 42)
            or review.get('distinct_checkpoints') != 2*n):
        raise ValueError('Final independently verified output-gate review required')
    run = path.parent
    plan = json.loads((run/'plan.json').read_text())
    gate.validate_plan(plan)
    if plan['returned_half_life'] != 200 or plan['recipe']['qk_family'] != 'standard':
        raise ValueError('FFN normalization requires standard-QK MHA and H200')
    report = json.loads((run/'report.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    if (digest(plan) != review['plan_sha256']
            or plan['git_commit'] != review['source_commit']
            or plan['source_sha256'] != review['source_sha256']
            or file_digest(run/'report.json') != review['report_sha256']
            or file_digest(run/'selection.json') != review['selection_sha256']
            or report['selection'] != selection or len(report['records']) != n):
        raise ValueError('Output-gate predecessor lineage changed')
    rebuilt = gate.summarize(plan, dict(records=report['records'], selection=selection))
    if report != rebuilt or review['quality_decision'] != rebuilt['quality_decision']:
        raise ValueError('Final output-gate adoption disagrees with complete evidence')
    decision = rebuilt['quality_decision']
    if decision['qualifies_for_incumbent_replacement']:
        family = decision['frozen_screen_winner']
        lr = selection['choices'][family]['lr']
    else:
        family, lr = 'none', plan['recipe']['learning_rate']
    cfg = copy.deepcopy(gate.suite_spec(plan, family, lr)['candidate']['config'])
    return dict(attention_output_gate=family, learning_rate=lr, config=cfg,
                review=dict(path=str(path.resolve()), sha256=file_digest(path)),
                plan_sha256=digest(plan), source_commit=plan['git_commit'])
