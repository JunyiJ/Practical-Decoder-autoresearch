"""Resolve common FFN normalization only from complete, reviewed predecessor evidence."""
import copy
import json
from pathlib import Path

from . import ffn_hidden_norm_study as ffn
from .protocol import digest, file_digest


def reviewed_ffn_hidden_norm(path):
    path = Path(path)
    review = json.loads(path.read_text())
    n = review.get('distinct_trajectories')
    if (review.get('status') != 'verified' or n not in (36, 42)
            or review.get('distinct_checkpoints') != 2*n):
        raise ValueError('Final independently verified FFN-hidden-norm review required')
    run = path.parent
    plan = json.loads((run/'plan.json').read_text())
    ffn.validate_plan(plan)
    if plan['returned_half_life'] != 200 or plan['config']['training']['weight_decay'] != .00006:
        raise ValueError('Decay comparison requires incumbent independent decay and H200')
    report = json.loads((run/'report.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    if (digest(plan) != review['plan_sha256']
            or plan['git_commit'] != review['source_commit']
            or plan['source_sha256'] != review['source_sha256']
            or file_digest(run/'report.json') != review['report_sha256']
            or file_digest(run/'selection.json') != review['selection_sha256']
            or report['selection'] != selection or len(report['records']) != n):
        raise ValueError('FFN-hidden-norm predecessor lineage changed')
    rebuilt = ffn.summarize(plan, dict(records=report['records'], selection=selection))
    if report != rebuilt or review['quality_decision'] != rebuilt['quality_decision']:
        raise ValueError('Final FFN-hidden-norm adoption disagrees with complete evidence')
    decision = rebuilt['quality_decision']
    if decision['qualifies_for_incumbent_replacement']:
        family = decision['frozen_screen_winner']
        lr = selection['choices'][family]['lr']
    else:
        family, lr = 'none', plan['recipe']['learning_rate']
    cfg = copy.deepcopy(ffn.suite_spec(plan, family, lr)['candidate']['config'])
    return dict(attention_output_gate=plan['recipe']['attention_output_gate'],
                ffn_hidden_norm=family, learning_rate=lr, config=cfg,
                review=dict(path=str(path.resolve()), sha256=file_digest(path)),
                plan_sha256=digest(plan), source_commit=plan['git_commit'])
