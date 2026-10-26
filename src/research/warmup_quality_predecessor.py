"""Resolve a warmup comparison only from the fully audited KDA decision."""
import copy
import json
from pathlib import Path

from . import kda_quality_study as kda
from .protocol import digest, file_digest


def reviewed_kda(path):
    path = Path(path)
    review = json.loads(path.read_text())
    n = review.get('distinct_trajectories')
    if (review.get('status') != 'verified' or n not in (24, 30)
            or review.get('distinct_checkpoints') != 2*n):
        raise ValueError('Final independently verified KDA review required')
    run = path.parent
    plan = json.loads((run/'plan.json').read_text())
    kda.validate_plan(plan)
    report = json.loads((run/'report.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    if (digest(plan) != review['plan_sha256']
            or plan['git_commit'] != review['source_commit']
            or plan['source_sha256'] != review['source_sha256']
            or file_digest(run/'report.json') != review['report_sha256']
            or file_digest(run/'selection.json') != review['selection_sha256']
            or report['selection'] != selection or len(report['records']) != n):
        raise ValueError('KDA predecessor lineage changed')
    rebuilt = kda.summarize(plan, dict(records=report['records'], selection=selection))
    if report != rebuilt or review['quality_decision'] != rebuilt['quality_decision']:
        raise ValueError('Final KDA adoption disagrees with complete evidence')
    decision = rebuilt['quality_decision']
    if decision['qualifies_for_incumbent_replacement']:
        family = decision['frozen_screen_winner']
        lr = selection['choices'][family]['lr']
        selected = family
    else:
        family, selected, lr = 'mha', kda.REFERENCE, plan['recipe']['learning_rate']
    cfg = copy.deepcopy(kda.suite_spec(plan, selected, lr)['candidate']['config'])
    if (cfg['model']['attn_type'] != family
            or cfg['training']['lr_schedule'] != dict(kind='linear_cooldown',
                total_steps=3000, min_ratio=.1, start_step=2000)
            or cfg['training']['weight_averaging'] != dict(kind='spaced_tail_ema',
                total_steps=3000, start_step=2000, interval=10, half_lives=[200])):
        raise ValueError('Warmup proposal requires the exact reviewed cooldown/H200 incumbent')
    return dict(attention_family=family, learning_rate=lr,
                weight_decay=cfg['training']['weight_decay'], config=cfg,
                review=dict(path=str(path.resolve()), sha256=file_digest(path)),
                plan_sha256=digest(plan), source_commit=plan['git_commit'])
