"""Resolve the KDA proposal's common recipe only after a complete decay audit."""
import copy
import json
from pathlib import Path

from . import decay_quality_study as decay
from .protocol import digest, file_digest


def reviewed_decay(path):
    path = Path(path)
    review = json.loads(path.read_text())
    n = review.get('distinct_trajectories')
    if (review.get('status') != 'verified' or n not in (36, 42)
            or review.get('distinct_checkpoints') != 2*n):
        raise ValueError('Final independently verified decay review required')
    run = path.parent
    plan = json.loads((run/'plan.json').read_text())
    decay.validate_plan(plan)
    report = json.loads((run/'report.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    if (digest(plan) != review['plan_sha256']
            or plan['git_commit'] != review['source_commit']
            or plan['source_sha256'] != review['source_sha256']
            or file_digest(run/'report.json') != review['report_sha256']
            or file_digest(run/'selection.json') != review['selection_sha256']
            or report['selection'] != selection or len(report['records']) != n):
        raise ValueError('Decay predecessor lineage changed')
    rebuilt = decay.summarize(plan, dict(records=report['records'], selection=selection))
    if report != rebuilt or review['quality_decision'] != rebuilt['quality_decision']:
        raise ValueError('Final decay adoption disagrees with complete evidence')
    decision = rebuilt['quality_decision']
    if decision['qualifies_for_incumbent_replacement']:
        family = decision['frozen_screen_winner']
        lr = selection['choices'][family]['lr']
    else:
        family, lr = 'anchor', plan['recipe']['learning_rate']
    cfg = copy.deepcopy(decay.suite_spec(plan, family, lr)['candidate']['config'])
    # This first KDA comparison does not also combine an unrelated output/FFN norm.
    if (plan['recipe']['attention_output_gate'] != 'none'
            or plan['recipe']['ffn_hidden_norm'] != 'none'
            or cfg['model']['attn_type'] != 'mha'
            or cfg['training']['weight_averaging']['half_lives'] != [200]):
        raise ValueError('KDA proposal requires the unchanged-FFN standard-MHA/H200 incumbent')
    return dict(learning_rate=lr, weight_decay=decay.DECAYS[family],
                config=cfg, review=dict(path=str(path.resolve()), sha256=file_digest(path)),
                plan_sha256=digest(plan), source_commit=plan['git_commit'])
