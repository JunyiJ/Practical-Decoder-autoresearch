"""Resolve the exact schedule only from a complete independently audited warmup round."""
import copy
import json
from pathlib import Path

from . import warmup_quality_study as warmup
from .protocol import digest, file_digest


def reviewed_warmup(path):
    path = Path(path)
    review = json.loads(path.read_text())
    if (review.get('status') != 'verified' or review.get('distinct_trajectories') != 24
            or review.get('distinct_checkpoints') != 48):
        raise ValueError('Final independently verified24-run warmup review required')
    run = path.parent
    plan = json.loads((run/'plan.json').read_text())
    warmup.validate_plan(plan)
    report = json.loads((run/'report.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    state = json.loads((run/'state.json').read_text())
    if (state.get('status') != 'completed' or state.get('records') != report['records']
            or state.get('selection') != selection):
        raise ValueError('Completed warmup state matching the reviewed report required')
    if (digest(plan) != review['plan_sha256'] or plan['git_commit'] != review['source_commit']
            or plan['source_sha256'] != review['source_sha256']
            or file_digest(run/'report.json') != review['report_sha256']
            or file_digest(run/'selection.json') != review['selection_sha256']
            or report['selection'] != selection or len(report['records']) != 24):
        raise ValueError('Warmup predecessor lineage changed')
    rebuilt = warmup.summarize(plan, dict(records=report['records'],selection=selection))
    if report != rebuilt or review['quality_decision'] != rebuilt['quality_decision']:
        raise ValueError('Final warmup adoption disagrees with complete evidence')
    decision = rebuilt['quality_decision']
    family = decision['frozen_screen_winner'] if decision['qualifies_for_incumbent_replacement'] else 'baseline'
    cfg = copy.deepcopy(warmup.suite_spec(plan,family,plan['recipe']['learning_rate'])['candidate']['config'])
    steps = warmup.WARMUP_STEPS[family]
    if (cfg['model']['attn_type'] != 'mha' or cfg['training']['learning_rate'] != .005
            or cfg['training']['weight_decay'] != .00006
            or cfg['training']['weight_averaging'] != dict(kind='spaced_tail_ema',
                total_steps=3000,start_step=2000,interval=10,half_lives=[200])):
        raise ValueError('Embedding proposal requires reviewed MHA/LR.005/decay.00006/H200')
    return dict(warmup_steps=steps,learning_rate=.005,weight_decay=.00006,config=cfg,
        review=dict(path=str(path.resolve()),sha256=file_digest(path)),
        plan_sha256=digest(plan),source_commit=plan['git_commit'])
