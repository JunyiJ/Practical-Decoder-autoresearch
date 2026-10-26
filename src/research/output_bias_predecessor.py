"""Resolve the full normalization recipe only from its complete independently reviewed evidence."""
import copy
import json
from pathlib import Path

from . import readout_norm_study as normalization
from .protocol import digest, file_digest


def reviewed_normalization(path):
    path = Path(path)
    review = json.loads(path.read_text())
    count = review.get('distinct_trajectories')
    if (review.get('status') != 'verified' or type(count) is not int or count not in (36,42)
            or review.get('distinct_checkpoints') != 2*count):
        raise ValueError('Final independently verified36/42-run normalization review required')
    run = path.parent
    plan = json.loads((run/'plan.json').read_text())
    normalization.validate_plan(plan)
    report = json.loads((run/'report.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    state = json.loads((run/'state.json').read_text())
    if (state.get('status') != 'completed' or state.get('records') != report['records']
            or state.get('selection') != selection):
        raise ValueError('Completed normalization state matching reviewed report required')
    if (digest(plan) != review['plan_sha256'] or plan['git_commit'] != review['source_commit']
            or plan['source_sha256'] != review['source_sha256']
            or file_digest(run/'report.json') != review['report_sha256']
            or file_digest(run/'selection.json') != review['selection_sha256']
            or report['selection'] != selection or len(report['records']) != count):
        raise ValueError('Normalization predecessor lineage changed')
    rebuilt = normalization.summarize(plan, dict(records=report['records'],selection=selection))
    if report != rebuilt or review['quality_decision'] != rebuilt['quality_decision']:
        raise ValueError('Final normalization adoption disagrees with complete evidence')
    decision = rebuilt['quality_decision']
    if decision['qualifies_for_incumbent_replacement']:
        family = decision['frozen_screen_winner']
        lr = selection['choices'][family]['lr']
    else:
        family, lr = normalization.REFERENCE, plan['recipe']['learning_rate']
    cfg = copy.deepcopy(normalization.suite_spec(plan,family,lr)['candidate']['config'])
    scale = cfg['model']['input_embedding_scale']
    norm = cfg['model']['readout_norm_kind']
    if (plan['recipe']['warmup_steps'] != 0 or cfg['model']['attn_type'] != 'mha'
            or lr not in normalization.LRS or scale not in (1.,.5,.25)
            or norm not in ('layer_affine','layer_no_bias','rms')
            or cfg['model']['readout_norm_width_policy'] != 'mha4_readout_norm'
            or cfg['model']['readout_norm_epsilon'] != 1e-5
            or cfg['training']['weight_decay'] != .00006
            or cfg['training']['weight_averaging'] != dict(kind='spaced_tail_ema',
                total_steps=3000,start_step=2000,interval=10,half_lives=[200])):
        raise ValueError('Output-bias proposal requires reviewed MHA/zero warmup/input scale/final norm/H200')
    return dict(warmup_steps=0,learning_rate=lr,weight_decay=.00006,
        input_embedding_scale=scale,readout_norm_kind=norm,readout_norm_epsilon=1e-5,config=cfg,
        review=dict(path=str(path.resolve()),sha256=file_digest(path)),
        plan_sha256=digest(plan),source_commit=plan['git_commit'])
