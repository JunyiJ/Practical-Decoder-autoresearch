"""Resolve the full position-quality recipe only from its complete independently reviewed evidence."""
import copy
import json
from pathlib import Path

from . import position_quality_study as prior_study
from .protocol import digest, file_digest


def reviewed_position(path):
    path = Path(path)
    review = json.loads(path.read_text())
    count = review.get('distinct_trajectories')
    if (review.get('status') != 'verified' or type(count) is not int or count not in (36,42)
            or review.get('distinct_checkpoints') != 2*count):
        raise ValueError('Final independently verified36/42-run position-quality review required')
    run = path.parent
    plan = json.loads((run/'plan.json').read_text())
    prior_study.validate_plan(plan)
    report = json.loads((run/'report.json').read_text())
    selection = json.loads((run/'selection.json').read_text())
    state = json.loads((run/'state.json').read_text())
    if (state.get('status') != 'completed' or state.get('records') != report['records']
            or state.get('selection') != selection):
        raise ValueError('Completed position-quality state matching reviewed report required')
    if (digest(plan) != review['plan_sha256'] or plan['git_commit'] != review['source_commit']
            or plan['source_sha256'] != review['source_sha256']
            or file_digest(run/'report.json') != review['report_sha256']
            or file_digest(run/'selection.json') != review['selection_sha256']
            or report['selection'] != selection or len(report['records']) != count):
        raise ValueError('Position predecessor lineage changed')
    rebuilt = prior_study.summarize(plan, dict(records=report['records'],selection=selection))
    if report != rebuilt or review['quality_decision'] != rebuilt['quality_decision']:
        raise ValueError('Final position-quality adoption disagrees with complete evidence')
    from .position_quality_audit import audit
    audit(plan,state,report)
    decision = rebuilt['quality_decision']
    if decision['qualifies_for_incumbent_replacement']:
        family = decision['frozen_screen_winner']
        lr = selection['choices'][family]['lr']
    else:
        family, lr = prior_study.REFERENCE, plan['recipe']['learning_rate']
    cfg = copy.deepcopy(prior_study.suite_spec(plan,family,lr)['candidate']['config'])
    cfg['model']['vocab_size']=len(plan['output_bias_prior']['alphabet'])
    from src.models.output_bias import output_bias_contract
    contract=output_bias_contract(cfg['model'])
    if contract is None or contract['output_bias_prior']!=plan['output_bias_prior']:
        raise ValueError('Reviewed training prior changed')
    from src.models.weight_tying import weight_tying_contract
    tying=weight_tying_contract(cfg['model'])
    if tying is None:
        raise ValueError('Reviewed table-sharing contract missing')
    from src.models.residual_topology import residual_topology_contract
    topology=residual_topology_contract(cfg['model'])
    if topology is None:
        raise ValueError('Reviewed residual topology contract missing')
    from src.models.attention_scale import attention_scale_contract
    attention=attention_scale_contract(cfg['model'])
    if attention is None:
        raise ValueError('Reviewed attention-scale contract missing')
    from src.models.gated_activation import gated_activation_contract
    activation=gated_activation_contract(cfg['model'])
    if activation is None:
        raise ValueError('Reviewed gated-activation contract missing')
    from src.models.residual_gain import residual_gain_contract
    gain=residual_gain_contract(cfg['model'])
    if gain is None:
        raise ValueError('Reviewed residual-gain contract missing')
    from src.models.value_residual import value_residual_contract
    from src.models.position_family import position_family_contract
    value=value_residual_contract(cfg['model'])
    if value is None:
        raise ValueError('Reviewed value-residual contract missing')
    position=position_family_contract(cfg['model'])
    if position is None:
        raise ValueError('Reviewed position-family contract missing')
    scale = cfg['model']['input_embedding_scale']
    norm = cfg['model']['readout_norm_kind']
    if (plan['recipe']['warmup_steps'] != 0 or cfg['model']['attn_type'] != 'mha'
            or lr not in prior_study.LRS or scale not in (1.,.5,.25)
            or norm not in ('layer_affine','layer_no_bias','rms')
            or cfg['model']['readout_norm_width_policy'] != 'mha4_readout_norm'
            or cfg['model']['readout_norm_epsilon'] != 1e-5
            or cfg['training']['weight_decay'] != .00006
            or cfg['training']['weight_averaging'] != dict(kind='spaced_tail_ema',
                total_steps=3000,start_step=2000,interval=10,half_lives=[200])):
        raise ValueError('Logit-Z proposal requires reviewed MHA/zero warmup/input scale/final norm/H200')
    return dict(warmup_steps=0,learning_rate=lr,weight_decay=.00006,
        input_embedding_scale=scale,readout_norm_kind=norm,readout_norm_epsilon=1e-5,
        position_family_kind=cfg['model']['position_family_kind'],
        value_residual_kind=cfg['model']['value_residual_kind'],
        residual_gain_kind=cfg['model']['residual_gain_kind'],
        gated_activation_kind=cfg['model']['gated_activation_kind'],
        attention_logit_scale=cfg['model']['attention_logit_scale'],
        residual_topology_kind=cfg['model']['residual_topology_kind'],
        weight_tying_kind=cfg['model']['weight_tying_kind'],
        output_bias_kind=cfg['model']['output_bias_kind'],output_bias_prior=copy.deepcopy(plan['output_bias_prior']),config=cfg,
        review=dict(path=str(path.resolve()),sha256=file_digest(path)),
        plan_sha256=digest(plan),source_commit=plan['git_commit'])
