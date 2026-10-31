"""Regression for a concrete fixture-to-pilot gap: sparse evaluation metrics."""
import copy
import pytest
from omegaconf import OmegaConf
from research.block_norm_v1 import artifact_acceptance as a,campaign_contract as c,campaign_rules as rules
from research.block_norm_v1.runtime.train.lr_schedule import schedule_contract,multiplier
SEEDS=dict(pilot=374001,screen=374002,fresh=[374011,374012,374013],evaluation=374003,bootstrap=374004)


@pytest.fixture(params=['pilot','screen'])
def sparse(request):
    pilot=request.param=='pilot';spec=(rules.pilot_specs(SEEDS) if pilot else rules.screen_specs(SEEDS))[0]
    cfg=OmegaConf.create(c.trial_config(c.ROOT/'research_runs/tokenizer-block-norm-v2-contract-test/evidence'/('pilots' if pilot else 'quality')/'00',spec,SEEDS))
    # Explicit production eval indices:20step pilot logs0/19;1200step screen
    # logs0/300/600/900/1199. Source trainer condition verified before this test.
    steps=[0,19] if pilot else [0,300,600,900,1199];schedule=schedule_contract(cfg.training)
    rows=[dict(step=i,tokens_processed=(i+1)*1024,device='mps',base_learning_rate=spec['lr'],lr_multiplier=multiplier(schedule,i),scheduled_learning_rate=spec['lr']*multiplier(schedule,i),train_ce_loss=5.) for i in steps]
    return cfg,rows


def test_production_sparse_metrics_accepted(sparse):a.audit_metric_rows(*sparse)


@pytest.mark.parametrize('case',['missing_last','extra_unscheduled','wrong_step','wrong_tokens','wrong_lr'])
def test_incomplete_or_unscheduled_metrics_refused(sparse,case):
    cfg,original=sparse;rows=copy.deepcopy(original)
    if case=='missing_last':rows.pop()
    elif case=='extra_unscheduled':rows.insert(1,dict(rows[0],step=1,tokens_processed=2048))
    elif case=='wrong_step':rows[-1]['step']-=1
    elif case=='wrong_tokens':rows[-1]['tokens_processed']-=1024
    else:rows[-1]['scheduled_learning_rate']*=2
    with pytest.raises(ValueError):a.audit_metric_rows(cfg,rows)
