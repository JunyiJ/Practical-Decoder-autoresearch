import copy
import json

import pytest
import torch
from omegaconf import OmegaConf

from src.research.tokenizer_horizon_study import (
    RECIPE,
    choose,
    conclusion,
    config,
    specs,
)
from src.train.lr_schedule import multiplier, schedule_contract
from src.train.weight_averaging import TailWeightAverages


@pytest.mark.parametrize('steps,snapshots',[(60,3),(1200,41),(1800,61),(3000,101)])
def test_only_declared_recipe_changes_and_actual_schedule_ema_resume(tmp_path,steps,snapshots):
    original=json.loads(RECIPE.read_text())['config'];changed=config(tmp_path,2811,steps)
    assert changed['model']==original['model'] and changed['data']==original['data']
    allowed={'seed','max_iters','eval_every','checkpoint_dir','metrics_file','summary_file','lr_schedule','weight_averaging'}
    assert {k:v for k,v in changed['training'].items() if k not in allowed}=={k:v for k,v in original['training'].items() if k not in allowed}
    cfg=OmegaConf.create(changed);sched=schedule_contract(cfg.training)
    assert multiplier(sched,0)==1 and multiplier(sched,2*steps//3)==1 and multiplier(sched,steps-1)==.1
    model=torch.nn.Linear(1,1,bias=False);avg=TailWeightAverages(model,cfg.training);expected=None
    start=2*steps//3
    for done in range(1,steps+1):
        with torch.no_grad():model.weight.fill_(done)
        avg.update(done)
        if done==start:expected=float(done)
        elif done>start and (done-start)%10==0:expected=expected*2**(-10/200)+done*(1-2**(-10/200))
        if done==start+5:
            replacement=TailWeightAverages(model,cfg.training);replacement.load_state_dict(avg.state_dict(),done);avg=replacement
    assert avg.snapshots==snapshots
    assert avg.averages['200']['weight'].item()==pytest.approx(expected,rel=2e-6)


def record(stage,seed,steps,ce):
    return {'stage': stage,'seed': seed,'steps': steps,'h200': {'mean_ce': ce,'context': 128,'block_tokens': 2048,'block_target_counts': [2048,2048],'block_loss_sums': [ce*2048,ce*2048]}}


def test_selection_needs_exact_screen_and_confirm_seeds():
    plan={'pilot': False,'durations': [1200,1800,3000],'confirm_seeds': [2811,2812,2813]}
    rows=[record('screen',2801,n,v) for n,v in [(1200,4.),(1800,4.),(3000,5.)]]
    assert choose(rows,plan)==1200
    with pytest.raises(ValueError):choose(rows[:-1],plan)
    with pytest.raises(ValueError):choose(rows+[rows[0]],plan)
    queue=specs(plan,1200);assert len(queue)==9
    assert {r['seed'] for r in queue if r['stage']=='confirm'}=={2811,2812,2813}
    assert len(specs(plan,3000))==3
    fresh=[record('confirm',s,n,v) for s in plan['confirm_seeds'] for n,v in [(1200,4.),(3000,5.)]]
    assert conclusion(fresh,1200,plan)['quality_eligible']
    with pytest.raises(ValueError):conclusion(fresh[:-1],1200,plan)
    with pytest.raises(ValueError):conclusion(fresh+[fresh[0]],1200,plan)
    failed=copy.deepcopy(fresh);failed[0]=record('confirm',2811,1200,5.01)
    assert not conclusion(failed,1200,plan)['quality_eligible']
    no_gain=[record('screen',2801,n,5.) for n in plan['durations']]
    assert choose(no_gain,plan)==3000


def test_undeclared_horizon_rejected(tmp_path):
    for n in [True,1200.,1000,2999]:
        with pytest.raises(ValueError):config(tmp_path,2801,n)
