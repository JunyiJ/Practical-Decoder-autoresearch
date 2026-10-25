"""Tiny tensor recurrence/configuration tests only; no model training."""
import copy
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf
from src.train.weight_averaging import TailWeightAverages, averaging_contract


class Parameters:
    def __init__(self):
        self.w = torch.tensor([1., -2.], requires_grad=True)
        self.b = torch.tensor([.5], requires_grad=True)
    def named_parameters(self):
        return iter([('w', self.w), ('b', self.b)])


def training():
    return OmegaConf.create({'max_iters':12,'weight_averaging':{'kind':'spaced_tail_ema',
        'total_steps':12,'start_step':4,'interval':2,'half_lives':[2,4,8]}})


def values(model,step):
    with torch.no_grad():
        model.w.copy_(torch.tensor([step/3., -step/7.]));model.b.fill_(step/11.)


def test_recurrence_matches_explicit_weighted_sum_and_preserves_online_rng():
    torch.set_num_threads(1);model=Parameters();tracker=TailWeightAverages(model,training());rng=torch.get_rng_state().clone();samples=[]
    for step in range(1,13):
        values(model,step);before={n:p.clone() for n,p in model.named_parameters()}
        tracker.update(step)
        for name,p in model.named_parameters():torch.testing.assert_close(p,before[name],rtol=0,atol=0)
        assert torch.equal(torch.get_rng_state(),rng)
        if step>=4 and (step-4)%2==0:samples.append(before)
    assert tracker.snapshots==5
    for h in [2,4,8]:
        beta=2**(-2/h);actual=tracker.averaged_parameters(h)
        for name in actual:
            expected=samples[0][name].double()*beta**4
            for j in range(1,5):expected+=samples[j][name].double()*(1-beta)*beta**(4-j)
            torch.testing.assert_close(actual[name].double(),expected,rtol=2e-7,atol=2e-7)
        actual['w'].zero_();assert tracker.averages[str(h)]['w'].abs().sum()>0


@pytest.mark.parametrize('split',[0,1,3,4,5,6,11,12])
def test_exact_tensor_continuation_before_at_and_after_start(split):
    model=Parameters();full=TailWeightAverages(model,training());saved=None
    if split==0:saved=full.state_dict()
    for step in range(1,13):
        values(model,step);full.update(step)
        if step==split:saved=full.state_dict()
    clone=Parameters();resumed=TailWeightAverages(clone,training());resumed.load_state_dict(saved,split)
    for step in range(split+1,13):values(clone,step);resumed.update(step)
    for h in [2,4,8]:
        for name,value in full.averaged_parameters(h).items():torch.testing.assert_close(value,resumed.averaged_parameters(h)[name],rtol=0,atol=0)


@pytest.mark.parametrize('field,value',[('kind','swa'),('total_steps',13),('total_steps',True),('start_step',0),('start_step',12),('start_step',True),('interval',0),('interval',3),('interval',True),('half_lives',[]),('half_lives',[2,2]),('half_lives',[4,2]),('half_lives',[True]),('half_lives',[.5]),('half_lives',[1,2,3,4]),('half_lives','2')])
def test_invalid_contract_rejected(field,value):
    t=training();t.weight_averaging[field]=value
    with pytest.raises(ValueError):averaging_contract(t)


def test_disabled_preserves_legacy_state():
    model=Parameters();t=TailWeightAverages(model,SimpleNamespace(max_iters=12));t.update(1);assert t.state_dict() is None
    with pytest.raises(ValueError):t.load_state_dict({},1)


@pytest.mark.parametrize('mutation',['missing','extra','step','snapshots','names','half_life','missing_tensor','shape','dtype','nonfinite'])
def test_reject_corrupt_saved_state(mutation):
    model=Parameters();a=TailWeightAverages(model,training())
    for step in range(1,7):values(model,step);a.update(step)
    s=a.state_dict()
    if mutation=='missing':s.pop('snapshots')
    elif mutation=='extra':s['surprise']=1
    elif mutation=='step':s['completed_steps']=5
    elif mutation=='snapshots':s['snapshots']=3
    elif mutation=='names':s['parameter_names'].reverse()
    elif mutation=='half_life':s['averages']['3']=s['averages'].pop('2')
    elif mutation=='missing_tensor':s['averages']['2'].pop('b')
    elif mutation=='shape':s['averages']['2']['w']=torch.zeros(3)
    elif mutation=='dtype':s['averages']['2']['w']=s['averages']['2']['w'].double()
    elif mutation=='nonfinite':s['averages']['2']['w'][0]=float('nan')
    with pytest.raises(ValueError):TailWeightAverages(Parameters(),training()).load_state_dict(s,6)


def test_sequential_steps_identity_and_final_export_enforced():
    model=Parameters();a=TailWeightAverages(model,training())
    with pytest.raises(ValueError):a.update(2)
    a.update(1)
    with pytest.raises(ValueError):a.update(1)
    with pytest.raises(ValueError):a.averaged_parameters(2)
    model.w=model.w.clone()
    with pytest.raises(ValueError):a.update(2)


def test_inference_average_is_explicitly_rejected_for_optimizer_resume():
    from src.utils.checkpoint import load_model_weights
    with pytest.raises(ValueError,match='Averaged inference weights'):
        load_model_weights(Parameters(),{'averaged_weights':{},'model_state':{}},for_resume=True)


def test_suite_accepts_explicit_valid_averaging_and_rejects_changed_horizon():
    from src.research.capacity_study import suite_spec
    from src.research.team.suite import validate_suite
    from tests.test_capacity_study import plan
    spec=suite_spec(plan('linear_cooldown'),'dense4',.005)
    for arm in ('baseline','candidate'):
        spec[arm]['config']['training']['weight_averaging']={'kind':'spaced_tail_ema',
            'total_steps':3000,'start_step':2000,'interval':10,'half_lives':[10,50,200]}
    assert validate_suite(spec)['candidate']['config']['training']['weight_averaging']['half_lives']==[10,50,200]
    spec['candidate']['config']['training']['weight_averaging']['total_steps']=4000
    with pytest.raises(ValueError):validate_suite(spec)
