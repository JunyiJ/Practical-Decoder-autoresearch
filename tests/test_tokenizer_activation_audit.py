"""Independent audit on actual activation shapes and synthetic CPU optimizer updates.

No corpus split, research trial, disk checkpoint or MPS training. Resource traces
below are synthetic test inputs and cannot be used as pilot acceptance evidence.
"""
import copy
import gc

import pytest
import torch
from omegaconf import OmegaConf

from src.models.gpt import GPT
from src.train.optimizers import build_optimizer
from src.train.lr_schedule import LearningRateSchedule
from src.train.weight_averaging import TailWeightAverages
from src.research import tokenizer_activation_audit as a
from src.research.tokenizer_activation_config import config
from src.research.tokenizer_activation_rules import pilot_specs
from src.research.tokenizer_wide_hybrid_resources import _header


@pytest.fixture(scope='module', params=pilot_specs(), ids=['silu', 'gelu_exact'])
def pair(request):
    torch.set_num_threads(1); torch.manual_seed(7193)
    cfg_dict=config('/tmp/activation-audit-ram-only', request.param); cfg=OmegaConf.create(cfg_dict)
    model=GPT(cfg.model); optimizer=build_optimizer(model,cfg.training)
    schedule=LearningRateSchedule(optimizer,cfg.training); ema=TailWeightAverages(model,cfg.training)
    # Twenty synthetic updates instantiate all real optimizer states and the
    # exact eight-snapshot pilot EMA schedule, without reading any data loader.
    for step in range(20):
        schedule.apply(step); optimizer.zero_grad()
        ids=torch.randint(0,50257,(1,3));targets=torch.randint(0,50257,(1,3))
        _,loss=model(ids,targets);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();ema.update(step+1)
    contract=dict(tokenizer={'purpose':'synthetic CPU fixture'}, model=cfg_dict['model'],
                  weight_averaging=cfg_dict['training']['weight_averaging'])
    raw=dict(cfg=cfg_dict,step=20,model_math_version=2,parameter_layout_version=2,
             model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),
             training_state=dict(contract=contract,weight_averaging_state=ema.state_dict()))
    raw_ref=dict(path='/tmp/activation-audit-ram-only/raw.pt',sha256='a'*64)
    state=dict(raw['model_state']);state.update(ema.averaged_parameters(200))
    averaged=dict(cfg=cfg_dict,step=20,model_math_version=2,parameter_layout_version=2,model_state=state,
        tokenizer_contract=contract['tokenizer'],parent_checkpoint=raw_ref,
        averaged_weights=dict(spec=cfg_dict['training']['weight_averaging'],completed_steps=20,snapshots=8,
                              half_life=200,purpose='inference_only_not_optimizer_resume',parameter_names=list(dict(model.named_parameters()))))
    yield cfg_dict,contract,raw,averaged,raw_ref
    del model,optimizer,ema,state,raw,averaged;gc.collect()


def test_real_activation_payload_optimizer_and_export(pair):
    result=a.check_payloads(*pair)
    assert result['parameter_count']==28492416 and result['snapshots']==8
    assert result['optimizer_structure_audited'] is True
    assert result['exact_export_matches_stored_ema'] is True
    assert result['optimizer_resume_audited'] is False and result['ema_recurrence_audited'] is False


@pytest.mark.parametrize('mutation',['gate','step','ema_count','missing_optimizer','lr','counter','parent','export','nonfinite'])
def test_independent_audit_rejects_corruption(pair,mutation):
    cfg,contract,raw,avg,ref=pair
    # Copy only containers that change. Tensor payloads remain shared and are
    # restored in finally for the two numerical mutation cases.
    raw=dict(raw);avg=dict(avg);restore=None
    try:
        if mutation=='gate':
            raw['cfg']=copy.deepcopy(cfg);raw['cfg']['model']['gated_activation_kind']='invalid'
        elif mutation=='step': raw['step']=True
        elif mutation=='ema_count':
            avg['averaged_weights']=dict(avg['averaged_weights'],snapshots=7)
        elif mutation in ('missing_optimizer','lr','counter'):
            opt=raw['optimizer_state'];children=list(opt['optimizers']);child=dict(children[-1]);children[-1]=child
            raw['optimizer_state']=dict(opt,optimizers=children)
            if mutation=='missing_optimizer': child['state']={}
            elif mutation=='lr':
                child['param_groups']=copy.deepcopy(child['param_groups']);child['param_groups'][0]['lr']*=2
            else:
                child['state']=dict(child['state']);key=next(iter(child['state']))
                child['state'][key]=dict(child['state'][key],step=torch.tensor(19.))
        elif mutation=='parent': avg['parent_checkpoint']=dict(ref,sha256='b'*64)
        else:
            name=raw['training_state']['weight_averaging_state']['parameter_names'][0]
            tensor=avg['model_state'][name].view(-1);old=tensor[0].clone();restore=(tensor,old)
            tensor[0]=float('nan') if mutation=='nonfinite' else old+1
        with pytest.raises(ValueError): a.check_payloads(cfg,contract,raw,avg,ref)
    finally:
        if restore is not None:restore[0][0]=restore[1]


def trace(pilot):
    limits=a.profile(pilot);sample=dict(elapsed_seconds=0.,peak_rss_bytes=200000000,mps_driver_bytes=100000000,free_bytes=16*1024**3)
    report=dict(kind=a.RESOURCE_KIND,status='completed',violations=[],limits=limits,
        plan_sha256='a'*64,request_sha256='b'*64,result_sha256='c'*64,
        samples=[sample,dict(sample,elapsed_seconds=2.)],events=dict(path='resource-events.jsonl'))
    events=[_header('a'*64,'b'*64,a.RESOURCE_KIND,limits)]+[dict(event='sample',**s) for s in report['samples']]
    events.append(dict(event='terminal',status='completed',result_sha256='c'*64,reason=None))
    return report,events


@pytest.mark.parametrize('pilot',[False,True])
def test_resource_profiles_and_complete_trace(pilot):
    report,events=trace(pilot)
    assert a.check_resources(report,events,'a'*64,'b'*64,'c'*64,pilot)['samples']==2
    assert report['limits']['timeout_seconds']==(600 if pilot else 7200)
    for broken in (events[:-1],events[::-1]):
        with pytest.raises(ValueError):a.check_resources(report,broken,'a'*64,'b'*64,'c'*64,pilot)
    with pytest.raises(ValueError):a.check_resources(report,events,'d'*64,'b'*64,'c'*64,pilot)
    report['samples'][-1]['peak_rss_bytes']=4*1024**3
    with pytest.raises(ValueError):a.check_resources(report,events,'a'*64,'b'*64,'c'*64,pilot)


def measured_fixture():
    from src.research.protocol import digest
    from src.train.lr_schedule import schedule_contract,multiplier
    spec=pilot_specs()[0];cfg=config('/tmp/activation-measurement-fixture',spec);t=cfg['training']
    req=dict(spec=spec,config=cfg);counts=[min(2048,34278-i) for i in range(0,34278,2048)]
    outputs=dict(status='completed',config_sha256=digest(cfg),raw=dict(path=t['checkpoint_dir']+'/raw.pt',sha256='a'*64),
                 h200=dict(path=t['checkpoint_dir']+'/h200.pt',sha256='b'*64))
    import math
    measured=dict(training_contract={},cache={})
    for role in ('raw','h200'):
        measured[role]=dict(context=128,eval_batch_size=8,block_tokens=2048,target_count=34278,target_bytes=111536,
            window_count=268,parameters=a.PARAMETERS,block_target_counts=counts,block_loss_sums=[v*5. for v in counts],
            mean_ce=5.,bits_per_byte=34278*5./111536/math.log(2),perplexity=math.exp(5.),
            checkpoint=outputs[role]['path'],checkpoint_sha256=outputs[role]['sha256'])
    metrics=[]
    for i in (0,19):
        ratio=multiplier(schedule_contract(OmegaConf.create(t)),i)
        metrics.append(dict(step=i,train_loss=5.,train_total_loss=5.,train_ce_loss=5.,train_aux_loss=None,
            val_loss=5.,val_ce_loss=5.,val_aux_loss=None,tokens_processed=(i+1)*1024,
            lr_multiplier=ratio,base_learning_rate=t['learning_rate'],scheduled_learning_rate=t['learning_rate']*ratio))
    summary=dict(device='mps',stop_reason='max_iters',max_iters=20,completed_steps=20,best_checkpoint=None,
        final_checkpoint=outputs['raw']['path'],metrics=metrics,final_metric=metrics[-1],
        weight_averaging=dict(spec=t['weight_averaging'],completed_steps=20,snapshots=8))
    return req,measured,outputs,summary,metrics


def test_measurement_arithmetic_training_schedule_and_coverage():
    fixture=measured_fixture();result=a.check_measurements(*fixture)
    assert result['target_count']==34278 and result['h200_ce']==5.
    for case in ('coverage','ce','endpoint','aux','schedule','missing_metric','snapshots'):
        req,measured,outputs,summary,metrics=copy.deepcopy(fixture)
        if case=='coverage':measured['h200']['block_target_counts'][-1]-=1
        elif case=='ce':measured['h200']['mean_ce']=4.
        elif case=='endpoint':measured['h200']['checkpoint_sha256']='e'*64
        elif case=='aux':metrics[0]['train_aux_loss']=.1
        elif case=='schedule':metrics[-1]['lr_multiplier']=1.
        elif case=='missing_metric':metrics.pop()
        else:summary['weight_averaging']['snapshots']=7
        with pytest.raises(ValueError):a.check_measurements(req,measured,outputs,summary,metrics)
