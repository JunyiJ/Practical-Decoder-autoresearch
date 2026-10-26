"""Resume and trusted metric metadata only; no training or checkpoint evaluation."""
import copy
from types import SimpleNamespace
import pytest
from omegaconf import OmegaConf

from src.train.train import _training_contract, _contracts_match
from src.train.lr_schedule import schedule_contract, multiplier
from src.research.runner import validate_result


@pytest.mark.parametrize('field,value', [('kind','linear_cooldown'),('warmup_steps',300),
    ('initial_ratio',.2),('start_step',1999),('min_ratio',.2),('total_steps',3001)])
def test_resume_contract_binds_every_warmup_field(tmp_path,field,value):
    corpus=tmp_path/'text.txt'; corpus.write_text('synthetic metadata only')
    cfg=OmegaConf.create({'data':{'path':str(corpus)},'model':{'dim':64},
        'training':{'max_iters':3000,'batch_size':8,'learning_rate':.005,
        'weight_decay':.00006,'optimizer':'muon','decay_mode':'independent',
        'lr_schedule':dict(kind='warmup_linear_cooldown',total_steps=3000,
            start_step=2000,min_ratio=.1,warmup_steps=100,initial_ratio=.1)}})
    loader=SimpleNamespace(batch_layout='paired_blocks512',train_fraction=.8,val_fraction=.1,eval_seed=2026)
    contract=_training_contract(cfg,'cpu',loader)
    assert contract['lr_schedule']==dict(cfg.training.lr_schedule)
    assert _contracts_match(copy.deepcopy(contract),contract)
    changed=copy.deepcopy(contract); changed['lr_schedule'][field]=value
    assert not _contracts_match(changed,contract)


@pytest.mark.parametrize('corrupt',[None,'first_ratio','peak_ratio','final_ratio',
    'base_rate','scheduled_rate','missing_metric','missing_field','nonfinite'])
def test_worker_requires_complete_exact_absolute_schedule_history(tmp_path,corrupt):
    t=dict(max_iters=12,batch_size=8,block_size=128,learning_rate=.005,eval_every=1,
        lr_schedule=dict(kind='warmup_linear_cooldown',total_steps=12,
            start_step=8,min_ratio=.1,warmup_steps=4,initial_ratio=.1))
    request={'stage':'confirmation','config':{'training':t}}
    p={'device':'cpu','training':t}
    checkpoint=tmp_path/'final.pt'; checkpoint.write_bytes(b'not loaded in this metadata test')
    metrics=[]
    spec=schedule_contract(OmegaConf.create(t))
    for step in range(12):
        ratio=multiplier(spec,step)
        metrics.append(dict(step=step,tokens_processed=(step+1)*1024,val_ce_loss=1.5,
            rss_mb=1.,accel_mem_mb=None,accel_mem_kind=None,lr_multiplier=ratio,
            base_learning_rate=.005,scheduled_learning_rate=.005*ratio))
    result=dict(device='cpu',completed_steps=12,stop_reason='max_iters',
        final_metric=metrics[-1],metrics=metrics,final_checkpoint=str(checkpoint),training_time_sec=1.)
    if corrupt=='first_ratio': metrics[0]['lr_multiplier']=1.
    elif corrupt=='peak_ratio': metrics[3]['lr_multiplier']=.9
    elif corrupt=='final_ratio': metrics[-1]['lr_multiplier']=1.
    elif corrupt=='base_rate': metrics[5]['base_learning_rate']=.004
    elif corrupt=='scheduled_rate': metrics[5]['scheduled_learning_rate']=.0001
    elif corrupt=='missing_metric': del metrics[5]
    elif corrupt=='missing_field': del metrics[5]['lr_multiplier']
    elif corrupt=='nonfinite': metrics[5]['scheduled_learning_rate']=float('nan')
    if corrupt is None:
        assert validate_result(result,request,p,tmp_path)['metric']==1.5
    else:
        with pytest.raises(ValueError): validate_result(result,request,p,tmp_path)
