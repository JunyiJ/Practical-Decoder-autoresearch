"""Actual retained output bundles and negative parent-acceptance checks; no training."""
import copy,gc,json,os
from pathlib import Path
import pytest,torch
from research.block_norm_v1 import campaign_contract as c,artifact_acceptance as a,validation as v
from research.block_norm_v1.runtime.utils.checkpoint import checkpoint_loader
from omegaconf import OmegaConf


@pytest.fixture(scope='module',params=['rms_incumbent','layer_matched_epsilon'])
def fixture(request):
    plan=json.loads(Path(os.environ['BLOCK_NORM_PLAN_PATH']).read_text())
    req=copy.deepcopy(plan['requests'][request.param]);bundle=a.read_bundle(req)
    cfg=bundle['cfg'];data=checkpoint_loader(copy.deepcopy(cfg),bundle['avg'],cfg.data.path,'cpu')
    report=plan['validation_reports'][request.param]
    yield dict(request=req,bundle=bundle,data=data,report=report,arm=request.param)
    gc.collect();torch.mps.synchronize();torch.mps.empty_cache()


def test_parent_accepts_real_bundle_and_recomputes_score(fixture,record_property):
    before=torch.get_rng_state().clone();mps_before=torch.mps.get_rng_state().clone()
    receipt=a.accept(fixture['request'],fixture['report'])
    assert receipt['accepted'] is True and receipt['launch_permitted'] is False and receipt['quality_adopted'] is False
    assert receipt['audit']==dict(completed_steps=2,tokens_processed=2048,ema_parameters=88 if fixture['arm']=='rms_incumbent' else 96,ema_snapshots=2)
    assert torch.equal(before,torch.get_rng_state()) and torch.equal(mps_before,torch.mps.get_rng_state())
    assert receipt['validation']==fixture['report']
    record_property('real_artifact_acceptance',dict(arm=fixture['arm'],receipt=receipt,rng_preserved=True))
    gc.collect();torch.mps.synchronize();torch.mps.empty_cache()


def replace_path(value,path,new):
    # Copy only dictionaries on the edited path; preserve large immutable tensor storage.
    out=dict(value);cursor=out
    for key in path[:-1]:cursor[key]=dict(cursor[key]);cursor=cursor[key]
    cursor[path[-1]]=new
    return out


@pytest.mark.parametrize('case',['raw_step','average_step','raw_config','signature','training_contract','parent_hash','export_hash','completed','stop_reason','summary_path','missing_metric','schedule','ema_progress','ema_names','h200_metadata','h200_tensor','raw_nonfinite','extra_h200_optimizer'])
def test_corrupt_artifact_payload_rejected(fixture,case):
    b=fixture['bundle'];changed=dict(b)
    if case=='raw_step':changed=replace_path(b,['raw','step'],1)
    elif case=='average_step':changed=replace_path(b,['avg','step'],1)
    elif case=='raw_config':changed=replace_path(b,['raw','cfg','training','seed'],1)
    elif case=='signature':changed=replace_path(b,['avg',a.CHECKPOINT_KEY],{})
    elif case=='training_contract':changed=replace_path(b,['raw','training_state','contract','batch_size'],1)
    elif case=='parent_hash':changed=replace_path(b,['avg','parent_checkpoint','sha256'],'0'*64)
    elif case=='export_hash':changed=replace_path(b,['export','checkpoint_sha256'],'0'*64)
    elif case=='completed':changed=replace_path(b,['summary','completed_steps'],1)
    elif case=='stop_reason':changed=replace_path(b,['summary','stop_reason'],'time_limit')
    elif case=='summary_path':changed=replace_path(b,['summary','final_checkpoint'],'/private/tmp/wrong.pt')
    elif case=='missing_metric':changed['metrics']=b['metrics'][:1]
    elif case=='schedule':
        rows=[dict(row) for row in b['metrics']];rows[0]['scheduled_learning_rate']=.5
        changed=replace_path(b,['raw','training_state','metrics'],rows);changed['metrics']=rows;changed['summary']=dict(b['summary'],metrics=rows)
    elif case=='ema_progress':changed=replace_path(b,['raw','training_state','weight_averaging_state','snapshots'],1)
    elif case=='ema_names':changed=replace_path(b,['raw','training_state','weight_averaging_state','parameter_names'],[])
    elif case=='h200_metadata':changed=replace_path(b,['avg','averaged_weights','half_life'],100)
    elif case in ('h200_tensor','raw_nonfinite'):
        which='avg' if case=='h200_tensor' else 'raw';name='transformer.h.0.ln1.weight'
        tensor=b[which]['model_state'][name].clone();tensor[0]=float('nan') if case=='raw_nonfinite' else tensor[0]+1
        changed=replace_path(b,[which,'model_state',name],tensor)
    else:changed['avg']=dict(b['avg'],optimizer_state={})
    with pytest.raises(ValueError):a.audit_payloads(fixture['request'],changed,fixture['data'])
    # Negative inputs never alter the original in-memory or disk fixture.
    a.audit_payloads(fixture['request'],b,fixture['data'])


@pytest.mark.parametrize('kind',['config','parent','h200','summary','metrics','export'])
def test_changed_file_pin_refused(fixture,kind):
    req=copy.deepcopy(fixture['request']);req['files'][kind]['sha256']='0'*64
    with pytest.raises(ValueError,match='hash'):a.verify_request(req)


def test_forged_consistent_score_refused_by_parent_recomputation(fixture):
    report=copy.deepcopy(fixture['report']);target=report['target']
    report['window_loss_sums']=[x+.1*n for x,n in zip(report['window_loss_sums'],target['window_counts'])]
    report['block_loss_sums']=[x+.1*n for x,n in zip(report['block_loss_sums'],target['block_counts'])]
    report['total_loss']+=.1*target['targets'];report['ce']=report['total_loss']/target['targets']
    import math
    report['bits_per_byte']=report['total_loss']/math.log(2)/target['target_bytes']
    v.validate_report(report,report['binding'],fixture['data'])  # Coherent fabricated arithmetic is insufficient.
    with pytest.raises(ValueError,match='Parent recomputation'):a.accept(fixture['request'],report)
    gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
