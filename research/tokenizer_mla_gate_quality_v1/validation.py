import copy
import gc
import math
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import require,plain
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_mla_gate_model_v1.model import GPT,load_weights
from research.tokenizer_mla_gate_model_v1.optimizer import build_optimizer
from research.tokenizer_mla_gate_resume_v1.data import load,architecture
from research.tokenizer_mla_gate_resume_v1.checkpoint import validate_resume,load_export
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule,schedule_contract,multiplier
from research.tokenizer_mla_gate_worker_v1.scoring import score


def verify_artifact(ref):
    p=Path(ref['path']);require(not p.is_symlink() and p.is_file() and p.stat().st_nlink==1 and p.stat().st_size==ref['size'] and reference(p)['sha256']==ref['sha256'],'Exact original artifact identity required')


def cache_check(model):
    rows=[];device=next(model.parameters()).device
    with torch.no_grad():
        for batch in [1,8]:
            x=(torch.arange(batch*128).reshape(batch,128)%400+1).to(device)
            full=model(x,include_aux_loss=False)[0];c=model.caches(batch)
            replay=torch.cat([model(x[:,:17],cache=c,include_aux_loss=False)[0],model(x[:,17:],cache=c,start_pos=17,include_aux_loss=False)[0]],1)
            torch.testing.assert_close(full,replay,rtol=3e-4,atol=3e-5);difference=float((full-replay).abs().max());del replay
            for item in c:item.reset()
            reset=model(x,cache=c,include_aux_loss=False)[0];torch.testing.assert_close(full,reset,rtol=3e-4,atol=3e-5)
            size=sum(t.numel()*t.element_size() for item in c for t in item.buffers())//batch;require(size==196608,'Unchanged attention cache storage required')
            rows.append(dict(device=str(device),batch=batch,tokens=128,maximum_logit_difference=difference,reset_maximum_logit_difference=float((full-reset).abs().max()),cache_bytes_per_item=size))
            del full,reset,c
    return rows


def accept(summary,cfg,binding,*,compare_cpu=False):
    require(compare_cpu is True and cfg.training.max_iters==1200,'Complete CPU AND MPS scientific scoring required')
    require(summary['kind']=='tokenizer_mla_gate_quality_worker_v1' and summary['config']==plain(cfg) and summary['step']==cfg.training.max_iters
        and summary['targets']==1024*cfg.training.max_iters and summary['architecture']==architecture(cfg),'Worker config/coverage differs')
    require(summary['parent']['path']!=summary['h200']['path'],'Distinct RAW/H200 artifacts required')
    for name in ['parent','h200']:verify_artifact(summary[name])
    data=load(cfg,'mps');require(summary['data']==data.data_identity and summary['holdout_scored'] is False,'Original data coverage required')
    steps=[i+1 for i in range(cfg.training.max_iters) if i%cfg.training.eval_every==0 or i==cfg.training.max_iters-1]
    require([r['step'] for r in summary['metrics']]==steps and len(summary['update_seconds'])==cfg.training.max_iters,'Complete sampled/update history required')
    require(all(type(v) is float and math.isfinite(v) and v>0 for v in summary['update_seconds']) and type(summary['training_seconds']) is float and math.isfinite(summary['training_seconds']) and summary['training_seconds']>0,'Finite durations required')
    torch.default_generator.manual_seed(cfg.training.seed);torch.mps.manual_seed(cfg.training.seed)
    model=GPT(plain(cfg.gate_model)).to('mps');expected_cpu=torch.get_rng_state().clone();expected_mps=torch.mps.get_rng_state().clone()
    opt=build_optimizer(model,cfg.training);ema=TailWeightAverages(model,cfg.training);LearningRateSchedule(opt,cfg.training).apply(cfg.training.max_iters-1)
    raw=torch.load(summary['parent']['path'],map_location='cpu',weights_only=True)
    require(raw['cfg']==plain(cfg),'Raw configuration differs');step=validate_resume(model,opt,ema,raw,cfg)
    require(torch.equal(expected_cpu,raw['cpu_rng']) and torch.equal(expected_mps,raw['mps_rng']),'Deterministic global RNG advancement differs')
    require(step==cfg.training.max_iters and raw['ema']['snapshots']==summary['ema_snapshots'] and len(raw['ema']['parameter_names'])==summary['parameter_tensors']==model.signature['muon_groups']+model.signature['adam_groups'],'Actual optimizer/EMA coverage differs')
    require(summary['parameter_gradient_steps']==raw['optimizer_state']['gradient_steps'],'Per-parameter gradient/decay exposure differs')
    for row in summary['metrics']:
        history=raw['clipping_history'][row['step']-1]
        require(row['targets']==row['step']*1024 and all(type(row[k]) is float and math.isfinite(row[k]) for k in ['train_ce','validation_ce','lr_multiplier','update_seconds']),'Complete finite sampled metrics required')
        require(row['train_ce']==history['ce_loss']
            and row['lr_multiplier']==multiplier(schedule_contract(cfg.training),row['step']-1) and row['update_seconds']==summary['update_seconds'][row['step']-1],'Sampled CE/aux/objective/history binding differs')
    require(summary['final_online_train_ce']==raw['clipping_history'][-1]['ce_loss'],'Final CE/aux/objective differs')
    for name,seed in [('train',cfg.training.seed),('val',cfg.training.eval_seed)]:
        generator=torch.Generator().manual_seed(seed);draws=2*step if name=='train' else 8*cfg.training.eval_iters*len(summary['metrics'])
        high=267688-512 if name=='train' else 34279-128
        torch.randint(high,(draws,),generator=generator);require(torch.equal(generator.get_state(),raw['sampler_state'][name]),'Actual private sampler advancement differs')
    clipped=sum(r['coefficient']<1 for r in raw['clipping_history']);require(clipped==summary['clipped_updates'],'Actual clipping summary differs')
    exported=torch.load(summary['h200']['path'],map_location='cpu',weights_only=True)
    require(exported['parent']=={k:summary['parent'][k] for k in ('path','sha256')} and exported['averaged_weights']==dict(spec=ema.spec,snapshots=raw['ema']['snapshots'],parameter_names=raw['ema']['parameter_names'],half_life=200,purpose='inference_only_not_optimizer_resume'),'H200 provenance differs')
    require(exported['model_state'].keys()==raw['model_state'].keys(),'H200 tensor inventory differs')
    for name,value in exported['model_state'].items():require(torch.equal(value,raw['ema']['averages']['200'].get(name,raw['model_state'][name])),'H200 must equal exact RAW learned EMA plus original buffers: '+name)
    del opt,ema;gc.collect();reports={};caches={}
    for name,payload in [('raw',raw),('h200',exported)]:
        if name=='h200':load_export(model,payload,cfg)
        else:load_weights(model,dict(kind='tokenizer_mla_gate_weights_v1',architecture=payload['architecture'],model_state=payload['model_state']))
        model.eval();mps_report=score(model,data,cfg)
        with torch.random.fork_rng(devices=[]):cpu=GPT(plain(cfg.gate_model)).eval()
        if name=='h200':load_export(cpu,payload,cfg)
        else:load_weights(cpu,dict(kind='tokenizer_mla_gate_weights_v1',architecture=payload['architecture'],model_state=payload['model_state']))
        cpu_report=score(cpu,data,cfg);difference=abs(cpu_report['ce']-mps_report['ce'])
        require(difference<=3e-5 and cpu_report['target']==mps_report['target'],'Complete CPU/MPS native CE/coverage differs')
        reports[name]=dict(mps=mps_report,cpu=cpu_report,cpu_mps_ce_difference=difference)
        if name=='h200':caches=dict(mps=cache_check(model),cpu=cache_check(cpu))
        del cpu;gc.collect()
    exposure=copy.deepcopy(summary['parameter_gradient_steps']);del payload,raw,exported,model,data;gc.collect();torch.mps.empty_cache()
    for name in ['parent','h200']:verify_artifact(summary[name])
    return dict(status='accepted',binding=copy.deepcopy(binding),step=step,targets=step*1024,ema_snapshots=summary['ema_snapshots'],parameter_tensors=summary['parameter_tensors'],
                architecture=summary['architecture'],clipped_updates=clipped,parameter_gradient_steps=exposure,full_validation=reports,cache=caches,
                complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked=True,holdout_scored=False,interpretation='Independent complete scientific endpoint acceptance;not panel selection or adoption.')
