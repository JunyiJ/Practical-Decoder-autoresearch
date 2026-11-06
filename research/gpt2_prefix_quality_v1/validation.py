"""Independent persisted RAW/H200 state and complete common-byte acceptance."""
import copy,gc,math
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import require,plain
from research.body_allocation_model_v1.evidence import reference
from research.body_allocation_model_v1.optimizer import build_optimizer
from research.gpt2_prefix_data_v1.data import load,architecture
from research.gpt2_prefix_tokenizer_v1.codec import Codec
from research.gpt2_prefix_model_v2.model import GPT,load_weights
from research.gpt2_prefix_resume_v1.checkpoint import validate_resume
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.gpt2_prefix_worker_v1.scoring import score

def verify_artifact(ref):
    path=Path(ref['path']);require(not path.is_symlink() and path.is_file() and path.stat().st_nlink==1
        and path.stat().st_size==ref['size'] and reference(path)['sha256']==ref['sha256'],'Artifact binding differs')

def cache_check(model):
    rows=[];device=next(model.parameters()).device
    with torch.no_grad():
        for batch in (1,8):
            x=(torch.arange(batch*128).reshape(batch,128)%400+1).to(device);full=model(x)[0];c=model.caches(batch)
            replay=torch.cat([model(x[:,:17],cache=c)[0],model(x[:,17:],cache=c,start_pos=17)[0]],1)
            torch.testing.assert_close(full,replay,rtol=3e-4,atol=3e-5);maximum=float((full-replay).abs().max());del replay
            for item in c:item.reset()
            reset=model(x,cache=c)[0];torch.testing.assert_close(full,reset,rtol=3e-4,atol=3e-5)
            size=sum(t.numel()*t.element_size() for item in c for t in item.buffers())//batch;require(size==196608,'Exact unchanged cache storage required')
            rows.append(dict(device=str(device),batch=batch,tokens=128,maximum_logit_difference=maximum,reset_maximum_logit_difference=float((full-reset).abs().max()),cache_bytes_per_item=size))
            del full,reset,c
    return rows

def sampled_bytes(data,cfg,saved_train_state):
    codec=Codec.from_descriptor(data.data_identity['tokenizer']);lengths=torch.tensor([len(codec._engine.decode_single_token_bytes(i)) for i in range(data.vocab_size)],dtype=torch.int64)
    cumulative=torch.cat([torch.zeros(1,dtype=torch.int64),lengths[data.train_data].cumsum(0)])
    generator=torch.Generator().manual_seed(cfg.training.seed);starts=torch.randint(len(data.train_data)-512,(2*cfg.training.max_iters,),generator=generator)
    require(torch.equal(generator.get_state(),saved_train_state),'Byte-exposure audit sampler differs')
    values=(cumulative[starts+513]-cumulative[starts+1]).reshape(-1,2).sum(1).tolist();total=sum(values)
    return dict(sampled_target_bytes=total,sampled_target_tokens=1024*cfg.training.max_iters,per_update_target_bytes=values,mean_bytes_per_sampled_token=total/(1024*cfg.training.max_iters),
        scope='Actual private sampler replay;repeated token-byte lengths,not unique text coverage or literal epochs.Equal token updates do not imply equal bytes/context/FLOPs/time.')

def accept(summary,cfg,binding,*,compare_cpu=False):
    require(compare_cpu is True and cfg.training.max_iters==1200,'Full scientific endpoints require BOTH complete CPU/MPS comparisons')
    require(summary['kind']=='gpt2_prefix_quality_worker_v1' and summary['config']==plain(cfg) and summary['step']==cfg.training.max_iters
        and summary['targets']==cfg.training.max_iters*1024 and summary['architecture']==architecture(cfg),'Worker configuration/coverage differs')
    require(summary['parent']['path']!=summary['h200']['path'],'Distinct original RAW/H200 artifacts required')
    for name in ('parent','h200'):verify_artifact(summary[name])
    data=load(cfg,'mps');require(summary['data']==data.data_identity and summary['holdout_scored'] is False,'Actual data coverage differs')
    expected_metric_steps=[i+1 for i in range(cfg.training.max_iters) if i%cfg.training.eval_every==0 or i==cfg.training.max_iters-1]
    require([r['step'] for r in summary['metrics']]==expected_metric_steps and len(summary['update_seconds'])==cfg.training.max_iters,'Complete sampled metrics/update coverage required')
    require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in summary['update_seconds']) and math.isfinite(summary['training_seconds']) and summary['training_seconds']>0,'Finite positive measured durations required')
    for row in summary['metrics']:
        require(row['targets']==row['step']*1024 and all(math.isfinite(row[n]) for n in ('train_ce','validation_ce','lr_multiplier','update_seconds')),'Finite complete sampled metrics required')
    torch.manual_seed(cfg.training.seed);torch.mps.manual_seed(cfg.training.seed)
    m=GPT(plain(cfg.prefix_model)).to('mps');expected_cpu=torch.get_rng_state();expected_mps=torch.mps.get_rng_state()
    o=build_optimizer(m,cfg);e=TailWeightAverages(m,cfg.training);LearningRateSchedule(o,cfg.training).apply(cfg.training.max_iters-1)
    raw=torch.load(summary['parent']['path'],map_location='cpu',weights_only=True)
    require(raw['cfg']==plain(cfg),'Raw config differs');step=validate_resume(m,o,e,raw,cfg)
    for child,stored in zip(o.optimizers,raw['optimizer_state']['optimizers']):
        require([g['lr'] for g in child.param_groups]==[g['lr'] for g in stored['param_groups']],'Actual terminal scheduled optimizer LR differs')
    require(torch.equal(expected_cpu,raw['cpu_rng']) and torch.equal(expected_mps,raw['mps_rng']),'Dropout-free global RNG advancement differs')
    require(step==cfg.training.max_iters and raw['ema']['snapshots']==summary['ema_snapshots'] and len(raw['ema']['parameter_names'])==summary['parameter_tensors']==88,'Raw completion/EMA differs')
    for name,seed in [('train',cfg.training.seed),('val',cfg.training.eval_seed)]:
        generator=torch.Generator().manual_seed(seed);draws=2*step if name=='train' else 8*cfg.training.eval_iters*len(summary['metrics'])
        high=len(data.train_data)-512 if name=='train' else len(data.val_data)-128
        torch.randint(high,(draws,),generator=generator);require(torch.equal(generator.get_state(),raw['sampler_state'][name]),'Actual vocabulary-specific sampler advancement differs')
    clipped=sum(row['coefficient']<1 for row in raw['clipping_history']);require(clipped==summary['clipped_updates'],'Clipping count differs')
    h=torch.load(summary['h200']['path'],map_location='cpu',weights_only=True)
    require(h['kind']=='h200_gpt2_prefix_model_v2' and h['cfg']==plain(cfg) and h['step']==step and h['architecture']==m.signature
        and h['parent']==summary['parent'] and h['averaged_weights']==dict(spec=e.spec,snapshots=raw['ema']['snapshots'],parameter_names=raw['ema']['parameter_names'],half_life=200,purpose='inference_only_not_optimizer_resume'),'H200 provenance differs')
    require(h['model_state'].keys()==raw['model_state'].keys(),'H200 state coverage differs')
    for name,t in h['model_state'].items():require(torch.equal(t,raw['ema']['averages']['200'].get(name,raw['model_state'][name])),'H200 differs from exact RAW EMA: '+name)
    exposure=sampled_bytes(data,cfg,raw['sampler_state']['train'])
    del o,e;gc.collect();reports={};caches={}
    for name,payload in [('raw',raw),('h200',h)]:
        load_weights(m,payload);m.eval();mps_report=score(m,data,cfg)
        with torch.random.fork_rng(devices=[]):cpu=GPT(plain(cfg.prefix_model)).eval()
        load_weights(cpu,payload);cpu_report=score(cpu,data,cfg)
        difference=abs(cpu_report['total_nats']-mps_report['total_nats'])/mps_report['target_tokens'];bpb=abs(cpu_report['bits_per_byte']-mps_report['bits_per_byte'])
        require(difference<=3e-5 and bpb<=3e-5,'CPU/MPS complete common-byte likelihood differs')
        require(cpu_report['target']==mps_report['target'],'CPU/MPS common byte/token coverage differs')
        reports[name]=dict(mps=mps_report,cpu=cpu_report,cpu_mps_mean_token_loss_difference=difference,cpu_mps_bpb_difference=bpb)
        if name=='h200':caches=dict(mps=cache_check(m),cpu=cache_check(cpu))
        del cpu;gc.collect()
    del payload,raw,h,m,data;gc.collect();torch.mps.empty_cache()
    for name in ('parent','h200'):verify_artifact(summary[name])
    return dict(status='accepted',binding=copy.deepcopy(binding),step=step,targets=step*1024,ema_snapshots=summary['ema_snapshots'],parameter_tensors=88,
        architecture=summary['architecture'],clipped_updates=clipped,full_validation=reports,cache=caches,raw_target_byte_exposure=exposure,
        complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked=True,holdout_scored=False,
        interpretation='Independent full trajectory accepted;panel review/selection remains separate.No cross-tokenizer CE comparison.')
