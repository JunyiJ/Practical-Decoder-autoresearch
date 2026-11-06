"""Independent complete artifact, clipping, full-validation and cache acceptance."""
import copy,gc,json,math
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import require,plain,signature
from research.body_allocation_model_v1.evidence import reference
from research.body_allocation_model_v1.model import GPT,load_weights
from research.body_allocation_model_v1.optimizer import build_optimizer
from research.body_allocation_model_v1.checkpoint import validate_resume
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.final_ln_scale_worker_v1.validation import score,target_contract,validate_report
from .data import load

def verify_artifact(ref):
    path=Path(ref['path']);require(not path.is_symlink() and path.is_file() and path.stat().st_nlink==1
        and path.stat().st_size==ref['size'] and reference(path)['sha256']==ref['sha256'],'Artifact binding differs')

def cache_check(model):
    rows=[]
    with torch.no_grad():
        for batch in (1,8):
            x=(torch.arange(batch*128).reshape(batch,128)%400+1).to('mps');full=model(x)[0];c=model.caches(batch)
            replay=torch.cat([model(x[:,:17],cache=c)[0],model(x[:,17:],cache=c,start_pos=17)[0]],1)
            torch.testing.assert_close(full,replay,rtol=3e-4,atol=3e-5)
            maximum=float((full-replay).abs().max());del replay
            for item in c:item.reset()
            torch.testing.assert_close(full,model(x,cache=c)[0],rtol=3e-4,atol=3e-5)
            rows.append(dict(batch=batch,tokens=128,maximum_logit_difference=maximum,cache_bytes_per_item=sum(t.numel()*t.element_size() for item in c for t in item.buffers())//batch))
    return rows

def accept(summary,cfg,binding,*,compare_cpu=False):
    require(summary['config']==plain(cfg) and summary['step']==cfg.training.max_iters
        and summary['targets']==cfg.training.max_iters*1024 and summary['architecture']==signature(cfg.model),'Worker configuration/coverage differs')
    for name in ('parent','h200'):verify_artifact(summary[name])
    data=load(cfg,'mps');require(summary['data']==data.data_identity and summary['holdout_scored'] is False,'Data coverage differs')
    torch.manual_seed(cfg.training.seed);torch.mps.manual_seed(cfg.training.seed)
    m=GPT(cfg.model).to('mps');expected_cpu=torch.get_rng_state();expected_mps=torch.mps.get_rng_state()
    o=build_optimizer(m,cfg);e=TailWeightAverages(m,cfg.training)
    raw=torch.load(summary['parent']['path'],map_location='cpu',weights_only=True)
    require(raw['cfg']==plain(cfg),'Raw config differs');step=validate_resume(m,o,e,raw,cfg)
    require(torch.equal(expected_cpu,raw['cpu_rng']) and torch.equal(expected_mps,raw['mps_rng']),'Global RNG advancement differs from dropout-free private-sampler contract')
    require(step==cfg.training.max_iters and raw['ema']['snapshots']==summary['ema_snapshots']
        and len(raw['ema']['parameter_names'])==summary['parameter_tensors'],'Raw completion/EMA differs')
    for n,seed in [('train',cfg.training.seed),('val',cfg.training.eval_seed)]:
        g=torch.Generator().manual_seed(seed)
        draws=2*step if n=='train' else 8*cfg.training.eval_iters*len(summary['metrics'])
        high=267688-512 if n=='train' else 34279-128
        torch.randint(high,(draws,),generator=g)
        require(torch.equal(g.get_state(),raw['sampler_state'][n]),'Sampler advancement differs')
    clipped=sum(r['coefficient']<1 for r in raw['clipping_history']);require(clipped==summary['clipped_updates'],'Clipping summary differs')
    h=torch.load(summary['h200']['path'],map_location='cpu',weights_only=True)
    require(h['kind']=='h200_body_allocation_v1' and h['cfg']==plain(cfg) and h['step']==step
        and h['parent']==summary['parent'] and h['averaged_weights']==dict(spec=e.spec,snapshots=raw['ema']['snapshots'],parameter_names=raw['ema']['parameter_names'],half_life=200,purpose='inference_only_not_optimizer_resume'),'H200 provenance differs')
    require(h['model_state'].keys()==raw['model_state'].keys(),'H200 state coverage differs')
    for name,t in h['model_state'].items():
        expected=raw['ema']['averages']['200'].get(name,raw['model_state'][name])
        require(torch.equal(t,expected),'H200 differs from exact RAW EMA: '+name)
    del raw,o,e;gc.collect();load_weights(m,h);m.eval();del h
    parent_report=dict(binding=copy.deepcopy(binding),**score(m,data));validate_report(parent_report,binding,data)
    caches=cache_check(m);del m;gc.collect();torch.mps.empty_cache()
    difference=None
    if compare_cpu:
        cpu=GPT(cfg.model).eval();h=torch.load(summary['h200']['path'],map_location='cpu',weights_only=True);load_weights(cpu,h);del h
        cpu_report=score(cpu,data);difference=abs(cpu_report['ce']-parent_report['ce'])
        require(difference<=3e-5,'CPU/MPS complete validation CE differs')
        del cpu;gc.collect()
    for name in ('parent','h200'):verify_artifact(summary[name])
    return dict(status='accepted',binding=binding,step=step,targets=step*1024,ema_snapshots=summary['ema_snapshots'],
        parameter_tensors=summary['parameter_tensors'],architecture=summary['architecture'],clipped_updates=clipped,
        full_validation=parent_report,cache=caches,cpu_mps_ce_difference=difference,
        complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked=True,holdout_scored=False)
