"""Strict MPS synthetic FFN model forward/gradient/cache prerequisite."""
import copy, datetime, gc, hashlib, json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.ema_horizon_v1 import contract as c, execution as e
from research.block_norm_v1.fixture_dispatch import read, reference
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.ffn_structure_v1.runtime.models.gpt import GPT
from research.ffn_structure_v1.runtime.models.block_norm import FFN_POLICY
from research.ffn_structure_v1.runtime.attention.cache import KVCache
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
B=Path(__file__).resolve().parent
torch.set_num_threads(4)
def compare(a,b,atol=3e-4,rtol=3e-4):
    a=a.detach().cpu();b=b.detach().cpu();torch.testing.assert_close(a,b,atol=atol,rtol=rtol)
    return float((a-b).abs().max())
def caches(cfg):return [KVCache(cfg,torch.device('mps'),batch_size=2,dtype=torch.float32,layer_index=i) for i in range(4)]
with session_lock(GLOBAL_LOCK.parent):
    e.idle();previous=read(reference(B/'model-integration001.json'))
    for n,h in previous['new_sources'].items():assert hashlib.sha256((c.ROOT/n).read_bytes()).hexdigest()==h
    frozen=c.source_inventory();assert frozen==previous['frozen_sources']
    with StrictMpsGuard(B/'mps-integration001-guard.json',c.ROOT):
        seeds=dict(pilot=376971,screen=376972,fresh=[376981,376982,376983],evaluation=376973,bootstrap=376974)
        base=OmegaConf.create(c.config(c.ROOT/'research_runs/tokenizer-ema-horizon-v1-ffn-mps-component/evidence/pilots/00','pilot',0,seeds))
        base.model.tokenizer_architecture_policy=FFN_POLICY
        records=[]
        for arm in ('swiglu','ungated_silu'):
            cfg=copy.deepcopy(base)
            if arm=='ungated_silu':cfg.model.ffn='silu';cfg.model.ffn_width_policy='ungated_silu_63_16'
            torch.manual_seed(376975);cpu=GPT(cfg.model);gpu=copy.deepcopy(cpu).to('mps')
            # Repeated IDs exercise grouped embedding accumulation, not only unique rows.
            x=(torch.arange(18).reshape(2,9)%5+23).long();target=(x+1)%50257
            a,la=cpu(x,target);b,lb=gpu(x.to('mps'),target.to('mps'))
            row=dict(arm=arm,logits_max_abs=compare(a,b),ce_abs=compare(la,lb,1e-4,1e-5))
            la.backward();lb.backward();max_grad=0.
            for (n,p),(m,q) in zip(cpu.named_parameters(),gpu.named_parameters()):
                assert n==m and q.device.type=='mps' and q.grad.device.type=='mps' and q.grad.dtype==torch.float32
                max_grad=max(max_grad,compare(p.grad,q.grad,3e-5,3e-3))
            row['gradient_max_abs']=max_grad;cpu.zero_grad();gpu.zero_grad();gpu.eval()
            with torch.no_grad():
                x=x.to('mps');y=x+31;full=gpu(x)[0];full_y=gpu(y)[0]
                ca,cb=caches(cfg),caches(cfg);parts=[];other=[];pos=0
                for count in (3,1,5):
                    parts.append(gpu(x[:,pos:pos+count],cache=ca,start_pos=pos)[0])
                    other.append(gpu(y[:,pos:pos+count],cache=cb,start_pos=pos)[0]);pos+=count
                row['cached_max_abs']=compare(full,torch.cat(parts,1));row['interleaved_max_abs']=compare(full_y,torch.cat(other,1))
                changed=x.clone();changed[:,4:]+=3;row['causal_max_abs']=compare(full[:,:4],gpu(changed)[0][:,:4],0.,0.)
                other_state=[{n:v.clone() for n,v in cache.state_dict().items()} for cache in cb]
                for cache in ca:cache.reset();assert cache.position==0
                row['reset_max_abs']=compare(gpu(y[:,:5],cache=ca)[0],gpu(y[:,:5],cache=caches(cfg))[0],0.,0.)
                assert all(torch.equal(state[n],v) for state,cache in zip(other_state,cb) for n,v in cache.state_dict().items())
                row['cache_bytes_per_item']=sum(v.numel()*v.element_size() for cache in cb for v in cache.buffers())//2
                assert row['cache_bytes_per_item']==196608
            torch.mps.synchronize();records.append(row)
            del cpu,gpu,a,b,la,lb,ca,cb,parts,other,other_state,full,full_y
            gc.collect();torch.mps.empty_cache()
        assert c.source_inventory()==frozen
        for n,h in previous['new_sources'].items():assert hashlib.sha256((c.ROOT/n).read_bytes()).hexdigest()==h
        result=dict(status='strict_mps_full_model_prerequisite_passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runner=reference(__file__),
            source_binding=reference(B/'model-integration001.json'),cases=records,
            tolerances=dict(logits_atol=3e-4,logits_rtol=3e-4,ce_atol=1e-4,ce_rtol=1e-5,gradient_atol=3e-5,gradient_rtol=3e-3,causality_and_reset='bitwise'),
            scope='Synthetic CPU/MPS logits, CE, all parameter gradients, repeated-token grouped embedding, full/chunk/interleaved/reset caches and causality for both arms. No optimizer/resume or quality claim.',
            corpus_read=False,holdout_read=False,checkpoint_io=False,scientific_training_launched=False,scientific_seeds_reserved=False)
    result['guard']=reference(B/'mps-integration001-guard.json');assert read(result['guard'])['status']=='finished'
    e.exclusive(B/'mps-integration001.json',result);print(json.dumps(dict(status=result['status'],cases=records)),flush=True)
