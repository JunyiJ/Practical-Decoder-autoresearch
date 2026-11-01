"""Synthetic arithmetic prerequisite only; no full-model policy or training dispatch."""
import copy, datetime, hashlib, json, math, shutil
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.ffn_quality_v1 import campaign_contract as c
from research.ffn_quality_v1.support import idle
from research.ffn_structure_v1 import pilot_contract as pilot
from research.ffn_structure_v1.runtime.models.gpt import GPT
from research.ffn_structure_v1.runtime.attention.cache import KVCache
from research.ffn_structure_v1.runtime.models.block_norm import contract
from research.ffn_structure_v1.runtime.models.rope import precompute_freqs_cis
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock

B=Path(__file__).resolve().parent
ROOT=c.ROOT

def ref(p):
    p=Path(p);return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def oracle(m,x,cos,sin):
    # Separate content and rotary contractions, complex rotation, explicit softmax.
    n,t,_=x.shape
    linear=lambda x,layer: x @ layer.weight.T + (layer.bias if layer.bias is not None else 0)
    cq=linear(x,m.dq);ckv=linear(x,m.dkv)
    q=(linear(cq,m.uq)).reshape(n,t,8,32)
    k=(linear(ckv,m.uk)).reshape(n,t,8,32)
    v=(linear(ckv,m.uv)).reshape(n,t,8,32)
    qr=(linear(cq,m.qr)).reshape(n,t,8,16,2).contiguous()
    kr=(linear(x,m.kr)).reshape(n,t,1,16,2).contiguous()
    phase=torch.complex(cos[:t],sin[:t])[None,:,None,:]
    qr=torch.view_as_real(torch.view_as_complex(qr)*phase).flatten(-2)
    kr=torch.view_as_real(torch.view_as_complex(kr)*phase).flatten(-2)
    score=(torch.einsum('bthd,bshd->bhts',q,k)+torch.einsum('bthd,bsjd->bhts',qr,kr))*(m.attention_logit_scale/8.)
    mask=torch.arange(t,device=x.device)[:,None]>=torch.arange(t,device=x.device)[None,:]
    score=score.masked_fill(~mask,-torch.inf)
    exponential=torch.exp(score-score.amax(-1,keepdim=True))
    probability=exponential/exponential.sum(-1,keepdim=True)
    return linear(torch.einsum('bhts,bshd->bthd',probability,v).reshape(n,t,256),m.out)

def compare(a,b,rtol,atol):
    a=a.detach().cpu();b=b.detach().cpu()
    torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
    return float((a-b).abs().max())

def gradients(m,x,cos,sin,fn):
    m.zero_grad(set_to_none=True);x=x.detach().clone().requires_grad_(True)
    y=fn(m,x,cos,sin)
    probe=torch.linspace(-1,1,y.numel(),device=x.device,dtype=x.dtype).reshape(y.shape)
    (y*probe).sum().backward()
    return y.detach(),[x.grad.detach(),*[p.grad.detach().clone() for p in m.parameters()]]

def forward(m,x,cos,sin):return m(x,freq_cos=cos,freq_sin=sin)[0]

torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    idle();assert not (B/'component001.json').exists()
    sources=c.source_inventory()
    frozen=json.loads((pilot.EVIDENCE/'quality001-plan.json').read_text())['sources']
    assert sources==frozen
    with StrictMpsGuard(B/'component001-guard.json',ROOT):
        plan=json.loads((pilot.EVIDENCE/'pilot001-plan.json').read_text())
        cfg=OmegaConf.create(pilot.config(0,plan['seeds']))
        assert cfg.model.attention_logit_scale==.5
        torch.manual_seed(617309)
        full_model=GPT(cfg.model)
        base=copy.deepcopy(full_model.transformer.h[0].attn)
        del full_model
        assert base.head_dim==base.head_dim_r==32 and base.dimc_kv==base.dimc_q==64
        generator=torch.Generator().manual_seed(617309)
        x=torch.randn(2,16,256,generator=generator)
        cos,sin=precompute_freqs_cis(32,128)
        rows=[]
        for scale in (.5,1.):
            # Research-only scalar override on a private component copy, NOT a
            # validated full model or checkpoint. Frozen constructor is unchanged.
            m=copy.deepcopy(base).double();m.attention_logit_scale=scale
            o=copy.deepcopy(m)
            y,grad=gradients(m,x.double(),cos.double(),sin.double(),forward)
            expected,expected_grad=gradients(o,x.double(),cos.double(),sin.double(),oracle)
            output_error=compare(y,expected,1e-10,1e-10)
            gradient_error=max(compare(a,b,1e-9,1e-10) for a,b in zip(grad,expected_grad))
            cpu=copy.deepcopy(base);cpu.attention_logit_scale=scale
            yc,gc=gradients(cpu,x,cos,sin,forward)
            gpu=copy.deepcopy(cpu).to('mps')
            ym,gm=gradients(gpu,x.to('mps'),cos.to('mps'),sin.to('mps'),forward)
            mps_output_error=compare(ym,yc,3e-4,3e-5)
            mps_gradient_error=max(compare(a,b,3e-4,3e-5) for a,b in zip(gm,gc))
            cache_rows=[]
            for device,model in [('cpu',cpu),('mps',gpu)]:
                xx=x.to(device);cc=cos.to(device);ss=sin.to(device)
                with torch.no_grad():
                    full=forward(model,xx,cc,ss)
                    cache=KVCache(cfg,device,batch_size=2,dtype=torch.float32,layer_index=0)
                    chunks=[];start=0
                    for size in (3,1,5,7):
                        chunks.append(model(xx[:,start:start+size],cache=cache,start_pos=start,freq_cos=cc,freq_sin=ss)[0]);start+=size
                    error=compare(torch.cat(chunks,1),full,3e-4,3e-5)
                    cache.reset();reset=compare(model(xx,cache=cache,start_pos=0,freq_cos=cc,freq_sin=ss)[0],full,3e-4,3e-5)
                    changed=xx.clone();changed[:,8:]+=7
                    causal=compare(forward(model,changed,cc,ss)[:,:8],full[:,:8],0,0)
                    cache_rows.append(dict(device=device,uneven_cache_max_abs=error,reset_max_abs=reset,causality_max_abs=causal,cache_bytes=sum(t.numel()*t.element_size() for t in cache.buffers())))
            rows.append(dict(multiplier=scale,oracle_output_max_abs=output_error,oracle_all_gradients_max_abs=gradient_error,mps_output_max_abs=mps_output_error,mps_all_gradients_max_abs=mps_gradient_error,caches=cache_rows))
        changed=copy.deepcopy(cfg.model);changed.attention_logit_scale=1.
        try:contract(changed)
        except ValueError as exc:rejection=str(exc)
        else:raise AssertionError('Frozen full-model policy unexpectedly accepted new scale')
        assert c.source_inventory()==sources
        result=dict(status='component_arithmetic_passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runner=ref(__file__),frozen_sources=sources,
            rows=rows,component_parameters=sum(p.numel() for p in base.parameters()),full_model_policy_rejection=rejection,
            synthetic_seed=617309,scientific_seeds_reserved=False,corpus_read=False,checkpoint_io=False,quality_training=False,
            scope='Private scalar override in inherited MLA component only. Independent complex-RoPE/separate-score oracle and all input/parameter gradients, CPU/MPS full/cache/offset/reset/causality. Full-model, optimizer, checkpoint/resume, worker and inference acceptance remain prerequisites.',
            observed_free_bytes=shutil.disk_usage(ROOT).free)
    result['guard']=ref(B/'component001-guard.json')
    assert json.loads((B/'component001-guard.json').read_text())['status']=='finished'
    with (B/'component001.json').open('x') as f:json.dump(result,f,sort_keys=True);f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','rows','component_parameters','full_model_policy_rejection','observed_free_bytes')}))
