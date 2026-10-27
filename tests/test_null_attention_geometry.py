"""Deferred actual tensor/model gates. AST review only while predecessor runs."""
import copy
import math
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.models.gpt import GPT
from src.models.logit_z import POLICY, COEFFICIENTS, FIELDS, CONVENTION, training_logit_z, validate_logit_z_model
from src.train.optimizers import build_optimizer
from src.train.train import _evaluate
from src.research.coverage_eval import score_validation
from tests.test_position_family_contract import config as inherited_config
from tests.test_position_family_geometry import caches, chunked
from tests.test_attention_scale_geometry import rotate
from tests.test_gated_activation_geometry import mlp_reference
from tests.test_residual_gain_geometry import gain
from src.models.null_attention import POLICY as NULL_POLICY, FIELDS as NULL_FIELDS, KINDS, normalize_attention, validate_null_attention_model


def config(kind,position='rope',value='local',gain='unit',activation='silu',width=64,scale=.5,topology='serial',table='legacy_untied',z_kind='none'):
    c=inherited_config(position,value,gain,activation,width,scale,topology,table)
    c.update(output_bias_kind='learned_zero',logit_z_kind=z_kind,logit_z_coefficient=COEFFICIENTS[z_kind],logit_z_objective_policy=POLICY, null_attention_kind=kind,null_attention_policy=NULL_POLICY)
    return c


def reference(m,tokens,kind,gain_kind,activation,scale,topology,*,position="rope",start_pos=0,capture=None,split_first=False,null_kind="standard"):
    """Independent masked exp/sum attention and attached first-value graph.

    split_first replaces the four roles of V1 with independent leaves solely
    to audit the chain-rule sum. No production mixing/attention helper is used.
    """
    x=m.transformer.wte(tokens)*m.input_embedding_scale
    first=None;roles=[]
    for index,block in enumerate(m.transformer.h):
        original=x;normalized=block.ln1(original);b,t,d=normalized.shape;a=block.attn
        packed=F.linear(normalized,a.qkv.weight,a.qkv.bias)
        q,k,raw=[z.reshape(b,t,d//32,32).transpose(1,2) for z in packed.chunk(3,-1)]
        if index==0:
            first=raw
            if capture is not None:capture['first_raw']=raw
            if split_first:
                first=raw.detach().clone().requires_grad_(True);roles.append(first)
            value=first
        elif kind=='local':value=raw
        elif kind=='local_half':value=.5*raw
        else:
            reuse=first
            if split_first:
                reuse=first.detach().clone().requires_grad_(True);roles.append(reuse)
            value=.5*raw+.5*reuse
        if capture is not None:capture.setdefault('mixed_values',[]).append(value)
        if position=='rope':
            q=rotate(q,block.freq_cos[start_pos:],block.freq_sin[start_pos:])
            k=rotate(k,block.freq_cos[start_pos:],block.freq_sin[start_pos:])
        if capture is not None:capture.setdefault('keys',[]).append(k)
        scores=torch.einsum('bhid,bhjd->bhij',q,k)*(scale/math.sqrt(32))
        if position=='alibi':
            # Independent formula, not the production slope or bias helpers.
            slopes=torch.tensor([2.**(-8.*(h+1)/(d//32)) for h in range(d//32)],device=x.device,dtype=x.dtype)
            absolute=torch.arange(start_pos,start_pos+t,device=x.device,dtype=x.dtype)
            scores=scores-slopes[None,:,None,None]*(absolute[:,None]-absolute[None,:])
        mask=torch.arange(t,device=x.device)[:,None]>=torch.arange(t,device=x.device)[None,:]
        scores=scores.masked_fill(~mask,float('-inf'))
        # Independent exp/sum oracle; include the fixed null in the maximum.
        maximum=scores.max(-1,keepdim=True).values
        if null_kind=='null_zero':maximum=torch.maximum(maximum,torch.zeros_like(maximum))
        exponent=(scores-maximum).exp()
        denominator=exponent.sum(-1,keepdim=True)
        if null_kind=='null_zero':denominator=denominator+(-maximum).exp()
        probability=exponent/denominator
        if null_kind=='prefix_control':
            counts=mask.sum(-1,keepdim=True).to(scores.dtype)
            probability=probability*(counts/(counts+1))
        assert null_kind in ('standard','prefix_control','null_zero')
        attended=torch.einsum('bhij,bhjd->bhid',probability,value).transpose(1,2).reshape(b,t,d)
        attention=F.linear(attended,a.out.weight,a.out.bias)
        u=original+gain(block,attention,'attention',gain_kind)
        f=mlp_reference(block.mlp,block.ln2(original if topology=='parallel_separate_norms' else u),activation)
        x=u+gain(block,f,'ffn',gain_kind)
    if capture is not None:capture['first_roles']=roles
    h=m.transformer.ln_f(x)
    w=m.transformer.wte.weight if m.lm_head is None else m.lm_head.weight
    logits=F.linear(h,w)*m.table_readout_scale/m.width_mult
    return logits if m.output_bias is None else logits+m.output_bias


def independent_objective(logits,targets,coefficient):
    # FP64 max-shifted exp/sum oracle, not production logsumexp or CE helpers.
    z=logits.reshape(-1,logits.shape[-1]).double()
    y=targets.reshape(-1);valid=y!=-100;z=z[valid];y=y[valid]
    maximum=z.max(-1,keepdim=True).values
    exponent=(z-maximum).exp();prob=exponent/exponent.sum(-1,keepdim=True)
    logz=maximum.squeeze(-1)+exponent.sum(-1).log()
    ce=(logz-z.gather(1,y[:,None]).squeeze(-1)).mean()
    squared=logz.square().mean()
    return ce+coefficient*squared,ce,logz,prob


@pytest.mark.parametrize('width',[64,256])
@pytest.mark.parametrize('topology',['serial','parallel_separate_norms'])
@pytest.mark.parametrize('table',['legacy_untied','matched_untied','scaled_tied'])
def test_actual_null_attention_graph_rng_groups_and_cache(width,topology,table):
    torch.set_num_threads(1)
    x=torch.arange(16).reshape(2,8)%4;y=(x+1)%4;y[0,2]=-100;y[1,7]=-100
    routes=[('rope','local','unit','silu',.5),('nope','local_half','static_tenth','gelu_exact',1.),('alibi','first_blend','learned_tenth','relu',2.)]
    for route_index,(position,value,gain_kind,activation,scale) in enumerate(routes):
        z_kind=('none','z1e4','z1e3')[route_index]
        models=[];rngs=[];groups=[]
        for kind in KINDS:
            torch.manual_seed(9481)
            m=GPT(OmegaConf.create(config(kind,position,value,gain_kind,activation,width,scale,topology,table,z_kind)))
            validate_logit_z_model(m);validate_null_attention_model(m);models.append(m);rngs.append(torch.get_rng_state().clone())
            opt=build_optimizer(m,NS(optimizer='muon',decay_mode='independent',learning_rate=.01,weight_decay=.00006))
            groups.append([(g['parameter_names'],g['lr'],g['width_lr_scale']) for g in opt.param_groups])
            ids=[id(p) for g in opt.param_groups for p in g['params']]
            assert len(ids)==len(set(ids))==len(list(m.parameters()))
            assert not any('null_attention' in n for n in m.state_dict())
        assert groups[0]==groups[1]==groups[2]
        d=width;h=21*d//8
        expected_count=8*d+4*(4*d*d+3*d*h+6*d)+2*d+4-(4*d if table=='scaled_tied' else 0)+(8*d if gain_kind=='learned_tenth' else 0)
        assert all(sum(p.numel() for p in m.parameters())==expected_count for m in models)
        assert all(torch.equal(rngs[0],r) for r in rngs)
        for m in models[1:]:
            assert m.state_dict().keys()==models[0].state_dict().keys()
            assert all(torch.equal(v,m.state_dict()[n]) for n,v in models[0].state_dict().items())
        old=copy.deepcopy(models[0].cfg)
        for name in NULL_FIELDS:del old[name]
        torch.manual_seed(9481);historical=GPT(old)
        assert torch.equal(torch.get_rng_state(),rngs[0])
        assert sum(p.numel() for p in historical.parameters())==sum(p.numel() for p in models[0].parameters())
        historical_logits,historical_loss=historical(x,y,include_training_objective=True)
        zero_logits,zero_loss=models[0](x,y,include_training_objective=True)
        assert torch.equal(zero_logits,historical_logits) and torch.equal(zero_loss,historical_loss)
        zero_loss.backward();historical_loss.backward()
        assert all(torch.equal(p.grad,dict(historical.named_parameters())[n].grad) for n,p in models[0].named_parameters())
        for kind,m in zip(KINDS,models):
            # Make bias and signed gain paths visible in the complete graph.
            with torch.no_grad():
                for i,b in enumerate(m.transformer.h):
                    b.attn.qkv.bias.copy_(torch.linspace(-.03,.04,3*width)+.01*i)
                    b.attn.out.bias.copy_(torch.linspace(-.02,.03,width))
                    if gain_kind=='learned_tenth':
                        b.attention_residual_gain.copy_(torch.linspace(-.2,.4,width))
                        b.ffn_residual_gain.copy_(torch.linspace(.3,-.1,width))
                if m.output_bias is not None:m.output_bias.copy_(torch.linspace(-.4,.5,4))
            oracle=copy.deepcopy(m);oracle.zero_grad(set_to_none=True);m.zero_grad(set_to_none=True)
            actual,total,breakdown=m(x,y,include_training_objective=True,return_loss_breakdown=True)
            expected_logits=reference(oracle,x,value,gain_kind,activation,scale,topology,position=position,null_kind=kind)
            expected,ce,logz,_=independent_objective(expected_logits,y,COEFFICIENTS[z_kind])
            assert actual.dtype==torch.float32
            assert torch.allclose(actual,expected_logits,atol=1e-5,rtol=2e-5)
            assert total.item()==pytest.approx(expected.item(),abs=2e-5,rel=2e-5)
            assert breakdown['ce_loss'].item()==pytest.approx(ce.item(),abs=2e-5,rel=2e-5)
            assert breakdown['aux_loss'] is None
            if z_kind=='none':
                assert total is breakdown['ce_loss']
                assert all(breakdown[k] is None for k in ('logit_z_mean','logit_z_squared_mean','logit_z_penalty'))
            else:
                assert breakdown['logit_z_mean'].item()==pytest.approx(logz.mean().item(),abs=2e-5)
                assert breakdown['logit_z_penalty'].item()==pytest.approx(COEFFICIENTS[z_kind]*logz.square().mean().item(),abs=2e-6)
            total.backward();expected.backward();op=dict(oracle.named_parameters())
            for name,p in m.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(),name
                assert torch.allclose(p.grad,op[name].grad,atol=2e-5,rtol=3e-4),name
            # Explicit objective opt-in, not training mode, determines the loss.
            for train_mode in (True,False):
                m.train(train_mode)
                _,pure=m(x,y);_,enabled=m(x,y,include_training_objective=True)
                assert torch.equal(pure,F.cross_entropy(m(x)[0].reshape(-1,4),y.reshape(-1)))
                assert enabled.item()==pytest.approx(total.item(),abs=1e-7)
            with torch.no_grad():
                full=m(x)[0];cache=caches(m)
                assert torch.allclose(chunked(m,x,cache,[3,1,4]),full,atol=1e-4,rtol=2e-5)
                for c in cache:c.reset()
                assert torch.allclose(chunked(m,x,cache,[1]*8),full,atol=1e-4,rtol=2e-5)
                captured={};reference(m,x,value,gain_kind,activation,scale,topology,position=position,null_kind=kind,capture=captured)
                for c,k,v in zip(cache,captured['keys'],captured['mixed_values']):
                    assert c.k_cache.shape==c.v_cache.shape==(2,width//32,128,32)
                    assert torch.allclose(c.k_cache[:,:,:8],k,atol=1e-5,rtol=2e-5)
                    assert torch.allclose(c.v_cache[:,:,:8],v,atol=1e-5,rtol=2e-5)
                changed=x.clone();changed[:,4:]=(changed[:,4:]+1)%4
                assert torch.allclose(full[:,:4],m(changed)[0][:,:4],atol=1e-6,rtol=1e-5)
                other=(x+1)%4;ca,cb=caches(m),caches(m);aa=[];bb=[];pos=0
                for n in (2,1,5):
                    aa.append(m(x[:,pos:pos+n],cache=ca,start_pos=pos)[0])
                    bb.append(m(other[:,pos:pos+n],cache=cb,start_pos=pos)[0]);pos+=n
                assert torch.allclose(torch.cat(aa,1),full,atol=1e-4,rtol=2e-5)
                assert torch.allclose(torch.cat(bb,1),m(other)[0],atol=1e-4,rtol=2e-5)
                for c in ca:c.reset()
                assert torch.allclose(chunked(m,other,ca,[5,3]),m(other)[0],atol=1e-4,rtol=2e-5)
                assert torch.allclose(m(x,start_pos=120)[0],reference(m,x,value,gain_kind,activation,scale,topology,position=position,start_pos=120,null_kind=kind),atol=1e-5,rtol=2e-5)



def test_actual_null_primitive_mask_extremes_and_analytic_vjp():
    torch.set_num_threads(1)
    for start,length in ((0,3),(4,2),(120,3)):
        keys=start+length
        mask=torch.arange(keys)[None,:]<=torch.arange(start,start+length)[:,None]
        for offset in (0.,10000.,-10000.):
            raw=torch.linspace(-2.,3.,2*2*length*keys).reshape(2,2,length,keys)+offset
            for kind in KINDS:
                scores=raw.clone().requires_grad_();masked=scores.masked_fill(~mask,float('-inf'))
                values=torch.linspace(-1.,2.,2*2*keys*5).reshape(2,2,keys,5).requires_grad_()
                rng=torch.get_rng_state().clone()
                probability=normalize_attention(masked,mask,kind);out=probability@values
                assert torch.equal(rng,torch.get_rng_state())
                z=masked.detach().double();maximum=z.max(-1,keepdim=True).values
                if kind=='null_zero':maximum=torch.maximum(maximum,torch.zeros_like(maximum))
                exponent=(z-maximum).exp();denominator=exponent.sum(-1,keepdim=True)
                if kind=='null_zero':denominator=denominator+(-maximum).exp()
                expected=exponent/denominator
                factor=mask.sum(-1,keepdim=True).double();factor=factor/(factor+1)
                if kind=='prefix_control':expected=expected*factor
                assert torch.allclose(probability.double(),expected,atol=1e-6,rtol=2e-5)
                assert torch.count_nonzero(probability.masked_select(~mask))==0
                upstream=torch.linspace(-.3,.7,out.numel()).reshape_as(out)
                ds,dv=torch.autograd.grad((out*upstream).sum(),(scores,values))
                ey=expected@values.detach().double()
                center=ey/factor if kind=='prefix_control' else ey
                expected_ds=expected*torch.einsum('bhid,bhijd->bhij',upstream.double(),values.detach().double()[:,:,None,:,:]-center[:,:,:,None,:])
                expected_dv=expected.transpose(-1,-2)@upstream.double()
                assert torch.allclose(ds.double(),expected_ds,atol=2e-6,rtol=3e-4)
                assert torch.allclose(dv.double(),expected_dv,atol=2e-6,rtol=2e-5)
                assert torch.isfinite(ds).all() and torch.isfinite(out).all()
                if kind=='standard':assert torch.equal(probability,torch.softmax(masked,-1))
    valid=torch.ones(2,3,dtype=torch.bool);scores=torch.zeros(1,1,2,3)
    for bad,visibility in ((scores.double(),valid),(scores,valid.float()),(scores,valid[:1]),(scores,torch.zeros_like(valid))):
        for kind in ('prefix_control','null_zero'):
            with pytest.raises(ValueError):normalize_attention(bad,visibility,kind)


def test_actual_null_zero_score_control_shift_and_output_bias():
    mask=torch.arange(7)[None,:]<=torch.arange(4,7)[:,None]
    zero=torch.zeros(2,2,3,7).masked_fill(~mask,float('-inf'))
    p=normalize_attention(zero,mask,'prefix_control');n=normalize_attention(zero,mask,'null_zero')
    expected=mask.double()/(mask.sum(-1,keepdim=True).double()+1)
    assert torch.allclose(p.double(),expected,atol=1e-7,rtol=1e-6)
    assert torch.allclose(n.double(),expected,atol=1e-7,rtol=1e-6)
    scores=(torch.arange(7,dtype=torch.float32)*.3).expand(2,2,3,7).masked_fill(~mask,float('-inf'))
    for kind in KINDS:
        before=normalize_attention(scores,mask,kind);after=normalize_attention(scores+3.,mask,kind)
        if kind=='null_zero':assert torch.all(after.sum(-1)>before.sum(-1))
        else:assert torch.allclose(before,after,atol=1e-7,rtol=2e-6)
    # Even vanishing weighted real values leave the output projection bias.
    for kind in KINDS:
        torch.manual_seed(9501);m=GPT(OmegaConf.create(config(kind)))
        a=m.transformer.h[0].attn
        with torch.no_grad():
            a.qkv.weight.zero_();a.qkv.bias.zero_();a.out.bias.fill_(.25)
            output,_=a(torch.zeros(2,3,64),freq_cos=m.transformer.h[0].freq_cos,freq_sin=m.transformer.h[0].freq_sin)
        assert torch.equal(output,torch.full_like(output,.25))


def test_actual_null_attached_first_value_gradient_sum():
    torch.set_num_threads(1);x=torch.arange(16).reshape(2,8)%4;y=(x+1)%4
    for kind in KINDS:
        torch.manual_seed(9503);m=GPT(OmegaConf.create(config(kind,value='first_blend',z_kind='z1e3')))
        packed=[];hook=m.transformer.h[0].attn.qkv.register_forward_hook(lambda mod,args,out:packed.append(out))
        _,loss=m(x,y,include_training_objective=True);hook.remove()
        actual=torch.autograd.grad(loss,packed[0])[0].chunk(3,-1)[2].reshape(2,8,2,32).transpose(1,2)
        attached={};expected=reference(copy.deepcopy(m),x,'first_blend','unit','silu',.5,'serial',capture=attached,null_kind=kind)
        expected_loss=independent_objective(expected,y,.001)[0]
        attached_grad,=torch.autograd.grad(expected_loss,attached['first_raw'])
        split={};split_logits=reference(copy.deepcopy(m),x,'first_blend','unit','silu',.5,'serial',capture=split,split_first=True,null_kind=kind)
        grads=torch.autograd.grad(independent_objective(split_logits,y,.001)[0],split['first_roles'])
        assert len(grads)==4 and all(torch.count_nonzero(g)>0 for g in grads)
        assert torch.allclose(actual,attached_grad,atol=2e-5,rtol=3e-4)
        assert torch.allclose(attached_grad,sum(grads),atol=2e-5,rtol=3e-4)
        assert not torch.allclose(attached_grad,grads[0],atol=1e-8,rtol=1e-5)

def test_actual_null_sampled_and_full_validation_are_pure_ce():
    torch.set_num_threads(1);x=torch.arange(16).reshape(2,8)%4;y=(x+1)%4
    class Loader:
        eval_seed=2026
        def __init__(self):self.generators={'val':torch.Generator().manual_seed(99)}
        def get_batch(self,split):assert split=='val';return x,y
    for kind in KINDS:
        torch.manual_seed(9505);m=GPT(OmegaConf.create(config(kind,z_kind="z1e3")))
        with torch.no_grad():expected=F.cross_entropy(m(x)[0].reshape(-1,4),y.reshape(-1)).item()
        # Any accidental penalty invocation fails, even for the zero arm.
        with patch('src.models.logit_z.training_logit_z',side_effect=AssertionError('Evaluation invoked training penalty')):
            result,aux=_evaluate(m,Loader(),'val',2)
            assert result==expected and aux is None
            data=torch.arange(22)%4
            result=score_validation(m,data,context=8,batch_size=2,block_tokens=16)
        sums=[]
        with torch.no_grad():
            for start in (0,8,16):
                a=data[start:min(start+8,21)];b=data[start+1:min(start+9,22)]
                logits=m(a[None])[0]
                sums.extend(F.cross_entropy(logits.reshape(-1,4),b,reduction='none').double().tolist())
        assert result['target_count']==21 and result['block_target_counts']==[16,5]
        assert result['mean_ce']==pytest.approx(sum(sums)/21,abs=2e-7)
        assert result['block_loss_sums']==pytest.approx([sum(sums[:16]),sum(sums[16:])],abs=3e-6)
