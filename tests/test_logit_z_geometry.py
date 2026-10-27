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
from tests.test_position_family_geometry import reference, caches, chunked


def config(kind,position='rope',value='local',gain='unit',activation='silu',width=64,scale=.5,topology='serial',table='legacy_untied'):
    c=inherited_config(position,value,gain,activation,width,scale,topology,table)
    c.update(output_bias_kind='learned_zero',logit_z_kind=kind,logit_z_coefficient=COEFFICIENTS[kind],logit_z_objective_policy=POLICY)
    return c


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
def test_actual_logit_z_graph_rng_groups_and_cache(width,topology,table):
    torch.set_num_threads(1)
    x=torch.arange(16).reshape(2,8)%4;y=(x+1)%4;y[0,2]=-100;y[1,7]=-100
    routes=[('rope','local','unit','silu',.5),('nope','local_half','static_tenth','gelu_exact',1.),('alibi','first_blend','learned_tenth','relu',2.)]
    for position,value,gain,activation,scale in routes:
        models=[];rngs=[];groups=[]
        for kind in COEFFICIENTS:
            torch.manual_seed(9481)
            m=GPT(OmegaConf.create(config(kind,position,value,gain,activation,width,scale,topology,table)))
            validate_logit_z_model(m);models.append(m);rngs.append(torch.get_rng_state().clone())
            opt=build_optimizer(m,NS(optimizer='muon',decay_mode='independent',learning_rate=.01,weight_decay=.00006))
            groups.append([(g['parameter_names'],g['lr'],g['width_lr_scale']) for g in opt.param_groups])
            ids=[id(p) for g in opt.param_groups for p in g['params']]
            assert len(ids)==len(set(ids))==len(list(m.parameters()))
            assert not any('logit_z' in n for n in m.state_dict())
        assert groups[0]==groups[1]==groups[2]
        assert all(torch.equal(rngs[0],r) for r in rngs)
        for m in models[1:]:
            assert m.state_dict().keys()==models[0].state_dict().keys()
            assert all(torch.equal(v,m.state_dict()[n]) for n,v in models[0].state_dict().items())
        old=copy.deepcopy(models[0].cfg)
        for name in FIELDS:del old[name]
        torch.manual_seed(9481);historical=GPT(old)
        assert torch.equal(torch.get_rng_state(),rngs[0])
        assert sum(p.numel() for p in historical.parameters())==sum(p.numel() for p in models[0].parameters())
        historical_logits,historical_loss=historical(x,y)
        zero_logits,zero_loss=models[0](x,y,include_training_objective=True)
        assert torch.equal(zero_logits,historical_logits) and torch.equal(zero_loss,historical_loss)
        zero_loss.backward();historical_loss.backward()
        assert all(torch.equal(p.grad,dict(historical.named_parameters())[n].grad) for n,p in models[0].named_parameters())
        for kind,m in zip(COEFFICIENTS,models):
            # Make bias and signed gain paths visible in the complete graph.
            with torch.no_grad():
                for i,b in enumerate(m.transformer.h):
                    b.attn.qkv.bias.copy_(torch.linspace(-.03,.04,3*width)+.01*i)
                    b.attn.out.bias.copy_(torch.linspace(-.02,.03,width))
                    if gain=='learned_tenth':
                        b.attention_residual_gain.copy_(torch.linspace(-.2,.4,width))
                        b.ffn_residual_gain.copy_(torch.linspace(.3,-.1,width))
                if m.output_bias is not None:m.output_bias.copy_(torch.linspace(-.4,.5,4))
            oracle=copy.deepcopy(m);m.zero_grad(set_to_none=True)
            actual,total,breakdown=m(x,y,include_training_objective=True,return_loss_breakdown=True)
            expected_logits=reference(oracle,x,value,gain,activation,scale,topology,position=position)
            expected,ce,logz,_=independent_objective(expected_logits,y,COEFFICIENTS[kind])
            assert actual.dtype==torch.float32
            assert torch.allclose(actual,expected_logits,atol=1e-5,rtol=2e-5)
            assert total.item()==pytest.approx(expected.item(),abs=2e-5,rel=2e-5)
            assert breakdown['ce_loss'].item()==pytest.approx(ce.item(),abs=2e-5,rel=2e-5)
            assert breakdown['aux_loss'] is None
            if kind=='none':
                assert total is breakdown['ce_loss']
                assert all(breakdown[k] is None for k in ('logit_z_mean','logit_z_squared_mean','logit_z_penalty'))
            else:
                assert breakdown['logit_z_mean'].item()==pytest.approx(logz.mean().item(),abs=2e-5)
                assert breakdown['logit_z_penalty'].item()==pytest.approx(COEFFICIENTS[kind]*logz.square().mean().item(),abs=2e-6)
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
                other=(x+1)%4;ca,cb=caches(m),caches(m);aa=[];bb=[];pos=0
                for n in (2,1,5):
                    aa.append(m(x[:,pos:pos+n],cache=ca,start_pos=pos)[0])
                    bb.append(m(other[:,pos:pos+n],cache=cb,start_pos=pos)[0]);pos+=n
                assert torch.allclose(torch.cat(aa,1),full,atol=1e-4,rtol=2e-5)
                assert torch.allclose(torch.cat(bb,1),m(other)[0],atol=1e-4,rtol=2e-5)
                assert torch.allclose(m(x,start_pos=120)[0],reference(m,x,value,gain,activation,scale,topology,position=position,start_pos=120),atol=1e-5,rtol=2e-5)


def test_actual_logit_z_mask_extremes_analytic_gradient_and_zero_identity():
    for values in ([[0.,1.,-2.,3.],[-3.,4.,2.,-1.],[1.,1.,1.,1.]],
                   [[10000.,9999.,-10000.,0.],[-10000.,-9999.,-9998.,-10001.],[10000.]*4]):
        for kind,coef in COEFFICIENTS.items():
            z=torch.tensor(values,dtype=torch.float32,requires_grad=True);y=torch.tensor([2,-100,1])
            ce=F.cross_entropy(z,y);sig=(kind,coef,POLICY,CONVENTION)
            loss,metrics=training_logit_z(z,y,ce,sig)
            expected,_,logz,prob=independent_objective(z,y,coef)
            gradient=torch.zeros_like(z,dtype=torch.float64)
            onehot=F.one_hot(y[y!=-100],4).double()
            gradient[y!=-100]=(prob-onehot+2*coef*logz[:,None]*prob)/2
            loss.backward()
            assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
            assert torch.allclose(z.grad.double(),gradient,atol=2e-3,rtol=2e-5)
            assert torch.count_nonzero(z.grad[1])==0
            assert loss.item()==pytest.approx(expected.item(),rel=2e-6,abs=2e-3)
            if kind=='none':assert loss is ce and metrics['logit_z_penalty'] is None
            else:
                assert metrics['logit_z_penalty'].item()==pytest.approx(coef*logz.square().mean().item(),rel=2e-6)
        # Zero path must neither inspect logits nor perform extra tensor work.
        ce=torch.tensor(2.,requires_grad=True)
        loss,_=training_logit_z(object(),object(),ce,('none',0.,POLICY,CONVENTION))
        assert loss is ce
    sig=('z1e4',.0001,POLICY,CONVENTION)
    for bad,y in [(torch.zeros(2,4,dtype=torch.float64),torch.zeros(2,dtype=torch.long)),
                  (torch.zeros(2,4),torch.zeros(3,dtype=torch.long)),
                  (torch.zeros(2,4),torch.full((2,),-100))]:
        with pytest.raises(ValueError):training_logit_z(bad,y,torch.tensor(0.),sig)


def test_actual_logit_z_common_shift_coordinate_and_coefficient_once():
    base=torch.tensor([[.1,1.2,-.4,2.],[-1.,.3,.5,.8]],dtype=torch.float32)
    y=torch.tensor([0,3]);penalties={}
    for kind in ('z1e4','z1e3'):
        coef=COEFFICIENTS[kind]
        for shift in (0.,3.):
            z=(base+shift).requires_grad_();ce=F.cross_entropy(z,y)
            loss,meta=training_logit_z(z,y,ce,(kind,coef,POLICY,CONVENTION))
            loss.backward();logz=torch.logsumexp(z.double(),-1)
            assert torch.allclose(z.grad.sum(-1).double(),2*coef*logz/2,atol=2e-7,rtol=2e-5)
            penalties[kind,shift]=meta['logit_z_penalty'].item()
            assert ce.item()==pytest.approx(F.cross_entropy(base,y).item(),abs=3e-7)
        assert penalties[kind,3.]>penalties[kind,0.]
    for shift in (0.,3.):assert penalties['z1e3',shift]==pytest.approx(10*penalties['z1e4',shift],rel=2e-6)


def test_actual_sampled_and_full_validation_are_pure_ce():
    torch.set_num_threads(1);x=torch.arange(16).reshape(2,8)%4;y=(x+1)%4
    class Loader:
        eval_seed=2026
        def __init__(self):self.generators={'val':torch.Generator().manual_seed(99)}
        def get_batch(self,split):assert split=='val';return x,y
    for kind in COEFFICIENTS:
        torch.manual_seed(9483);m=GPT(OmegaConf.create(config(kind)))
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
