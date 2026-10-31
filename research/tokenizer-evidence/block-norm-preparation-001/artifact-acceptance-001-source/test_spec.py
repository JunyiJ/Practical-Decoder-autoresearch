"""Small synthetic CPU arithmetic and contract tests; no corpus or model weights.

Analytic oracle uses independent FP64 formulas/derivatives, never autograd through
native normalization. Tolerances are fixed before execution and retained with
source snapshots. Near-constant shifted inputs are separately conditioned.
"""
import copy,json
import pytest
import torch
from torch import nn
from research.block_norm_v1 import spec as s


def inputs(case):
    g=torch.Generator().manual_seed(371001)
    x=torch.randn((2,3,256),generator=g)
    if case=='shifted':x=x+3.
    elif case=='zero':x.zero_()
    elif case=='constant':x.fill_(1.)
    elif case=='near_zero':x=x*1e-5
    elif case=='near_constant':x=1.+x*1e-5
    return x


def oracle(kind,x,gamma,beta,upstream):
    z=x.double();g=gamma.double();u=upstream.double()
    if kind=='layer_matched_epsilon':z=z-z.mean(-1,keepdim=True)
    inv=torch.rsqrt((z*z).mean(-1,keepdim=True)+s.EPSILON)
    normalized=z*inv;y=normalized*g
    if beta is not None:y=y+beta.double()
    dy=u*g
    dx=(dy-z*inv.square()*(dy*z).mean(-1,keepdim=True))*inv
    if kind=='layer_matched_epsilon':dx=dx-dy.mean(-1,keepdim=True)*inv
    dg=(u*normalized).sum((0,1));db=u.sum((0,1)) if beta is not None else None
    return y,dx,dg,db


@pytest.mark.parametrize('kind',s.KINDS)
@pytest.mark.parametrize('case',['random','shifted','zero','constant','near_zero','near_constant'])
def test_forward_and_input_gain_bias_gradients_against_analytic_fp64(kind,case,record_property):
    spec=s.Spec(kind);module=s.make(spec)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(.5,1.5,256))
        if spec.bias:module.bias.copy_(torch.linspace(-.2,.2,256))
    x=inputs(case).requires_grad_();u=torch.randn(x.shape,generator=torch.Generator().manual_seed(371002))
    y=s.checked_forward(spec,module,x);(y*u).sum().backward()
    expected=oracle(kind,x.detach(),module.weight.detach(),module.bias.detach() if spec.bias else None,u)
    # A shifted near-constant FP32 mean loses precision before reciprocal
    # amplification (~2896). The absolute bound is explicit for that case only.
    sensitive=(kind=='layer_matched_epsilon' and case=='near_constant')
    atol_y,atol_dx,atol_g=(8e-4,.2,3e-3) if sensitive else (3e-5,3e-3,3e-4)
    for actual,wanted,atol in [(y,expected[0],atol_y),(x.grad,expected[1],atol_dx),
                              (module.weight.grad,expected[2],atol_g)]:
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.double(),wanted,rtol=2e-4,atol=atol)
    if spec.bias:torch.testing.assert_close(module.bias.grad.double(),expected[3],rtol=2e-5,atol=3e-5)
    for name,actual,wanted in [('output',y,expected[0]),('input_gradient',x.grad,expected[1]),('gain_gradient',module.weight.grad,expected[2])]:
        record_property(name+'_max_absolute_error',float((actual.detach().double()-wanted).abs().max()))


@pytest.mark.parametrize('case',['random','shifted','zero','constant','near_zero','near_constant'])
def test_control_is_bit_exact_incumbent_forward_and_backward(case):
    a=s.make(s.Spec('rms_incumbent'));b=nn.RMSNorm(256,eps=None,dtype=torch.float32)
    with torch.no_grad():a.weight.copy_(torch.linspace(.5,1.5,256));b.weight.copy_(a.weight)
    x=inputs(case).requires_grad_();z=x.detach().clone().requires_grad_()
    u=torch.randn(x.shape,generator=torch.Generator().manual_seed(371003))
    y=s.checked_forward(s.Spec('rms_incumbent'),a,x);v=b(z)
    (y*u).sum().backward();(v*u).sum().backward()
    assert torch.equal(y,v) and torch.equal(x.grad,z.grad) and torch.equal(a.weight.grad,b.weight.grad)


@pytest.mark.parametrize('kind',s.KINDS)
def test_eight_independent_initializers_preserve_rng_and_parameter_counts(kind):
    torch.manual_seed(371004);before=torch.get_rng_state().clone();spec=s.Spec(kind)
    norms=[s.make(spec) for _ in range(8)];assert torch.equal(before,torch.get_rng_state())
    s.validate_eight(spec,norms)
    assert sum(p.numel() for n in norms for p in n.parameters())==(4096 if spec.bias else 2048)
    for module in norms:
        assert torch.equal(module.weight,torch.ones(256))
        if spec.bias:assert torch.equal(module.bias,torch.zeros(256))
    # One following random draw matches the path with no normalizer construction.
    after=torch.rand(8);torch.set_rng_state(before);assert torch.equal(after,torch.rand(8))


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float64])
@pytest.mark.parametrize('kind',s.KINDS)
def test_non_fp32_input_or_parameters_rejected(kind,dtype):
    spec=s.Spec(kind);module=s.make(spec)
    with pytest.raises(ValueError):s.checked_forward(spec,module,torch.ones(2,256,dtype=dtype))
    module.to(dtype=dtype)
    with pytest.raises(ValueError):s.checked_forward(spec,module,torch.ones(2,256))


@pytest.mark.parametrize('kind',s.KINDS)
def test_default_dtype_change_does_not_change_normalizer_dtype(kind):
    original=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64);module=s.make(s.Spec(kind))
        assert all(p.dtype==torch.float32 for p in module.parameters())
    finally:torch.set_default_dtype(original)


@pytest.mark.parametrize('kind',s.KINDS)
def test_autocast_rejected(kind):
    spec=s.Spec(kind);module=s.make(spec)
    with torch.autocast('cpu',dtype=torch.bfloat16),pytest.raises(ValueError,match='Autocast'):
        s.checked_forward(spec,module,torch.ones(2,256))


@pytest.mark.parametrize('key,value',[('kind','layer_default'),('policy','old'),('width',128),('width',True),
    ('dtype','float64'),('epsilon',1e-5),('epsilon',None),('epsilon',True),('affine',False),('affine',1),('placement','final')])
def test_invalid_specs_rejected(key,value):
    fields=dict(kind='rms_incumbent');fields[key]=value
    with pytest.raises(ValueError):s.Spec(**fields)


@pytest.mark.parametrize('kind',s.KINDS)
@pytest.mark.parametrize('change',['epsilon','shape','affine','bias','frozen_gain','subclass','buffer'])
def test_constructed_module_mutations_rejected(kind,change):
    spec=s.Spec(kind);module=s.make(spec)
    if change=='epsilon':module.eps=1e-5
    elif change=='shape':module.normalized_shape=(128,)
    elif change=='affine':module.elementwise_affine=False
    elif change=='bias':
        if spec.bias:module.register_parameter('bias',None)
        else:module.register_parameter('bias',nn.Parameter(torch.zeros(256)))
    elif change=='frozen_gain':module.weight.requires_grad_(False)
    elif change=='subclass':
        class Custom(type(module)):pass
        module.__class__=Custom
    else:module.register_buffer('unknown',torch.zeros(1))
    with pytest.raises(ValueError):s.validate(spec,module)


@pytest.mark.parametrize('shape',[(256,),(2,128),(0,256),(2,0,256)])
def test_invalid_input_shapes_rejected(shape):
    spec=s.Spec('rms_incumbent')
    with pytest.raises(ValueError):s.checked_forward(spec,s.make(spec),torch.ones(shape))


@pytest.mark.parametrize('change',['count','mixed','shared_module','shared_parameter','aliased_storage'])
def test_mixed_or_shared_block_normalizers_rejected(change):
    spec=s.Spec('rms_incumbent');norms=[s.make(spec) for _ in range(8)]
    if change=='count':norms.pop()
    elif change=='mixed':norms[2]=s.make(s.Spec('layer_matched_epsilon'))
    elif change=='shared_module':norms[2]=norms[1]
    elif change=='shared_parameter':norms[2].weight=norms[1].weight
    else:norms[2].weight=nn.Parameter(norms[1].weight.detach())
    with pytest.raises(ValueError):s.validate_eight(spec,norms)


@pytest.mark.parametrize('kind',s.KINDS)
def test_signature_round_trip_and_same_shape_mismatch(kind):
    spec=s.Spec(kind);saved=json.loads(json.dumps(s.encode(spec)))
    assert s.decode(saved)==spec;s.require_same_signature(saved,spec)
    for key,value in [('epsilon',1e-5),('bias',not spec.bias),('dtype','float64'),('extra',True)]:
        bad=copy.deepcopy(saved);bad[key]=value
        with pytest.raises(ValueError):s.require_same_signature(bad,spec)
    other=s.Spec(s.KINDS[1-s.KINDS.index(kind)])
    with pytest.raises(ValueError,match='signature differs'):s.require_same_signature(saved,other)
