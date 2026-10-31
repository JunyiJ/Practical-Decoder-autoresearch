"""Prospective standalone formula, repeatability and contract checks."""
import gc
import hashlib
import json
import os
import time
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.research.protocol import ROOT
from research.block_norm_v1.runtime.train.train import _make_loader
from research.block_norm_v1.grouped_embedding_prototype import grouped_embedding


def sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def test_formula_native_forward_and_repeated_backward(record_property):
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    cfg=OmegaConf.create(json.loads((ROOT/'research_state/block-norm-trainer-fixture-002/rms_incumbent/config.json').read_text()))
    cfg.training.seed=372903
    one,_=_make_loader(cfg,'cpu').get_batch('train')
    cfg.training.batch_size=8;cfg.training.batch_layout='paired_blocks512'
    eight,_=_make_loader(cfg,'cpu').get_batch('train')
    cases=[('generated1',one,'random'),('generated8paired',eight,'random'),
        ('unique1',torch.arange(128).reshape(1,128),'random'),
        ('unique8',torch.arange(1024).reshape(8,128),'random'),
        ('same1',torch.full((1,128),17),'random'),('same8',torch.full((8,128),17),'random'),
        ('alternating8',(torch.arange(1024)%2).reshape(8,128),'random'),
        ('zero8',eight,'zero'),('constant_same8',torch.full((8,128),17),'constant'),
        ('noncontiguous_ids',eight.T,'random'),('noncontiguous_upstream',eight,'noncontiguous')]
    generator=torch.Generator().manual_seed(372901)
    weight=torch.nn.Parameter((torch.randn(50257,256,generator=generator)*.01).to('mps'))
    initial=sha(weight);summaries=[]
    with Path(os.environ['BLOCK_NORM_TRACE_PATH']).open('x') as stream:
        def emit(event,**fields):
            stream.write(json.dumps(dict(event=event,**fields),sort_keys=True)+'\n');stream.flush()
            assert stream.tell()<2*1024**2
        try:
            for name,ids,kind in cases:
                upstream=torch.randn(*ids.shape,256,generator=generator)*.001
                if kind=='zero':upstream.zero_()
                elif kind=='constant':upstream.fill_(.125)
                elif kind=='noncontiguous':upstream=upstream.transpose(1,2).contiguous().transpose(1,2)
                active=torch.unique(ids);rows={int(v):i for i,v in enumerate(active)}
                oracle=torch.zeros(len(active),256,dtype=torch.float64)
                for index,value in zip(ids.reshape(-1),upstream.reshape(-1,256)):
                    oracle[rows[int(index)]]+=value.double()
                tokens=ids.to('mps');grad=upstream.to('mps')
                with torch.no_grad():native=F.embedding(tokens,weight)
                case_hashes=[]
                for deterministic in (False,True):
                    torch.use_deterministic_algorithms(deterministic,warn_only=False)
                    hashes=[];times=[];max_error=0.
                    for repeat in range(4):
                        weight.grad=None;torch.mps.synchronize();start=time.perf_counter()
                        output=grouped_embedding(weight,tokens)
                        forward_exact=bool(torch.equal(native,output))
                        output.backward(grad);torch.mps.synchronize();times.append(time.perf_counter()-start)
                        actual=weight.grad.detach().cpu();values=actual[active]
                        h=sha(actual);hashes.append(h);case_hashes.append(h)
                        error=float((values.double()-oracle).abs().max());max_error=max(max_error,error)
                        finite=bool(torch.isfinite(actual).all())
                        unused_zero=int(torch.count_nonzero(actual))==int(torch.count_nonzero(values))
                        emit('backward',case=name,deterministic=deterministic,repeat=repeat,
                            gradient_sha256=h,forward_exact=forward_exact,finite=finite,unused_rows_zero=unused_zero,
                            max_abs_vs_fp64=error,elapsed_seconds=times[-1],indices_contiguous=tokens.is_contiguous(),
                            upstream_contiguous=grad.is_contiguous())
                        assert forward_exact and finite and unused_zero
                        assert weight.grad.device.type=='mps' and weight.grad.dtype==torch.float32
                        torch.testing.assert_close(values.double(),oracle,atol=1e-7,rtol=2e-5)
                        if kind in ('zero','constant') or name.startswith('unique'):
                            assert torch.equal(values.double(),oracle)
                    summary=dict(case=name,deterministic=deterministic,active_ids=len(active),
                        unique_gradient_hashes=len(set(hashes)),max_abs_vs_fp64=max_error,
                        elapsed_seconds=times,token_shape=list(ids.shape))
                    emit('case_summary',**summary);summaries.append(summary)
                    assert len(set(hashes))==1
                assert len(set(case_hashes))==1
                del native,output,actual,values,grad,tokens;gc.collect();torch.mps.empty_cache()
            assert sha(weight)==initial
            emit('finished',backward_calls=88,standalone_checks_only=True,runtime_adopted=False)
            record_property('case_summaries',summaries)
        finally:
            torch.use_deterministic_algorithms(enabled,warn_only=warn)
            del weight;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
    assert torch.are_deterministic_algorithms_enabled()==enabled
    assert torch.is_deterministic_algorithms_warn_only_enabled()==warn


@pytest.mark.parametrize('case',['dtype','width','indices_dtype','rank','empty','too_many','cpu_weight','negative','out_of_range','autocast'])
def test_reject_unsupported_contract(case):
    weight=torch.zeros(50257,256,device='mps',requires_grad=True)
    ids=torch.zeros(1,2,dtype=torch.long,device='mps')
    if case=='dtype':weight=weight.half()
    elif case=='width':weight=weight[:,:128]
    elif case=='indices_dtype':ids=ids.int()
    elif case=='rank':ids=ids.flatten()
    elif case=='empty':ids=ids[:,:0]
    elif case=='too_many':ids=torch.zeros(1,1025,dtype=torch.long,device='mps')
    elif case=='cpu_weight':weight=weight.cpu()
    elif case=='negative':ids[0,0]=-1
    elif case=='out_of_range':ids[0,0]=50257
    if case=='autocast':
        with torch.autocast('mps',dtype=torch.float16),pytest.raises(ValueError,match='autocast'):
            grouped_embedding(weight,ids)
    else:
        with pytest.raises(ValueError):grouped_embedding(weight,ids)
    gc.collect();torch.mps.empty_cache()
