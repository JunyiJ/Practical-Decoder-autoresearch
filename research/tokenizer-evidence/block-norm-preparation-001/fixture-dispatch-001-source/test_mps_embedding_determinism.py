"""Isolated embedding-backward diagnostic; no model or optimizer training."""
import gc
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from src.research.protocol import ROOT
from research.block_norm_v1.runtime.train.train import _make_loader


def test_embedding_repeat_and_deterministic_control(record_property):
    enabled=torch.are_deterministic_algorithms_enabled()
    warn=torch.is_deterministic_algorithms_warn_only_enabled()
    cfg=OmegaConf.create(json.loads((ROOT/'research_state/block-norm-trainer-fixture-002/rms_incumbent/config.json').read_text()))
    cfg.training.seed=372701
    data=_make_loader(cfg,'cpu');sample,_=data.get_batch('train')
    assert sample.shape==(1,128) and set(data.splits)=={'train','val'}
    generator=torch.Generator().manual_seed(372702)
    upstream=torch.randn(1,128,256,generator=generator,dtype=torch.float32)*.001
    upstream_mps=upstream.to('mps')
    path=Path(os.environ['BLOCK_NORM_TRACE_PATH'])
    summaries=[]
    with path.open('x') as stream:
        def emit(event,**fields):
            stream.write(json.dumps(dict(event=event,**fields),sort_keys=True)+'\n');stream.flush()
            assert stream.tell()<1024**2
        emit('begin',torch_version=str(torch.__version__),initial_deterministic=enabled,
            initial_warn_only=warn,upstream_sha256=hashlib.sha256(upstream.numpy().tobytes()).hexdigest(),
            cases=['generated_text_duplicates','unique_ids'],passes_per_mode=8,optimizer_updates=0)
        try:
            for name,indices in [('generated_text_duplicates',sample),('unique_ids',torch.arange(128).reshape(1,128))]:
                active=torch.unique(indices)
                # Exact analytic embedding derivative: sequential FP64 sums of upstream rows.
                oracle=torch.zeros(len(active),256,dtype=torch.float64)
                row={int(token):i for i,token in enumerate(active)}
                for token,grad in zip(indices.flatten(),upstream.reshape(128,256)):
                    oracle[row[int(token)]]+=grad.double()
                ids=indices.to('mps')
                for deterministic in (False,True):
                    torch.use_deterministic_algorithms(deterministic,warn_only=False)
                    assert torch.are_deterministic_algorithms_enabled()==deterministic
                    assert not torch.is_deterministic_algorithms_warn_only_enabled()
                    weight=torch.nn.Parameter(torch.zeros(50257,256,device='mps',dtype=torch.float32))
                    first=None;hashes=[];errors=[];changes=[];failure=None
                    emit('case_begin',case=name,deterministic=deterministic,active_ids=len(active),
                        input_sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest())
                    for repeat in range(8):
                        weight.grad=None
                        try:
                            F.embedding(ids,weight).backward(upstream_mps);torch.mps.synchronize()
                        except RuntimeError as exc:
                            emit('backend_error',case=name,deterministic=deterministic,repeat=repeat,error=str(exc))
                            if deterministic and 'determin' in str(exc).lower():
                                failure=str(exc);break
                            raise
                        assert weight.grad.device.type=='mps' and weight.grad.dtype==torch.float32
                        gradient=weight.grad.detach().cpu()
                        actual=gradient[active]
                        h=hashlib.sha256(gradient.numpy().tobytes()).hexdigest()
                        error=float((actual.double()-oracle).abs().max())
                        change=0. if first is None else float((actual-first).abs().max())
                        differing=0 if first is None else int((actual!=first).sum())
                        finite=bool(torch.isfinite(gradient).all())
                        unused_zero=int(torch.count_nonzero(gradient))==int(torch.count_nonzero(actual))
                        emit('backward',case=name,deterministic=deterministic,repeat=repeat,
                            gradient_sha256=h,max_abs_vs_fp64_oracle=error,max_abs_vs_first=change,
                            differing_elements_vs_first=differing,finite=finite,unused_rows_zero=unused_zero)
                        assert finite and unused_zero
                        if first is None:first=actual.clone()
                        hashes.append(h);errors.append(error);changes.append(change)
                    assert torch.count_nonzero(weight)==0
                    summary=dict(case=name,deterministic=deterministic,completed_passes=len(hashes),
                        unique_gradient_hashes=len(set(hashes)),max_repeat_difference=max(changes,default=None),
                        max_oracle_error=max(errors,default=None),backend_error=failure)
                    emit('case_summary',**summary);summaries.append(summary)
                    del weight,first;gc.collect();torch.mps.empty_cache()
            emit('finished',summaries=summaries,numerical_resume_accepted=False,
                scope='Isolated embedding only; no optimizer updates or full-model deterministic claim.')
            record_property('case_summaries',summaries)
        finally:
            torch.use_deterministic_algorithms(enabled,warn_only=warn)
            torch.mps.synchronize();gc.collect();torch.mps.empty_cache()
    assert torch.are_deterministic_algorithms_enabled()==enabled
    assert torch.is_deterministic_algorithms_warn_only_enabled()==warn
    record_property('global_deterministic_settings_restored',True)
