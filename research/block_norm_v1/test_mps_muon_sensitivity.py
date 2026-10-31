"""Synthetic identical-gradient repeat and perturbation probe; no model training."""
import gc
import hashlib
import json
import os
from pathlib import Path
import torch


def sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def test_muon_same_gradient_and_small_perturbation(record_property):
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    generator=torch.Generator().manual_seed(372801)
    summaries=[]
    with Path(os.environ['BLOCK_NORM_TRACE_PATH']).open('x') as stream:
        def emit(event,**fields):
            stream.write(json.dumps(dict(event=event,**fields),sort_keys=True)+'\n');stream.flush()
            assert stream.tell()<1024**2
        emit('begin',torch_version=str(torch.__version__),seed=372801,perturbation_magnitude=2**-29,
            shapes=[[256,64],[672,256]],synthetic_probe_not_measured_gradient_replay=True)
        try:
            for shape in [(256,64),(672,256)]:
                initial=torch.randn(shape,generator=generator)*.01
                gradient=torch.randn(shape,generator=generator)*.0001
                momentum=torch.randn(shape,generator=generator)*.0001
                perturb=(torch.randint(2,shape,generator=generator)*2-1).float()*(2**-29)
                perturbed=gradient+perturb
                actual_delta=float((perturbed-gradient).abs().max())
                emit('inputs',shape=shape,weight_sha256=sha(initial),gradient_sha256=sha(gradient),
                    momentum_sha256=sha(momentum),perturbed_gradient_sha256=sha(perturbed),actual_gradient_delta=actual_delta)
                for deterministic in (False,True):
                    torch.use_deterministic_algorithms(deterministic,warn_only=False)
                    first={};quantized={};rows=[]
                    for name,grad in [('base',gradient),('perturbed',perturbed)]:
                        for repeat in range(4):
                            param=torch.nn.Parameter(initial.to('mps').clone());param.grad=grad.to('mps').clone()
                            opt=torch.optim.Muon([param],lr=.000625,weight_decay=0.,adjust_lr_fn='match_rms_adamw')
                            opt.state[param]['momentum_buffer']=momentum.to('mps').clone()
                            # Observe the input to installed Muon's BF16 orthogonalizer.
                            with torch.no_grad():
                                buf=opt.state[param]['momentum_buffer'].lerp(param.grad,1-opt.defaults['momentum'])
                                update=param.grad.lerp(buf,opt.defaults['momentum'])
                                q=update.bfloat16().cpu()
                                param.mul_(1-.00006)
                            opt.step();torch.mps.synchronize()
                            result=param.detach().cpu();slot=opt.state[param]['momentum_buffer']
                            assert param.dtype==param.grad.dtype==slot.dtype==torch.float32
                            assert param.device.type==param.grad.device.type==slot.device.type=='mps'
                            assert torch.isfinite(result).all() and torch.isfinite(slot).all()
                            if repeat==0:first[name]=result.clone();quantized[name]=q.clone()
                            row=dict(shape=shape,deterministic=deterministic,input=name,repeat=repeat,
                                parameter_sha256=sha(result),momentum_sha256=sha(slot),orthogonalizer_bf16_input_sha256=sha(q),
                                max_parameter_repeat_difference=float((result-first[name]).abs().max()))
                            emit('step',**row);rows.append(row)
                            del opt,param,slot,buf,update,q
                        gc.collect();torch.mps.empty_cache()
                    difference=(first['perturbed']-first['base']).abs()
                    summary=dict(shape=shape,deterministic=deterministic,actual_gradient_delta=actual_delta,
                        perturbed_parameter_max_difference=float(difference.max()),
                        changed_parameter_elements=int((difference!=0).sum()),
                        changed_bf16_orthogonalizer_input_elements=int((quantized['base']!=quantized['perturbed']).sum()),
                        max_identical_input_repeat_difference=max(r['max_parameter_repeat_difference'] for r in rows),
                        unique_base_parameter_hashes=len({r['parameter_sha256'] for r in rows if r['input']=='base'}),
                        unique_perturbed_parameter_hashes=len({r['parameter_sha256'] for r in rows if r['input']=='perturbed'}))
                    emit('case_summary',**summary);summaries.append(summary)
            emit('finished',synthetic_single_matrix_updates=32,quality_training=False,numerical_resume_accepted=False)
            record_property('case_summaries',summaries)
        finally:
            torch.use_deterministic_algorithms(enabled,warn_only=warn)
            gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
    assert torch.are_deterministic_algorithms_enabled()==enabled
    assert torch.is_deterministic_algorithms_warn_only_enabled()==warn
