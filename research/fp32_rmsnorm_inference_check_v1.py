"""Separate, bounded correctness qualification; never changes quality reports."""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
import shutil
import traceback
from pathlib import Path

from research.byte_context_transform_pilot_v1 import ROOT, BASE, sha, put, read, runtime, pins

OUT = BASE/'inference-epsilon001'
Q = BASE/'quality-recovery001'


def prepare():
    assert not OUT.exists()
    original = read(BASE/'quality-plan.json')
    assert read(Q/'00/acceptance.json')['accepted']
    inputs = [BASE/'quality-plan.json', Q/'00/acceptance.json', Q/'00/result.json',
              Q/'00/validation-cpu.json', Q/'00/validation-mps.json',
              BASE/'rms-diagnostic001/result.json', ROOT/'research/FP32_RMSNORM_INFERENCE_REPAIR.md']
    checkpoint = ROOT/'research_runs/byte-context-transform-001/00/h200.pt'
    assert sha(checkpoint) == read(Q/'00/result.json')['h200']['sha256']
    plan = dict(kind='fp32_rmsnorm_inference_check_v1', runtime=runtime(),
        sources={str(p.relative_to(ROOT)):sha(p) for folder in ('research','src')
                 for p in (ROOT/folder).rglob('*.py')},
        assets=original['assets']+[dict(path=str(p),sha256=sha(p)) for p in inputs+[checkpoint]],
        checkpoint=str(checkpoint), block_starts=[0,15616], batches=[1,8],
        prefill=96, decode_steps=32, full_targets=34278,
        mean_ce_atol=3e-5, target_ce_atol=3e-4, cache_atol=3e-5, cache_rtol=3e-4,
        seconds=600, rss=4*1024**3, mps=6*1024**3, reserve=8*1024**3,
        output_cap=32*1024**2, extra_headroom=2*1024**3,
        no_training=True, no_quality_reselection=True)
    assert shutil.disk_usage(ROOT).free>=plan['reserve']+plan['extra_headroom']+plan['output_cap']
    OUT.mkdir();put(OUT/'plan.json',plan);print(sha(OUT/'plan.json'),flush=True)


def run(expected):
    import torch
    from torch.nn import functional as F
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    from research.byte_context_transform_v1 import construct,load_weights
    from research.byte_input_model_v3 import build_codebook
    from research.byte_input_training_v6 import load_view
    from research.byte_input_cache_v4 import Session

    assert sha(OUT/'plan.json') == expected
    plan=read(OUT/'plan.json');pins(plan)
    assert not (OUT/'attempt.json').exists()

    class Guard(CpuPrerequisiteGuard):
        def _sample(self):
            sample,reason=super()._sample()
            sample['mps_bytes']=torch.mps.driver_allocated_memory()
            sample['output_bytes']=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
            if sample['mps_bytes']>=plan['mps']:reason='mps_limit'
            if sample['output_bytes']>plan['output_cap']:reason='output_limit'
            return sample,reason

    def digest(t):
        return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def numerical(a,b):
        return dict(mean_ce_error=abs(math.fsum(a)-math.fsum(b))/len(a),
                    max_target_ce_error=max(abs(x-y) for x,y in zip(a,b,strict=True)))

    def check_error(r):
        assert r['mean_ce_error']<=plan['mean_ce_atol']
        assert r['max_target_ce_error']<=plan['target_ce_atol']

    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free>=plan['reserve']+plan['extra_headroom']+plan['output_cap']
        put(OUT/'attempt.json',dict(pid=os.getpid(),plan_sha256=expected))
        try:
            with Guard(OUT/'guard.json',ROOT,timeout_seconds=plan['seconds'],
                       rss_limit_bytes=plan['rss'],reserve_bytes=plan['reserve'],poll_seconds=.5):
                assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
                assert not torch.is_inference_mode_enabled()
                torch.set_num_threads(4)
                original=read(BASE/'quality-plan.json');spec=original['specs'][0]
                book=build_codebook(*(Path(r['path']).read_bytes() for r in original['codebook_assets']),'raw_bytes')
                saved=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
                scores={};cases=[]
                for backend in ('cpu','mps'):
                    model=construct('byte',spec['roles'],spec['projection_seed'],book,backend)
                    load_weights(model,saved['weights']);model.eval()
                    view=load_view(spec['roles'],'cpu')
                    before={n:digest(p) for n,p in model.ordered_parameters()}
                    rng=(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
                    semantics=model.base._semantic_inventory()
                    probes=[]
                    with torch.no_grad():
                        for start in plan['block_starts']:
                            ids=view.val_data[start:start+128].unsqueeze(0).to(backend)
                            old,_=model(ids);probes.append(digest(old));del old
                        losses=[]
                        with FP32RMSNormInference() as correction:
                            for start in range(0,plan['full_targets'],128):
                                count=min(128,plan['full_targets']-start)
                                ids=view.val_data[start:start+count].unsqueeze(0).to(backend)
                                targets=view.val_data[start+1:start+count+1].to(backend)
                                logits,_=model(ids)
                                assert not logits.requires_grad
                                values=F.cross_entropy(logits[0],targets,reduction='none').cpu().double()
                                assert torch.isfinite(values).all() and (values>=0).all()
                                losses.extend(values.tolist());del logits,values
                        reference=read(Q/f'00/validation-{backend}.json')['per_target_losses']
                        error=numerical(losses,reference)
                        put(OUT/f'{backend}-score.json',dict(ce=math.fsum(losses)/len(losses),
                            per_target_losses=losses,reference_error=error,corrected_calls=correction.corrected_calls))
                        assert correction.corrected_calls==8*math.ceil(plan['full_targets']/128)
                        check_error(error);scores[backend]=losses
                        for start,prior in zip(plan['block_starts'],probes,strict=True):
                            old,_=model(view.val_data[start:start+128].unsqueeze(0).to(backend))
                            assert digest(old)==prior;del old
                        weight=model.base.blocks[0].ln1.weight
                        x=torch.ones((1,1,256),device=backend)
                        explicit=F.rms_norm(x,(256,),weight,1e-5)
                        with FP32RMSNormInference() as correction:
                            assert torch.equal(F.rms_norm(x,(256,),weight,1e-5),explicit)
                        assert correction.corrected_calls==0
                        for bad,shape in [(x.double(),(256,)),(x[...,:128],(128,))]:
                            try:
                                with FP32RMSNormInference():F.rms_norm(bad,shape,weight)
                            except ValueError:pass
                            else:raise AssertionError('Unsupported input accepted')
                    try:
                        with torch.enable_grad(),FP32RMSNormInference():F.rms_norm(x,(256,),weight)
                    except ValueError:pass
                    else:raise AssertionError('Training call accepted')
                    try:
                        with torch.inference_mode(),FP32RMSNormInference():F.rms_norm(x,(256,),weight)
                    except ValueError:pass
                    else:raise AssertionError('Unqualified inference_mode call accepted')
                    for batch in plan['batches']:
                        ids=view.train_data[:batch*128].reshape(batch,128).to(backend)
                        with torch.no_grad(),FP32RMSNormInference() as correction:
                            full,_=model(ids)
                        assert correction.corrected_calls==8
                        session=Session(model.base,batch);checks=[];first=[];state=None
                        chunks=[(0,96)]+[(i,i+1) for i in range(96,128)]
                        for replay in (0,1):
                            if replay:
                                session.reset();assert all(not bool(t.any()) for t in session._tensors)
                            for i,(lo,hi) in enumerate(chunks):
                                with FP32RMSNormInference() as correction:
                                    actual=session.logits(ids[:,lo:hi],lo)
                                assert correction.corrected_calls==8 and not actual.requires_grad
                                target=full[:,lo:hi]
                                error=(actual-target).abs()
                                ratio=float((error/(plan['cache_atol']+plan['cache_rtol']*target.abs())).max())
                                row=dict(replay=replay,start=lo,end=hi,max_gate_ratio=ratio,
                                         max_absolute_error=float(error.max()),sha256=digest(actual))
                                checks.append(row)
                                if ratio>1:
                                    put(OUT/f'{backend}-b{batch}-failed-cache.json',dict(checks=checks))
                                torch.testing.assert_close(actual,target,atol=plan['cache_atol'],rtol=plan['cache_rtol'])
                                if replay:assert row['sha256']==first[i]
                                else:first.append(row['sha256'])
                                del actual,target,error
                            snapshot=[digest(t) for t in session._tensors]
                            assert session.position==128 and session.storage_bytes==196608*batch
                            if replay:assert snapshot==state
                            else:state=snapshot
                        row=dict(backend=backend,batch=batch,checks=checks,reset_exact=True)
                        put(OUT/f'{backend}-b{batch}-cache.json',row);cases.append(row)
                        del full,session,ids;gc.collect();torch.mps.empty_cache()
                    model.validate()
                    assert model.base._semantic_inventory()==semantics
                    assert {n:digest(p) for n,p in model.ordered_parameters()}==before
                    assert all(p.grad is None for _,p in model.ordered_parameters())
                    assert rng==(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
                    del model,view;gc.collect();torch.mps.empty_cache()
                cross=numerical(scores['cpu'],scores['mps'])
                put(OUT/'numerical.json',cross);check_error(cross);pins(plan)
                put(OUT/'result.json',dict(completed=True,adopted=False,no_training=True,
                    numerical=cross,cache_cases=len(cases),weights_semantics_rng_unchanged=True,
                    no_gradients=True,scoped_exit_verified=True,invalid_inputs_rejected=True,
                    explicit_epsilon_unchanged=True,plan_sha256=expected))
            put(OUT/'acceptance.json',dict(accepted=True,result_sha256=sha(OUT/'result.json'),guard_sha256=sha(OUT/'guard.json')))
        except BaseException:
            put(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));p.add_argument('plan_sha',nargs='?');a=p.parse_args()
    prepare() if a.action=='prepare' else run(a.plan_sha)
