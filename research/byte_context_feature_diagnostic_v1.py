"""Fixed training features and analytic map sensitivity; no logits or losses."""
import argparse
import fcntl
import gc
import math
import os
from pathlib import Path
import shutil
import traceback
from research.byte_context_transform_pilot_v1 import ROOT,BASE,sha,put,read,runtime,pins

QUALITY=BASE/'quality-recovery002'
OUT=BASE/'feature-diagnostic001'


def prepare():
    assert not OUT.exists()
    original=read(QUALITY/'quality-plan.json')
    assert read(QUALITY/'final-review.json')['accepted']
    evidence=read(BASE/'weight-dynamics001/acceptance.json');assert evidence['accepted']
    assert evidence['result_sha256']==sha(BASE/'weight-dynamics001/result.json')
    prior=read(ROOT/'research/tokenizer-evidence/public-inference-correction-001/plan.json')
    sources=dict(prior['sources'])
    for name in [__file__,'research/byte_context_transform_v1.py','research/byte_context_transform_pilot_v1.py',
                 'research/fp32_rmsnorm_inference_v1.py','src/research/cpu_prerequisite_guard.py']:
        p=Path(name);p=p if p.is_absolute() else ROOT/p
        sources[str(p.relative_to(ROOT))]=sha(p)
    assets=[ROOT/'research/BYTE_CONTEXT_FEATURE_DIAGNOSTIC_PROTOCOL.md',QUALITY/'quality-plan.json',
        QUALITY/'final-review.json',BASE/'weight-dynamics001/result.json',BASE/'weight-dynamics001/acceptance.json']
    assets.extend(Path(r['path']) for r in original['assets']+original['codebook_assets'])
    for slot in range(9):
        folder=QUALITY/f'{slot:02d}';row=read(folder/'result.json');accepted=read(folder/'acceptance.json')
        assert accepted['accepted'] and accepted['result_sha256']==sha(folder/'result.json')
        assets.extend([folder/'result.json',folder/'acceptance.json'])
        for key in ('raw','h200'):
            p=Path(row[key]['path']);assert p.stat().st_size==row[key]['size'] and sha(p)==row[key]['sha256']
            assets.append(p)
    plan=dict(kind='byte_context_feature_diagnostic_v1',sources=sources,runtime=runtime(),
        assets=[dict(path=str(p),sha256=sha(p)) for p in dict.fromkeys(assets)],specs=original['specs'],
        codebook_assets=original['codebook_assets'],starts=[0,128],length=128,endpoint_order=['raw','h200'],
        saturation_absolute_threshold=.95,seconds=180,rss=4*1024**3,reserve=8*1024**3,
        extra_headroom=2*1024**3,output_cap=4*1024**2,no_logits=True,no_losses=True,no_training=True,
        no_validation_scoring=True,no_holdout_scoring=True,no_adoption=True)
    assert shutil.disk_usage(ROOT).free>=plan['reserve']+plan['extra_headroom']+plan['output_cap']
    pins(plan);OUT.mkdir();put(OUT/'plan.json',plan);print(sha(OUT/'plan.json'),flush=True)


def run(expected):
    import torch
    from torch.nn import functional as F
    from research.byte_context_transform_v1 import construct,load_weights
    from research.byte_input_model_v3 import build_codebook
    from research.byte_input_training_v6 import load_view
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    assert sha(OUT/'plan.json')==expected
    plan=read(OUT/'plan.json');pins(plan);assert not (OUT/'attempt.json').exists()
    class Guard(CpuPrerequisiteGuard):
        def _sample(self):
            sample,reason=super()._sample()
            sample['output_bytes']=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
            if sample['output_bytes']>plan['output_cap']:reason='output_limit'
            return sample,reason
    def digest(t):
        import hashlib
        return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    def rms(x):return math.sqrt(float(x.double().square().mean()))
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free>=plan['reserve']+plan['extra_headroom']+plan['output_cap']
        put(OUT/'attempt.json',dict(pid=os.getpid(),plan_sha256=expected))
        try:
            with Guard(OUT/'guard.json',ROOT,timeout_seconds=plan['seconds'],rss_limit_bytes=plan['rss'],reserve_bytes=plan['reserve'],poll_seconds=.5):
                torch.set_num_threads(4)
                rng=(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
                book=build_codebook(*(Path(r['path']).read_bytes() for r in plan['codebook_assets']),'raw_bytes')
                cases=[];input_hashes={}
                for slot,spec in enumerate(plan['specs']):
                    model=construct(spec['arm'],spec['roles'],spec['projection_seed'],book,'cpu');model.eval()
                    semantics=model.base._semantic_inventory();view=load_view(spec['roles'],'cpu')
                    sampler=digest(view.generators['train'].get_state())
                    saved=read(QUALITY/f'{slot:02d}/result.json')
                    for endpoint in plan['endpoint_order']:
                        payload=torch.load(saved[endpoint]['path'],map_location='cpu',weights_only=True)
                        assert payload['spec']==spec
                        load_weights(model,payload['weights']);model.eval()
                        before={n:digest(p) for n,p in model.ordered_parameters()}
                        assert before=={n:digest(p) for n,p in payload['weights'].items()}
                        for start in plan['starts']:
                            ids=view.train_data[start:start+plan['length']].unsqueeze(0)
                            input_sha=digest(ids)
                            if start in input_hashes:assert input_hashes[start]==input_sha
                            else:input_hashes[start]=input_sha
                            model.base._ids(ids,torch.device('cpu'))
                            with torch.no_grad(),FP32RMSNormInference() as correction:
                                z=model.base._composed_hidden(ids)
                            assert correction.corrected_calls==8 and not z.requires_grad
                            with torch.no_grad():
                                u=z if spec['arm']=='byte' else F.linear(z,model.context_weight)
                                feature=u.tanh() if spec['arm']=='tanh' else u
                            assert z.shape==u.shape==feature.shape==(1,128,256)
                            assert all(torch.isfinite(t).all() and not t.requires_grad for t in (z,u,feature))
                            derivative=1-feature.double().square() if spec['arm']=='tanh' else torch.ones_like(feature,dtype=torch.float64)
                            assert (derivative>=0).all() and (derivative<=1).all()
                            row_sq=torch.ones(256,dtype=torch.float64) if spec['arm']=='byte' else model.context_weight.detach().double().square().sum(dim=1)
                            linear_gain=math.sqrt(float(row_sq.sum())/256)
                            local_gain=math.sqrt(float((derivative.square()*row_sq).sum(dim=-1).mean())/256)
                            assert 0<=local_gain<=linear_gain+1e-12
                            row=dict(slot=slot,arm=spec['arm'],bundle=spec['bundle'],endpoint=endpoint,start=start,input_sha256=input_sha,
                                z_rms=rms(z),preactivation_rms=rms(u),feature_rms=rms(feature),feature_to_z_rms=rms(feature)/rms(z),
                                output_weight_rms=rms(model.base.readout.weight),mean_derivative=float(derivative.mean()),
                                mean_squared_derivative=float(derivative.square().mean()),linear_map_gain=linear_gain,
                                local_map_gain=local_gain,local_to_linear_gain=local_gain/linear_gain,
                                saturation_fraction=float((feature.abs()>=plan['saturation_absolute_threshold']).double().mean()) if spec['arm']=='tanh' else None)
                            assert abs(row['mean_derivative']-(1-row['feature_rms']**2))<1e-12 if spec['arm']=='tanh' else row['mean_derivative']==1
                            cases.append(row);put(OUT/f'{slot:02d}-{endpoint}-{start}.json',row)
                            del z,u,feature,derivative,row_sq
                        model.validate();assert model.base._semantic_inventory()==semantics
                        assert {n:digest(p) for n,p in model.ordered_parameters()}==before
                        assert all(p.grad is None for _,p in model.ordered_parameters())
                        assert sampler==digest(view.generators['train'].get_state())
                        del payload
                    del model,view;gc.collect()
                assert len(cases)==36
                assert rng==(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
                pins(plan)
                put(OUT/'result.json',dict(completed=True,plan_sha256=expected,cases=cases,input_hashes=input_hashes,
                    weights_semantics_rng_unchanged=True,no_gradients=True,no_logits=True,no_losses=True,
                    no_training=True,no_validation_scoring=True,no_holdout_scoring=True,no_adoption=True))
            put(OUT/'acceptance.json',dict(accepted=True,result_sha256=sha(OUT/'result.json'),guard_sha256=sha(OUT/'guard.json')))
        except BaseException:
            put(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));p.add_argument('plan_sha',nargs='?');a=p.parse_args()
    prepare() if a.action=='prepare' else run(a.plan_sha)
