"""CPU-only RAW/H200 and saved-clipping analysis; no model or scoring calls."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import traceback
from research.byte_context_transform_pilot_v1 import ROOT, BASE, sha, put, read, runtime, pins

QUALITY=BASE/'quality-recovery002'
OUT=BASE/'weight-dynamics001'


def prepare():
    assert not OUT.exists()
    review=read(QUALITY/'final-review.json');assert review['accepted'] and not review['quality_success']
    assets=[QUALITY/'quality-plan.json',QUALITY/'final.json',QUALITY/'final-review.json',
            BASE/'final-analysis001/result.json',ROOT/'research/BYTE_CONTEXT_WEIGHT_DYNAMICS_PROTOCOL.md']
    specs=[]
    for slot in range(9):
        path=QUALITY/f'{slot:02d}';row=read(path/'result.json');acceptance=read(path/'acceptance.json')
        assert acceptance['accepted'] and acceptance['result_sha256']==sha(path/'result.json')
        assets.extend(path/name for name in ('result.json','acceptance.json','training.jsonl'))
        for key in ('raw','h200'):
            f=Path(row[key]['path']);assert f.stat().st_size==row[key]['size'] and sha(f)==row[key]['sha256']
            assets.append(f)
        specs.append(row['spec'])
    sources=[Path(__file__),ROOT/'research/byte_context_transform_pilot_v1.py',ROOT/'src/research/cpu_prerequisite_guard.py']
    plan=dict(kind='byte_context_weight_dynamics_v1',sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        assets=[dict(path=str(p),sha256=sha(p)) for p in assets],runtime=runtime(),specs=specs,
        groups=['output','composer','transformer','context'],windows=[300,600,900,1200],
        seconds=120,rss=4*1024**3,reserve=8*1024**3,extra_headroom=2*1024**3,output_cap=4*1024**2,
        no_model_execution=True,no_training=True,no_rescoring=True,no_adoption=True)
    assert shutil.disk_usage(ROOT).free>=plan['reserve']+plan['extra_headroom']+plan['output_cap']
    OUT.mkdir();put(OUT/'plan.json',plan);print(sha(OUT/'plan.json'),flush=True)


def group(name):
    if name.startswith('lm_head.'):return 'output'
    if name.startswith('composer.'):return 'composer'
    if name.startswith('transformer.'):return 'transformer'
    assert name=='context_weight',name
    return 'context'


def metrics(values):
    n=values['elements'];r=values['raw_sq'];h=values['h200_sq'];d=values['delta_sq'];dot=values['dot']
    assert n>0 and min(r,h,d)>=0 and all(math.isfinite(v) for v in values.values())
    # Independently equivalent Euclidean identity; cancellation error is FP64 only.
    assert abs(d-(r+h-2*dot))<=1e-10*max(1,r,h)
    radial=(dot-r)/r if r else None
    orth2=d/r-radial**2 if r else None
    if orth2 is not None:assert orth2>=-1e-12
    return dict(values,raw_rms=math.sqrt(r/n),h200_rms=math.sqrt(h/n),delta_rms=math.sqrt(d/n),
        relative_delta=math.sqrt(d/r) if r else None,norm_ratio=math.sqrt(h/r) if r else None,
        cosine=dot/math.sqrt(r*h) if r and h else None,radial_relative=radial,
        orthogonal_relative=math.sqrt(max(0,orth2)) if orth2 is not None else None)


def run(expected):
    import torch
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    assert sha(OUT/'plan.json')==expected
    plan=read(OUT/'plan.json');pins(plan)
    assert not (OUT/'attempt.json').exists()
    class Guard(CpuPrerequisiteGuard):
        def _sample(self):
            sample,reason=super()._sample()
            sample['output_bytes']=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
            if sample['output_bytes']>plan['output_cap']:reason='output_limit'
            return sample,reason
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free>=plan['reserve']+plan['extra_headroom']+plan['output_cap']
        put(OUT/'attempt.json',dict(pid=os.getpid(),plan_sha256=expected))
        try:
            with Guard(OUT/'guard.json',ROOT,timeout_seconds=plan['seconds'],rss_limit_bytes=plan['rss'],reserve_bytes=plan['reserve'],poll_seconds=.5):
                torch.set_num_threads(4);rng=torch.random.get_rng_state().clone()
                curves=read(BASE/'final-analysis001/result.json');rows=[]
                for slot,spec in enumerate(plan['specs']):
                    folder=QUALITY/f'{slot:02d}';original=read(folder/'result.json')
                    raw=torch.load(original['raw']['path'],map_location='cpu',weights_only=True)
                    avg=torch.load(original['h200']['path'],map_location='cpu',weights_only=True)
                    assert raw['spec']==avg['spec']==spec
                    assert raw['kind']=='final_raw_weights_not_resume' and avg['kind']=='h200_inference_weights'
                    a,b=raw['weights'],avg['weights'];assert list(a)==list(b)
                    parameters=[]
                    for name,x in a.items():
                        y=b[name];assert x.shape==y.shape and x.dtype==y.dtype==torch.float32
                        assert torch.isfinite(x).all() and torch.isfinite(y).all()
                        x=x.double().reshape(-1);y=y.double().reshape(-1);delta=y-x
                        values=dict(elements=x.numel(),raw_sq=float(torch.dot(x,x)),h200_sq=float(torch.dot(y,y)),delta_sq=float(torch.dot(delta,delta)),dot=float(torch.dot(x,y)))
                        parameters.append(dict(name=name,group=group(name),**metrics(values)))
                        del x,y,delta
                    expected_count=105 if spec['arm']=='byte' else 106
                    assert len(parameters)==expected_count and sum(r['elements'] for r in parameters)==original['parameter_count']
                    groups={}
                    for kind in plan['groups']:
                        members=[r for r in parameters if r['group']==kind]
                        if not members:
                            assert kind=='context' and spec['arm']=='byte';continue
                        totals={k:(sum(r[k] for r in members) if k=='elements' else math.fsum(r[k] for r in members)) for k in ('elements','raw_sq','h200_sq','delta_sq','dot')}
                        groups[kind]=metrics(totals)
                    total_delta=math.fsum(g['delta_sq'] for g in groups.values())
                    for g in groups.values():g['share_of_squared_displacement']=g['delta_sq']/total_delta
                    history=[json.loads(s) for s in (folder/'training.jsonl').read_text().splitlines()]
                    assert len(history)==1200 and all(r['step']==i for i,r in enumerate(history,1))
                    windows=[]
                    for end in plan['windows']:
                        data=history[end-100:end]
                        windows.append(dict(end_step=end,mean_pre_clip_norm=math.fsum(r['pre_norm'] for r in data)/100,
                            mean_clip_coefficient=math.fsum(r['coefficient'] for r in data)/100,
                            clipped_fraction=sum(r['coefficient']<1 for r in data)/100))
                    prior=curves['rows'][slot];assert prior['slot']==slot and prior['arm']==spec['arm']
                    row=dict(slot=slot,spec=spec,groups=groups,parameters=parameters,clipping=windows,
                        h200_minus_final_online=prior['h200_minus_final_online'],online_score_caveat=prior['online_score_caveat'])
                    put(OUT/f'{slot:02d}.json',row);rows.append({k:v for k,v in row.items() if k!='parameters'})
                    del a,b,raw,avg
                assert torch.equal(rng,torch.random.get_rng_state());pins(plan)
                put(OUT/'result.json',dict(completed=True,plan_sha256=expected,rows=rows,
                    no_model_execution=True,no_training=True,no_rescoring=True,no_adoption=True,rng_unchanged=True))
            put(OUT/'acceptance.json',dict(accepted=True,result_sha256=sha(OUT/'result.json'),guard_sha256=sha(OUT/'guard.json')))
        except BaseException:
            put(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));p.add_argument('plan_sha',nargs='?');a=p.parse_args()
    prepare() if a.action=='prepare' else run(a.plan_sha)
