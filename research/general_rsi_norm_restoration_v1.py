"""Frozen retained-weight factorial inference diagnostic. No optimizer or new weights."""
import fcntl,gc,json,math,os,shutil,signal,sys,time,traceback
from pathlib import Path
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,digest,batch_hash,evaluate,check_stop
OUT=ROOT/'research/tokenizer-evidence/general-rsi-norm-restoration-001'
N=ROOT/'research/tokenizer-evidence/general-rsi-native-horizon-001'
P=ROOT/'research/tokenizer-evidence/general-rsi-pseudo-repetition-001'
CONDITIONS=['original','readout','other','both']
G=1024**3

def read(p):return json.loads(Path(p).read_text())
def payload(rec):
    import torch
    p=Path(rec['path']);assert not p.is_symlink() and p.stat().st_size==rec['size'] and sha(p)==rec['sha256']
    v=torch.load(p,map_location='cpu',weights_only=True)
    assert v['kind']=='general_rsi_equal_time_inference_v1'
    assert all(t.dtype==torch.float32 and bool(torch.isfinite(t).all()) for t in v['parameters'].values())
    return v

def norms(weights):
    return {g:math.sqrt(math.fsum(float(t.double().square().sum()) for n,t in weights.items() if ('readout' if n=='lm_head.weight' else 'other')==g)) for g in ['readout','other']}

def apply(model,original,factors,condition):
    import torch
    items=dict(model.ordered_parameters());assert list(items)==list(original)
    with torch.no_grad():
        for n,p in items.items():
            t=original[n];assert t.dtype==p.dtype and t.shape==p.shape
            p.copy_(t)
        # Copy first, then check exact restoration before any scaling.
        assert all(torch.equal(p.detach().cpu(),original[n]) for n,p in items.items())
        for n,p in items.items():
            group='readout' if n=='lm_head.weight' else 'other'
            if condition in [group,'both']:p.mul_(factors[group])
            expected=original[n]*factors[group] if condition in [group,'both'] else original[n]
            assert torch.equal(p.detach().cpu(),expected) and bool(torch.isfinite(p).all())
    return True

def prepare():
    import numpy as np,torch
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(4);tick=time.monotonic()
    prior=read(N/'plan.json');panel=read(P/'plan.json');bindings=dict(prior['sources'])
    for path in [N/'plan.json',P/'plan.json',ROOT/'research/GENERAL_RSI_NORM_RESTORATION_NEXT.md',Path(__file__).resolve(),ROOT/'research/general_rsi_transfer_worker_v1.py',ROOT/'research/general_rsi_transfer_data_v1.py',ROOT/'research/general_rsi_pseudo_update_diagnostic_v1.py',ROOT/'src/research/cpu_prerequisite_guard.py']:
        bindings[str(path.relative_to(ROOT))]=sha(path)
    models=[]
    for c in panel['slots']:
        if c['recipe']!='native_ce':continue
        parent=c['parent'];pv=payload(parent);pn=norms(pv['parameters']);endpoints=[]
        for f in [P/f"slot-{c['slot']:02d}/review.json",P/f"slot-{c['slot']:02d}/result.json"]:bindings[str(f.relative_to(ROOT))]=sha(f)
        assert read(P/f"slot-{c['slot']:02d}/review.json")['accepted_execution']
        for e in read(P/f"slot-{c['slot']:02d}/result.json")['endpoints']:
            rec=e['weights'];v=payload(rec);assert v['binding']==sha(P/'plan.json') and v['slot']==c and v['endpoint']==e['endpoint']
            assert v['model_identity']==pv['model_identity'] and v['immutable_buffers']==pv['immutable_buffers']
            assert list(v['parameters'])==list(pv['parameters']);en=norms(v['parameters'])
            endpoints.append(dict(endpoint=e['endpoint'],weights=rec,norms=en,factors={g:pn[g]/en[g] for g in pn},parameters_sha256=digest(v['parameters'])))
            bindings[str(Path(rec['path']).relative_to(ROOT))]=rec['sha256'];del v
        bindings[str(Path(parent['path']).relative_to(ROOT))]=parent['sha256']
        models.append(dict(parent=dict(parent,arm='native'),parent_norms=pn,endpoints=endpoints));del pv;gc.collect()
    data=ROOT/'data/raw/rsi-general-10m-v1';manifest=read(data/'manifest.json')
    for name in ['manifest.json','train.bin','train.index.jsonl']:
        h=sha(data/name)
        if name!='manifest.json':assert h==manifest['files'][name]['sha256']
        bindings[str((data/name).relative_to(ROOT))]=h
    docs=[json.loads(l) for l in (data/'train.index.jsonl').read_text().splitlines()]
    values=torch.from_numpy(np.fromfile(data/'train.bin',dtype='<u2').astype('int64'))
    seed=621112093201;rng=torch.Generator().manual_seed(seed);eligible=[i for i,d in enumerate(docs) if d['tokens']>=129];windows=[]
    while len(windows)<64:
        doc=eligible[int(torch.randint(len(eligible),(),generator=rng))];d=docs[doc]
        start=d['offset']+int(torch.randint(d['tokens']-128,(),generator=rng))
        if any(start<w['start']+129 and w['start']<start+129 for w in windows):continue
        windows.append(dict(document=doc,source=d['source'],start=start,targets=128))
    batches=[(values[w['start']:w['start']+128][None,:],values[w['start']+1:w['start']+129][None,:]) for w in windows]
    assert len(models)==3 and sum(len(m['endpoints']) for m in models)==6
    plan=dict(kind='norm_restoration_factorial_v1',created_unix=time.time(),models=models,conditions=CONDITIONS,windows=windows,evaluation_seed=seed,
        evaluation_sha256=batch_hash(*(t for b in batches for t in b)),targets_per_condition=8192,model_conditions=27,torch_version=prior['torch_version'],
        checkpoint_source_plan_sha256=sha(N/'plan.json'),sources=bindings,
        order='Parent slots1,3,5; parent reference then original/readout/other/both for RAW, then same four for time EMA.',
        analysis='Per parent/endpoint: readout-original, other-original, both-original, interaction both-readout-other+original, and each condition-parent. Equal-parent means separately by endpoint. No selection/adoption or causal decay claim.',
        qualification=dict(logit_scaling_rtol=2e-5,logit_scaling_atol=2e-5,exact_tensor_restoration=True),
        total_seconds=600,rss_limit_bytes=4*G,mps_limit_bytes=6*G,reserve_bytes=10*G,metadata_cap_bytes=64*1024**2,overlap_bytes=64*1024**2,
        required_free_bytes=12*G+128*1024**2,checkpoints=0,training_updates=0,validation_scored=False,test_opened=False)
    assert all(sha(ROOT/p)==h for p,h in bindings.items())
    put(OUT/'plan.json',plan);put(OUT/'preparation.json',dict(seconds=time.monotonic()-tick,plan_sha256=sha(OUT/'plan.json'),factors_frozen_before_scores=True,model_scores=0,free_bytes=shutil.disk_usage(ROOT).free))
    print('Frozen27 conditions,64 shared windows and six mechanical factor pairs.',flush=True)

def execute(plan,attempt):
    import numpy as np,torch
    from research.general_rsi_transfer_worker_v1 import load_model
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    assert torch.__version__==plan['torch_version'] and torch.backends.mps.is_available()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0';torch.set_num_threads(4)
    values=torch.from_numpy(np.fromfile(ROOT/'data/raw/rsi-general-10m-v1/train.bin',dtype='<u2').astype('int64'))
    batches=[(values[w['start']:w['start']+128][None,:],values[w['start']+1:w['start']+129][None,:]) for w in plan['windows']]
    assert batch_hash(*(t for b in batches for t in b))==plan['evaluation_sha256']
    peak=0;results=[];tick=time.monotonic();qualified=False
    def boundary():
        nonlocal peak
        check_stop();torch.mps.synchronize();peak=max(peak,torch.mps.driver_allocated_memory());assert peak<plan['mps_limit_bytes']
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<plan['metadata_cap_bytes']
    for m in plan['models']:
        boundary();loaded=time.monotonic();model=load_model(m['parent'],None,plan);model.eval();load_seconds=time.monotonic()-loaded
        assert model.native.lm_head.bias is None
        if not qualified:
            e=m['endpoints'][0];v=payload(e['weights']);original=v['parameters'];assert digest(original)==e['parameters_sha256']
            q=time.monotonic();apply(model,original,e['factors'],'original');x=batches[0][0].to('mps')
            with torch.no_grad(),FP32RMSNormInference():
                a=model.primary_logits(x).values.detach().cpu()
                apply(model,original,e['factors'],'readout');b=model.primary_logits(x).values.detach().cpu();expected=a*e['factors']['readout']
            assert torch.allclose(b,expected,rtol=plan['qualification']['logit_scaling_rtol'],atol=plan['qualification']['logit_scaling_atol'])
            maxerr=float((b-expected).abs().max());argmax_changes=int((b.argmax(-1)!=a.argmax(-1)).sum())
            for condition in ['other','both','original']:apply(model,original,e['factors'],condition)
            qualification_seconds=time.monotonic()-q;del a,b,expected,v,original,x;gc.collect()
            # Reload exact parent after the preflight. No score has been computed.
            pv=payload(m['parent']);apply(model,pv['parameters'],{'readout':1.,'other':1.},'original');del pv
            free=shutil.disk_usage(ROOT).free;assert free>=plan['required_free_bytes']
            forecast=27*8192/100416*22.064+max(90.,9*load_seconds)+max(30.,6*qualification_seconds)+20
            assert forecast<plan['total_seconds']-(time.monotonic()-tick)
            put(attempt/'qualification.json',dict(passed=True,logit_scaling_max_abs_error=maxerr,argmax_changes=argmax_changes,seconds=qualification_seconds,exact_restore_and_application=True))
            put(attempt/'admission.json',dict(plan_sha256=sha(OUT/'plan.json'),free_bytes=free,required_free_bytes=plan['required_free_bytes'],forecast_seconds=forecast,
                measured_parent_load_seconds=load_seconds,measured_transformation_check_seconds=qualification_seconds,scoring_basis='Prior100416-target native inference22.064s;27*8192 targets.',
                reserve_bytes=plan['reserve_bytes'],extra_headroom_bytes=2*G,retained_bytes=plan['metadata_cap_bytes'],overlap_bytes=plan['overlap_bytes'],serialization_bytes=0,admitted_before_scores=True))
            qualified=True
        baseline=evaluate(model,batches);boundary();put(attempt/f"p{m['parent']['slot']:02d}-parent.json",baseline)
        for e in m['endpoints']:
            v=payload(e['weights']);assert v['model_identity']==model.identity and v['immutable_buffers']==model._own_expected_buffer_hashes
            original=v['parameters'];assert digest(original)==e['parameters_sha256'];scores={}
            for condition in plan['conditions']:
                boundary();apply(model,original,e['factors'],condition);start=time.monotonic();score=evaluate(model,batches);boundary()
                row=dict(parent=m['parent']['slot'],endpoint=e['endpoint'],condition=condition,score=score,score_seconds=time.monotonic()-start)
                put(attempt/f"p{m['parent']['slot']:02d}-{e['endpoint']}-{condition}.json",row);scores[condition]=score['ce']
                print(json.dumps({k:v for k,v in row.items() if k!='score'}|{'ce':score['ce']}),flush=True)
            assert digest(original)==e['parameters_sha256']
            results.append(dict(parent=m['parent']['slot'],endpoint=e['endpoint'],parent_ce=baseline['ce'],scores=scores,
                effects={c:scores[c]-scores['original'] for c in CONDITIONS[1:]},interaction=scores['both']-scores['readout']-scores['other']+scores['original'],
                residuals={c:scores[c]-baseline['ce'] for c in CONDITIONS}))
            del original,v;gc.collect()
        del model;gc.collect();torch.mps.empty_cache()
    assert len(results)==6
    return dict(rows=results,total_seconds=time.monotonic()-tick,peak_mps_bytes=peak,conditions_scored=27,test_opened=False,validation_scored=False,training_updates=0,new_checkpoints=0)

def main():
    os.chdir(ROOT);lock=open('research_state/tokenizer-worker-lock/.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop()
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    if sys.argv[1]=='prepare':
        assert shutil.disk_usage(ROOT).free>=12*G+128*1024**2
        prepare();return
    assert sys.argv[1]=='run';plan=read(OUT/'plan.json');assert read(OUT/'preparation.json')['plan_sha256']==sha(OUT/'plan.json')
    assert all(sha(ROOT/p)==h for p,h in plan['sources'].items());assert shutil.disk_usage(ROOT).free>=plan['required_free_bytes']
    attempt=OUT/'attempt-01';attempt.mkdir()
    def stop(*_):
        import research.general_rsi_pseudo_update_diagnostic_v1 as shared
        shared.STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with CpuPrerequisiteGuard(attempt/'guard.json',ROOT,timeout_seconds=plan['total_seconds'],rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):
            result=execute(plan,attempt);assert all(sha(ROOT/p)==h for p,h in plan['sources'].items());put(attempt/'result.json',result)
    except BaseException:
        put(attempt/'failure.json',dict(traceback=traceback.format_exc(),plan_sha256=sha(OUT/'plan.json')));raise
if __name__=='__main__':main()
