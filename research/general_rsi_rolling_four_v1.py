"""Add fixed older-token shuffles without touching the active rolling scorer."""
import argparse, datetime, fcntl, gc, hashlib, json, math, os, shutil, signal, time, traceback
from pathlib import Path
from research import general_rsi_rolling_context_v1 as base
ROOT, BC, PANEL = base.ROOT, base.OUT, base.PANEL
OUT=ROOT/'research/tokenizer-evidence/general-rsi-rolling-four-001'
DESIGN=ROOT/'research/GENERAL_RSI_ROLLING_FOUR_CONDITION_DESIGN.md'
read,put,sha=base.read,base.put,base.sha


def permutation(split,doc,p):
    import numpy as np
    key=f'rolling-older-shuffle-v1|621116220001|{split}|{doc}|{p}'.encode()
    seed=int.from_bytes(hashlib.sha256(key).digest()[:16],'big')
    return np.random.Generator(np.random.PCG64(seed)).permutation(128).astype(np.uint8)


def shuffled(values,start,positions,split,doc):
    import torch
    x,y=base.windows(values,start,positions,256);original=x.clone();perms=[]
    for i,p in enumerate(positions):
        perm=permutation(split,doc,p);perms.append(perm.tobytes())
        x[i,:128]=original[i,:128][torch.from_numpy(perm.astype('int64'))]
    assert torch.equal(x[:,128:],original[:,128:])
    assert torch.equal(torch.sort(x[:,:128],dim=1).values,torch.sort(original[:,:128],dim=1).values)
    return x,y,b''.join(perms)


def prepare():
    import numpy as np
    assert not OUT.exists();plan=read(BC/'plan.json');review=read(PANEL/'completion-review.json')
    docs=[json.loads(x) for x in (ROOT/'data/raw/rsi-general-10m-v1/val.index.jsonl').read_text().splitlines()]
    sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),DESIGN,BC/'plan.json',BC/'admission.json',BC/'qualification-01/result.json',PANEL/'completion-review.json',ROOT/'research/general_rsi_rolling_four_analysis_v1.py',ROOT/'research/general_rsi_rolling_four_queue_v1.py']}
    saved_a=[]
    for e in plan['endpoints']:
        path=PANEL/f"slot-{e['slot']:02d}"/'time_ema-common128-validation.json';name=str(path.relative_to(ROOT));digest=sha(path)
        assert digest==review['evidence_sha256'][name];a=read(path);positions=[]
        assert a['context']==128
        for chunk in a['chunks']:
            doc=docs[chunk['document']]
            assert chunk['source']==doc['source'] and chunk['targets']==len(chunk['target_ids'])==len(chunk['token_losses'])
            offset=chunk['target_start']-doc['offset']-1;assert offset%128==0
            assert chunk['targets']==min(128,doc['tokens']-1-offset)
            for j in range(chunk['targets']):
                p=chunk['target_start']+j-doc['offset']
                if p>=256:positions.append((chunk['document'],p))
        expected=[(i,p) for i,d in enumerate(docs) for p in range(256,d['tokens'])]
        assert positions==expected and len(positions)==72876
        sources[name]=digest;saved_a.append(dict(depth=e['depth'],bundle=e['bundle'],path=name,sha256=digest,matched_targets=len(positions)))
    q=read(BC/'qualification-01/result.json')
    forecast=600+1.5*3*72876*sum(m['seconds']['256']/128 for m in q['measurements'])
    p=dict(kind='four_condition_saved_model_extension_v1',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           sources=sources,base_plan_sha256=sha(BC/'plan.json'),endpoints=plan['endpoints'],saved_a=saved_a,
           target_count=72876,document_count=108,numpy_version=np.__version__,shuffle_seed_label=621116220001,
           bootstrap_seed=621116220002,bootstrap_resamples=100000,interval_quantiles=[.01/24,1-.01/24],
           contrasts=['B_minus_A','C_minus_B','C_minus_D','D_minus_B'],family_contrasts=12,
           max_seconds=43200,qualification_seconds=600,reserve_bytes=12*2**30,
           combined_allowance_bytes=512*2**20,provisional_forecast_seconds=forecast,test_opened=False,training=False)
    assert forecast<43200 and shutil.disk_usage(ROOT).free>=p['reserve_bytes']+p['combined_allowance_bytes']
    OUT.mkdir();put(OUT/'plan.json',p)
    print(json.dumps(dict(saved_a_coverage_verified=True,targets=72876,provisional_seconds=forecast)))


def bindings(plan):
    import numpy as np
    assert np.__version__==plan['numpy_version'] and sha(BC/'plan.json')==plan['base_plan_sha256']
    base.verify_bindings(read(BC/'plan.json'))
    for name,digest in plan['sources'].items():assert sha(ROOT/name)==digest,name


def qualify(plan,out):
    import torch
    import numpy as np
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    toy=torch.arange(1600,dtype=torch.int64)
    positions=[256,257,383,511]
    x,y,perms=shuffled(toy,37,positions,'train',0)
    for i,p in enumerate(positions):
        xx,yy,pp=shuffled(toy,37,[p],'train',0)
        assert torch.equal(x[i],xx[0]) and y[i]==yy[0] and perms[i*128:(i+1)*128]==pp
        assert sorted(pp)==list(range(128))
    # Identity permutation must reproduce C without any transformation.
    original,_=base.windows(toy,37,positions,256)
    identity=torch.cat([original[:,:128][:,torch.arange(128)],original[:,128:]],dim=1)
    assert torch.equal(original,identity)
    values,docs=base.data('train');doc=next(i for i,d in enumerate(docs) if d['tokens']>=640);row=docs[doc]
    measured=[]
    for depth in (4,16):
        e=next(e for e in plan['endpoints'] if e['depth']==depth and e['bundle']==1)
        m=base.load(e,out);before=base.parameter_digest(m)
        with torch.no_grad(),FP32RMSNormInference():
            x,y,_=shuffled(values,row['offset'],positions,'train',doc);x=x.to('mps')
            z=base.last_logits(m,x);ref=m.primary_logits(x).values[:,-1,:];error=0.
            torch.testing.assert_close(z,ref,atol=3e-5,rtol=3e-4)
            for i in range(4):
                scalar=m.primary_logits(x[i:i+1]).values[0,-1,:]
                torch.testing.assert_close(z[i],scalar,atol=3e-5,rtol=3e-4)
                error=max(error,(z[i]-scalar).abs().max().item())
            loss=torch.nn.functional.cross_entropy(z,y.to('mps'),reduction='none').cpu().double()
            zz=ref.cpu().double();oracle=torch.logsumexp(zz,-1)-zz.gather(1,y[:,None]).squeeze(1)
            ceerr=(loss-oracle).abs().max().item();assert ceerr<1e-5
            torch.mps.synchronize();start=time.monotonic()
            for pos in range(256,384,4):
                base.check_stop();xx,yy,_=shuffled(values,row['offset'],list(range(pos,pos+4)),'train',doc)
                ll=torch.nn.functional.cross_entropy(base.last_logits(m,xx.to('mps')),yy.to('mps'),reduction='none').cpu()
                assert torch.isfinite(ll).all() and torch.mps.driver_allocated_memory()<6*2**30
            torch.mps.synchronize();seconds=time.monotonic()-start
        assert before==base.parameter_digest(m);m.validate();base.checkpoint_memory(out,f'qualified_depth{depth}')
        measured.append(dict(depth=depth,targets=128,seconds=seconds,logits_max_abs_error=error,ce_max_abs_error=ceerr,parameters_unchanged=True))
        del m,x,y,z,ref,scalar,loss,zz,oracle,xx,yy,ll;gc.collect();torch.mps.empty_cache()
    forecast=600+1.5*3*72876*sum(r['seconds']/128 for r in measured)
    result=dict(accepted=forecast<plan['max_seconds'],measurements=measured,forecast_seconds=forecast,
                structural_checks_passed=True,identity_permutation_input_exact=True,validation_opened=False,test_opened=False,plan_sha256=sha(OUT/'plan.json'))
    put(out/'result.json',result);base.checkpoint_memory(out,'serialized');assert result['accepted']


def score(plan,out):
    import torch
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    admission=read(OUT/'admission.json');assert admission['accepted'] and admission['plan_sha256']==sha(OUT/'plan.json')
    values,docs=base.data('val');start=time.monotonic()
    for e in plan['endpoints']:
        folder=out/f"d{e['depth']}-b{e['bundle']}";folder.mkdir();m=base.load(e,out);before=base.parameter_digest(m);nt=0
        with torch.no_grad(),FP32RMSNormInference():
            for doc,row in enumerate(docs):
                base.check_stop();m.validate();positions=list(range(256,row['tokens']));losses=[];ids=[]
                ph=hashlib.sha256();ih=hashlib.sha256()
                for j in range(0,len(positions),4):
                    base.check_stop();x,y,permbytes=shuffled(values,row['offset'],positions[j:j+4],'val',doc)
                    ph.update(permbytes);ih.update(x.numpy().tobytes())
                    z=base.last_logits(m,x.to('mps'));ls=torch.nn.functional.cross_entropy(z,y.to('mps'),reduction='none').cpu().double().tolist()
                    assert all(math.isfinite(v) for v in ls) and torch.mps.driver_allocated_memory()<6*2**30
                    losses.extend(ls);ids.extend(y.tolist())
                assert len(losses)==len(ids)==len(positions)
                put(folder/f'doc-{doc:03d}.json',dict(document=doc,source=row['source'],offset=row['offset'],positions=positions,target_ids=ids,
                    losses=losses,loss_sum=math.fsum(losses),count=len(positions),permutations_sha256=ph.hexdigest(),inputs_sha256=ih.hexdigest()))
                nt+=len(positions)
                if doc%10==0 or doc==107:base.checkpoint_memory(out,f"d{e['depth']}_b{e['bundle']}_doc{doc}")
                base.append(out/'progress.jsonl',dict(depth=e['depth'],bundle=e['bundle'],document=doc,targets=nt,elapsed_seconds=time.monotonic()-start))
        assert nt==72876 and before==base.parameter_digest(m);m.validate()
        put(folder/'result.json',dict(completed=True,endpoint=e,targets=nt,parameters_unchanged=True,files={p.name:sha(p) for p in sorted(folder.glob('doc-*.json'))}))
        del m,x,y,z;gc.collect();torch.mps.empty_cache();base.checkpoint_memory(out,'model_released')
    bindings(plan);put(out/'result.json',dict(completed=True,models=6,condition='D',targets_per_model=72876,seconds=time.monotonic()-start,plan_sha256=sha(OUT/'plan.json'),test_opened=False,training_updates=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','qualify','score']);args=parser.parse_args()
    signal.signal(signal.SIGTERM,base.on_stop);signal.signal(signal.SIGINT,base.on_stop)
    if args.mode=='prepare':prepare();return
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.check_stop()
        plan=read(OUT/'plan.json');bindings(plan);out=OUT/('qualification-01' if args.mode=='qualify' else 'scoring-01');out.mkdir()
        from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
        with CpuPrerequisiteGuard(out/'guard.json',ROOT,rss_limit_bytes=4*2**30,reserve_bytes=12*2**30,
                                  timeout_seconds=plan['qualification_seconds'] if args.mode=='qualify' else plan['max_seconds'],poll_seconds=.2):
            try:
                h=base.host();assert int(h['pressure'])==1 and h['reclaimable_proxy_bytes']>=2**30+256*2**20
                assert shutil.disk_usage(ROOT).free>=plan['reserve_bytes']+plan['combined_allowance_bytes']
                put(out/'launch.json',dict(pid=os.getpid(),host=h,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(OUT/'plan.json')))
                import torch
                assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0' and torch.backends.mps.is_available();torch.set_num_threads(4)
                assert torch.__version__==read(BC/'plan.json')['torch_version']
                if args.mode=='qualify':qualify(plan,out)
                else:score(plan,out)
            except BaseException as exc:
                put(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
