"""New active-EMA, chunk-coverage and fixed-generation integration checks."""
import fcntl,json,os,shutil,signal,time,traceback,gc,math
from types import SimpleNamespace
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,check_stop,digest
OUT=ROOT/'research/tokenizer-evidence/general-rsi-context-length-integration-001'
RUN=ROOT/'research_runs/general-rsi-context-length-integration-001'

def execute(p,binding,a):
    import torch
    from torch.nn import functional as F
    from research.general_rsi_context_length_quality_v1 import Context,load_inference
    from research.general_rsi_context_length_evaluation_v1 import chunks,evaluate
    from research.general_rsi_context_length_generation_v1 import generate
    from research.general_rsi_training_v2 import write_checkpoint,read_checkpoint
    from research.general_rsi_data_v1 import GeneralData
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    assert torch.__version__==p['torch_version'] and os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0';torch.set_num_threads(4);RUN.mkdir()
    peak=0
    def memory():
        nonlocal peak
        peak=max(peak,torch.mps.driver_allocated_memory());assert peak<6*2**30
    def log(name,row):
        with open(a/name,'a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    # Synthetic coverage and original128 chunk adapter, with no validation data read.
    docs=[];numbers=[]
    for size in p['synthetic_lengths']:
        docs.append(dict(offset=len(numbers),tokens=size,source='synthetic'));numbers.extend(list(range(size-1))+[50256])
    values=torch.tensor(numbers);expected=[i for d in docs for i in range(d['offset']+1,d['offset']+d['tokens'])]
    for c in (128,256):
        rows=list(chunks(values,docs,c));positions=[i for _,_,start,_,y in rows for i in range(start,start+y.numel())];assert positions==expected and len(set(positions))==len(expected)
        for doc,_,start,x,y in rows:assert torch.equal(y.flatten(),values[start:start+y.numel()]) and int(values[docs[doc]['offset']+docs[doc]['tokens']-1])==50256
    old=GeneralData.__new__(GeneralData);old.view=SimpleNamespace(val_data=values);old.indexes={'val':docs}
    oldrows=list(old.validation_chunks());newrows=list(chunks(values,docs,128));assert len(oldrows)==len(newrows)
    for (doc,src,x,y),(dd,ss,_,xx,yy) in zip(oldrows,newrows):assert (doc,src)==(dd,ss) and torch.equal(x,xx) and torch.equal(y,yy)
    put(a/'coverage.json',dict(document_lengths=p['synthetic_lengths'],target_positions=len(expected),both_lengths_exact=True,original128_adapter_exact=True))
    ctx=Context(256,p['budget_ns'],p['half_life_ns'],p['roles'],binding)
    while ctx.clock.spent_ns<p['checkpoint_after_ns'] or ctx.snapshots<2:
        check_stop();assert ctx.step<p['max_updates'] and ctx.clock.spent_ns<ctx.clock.budget_ns;row=ctx.update();log('updates.jsonl',row);memory()
    assert ctx.snapshots>=2 and ctx.clock.spent_ns<ctx.clock.budget_ns and ctx.history[-1]['lr_ratio']<1
    tick=time.monotonic();payload=ctx.capture();before=digest(payload);receipt=write_checkpoint(RUN/'active-ema-restart.pt',payload);loaded=read_checkpoint(receipt);assert digest(loaded)==before;del loaded,payload
    restart_io=time.monotonic()-tick;step=ctx.step;snapshots=ctx.snapshots
    check_stop();refrow=ctx.update();log('reference-next.jsonl',refrow);duration=refrow['end_ns']-refrow['begin_ns'];reference=ctx.capture();refdigest=digest(reference);refreceipt=write_checkpoint(RUN/'active-ema-reference.pt',reference);del reference,ctx;gc.collect();torch.mps.empty_cache()
    ctx=Context(256,p['budget_ns'],p['half_life_ns'],p['roles'],binding);payload=read_checkpoint(receipt);ctx.restore(payload);assert digest(ctx.capture())==before;del payload
    ticks=iter([0,duration]);row=ctx.update(now_ns=lambda:next(ticks));log('replayed-next.jsonl',row);assert digest(ctx.capture())==refdigest;memory()
    receipts={endpoint:ctx.export(RUN/f'{endpoint}.pt',endpoint) for endpoint in ('raw','time_ema')}
    # Both endpoint receipts and a wrong-context load, with no additional training.
    before=digest(ctx.capture())
    try:load_inference(receipts['raw'],128,binding)
    except AssertionError:pass
    else:raise AssertionError('Wrong-context inference accepted')
    assert digest(ctx.capture())==before
    for endpoint in ('raw','time_ema'):
        model=load_inference(receipts[endpoint],256,binding);source=dict(ctx.model.ordered_parameters()) if endpoint=='raw' else ctx.ema
        assert all(torch.equal(q,source[n]) for n,q in model.ordered_parameters());memory()
        if endpoint=='raw':del model;gc.collect();torch.mps.empty_cache()
    assert digest(ctx.capture())==before
    # One predeclared training document, including EOT, no scientific validation.
    document=p['training_document'];values=ctx.data.view.train_data;eval_start=time.monotonic();scoring={}
    for c in (128,256):
        result=evaluate(model,values,[document],c);put(a/f'train-document-c{c}.json',result);scoring[c]=result
    assert [n for r in scoring[128]['rows'] for n in r['target_ids']]==[n for r in scoring[256]['rows'] for n in r['target_ids']]
    # Independent explicit reference slicing and FP64 CPU logsumexp CE.
    model.eval();maxerr=0.
    with torch.no_grad(),FP32RMSNormInference():
        for c in (128,256):
            refs=[]
            for offset in range(0,document['tokens']-1,c):
                start=document['offset']+offset;end=min(start+c,document['offset']+document['tokens']-1)
                z=model.primary_logits(values[start:end][None,:].to('mps')).values[0].cpu().double();targets=values[start+1:end+1]
                losses=torch.logsumexp(z,dim=-1)-z.gather(1,targets[:,None]).squeeze(1);refs.extend(losses.tolist())
            actual=[v for r in scoring[c]['rows'] for v in r['token_losses']];err=max(abs(x-y) for x,y in zip(refs,actual));maxerr=max(maxerr,err);assert err<=p['ce_atol']
    model.train();assert digest(ctx.capture())==before;eval_seconds=time.monotonic()-eval_start
    model_before=digest({n:q.detach().cpu().clone() for n,q in model.ordered_parameters()});tick=time.monotonic()
    rows=generate(model,p['generation_protocol'],a/'generations',receipts['time_ema'],binding);generation_seconds=time.monotonic()-tick
    assert model_before==digest({n:q.detach().cpu().clone() for n,q in model.ordered_parameters()}) and digest(ctx.capture())==before;model.validate();memory()
    retained=sum(q.stat().st_size for q in RUN.iterdir());assert retained<=2*512*2**20+4*128*2**20
    return dict(plan_sha256=binding,accepted=True,checkpoint_step=step,checkpoint_snapshots=snapshots,final_snapshots=ctx.snapshots,pilot_optimizer_updates=step+2,original128_chunk_adapter_exact=True,synthetic_targets=len(expected),train_document_targets=document['tokens']-1,per_token_ce_max_abs_vs_fp64=maxerr,full_active_ema_replay_exact=True,inference_parameters_exact=176,wrong_context_inference_rejected=True,state_unchanged_after_evaluation_generation=True,first_outputs=len(rows),generated_tokens=sum(len(r['generated_token_ids']) for r in rows),generation_seconds=generation_seconds,training_document_evaluation_and_oracle_seconds=eval_seconds,restart_io_seconds=restart_io,restart_receipts=[receipt,refreceipt],endpoint_receipts=receipts,retained_bytes=retained,peak_mps_bytes=peak,validation_opened=False,test_opened=False,quality_adopted=False)

def main():
    with open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop()
        p=json.loads((OUT/'plan.json').read_text());binding=sha(OUT/'plan.json');assert all(sha(ROOT/f)==h for f,h in p['sources'].items());assert shutil.disk_usage(ROOT).free>=p['required_free_bytes']
        a=OUT/'attempt-01';a.mkdir();put(a/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=binding))
        def stop(*_):
            import research.general_rsi_pseudo_update_diagnostic_v1 as shared
            shared.STOP=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
        try:
            with CpuPrerequisiteGuard(a/'guard.json',ROOT,timeout_seconds=600,rss_limit_bytes=4*2**30,reserve_bytes=10*2**30):
                result=execute(p,binding,a);assert all(sha(ROOT/f)==h for f,h in p['sources'].items());put(a/'result.json',result);print(json.dumps(result),flush=True)
        except BaseException:
            put(a/'failure.json',dict(traceback=traceback.format_exc(),plan_sha256=binding));raise
if __name__=='__main__':main()
