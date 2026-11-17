"""Fixed saved weights, explicit direct-RoPE extension, matched rolling histories."""
import argparse, contextlib, datetime, fcntl, gc, json, math, os, signal, time, traceback
from pathlib import Path
import torch
from research import general_rsi_rolling_context_v1 as base
from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
from research.general_rsi_depth_context_training_v1 import load_inference
from research.general_rsi_context_length_v1.runtime.models.rope import precompute_freqs_cis
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT, PANEL, BC = base.ROOT, base.PANEL, base.OUT
OUT = ROOT/'research/tokenizer-evidence/general-rsi-train-serve-context-001'
DESIGN = ROOT/'research/GENERAL_RSI_TRAIN_SERVE_CONTEXT_DESIGN.md'
read, put, sha = base.read, base.put, base.sha
HISTORIES = (128,256,512)


class ExtendedView:
    """Scoped deterministic buffer extension; original identities remain unmodified."""
    def __init__(self, model):
        self.model=model; self.saved=[]; self.expected={}

    def __enter__(self):
        m=self.model; m.validate(); assert not m.training
        try:
            for block in m.blocks:
                assert block.enable_rope and block.attention_family=='mla' and block.attn.causal
                cos,sin=precompute_freqs_cis(block.head_dim_r,512)
                theta=10000.**(-torch.arange(0,block.head_dim_r,2,dtype=torch.float64)/block.head_dim_r)
                angles=torch.arange(512,dtype=torch.float64)[:,None]*theta
                assert (cos.double()-angles.cos()).abs().max()<3e-5
                assert (sin.double()-angles.sin()).abs().max()<3e-5
                for owner,name,new in [(block,'freq_cos',cos),(block,'freq_sin',sin),
                                       (block.attn,'causal_mask',torch.tril(torch.ones(512,512,dtype=torch.bool)))]:
                    old=getattr(owner,name)
                    prefix=new[:old.shape[0]] if new.ndim==2 and name!='causal_mask' else new[:old.shape[0],:old.shape[1]]
                    assert torch.equal(prefix,old.cpu())
                    self.saved.append((owner,name,old))
                    new=new.to(old.device); setattr(owner,name,new)
                    self.expected[id(new)]=base.sha_tensor(new) if hasattr(base,'sha_tensor') else m._buffer_digest(new)
            self.validate(); return self
        except BaseException:
            self.__exit__(None,None,None); raise

    def validate(self):
        for owner,name,_ in self.saved:
            value=getattr(owner,name)
            assert id(value) in self.expected and self.model._buffer_digest(value)==self.expected[id(value)]

    def __exit__(self,*_):
        for owner,name,old in reversed(self.saved): setattr(owner,name,old)
        self.model.validate()

    def hidden(self,x):
        assert not torch.is_grad_enabled() and not self.model.training
        assert x.dtype==torch.int64 and x.ndim==2 and 1<=x.shape[0]<=2
        assert x.shape[1] in HISTORIES and x.numel()<=1024
        assert x.device==next(self.model.parameters()).device and bool(((x>=0)&(x<50257)).all())
        return self.model._hidden(x.contiguous())

    def logits(self,x):
        return self.model.readout(self.hidden(x)[:,-1,:])/4.


def windows(values,row,positions,h):
    assert h in HISTORIES and 1<=len(positions)<=2 and all(512<=p<row['tokens'] for p in positions)
    a=row['offset']; x=torch.stack([values[a+p-h:a+p] for p in positions]);y=values[[a+p for p in positions]]
    assert x.shape==(len(positions),h)
    return x,y


def name(e): return f"d{e['depth']}-b{e['bundle']}-t{e['train_context']}"


def prepare():
    assert not OUT.exists()
    old=read(BC/'plan.json'); panel=read(PANEL/'plan.json'); assert read(PANEL/'completion-review.json')['accepted']
    assert read(ROOT/'research/tokenizer-evidence/general-rsi-rolling-four-001/completion-review.json')['accepted']
    sources=dict(old['sources'])
    for p in [Path(__file__),DESIGN,BC/'plan.json',BC/'analysis.json',
              ROOT/'research/general_rsi_train_serve_analysis_v1.py',ROOT/'research/general_rsi_train_serve_supervisor_v1.py',
              ROOT/'research/tokenizer-evidence/general-rsi-rolling-four-001/analysis.json',
              ROOT/'research/tokenizer-evidence/general-rsi-rolling-four-001/completion-review.json']:
        sources[str(p.relative_to(ROOT))]=sha(p)
    endpoints=[];reused={}
    for depth in (4,16):
        for bundle in (1,2,3):
            for training in (128,256):
                slot,=[s for s in panel['slots'] if (s['depth'],s['bundle'],s['context'])==(depth,bundle,training)]
                result=read(PANEL/f"slot-{slot['slot']:02d}"/'result.json')
                assert result['status']=='completed' and result['binding']==sha(PANEL/'plan.json')
                endpoint,=[e for e in result['endpoints'] if e['endpoint']=='time_ema']
                rec=endpoint['weights'];assert sha(rec['path'])==rec['sha256']
                endpoints.append(dict(depth=depth,bundle=bundle,train_context=training,slot=slot['slot'],receipt=rec))
                sources[str((PANEL/f"slot-{slot['slot']:02d}"/'result.json').relative_to(ROOT))]=sha(PANEL/f"slot-{slot['slot']:02d}"/'result.json')
                if training==256:
                    folder=BC/'scoring-01'/f'd{depth}-b{bundle}';rr=read(folder/'result.json')
                    assert rr['parameters_unchanged']
                    for filename,digest in rr['files'].items():
                        path=folder/filename;assert sha(path)==digest;reused[str(path.relative_to(ROOT))]=digest
                    sources[str((folder/'result.json').relative_to(ROOT))]=sha(folder/'result.json')
    docs=[json.loads(x) for x in (ROOT/'data/raw/rsi-general-10m-v1/val.index.jsonl').read_text().splitlines()]
    assert sum(max(0,d['tokens']-512) for d in docs)==51457
    p=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,reused_scores=reused,endpoints=endpoints,
           target_count=51457,document_count=65,minimum_position=512,histories=list(HISTORIES),batch_size=2,
           bootstrap_seed=621117310001,bootstrap_resamples=100000,family_contrasts=36,interval_quantiles=[.01/72,1-.01/72],
           torch_version=torch.__version__,reserve_bytes=12*2**30,allowance_bytes=512*2**20,
           qualification_seconds=900,scoring_seconds=43200,analysis_seconds=600,training_updates=0,test_opened=False)
    OUT.mkdir();put(OUT/'plan.json',p)


def verify(plan):
    assert torch.__version__==plan['torch_version']
    for path,digest in {**plan['sources'],**plan['reused_scores']}.items():assert sha(ROOT/path)==digest,path
    for e in plan['endpoints']:assert sha(e['receipt']['path'])==e['receipt']['sha256']


def load(e,out):
    base.check_stop();h=base.checkpoint_memory(out,'before_load')
    assert int(h['host']['pressure'])==1 and h['host']['reclaimable_proxy_bytes']>=1.25*2**30
    m=load_inference(e['receipt'],e['depth'],e['train_context'],sha(PANEL/'plan.json'));m.eval();m.validate()
    base.checkpoint_memory(out,'loaded');return m


def qualify(plan,out):
    values,docs=base.data('train');row=next(d for d in docs if d['tokens']>=1024)
    toy=torch.arange(1600,dtype=torch.int64)
    for h in HISTORIES:
        x,y=windows(toy,dict(offset=37,tokens=1400),[512,777],h)
        for i,p in enumerate([512,777]): assert x[i].tolist()==list(range(37+p-h,37+p)) and y[i]==37+p
    measurements=[]
    for e in [e for e in plan['endpoints'] if e['bundle']==1]:
        m=load(e,out);before=base.parameter_digest(m);references={};errors=[];ce_errors=[];seconds={}
        with torch.no_grad(),FP32RMSNormInference():
            for h in HISTORIES:
                if h<=e['train_context']:
                    x,y=windows(values,row,[512,513],h)
                    # Native batch4 is the original saved B/C scorer's batching policy.
                    xx=torch.cat([x,x],dim=0).to('mps')
                    references[h]=m.primary_logits(xx).values[:2,-1,:].detach().cpu();del xx
            with ExtendedView(m) as view:
                base.checkpoint_memory(out,'buffers_extended')
                for h in HISTORIES:
                    x,y=windows(values,row,[512,513],h);x=x.to('mps');z=view.logits(x)
                    full=m.readout(view.hidden(x))/4.;ref=full[:,-1,:]
                    torch.testing.assert_close(z,ref,atol=3e-5,rtol=3e-4)
                    if h in references:torch.testing.assert_close(z.cpu(),references[h],atol=3e-5,rtol=3e-4)
                    for i in range(2):
                        scalar=(m.readout(view.hidden(x[i:i+1]))/4.)[0,-1,:]
                        torch.testing.assert_close(z[i],scalar,atol=3e-5,rtol=3e-4)
                        errors.append((z[i]-scalar).abs().max().item())
                    loss=torch.nn.functional.cross_entropy(z,y.to('mps'),reduction='none').cpu().double()
                    zz=ref.cpu().double();oracle=torch.logsumexp(zz,-1)-zz.gather(1,y[:,None]).squeeze(1)
                    ce_errors.append((loss-oracle).abs().max().item());assert max(ce_errors)<1e-5
                    if h==512:
                        # Future tokens cannot change representations in the first128 positions.
                        long_hidden=view.hidden(x[:1])[:,:128,:];short_hidden=view.hidden(x[:1,:128])
                        torch.testing.assert_close(long_hidden,short_hidden,atol=3e-5,rtol=3e-4)
                        del long_hidden,short_hidden
                    del full,ref,scalar,z,zz,oracle,loss
                    torch.mps.synchronize();t=time.monotonic()
                    for p in range(512,640,2):
                        base.check_stop();x,y=windows(values,row,[p,p+1],h);z=view.logits(x.to('mps'))
                        loss=torch.nn.functional.cross_entropy(z,y.to('mps'),reduction='none').cpu()
                        assert torch.isfinite(loss).all() and torch.mps.driver_allocated_memory()<6*2**30
                    torch.mps.synchronize();seconds[h]=time.monotonic()-t
                    base.checkpoint_memory(out,f"qualified_{name(e)}_h{h}")
                    del x,y,z,loss
                view.validate()
        assert before==base.parameter_digest(m);m.validate()
        measurements.append(dict(depth=e['depth'],train_context=e['train_context'],targets=128,seconds=seconds,
                                 max_logit_error=max(errors),max_ce_error=max(ce_errors),native_parity=True,causal_prefix=True,parameters_and_original_buffers_unchanged=True))
        del m,references;gc.collect();torch.mps.empty_cache();base.checkpoint_memory(out,'released')
    forecast=600+1.5*3*51457/128*sum(sum(r['seconds'][h] for h in HISTORIES if r['train_context']==128 or h==512) for r in measurements)
    put(out/'result.json',dict(accepted=forecast<43200,forecast_seconds=forecast,measurements=measurements,plan_sha256=sha(OUT/'plan.json'),validation_scored=False))
    assert forecast<43200


def score(plan,out):
    assert read(OUT/'admission.json')['accepted'];values,docs=base.data('val');started=time.monotonic()
    for e in plan['endpoints']:
        folder=out/name(e);folder.mkdir();m=load(e,out);before=base.parameter_digest(m);total=0;elapsed={h:0. for h in HISTORIES}
        hs=HISTORIES if e['train_context']==128 else (512,)
        with torch.no_grad(),FP32RMSNormInference(),ExtendedView(m) as view:
            base.checkpoint_memory(out,'extended')
            for doc,row in enumerate(docs):
                positions=list(range(512,row['tokens']))
                if not positions:continue
                base.check_stop();view.validate();losses={h:[] for h in HISTORIES};target_ids=values[row['offset']+512:row['offset']+row['tokens']].tolist();reuse=None
                if e['train_context']==256:
                    path=BC/'scoring-01'/f"d{e['depth']}-b{e['bundle']}"/f'doc-{doc:03d}.json'
                    assert sha(path)==plan['reused_scores'][str(path.relative_to(ROOT))];old=read(path)
                    ix=[i for i,p in enumerate(old['positions']) if p>=512]
                    assert [old['positions'][i] for i in ix]==positions and [old['target_ids'][i] for i in ix]==target_ids
                    for h in (128,256):losses[h]=[old['losses'][str(h)][i] for i in ix]
                    reuse=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
                for h in hs:
                    torch.mps.synchronize();t=time.monotonic()
                    for j in range(0,len(positions),2):
                        base.check_stop();x,y=windows(values,row,positions[j:j+2],h);z=view.logits(x.to('mps'))
                        ls=torch.nn.functional.cross_entropy(z,y.to('mps'),reduction='none').cpu().double().tolist()
                        assert all(math.isfinite(v) for v in ls);losses[h].extend(ls)
                        assert torch.mps.driver_allocated_memory()<6*2**30
                    torch.mps.synchronize();elapsed[h]+=time.monotonic()-t
                assert all(len(v)==len(positions) for v in losses.values())
                put(folder/f'doc-{doc:03d}.json',dict(document=doc,source=row['source'],offset=row['offset'],positions=positions,target_ids=target_ids,
                     count=len(positions),losses=losses,loss_sums={h:math.fsum(v) for h,v in losses.items()},reuse=reuse))
                total+=len(positions);base.checkpoint_memory(out,f'{name(e)}_doc{doc}')
                base.append(out/'progress.jsonl',dict(endpoint=name(e),document=doc,targets=total,seconds=time.monotonic()-started))
        assert total==51457 and before==base.parameter_digest(m);m.validate()
        put(folder/'result.json',dict(completed=True,endpoint=e,targets=total,parameters_and_buffers_unchanged=True,
             seconds=elapsed,newly_scored_histories=hs,files={p.name:sha(p) for p in sorted(folder.glob('doc-*.json'))}))
        del m,x,y,z;gc.collect();torch.mps.empty_cache();base.checkpoint_memory(out,'released')
    verify(plan);put(out/'result.json',dict(completed=True,models=12,cells=36,new_cells=24,reused_cells=12,targets_per_cell=51457,seconds=time.monotonic()-started,test_opened=False,training_updates=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','qualify','score']);args=parser.parse_args()
    signal.signal(signal.SIGTERM,base.on_stop);signal.signal(signal.SIGINT,base.on_stop)
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);base.check_stop()
        if args.mode=='prepare':prepare();return
        plan=read(OUT/'plan.json');verify(plan);out=OUT/('qualification-01' if args.mode=='qualify' else 'scoring-01');out.mkdir()
        with CpuPrerequisiteGuard(out/'guard.json',ROOT,rss_limit_bytes=4*2**30,reserve_bytes=12*2**30,
                                  timeout_seconds=plan['qualification_seconds' if args.mode=='qualify' else 'scoring_seconds'],poll_seconds=.2):
            try:
                h=base.host();put(out/'launch.json',dict(pid=os.getpid(),host=h,plan_sha256=sha(OUT/'plan.json')))
                assert int(h['pressure'])==1 and h['reclaimable_proxy_bytes']>=1.25*2**30
                assert __import__('shutil').disk_usage(ROOT).free>=plan['reserve_bytes']+plan['allowance_bytes']
                assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0' and torch.backends.mps.is_available()
                torch.set_num_threads(4)
                if args.mode=='qualify':qualify(plan,out)
                else:score(plan,out)
            except BaseException as exc:
                put(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
