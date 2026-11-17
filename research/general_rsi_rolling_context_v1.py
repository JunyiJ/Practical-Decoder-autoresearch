"""Exact stride-one fixed-weight history comparison, outside frozen namespaces."""
import argparse, datetime, fcntl, gc, hashlib, json, math, os, resource
from pathlib import Path
import re, shutil, signal, subprocess, time, traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/tokenizer-evidence/general-rsi-rolling-context-001'
PANEL = ROOT/'research/tokenizer-evidence/general-rsi-depth-context-quality-001'
DESIGN = ROOT/'research/GENERAL_RSI_ROLLING_CONTEXT_DESIGN.md'
GIB = 2**30
STOP = False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def put(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())


def on_stop(*_):
    global STOP
    STOP = True


def check_stop():
    if STOP or read(ROOT/'research_state/general-rsi-state.json').get('stop_requested', False):
        raise InterruptedError('Human stop')


def host():
    commands = {'vm_stat':['/usr/bin/vm_stat'],
                'pressure':['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],
                'swap':['/usr/sbin/sysctl','vm.swapusage'],
                'physical':['/usr/sbin/sysctl','-n','hw.memsize']}
    h = {k:subprocess.check_output(v, text=True).strip() for k,v in commands.items()}
    page = int(re.search(r'page size of (\d+) bytes', h['vm_stat']).group(1))
    h['reclaimable_proxy_bytes'] = page*sum(int(re.search(r'^'+re.escape(k)+r':\s+(\d+)\.', h['vm_stat'], re.M).group(1)) for k in ('Pages free','Pages inactive','Pages speculative'))
    return h


def checkpoint_memory(out, phase):
    import torch
    torch.mps.synchronize()
    h = host()
    row = dict(phase=phase, at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               mps_driver_bytes=torch.mps.driver_allocated_memory(),
               mps_current_bytes=torch.mps.current_allocated_memory(),
               free_bytes=shutil.disk_usage(ROOT).free, host=h)
    append(out/'memory.jsonl', row)
    assert row['peak_rss_bytes'] < 4*GIB and row['mps_driver_bytes'] < 6*GIB
    assert row['free_bytes'] >= 12*GIB and int(h['pressure']) <= 2
    assert h['reclaimable_proxy_bytes'] >= GIB
    return row


def data(split):
    import numpy as np
    import torch
    from research.general_rsi_data_v1 import MANIFEST_SHA
    assert split in ('train','val')
    root = ROOT/'data/raw/rsi-general-10m-v1'
    assert sha(root/'manifest.json') == MANIFEST_SHA
    manifest = read(root/'manifest.json')
    for suffix in ('bin','index.jsonl'):
        p = root/f'{split}.{suffix}'; r = manifest['files'][p.name]
        assert p.stat().st_size == r['bytes'] and sha(p) == r['sha256']
    values = torch.from_numpy(np.fromfile(root/f'{split}.bin', dtype='<u2').astype(np.int64))
    docs = [json.loads(x) for x in (root/f'{split}.index.jsonl').read_text().splitlines()]
    offset = 0
    for doc in docs:
        assert doc['offset'] == offset and doc['tokens'] > 1
        offset += doc['tokens']; assert values[offset-1].item() == 50256
    assert offset == len(values) and len(values) == manifest['tokens'][split]
    return values, docs


def windows(values, start, positions, history):
    import torch
    assert history in (128,256) and positions and len(positions) <= 4
    assert all(p >= 256 and start+p < len(values) for p in positions)
    x = torch.stack([values[start+p-history:start+p] for p in positions])
    y = values[[start+p for p in positions]]
    assert x.shape == (len(positions),history) and y.numel() == len(positions)
    return x, y


def last_logits(model, x):
    """Same hidden computation and scaling; project only the final position."""
    import torch
    assert not torch.is_grad_enabled() and not model.training
    model._ids(x, next(model.parameters()).device)
    return model.readout(model._hidden(x.contiguous())[:, -1, :]) / 4.0


def parameter_digest(model):
    h = hashlib.sha256()
    for name, p in model.ordered_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def endpoints():
    plan = read(PANEL/'plan.json')
    assert read(PANEL/'completion-review.json')['accepted']
    selected = []
    for depth in (4,16):
        for bundle in (1,2,3):
            s, = [s for s in plan['slots'] if s['depth']==depth and s['bundle']==bundle and s['context']==256]
            r = read(PANEL/f"slot-{s['slot']:02d}"/'result.json')
            assert r['status']=='completed' and r['binding']==sha(PANEL/'plan.json')
            e, = [e for e in r['endpoints'] if e['endpoint']=='time_ema']
            rec = e['weights']; assert Path(rec['path']).stat().st_size==rec['size'] and sha(rec['path'])==rec['sha256']
            selected.append(dict(depth=depth,bundle=bundle,slot=s['slot'],receipt=rec))
    return selected


def verify_bindings(plan):
    for name, digest in plan['sources'].items(): assert sha(ROOT/name)==digest, name
    for e in plan['endpoints']: assert sha(e['receipt']['path'])==e['receipt']['sha256']


def load(e, out):
    from research.general_rsi_depth_context_training_v1 import load_inference
    check_stop(); checkpoint_memory(out,'before_load')
    m = load_inference(e['receipt'],e['depth'],256,sha(PANEL/'plan.json'))
    m.eval(); m.validate(); checkpoint_memory(out,'loaded')
    return m


def prepare():
    import torch
    assert not OUT.exists()
    original = read(PANEL/'plan.json')
    sources = dict(original['sources'])
    extra = [Path(__file__),DESIGN,PANEL/'plan.json',PANEL/'completion-review.json',
             ROOT/'research/general_rsi_depth_context_training_v1.py',
             ROOT/'src/research/cpu_prerequisite_guard.py']
    for p in extra: sources[str(p.relative_to(ROOT))]=sha(p)
    docs = [json.loads(x) for x in (ROOT/'data/raw/rsi-general-10m-v1/val.index.jsonl').read_text().splitlines()]
    plan = dict(kind='saved_ema_rolling_context_v1', at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                sources=sources, endpoints=endpoints(), torch_version=torch.__version__,
                target_count=sum(max(0,d['tokens']-256) for d in docs),document_count=len(docs),
                history_lengths=[128,256],minimum_target_position=256,batch_size=4,
                bootstrap_seed=621116210001,bootstrap_resamples=100000,
                interval_quantiles=[.01/6,1-.01/6],max_scoring_seconds=43200,
                qualification_seconds=600,storage_allowance_bytes=256*2**20,
                reserve_bytes=12*GIB,rss_limit_bytes=4*GIB,mps_limit_bytes=6*GIB,
                scientific_training=False,test_opened=False)
    assert plan['target_count']==72876 and plan['document_count']==108
    verify_bindings(plan); assert shutil.disk_usage(ROOT).free>=12*GIB+256*2**20
    OUT.mkdir(); put(OUT/'plan.json',plan)


def qualify(plan, out):
    import torch
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    # Independent symbolic checks include the earliest full window and a tail.
    synthetic = torch.arange(1200, dtype=torch.int64)
    for c in (128,256):
        for positions in ([256,257,383,511],[599]):
            x,y=windows(synthetic,37,positions,c)
            for i,p in enumerate(positions):
                assert x[i].tolist()==list(range(37+p-c,37+p)) and y[i].item()==37+p
    train, docs = data('train')
    d = next(d for d in docs if d['tokens']>=640)
    measurements=[]
    for depth in (4,16):
        e=next(e for e in plan['endpoints'] if e['depth']==depth and e['bundle']==1)
        m=load(e,out); before=parameter_digest(m); maxerr=0.; ceerr=0.
        with torch.no_grad(), FP32RMSNormInference():
            for c in (128,256):
                x,y=windows(train,d['offset'],[256,257,383,511],c);x=x.to('mps')
                z=last_logits(m,x); reference=m.primary_logits(x).values[:,-1,:]
                torch.testing.assert_close(z,reference,atol=3e-5,rtol=3e-4)
                for i in range(4):
                    scalar=m.primary_logits(x[i:i+1]).values[0,-1,:]
                    torch.testing.assert_close(z[i],scalar,atol=3e-5,rtol=3e-4)
                    maxerr=max(maxerr,(z[i]-scalar).abs().max().item())
                loss=torch.nn.functional.cross_entropy(z,y.to('mps'),reduction='none').cpu().double()
                zz=reference.cpu().double(); oracle=torch.logsumexp(zz,-1)-zz.gather(1,y[:,None]).squeeze(1)
                ceerr=max(ceerr,(loss-oracle).abs().max().item());assert ceerr<1e-5
            elapsed={}
            for c in (128,256):
                torch.mps.synchronize(); t=time.monotonic()
                for pos in range(256,384,4):
                    check_stop();x,y=windows(train,d['offset'],list(range(pos,pos+4)),c)
                    z=last_logits(m,x.to('mps'))
                    loss=torch.nn.functional.cross_entropy(z,y.to('mps'),reduction='none').cpu()
                    assert torch.isfinite(loss).all()
                    assert torch.mps.driver_allocated_memory()<6*GIB
                torch.mps.synchronize();elapsed[c]=time.monotonic()-t
                checkpoint_memory(out,f'qualified_d{depth}_c{c}')
        assert before==parameter_digest(m);m.validate()
        measurements.append(dict(depth=depth,targets_per_history=128,seconds=elapsed,
                                 logits_max_abs_error=maxerr,ce_max_abs_error=ceerr,parameters_unchanged=True))
        del m,x,y,z,reference,scalar,loss,zz,oracle;gc.collect();torch.mps.empty_cache()
    forecast=600+1.5*3*plan['target_count']*sum(sum(m['seconds'].values())/128 for m in measurements)
    result=dict(accepted=forecast<=plan['max_scoring_seconds'],measurements=measurements,
                full_forecast_seconds=forecast,plan_sha256=sha(OUT/'plan.json'),validation_opened=False,test_opened=False)
    put(out/'result.json',result)
    checkpoint_memory(out,'serialized_qualification')
    assert result['accepted'], result
    print(json.dumps(result),flush=True)


def score_all(plan,out):
    import torch
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    qualification=read(OUT/'qualification-01/result.json')
    assert qualification['accepted'] and qualification['plan_sha256']==sha(OUT/'plan.json')
    assert shutil.disk_usage(ROOT).free>=plan['reserve_bytes']+plan['storage_allowance_bytes']
    values,docs=data('val'); t=time.monotonic()
    for e in plan['endpoints']:
        folder=out/f"d{e['depth']}-b{e['bundle']}";folder.mkdir()
        m=load(e,out); before=parameter_digest(m);nt=0
        with torch.no_grad(), FP32RMSNormInference():
            for doc, row in enumerate(docs):
                check_stop();m.validate();positions=list(range(256,row['tokens']))
                losses={128:[],256:[]};target_ids=[]
                for j in range(0,len(positions),4):
                    check_stop();pp=positions[j:j+4]
                    for c in (128,256):
                        x,y=windows(values,row['offset'],pp,c)
                        z=last_logits(m,x.to('mps'))
                        ls=torch.nn.functional.cross_entropy(z,y.to('mps'),reduction='none').cpu().double().tolist()
                        assert all(math.isfinite(v) for v in ls);losses[c].extend(ls)
                        assert torch.mps.driver_allocated_memory()<6*GIB
                    target_ids.extend(y.tolist())
                assert len(target_ids)==len(losses[128])==len(losses[256])==len(positions)
                record=dict(document=doc,source=row['source'],offset=row['offset'],
                            positions=positions,target_ids=target_ids,losses=losses,
                            loss_sums={c:math.fsum(v) for c,v in losses.items()},count=len(positions))
                put(folder/f'doc-{doc:03d}.json',record);nt+=len(positions)
                if doc%10==0 or doc==len(docs)-1:checkpoint_memory(out,f"d{e['depth']}_b{e['bundle']}_doc{doc}")
                append(out/'progress.jsonl',dict(depth=e['depth'],bundle=e['bundle'],document=doc,targets=nt,elapsed_seconds=time.monotonic()-t))
        assert nt==plan['target_count'] and before==parameter_digest(m);m.validate()
        put(folder/'result.json',dict(completed=True,endpoint=e,targets=nt,parameters_unchanged=True,
                                     files={p.name:sha(p) for p in sorted(folder.glob('doc-*.json'))}))
        del m,x,y,z;gc.collect();torch.mps.empty_cache();checkpoint_memory(out,'model_released')
    verify_bindings(plan)
    put(out/'result.json',dict(completed=True,plan_sha256=sha(OUT/'plan.json'),models=6,history_conditions=2,
                              targets_per_condition=plan['target_count'],seconds=time.monotonic()-t,test_opened=False,training_updates=0))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','qualify','score']);args=parser.parse_args()
    signal.signal(signal.SIGTERM,on_stop);signal.signal(signal.SIGINT,on_stop)
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop()
        if args.mode=='prepare':prepare();return
        plan=read(OUT/'plan.json');verify_bindings(plan)
        out=OUT/('qualification-01' if args.mode=='qualify' else 'scoring-01');out.mkdir()
        from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
        with CpuPrerequisiteGuard(out/'guard.json',ROOT,rss_limit_bytes=4*GIB,reserve_bytes=12*GIB,
                                  timeout_seconds=plan['qualification_seconds'] if args.mode=='qualify' else plan['max_scoring_seconds'],poll_seconds=.2):
            try:
                h=host();assert int(h['pressure'])==1 and h['reclaimable_proxy_bytes']>=GIB+256*2**20
                put(out/'launch.json',dict(pid=os.getpid(),host=h,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                         processes=subprocess.check_output(['/bin/ps','-axo','pid,ppid,rss,comm'],text=True),plan_sha256=sha(OUT/'plan.json')))
                import torch
                assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0' and torch.backends.mps.is_available()
                assert torch.__version__==plan['torch_version'];torch.set_num_threads(4)
                if args.mode=='qualify':qualify(plan,out)
                else:score_all(plan,out)
            except BaseException as exc:
                put(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
