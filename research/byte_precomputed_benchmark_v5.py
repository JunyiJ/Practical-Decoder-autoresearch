"""Bounded conversion, parity, latency and isolated-memory measurements.

No training, new quality selection, holdout access or frozen-source changes.
The parent holds the nonblocking research lock across all sequential workers.
"""
import argparse
import ctypes
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT = BASE/'precomputed-inference005'
PAYLOAD = ROOT/'research_runs/byte-precomputed-inference005'
RECIPE = BASE/'bulk-hash-adoption001/recipe.json'


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024**2), b''): h.update(part)
    return h.hexdigest()


def record(path):
    path = Path(path)
    return dict(path=str(path.resolve()), size=path.stat().st_size, sha256=sha(path))


def put(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2); stream.write('\n')


def prepare():
    recipe = read(RECIPE)
    assert recipe['status'] == 'adopted_equivalent_bulk_hash_implementation'
    sources = [ROOT/'research/byte_precomputed_inference_v3.py', Path(__file__).resolve()]
    plan = dict(kind='byte_precomputed_benchmark_v5', sources=[record(p) for p in sources],
        recipe=record(RECIPE), implementation=recipe['implementation'],
        original_plan=record(recipe['plan']['path']), acceptance=recipe['loader']['acceptance_record'],
        checkpoint=recipe['representative']['h200'],
        python=sys.version, torch_version=importlib.metadata.version('torch'), platform=platform.platform(),
        backends=['cpu','mps'], batches=[1,8], prefill=96, decode=32, context=128,
        vocabulary=50257, table_chunk=256, independent_chunk=1024,
        atol=3e-5, rtol=3e-4, mean_ce_atol=3e-5, max_ce_atol=3e-4,
        full_validation_targets=34278, warmups=2, repeats=7, greedy_steps=16,
        worker_seconds=900, total_seconds=2700, rss_limit=4*1024**3,
        mps_limit=6*1024**3, reserve=8*1024**3, headroom=2*1024**3,
        output_cap=160*1024**2, threads=4, no_training=True, holdout=False,
        semantics='Existing public no-grad FP32 behavior; no RMSNorm epsilon correction',
        memory_scope='Fresh process per backend/arm; current RSS, process high-water RSS, MPS current/driver and tensor storage separately',
        timing_scope='Paired alternating order; synchronize MPS; setup/reset excluded from steady-state latency; full logits retained until sync')
    assert not OUT.exists() and not PAYLOAD.exists()
    assert shutil.disk_usage(ROOT).free >= plan['reserve']+plan['headroom']+plan['output_cap']
    OUT.mkdir(); PAYLOAD.mkdir()
    put(OUT/'plan.json', plan)
    print(sha(OUT/'plan.json'), flush=True)


def verify_pins(plan):
    for r in plan['sources']+[plan['recipe'],plan['implementation'],plan['original_plan'],plan['acceptance']]:
        assert Path(r['path']).stat().st_size == r['size'] and sha(r['path']) == r['sha256']
    assert importlib.metadata.version('torch') == plan['torch_version']


def current_rss():
    # macOS proc_taskinfo: six uint64 counters followed by twelve int32 fields.
    class TaskInfo(ctypes.Structure):
        _fields_ = [('virtual_size', ctypes.c_uint64), ('resident_size', ctypes.c_uint64),
                    ('times', ctypes.c_uint64*4), ('counters', ctypes.c_int32*12)]
    info = TaskInfo()
    lib = ctypes.CDLL('/usr/lib/libproc.dylib', use_errno=True)
    lib.proc_pidinfo.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_int]
    lib.proc_pidinfo.restype = ctypes.c_int
    n = lib.proc_pidinfo(os.getpid(), 4, 0, ctypes.byref(info), ctypes.sizeof(info))
    if n != ctypes.sizeof(info): raise OSError(ctypes.get_errno(), 'proc_pidinfo current RSS')
    return info.resident_size


def worker(phase, backend, expected):
    import gc
    import math
    import resource
    import statistics
    import torch
    from torch.nn import functional as F
    from research import byte_precomputed_inference_v3 as compiled
    from research.byte_input_checkpoint_consumer_v4 import load as load_original
    from research.byte_input_cache_v4 import Session as OriginalSession
    from research.byte_input_inference_v5 import _logits
    from research.byte_input_training_v6 import load_view
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

    assert sha(OUT/'plan.json') == expected
    plan = read(OUT/'plan.json'); verify_pins(plan)
    name = f'{phase}-{backend}'
    torch.set_num_threads(plan['threads'])
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    # Initialize Metal on the main thread before the guard samples its allocator.
    if backend == 'mps':
        assert torch.backends.mps.is_available(), 'MPS unavailable: run through approved host GPU execution'
        probe = torch.empty(1, device='mps')
        torch.mps.synchronize()
        del probe

    def sync():
        if backend == 'mps': torch.mps.synchronize()

    def memory():
        sync()
        return dict(current_rss_bytes=current_rss(), process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    mps_current_bytes=torch.mps.current_allocated_memory() if backend=='mps' else None,
                    mps_driver_bytes=torch.mps.driver_allocated_memory() if backend=='mps' else None)

    class Guard(CpuPrerequisiteGuard):
        def _sample(self):
            sample, reason = super()._sample()
            if backend=='mps':
                sample['mps_driver_bytes'] = torch.mps.driver_allocated_memory()
                if sample['mps_driver_bytes'] >= plan['mps_limit']: reason='mps_limit'
            sample['output_bytes'] = sum(p.stat().st_size for root in (OUT,PAYLOAD) for p in root.rglob('*') if p.is_file())
            if sample['output_bytes'] > plan['output_cap']: reason='output_limit'
            return sample, reason

    def original():
        return load_original(plan['implementation'], plan['original_plan'], plan['acceptance'], backend)

    def digest(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def compare(a,b):
        assert a.shape==b.shape and a.dtype==b.dtype and torch.isfinite(a).all() and torch.isfinite(b).all()
        error=(a-b).abs()
        row=dict(max_abs=float(error.max()), max_ratio=float((error/(plan['atol']+plan['rtol']*b.abs())).max()),
                 exact=bool(torch.equal(a,b)))
        if row['max_ratio'] > 1:
            put(OUT/f'{name}-numerical-failure.json',row)
        assert row['max_ratio'] <= 1, row
        return row

    def measured(fn):
        sync(); start=time.perf_counter(); value=fn(); sync()
        elapsed=time.perf_counter()-start
        del value
        return elapsed

    with Guard(OUT/f'{name}-guard.json',ROOT,timeout_seconds=plan['worker_seconds'],
               rss_limit_bytes=plan['rss_limit'],reserve_bytes=plan['reserve'],poll_seconds=.5):
        baseline=memory()
        result=dict(phase=phase,backend=backend,plan_sha256=expected,baseline_memory=baseline)
        if phase=='verify':
            model, identity, source=original()
            weights={n:digest(t) for n,t in model.ordered_parameters()}
            rng=(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
            sync(); start=time.perf_counter()
            table_model=compiled.precompute(model,source,plan['table_chunk'])
            sync(); result['conversion_seconds']=time.perf_counter()-start
            result['conversion_memory']=memory()
            result['source_tensor_bytes']=compiled.tensor_bytes(model)
            result['compiled_tensor_bytes']=compiled.tensor_bytes(table_model)
            if backend=='cpu':
                start=time.perf_counter()
                artifact=compiled.save(table_model,PAYLOAD/'precomputed.pt')
                result['save_seconds']=time.perf_counter()-start
                put(OUT/'artifact.json',artifact)
            else: artifact=read(OUT/'artifact.json')
            loaded=compiled.load(artifact,source,backend)
            result['loaded_table_comparison']=compare(loaded.embedding_table,table_model.embedding_table)
            if backend=='cpu':
                assert all(torch.equal(t,loaded.state_dict()[n]) for n,t in table_model.state_dict().items())
            del table_model;gc.collect()
            table_checks=[]
            with torch.no_grad():
                for lo in range(0,50257,plan['independent_chunk']):
                    hi=min(50257,lo+plan['independent_chunk'])
                    actual=model.composer(model.code_ids[lo:hi],model.code_lengths[lo:hi])
                    table_checks.append(dict(lo=lo,hi=hi,**compare(loaded.embedding_table[lo:hi],actual)))
                    del actual
            result['all_vocabulary_checks']=table_checks
            view=load_view(identity['roles'],'cpu')
            errors=[]; rows=[]; argmax_changes=0
            with torch.no_grad():
                for lo in range(0,plan['full_validation_targets'],128):
                    n=min(128,plan['full_validation_targets']-lo)
                    ids=view.val_data[lo:lo+n].unsqueeze(0).to(backend)
                    target=view.val_data[lo+1:lo+n+1].to(backend)
                    a=_logits(model,ids);b=loaded(ids)
                    rows.append(dict(start=lo,targets=n,**compare(b,a)))
                    argmax_changes+=int((a.argmax(-1)!=b.argmax(-1)).sum())
                    errors.extend((F.cross_entropy(b[0],target,reduction='none')-F.cross_entropy(a[0],target,reduction='none')).cpu().double().tolist())
                    del a,b,ids,target
            result['validation']=dict(targets=len(errors),max_ce_error=max(abs(x) for x in errors),
                mean_absolute_ce_error=math.fsum(abs(x) for x in errors)/len(errors),
                mean_signed_ce_error=math.fsum(errors)/len(errors),argmax_changes=argmax_changes,windows=rows)
            put(OUT/f'{name}-validation.json',result['validation'])
            assert len(errors)==34278 and result['validation']['max_ce_error']<=plan['max_ce_atol']
            assert result['validation']['mean_absolute_ce_error']<=plan['mean_ce_atol']
            print(name,'full vocabulary and full validation passed',flush=True)
            cache_rows=[];timings=[]
            for batch in plan['batches']:
                ids=view.train_data[:batch*128].reshape(batch,128).to(backend)
                with torch.no_grad():full=loaded(ids)
                sessions=[OriginalSession(model,batch),compiled.Session(loaded,batch)]
                first=[]
                for replay in range(2):
                    if replay:
                        for s in sessions:
                            s.reset();assert s.position==0 and all(not bool(t.any()) for t in s._tensors)
                    for lo,hi in [(0,96)]+[(i,i+1) for i in range(96,128)]:
                        a=sessions[0].logits(ids[:,lo:hi],lo);b=sessions[1].logits(ids[:,lo:hi],lo)
                        cache_rows.append(dict(batch=batch,replay=replay,start=lo,vs_current=compare(b,a),vs_full=compare(b,full[:,lo:hi])))
                        h=digest(b)
                        if replay:assert h==first[0];first.pop(0)
                        else:first.append(h)
                        del a,b
                    assert all(s.position==128 and s.storage_bytes==196608*batch for s in sessions)
                del full
                # Greedy autoregressive comparison, distinct from teacher forcing.
                for s in sessions:s.reset()
                seq=[ids[:,:96],ids[:,:96]]
                for step in range(plan['greedy_steps']):
                    logits=[s.logits(x,0 if step==0 else 95+step) for s,x in zip(sessions,seq)]
                    compare(logits[1],logits[0])
                    seq=[x[:,-1].argmax(-1,keepdim=True) for x in logits]
                    assert torch.equal(seq[0],seq[1]);del logits
                for workload in ('full128','prefill96','decode32'):
                    samples=[[],[]]
                    for repeat in range(-plan['warmups'],plan['repeats']):
                        for arm in ([0,1] if repeat%2==0 else [1,0]):
                            s=sessions[arm]
                            if workload=='full128':
                                def fn():
                                    with torch.no_grad():return _logits(model,ids) if arm==0 else loaded(ids)
                            else:
                                s.reset()
                                if workload=='decode32':
                                    value=s.logits(ids[:,:96],0);del value;sync()
                                    def fn():
                                        for i in range(96,128):
                                            value=s.logits(ids[:,i:i+1],i)
                                            del value
                                else:
                                    def fn():return s.logits(ids[:,:96],0)
                            seconds=measured(fn)
                            if repeat>=0:samples[arm].append(seconds)
                    medians=[statistics.median(x) for x in samples]
                    timings.append(dict(batch=batch,workload=workload,seconds=samples,
                        median_seconds=medians,reference_over_precomputed=medians[0]/medians[1],
                        paired_speedups=[a/b for a,b in zip(*samples)]))
                    print(name,batch,workload,'speedup',round(medians[0]/medians[1],3),flush=True)
                # Negative checks must leave the session reusable and unmoved.
                s=sessions[1];s.reset()
                for bad,pos in [(ids[:,:1],1),(torch.full((batch,1),50257,device=backend,dtype=torch.int64),0),
                                (ids[:,:1].float(),0)]:
                    try:s.logits(bad,pos)
                    except ValueError:pass
                    else:raise AssertionError('Invalid input accepted')
                    assert s.position==0
                s.logits(ids,0)
                try:s.logits(ids[:,:1],128)
                except ValueError:pass
                else:raise AssertionError('Context overflow accepted')
                assert s.position==128
                del sessions,s,ids,seq;gc.collect()
            result['cache_checks']=cache_rows;result['timings']=timings
            result['greedy_exact']=True;result['reset_exact']=True;result['invalid_inputs_rejected']=True
            try:loaded.train()
            except ValueError:pass
            else:raise AssertionError('Training accepted')
            # A cache must reject a modified precomputed table.
            session=compiled.Session(loaded,1)
            with torch.no_grad():loaded.embedding_table[0,0].add_(1.)
            try:session.check()
            except ValueError:pass
            else:raise AssertionError('Mutation accepted')
            result['mutation_rejected']=True
            assert weights=={n:digest(t) for n,t in model.ordered_parameters()}
            assert all(p.grad is None for p in model.parameters())
            assert rng==(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
            assert sha(source['path'])==source['sha256']
            result['source_weights_rng_unchanged']=True
        else:
            sync();start=time.perf_counter()
            if phase=='memory-current':
                model,identity,source=original();session_type=OriginalSession
            else:
                source=plan['checkpoint'];model=compiled.load(read(OUT/'artifact.json'),source,backend);session_type=compiled.Session
            sync();result['load_seconds']=time.perf_counter()-start
            gc.collect()
            if backend=='mps':torch.mps.empty_cache()
            result['loaded_memory']=memory();result['model_tensor_bytes']=compiled.tensor_bytes(model)
            result['parameters_bytes']=sum(t.numel()*t.element_size() for t in model.parameters())
            result['buffers_bytes']=sum(t.numel()*t.element_size() for t in model.buffers())
            observations=[]
            for batch in plan['batches']:
                ids=(torch.arange(batch*128,dtype=torch.int64).reshape(batch,128)%50257).to(backend)
                session=session_type(model,batch)
                for warmup in range(2):
                    session.reset();value=session.logits(ids[:,:96],0);del value
                    for i in range(96,128):value=session.logits(ids[:,i:i+1],i);del value
                gc.collect()
                if backend=='mps':torch.mps.empty_cache()
                observations.append(dict(batch=batch,stage='idle_with_cache',cache_bytes=session.storage_bytes,**memory()))
                session.reset();value=session.logits(ids[:,:96],0)
                observations.append(dict(batch=batch,stage='prefill_logits_live',logits_bytes=value.numel()*value.element_size(),**memory()))
                del value
                for i in range(96,128):
                    value=session.logits(ids[:,i:i+1],i)
                    if i==127:observations.append(dict(batch=batch,stage='decode_logits_live',**memory()))
                    del value
                del session,ids;gc.collect()
                if backend=='mps':torch.mps.empty_cache()
            result['observations']=observations
        result['final_memory']=memory();verify_pins(plan)
        result['completed']=True
        put(OUT/f'{name}-result.json',result)
    print(name,'complete',flush=True)


def run(expected):
    assert sha(OUT/'plan.json')==expected
    plan=read(OUT/'plan.json');verify_pins(plan)
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (OUT/'attempt.json').exists()
        assert shutil.disk_usage(ROOT).free>=plan['reserve']+plan['headroom']+plan['output_cap']
        put(OUT/'attempt.json',dict(pid=os.getpid(),plan_sha256=expected,started=time.time()))
        inherited = plan['inherited_cpu']
        for r in inherited['receipts']:
            assert Path(r['path']).stat().st_size == r['size'] and sha(r['path']) == r['sha256']
        assert read(OUT/'verify-cpu-result.json')['completed']
        assert read(OUT/'verify-cpu-guard.json')['status'] == 'finished'
        start=time.monotonic();completed=['verify-cpu']
        try:
            for phase,backend in [('verify','mps'),('memory-current','cpu'),
                                  ('memory-precomputed','cpu'),('memory-precomputed','mps'),('memory-current','mps')]:
                timeout=min(plan['worker_seconds']+30,plan['total_seconds']-(time.monotonic()-start))
                assert timeout>0
                name=f'{phase}-{backend}'
                with (OUT/f'{name}.log').open('x') as log:
                    result=subprocess.run([sys.executable,'-m','research.byte_precomputed_benchmark_v5',
                        'worker',expected,'--phase',phase,'--backend',backend],cwd=ROOT,
                        env={**os.environ,'PYTORCH_ENABLE_MPS_FALLBACK':'0'},stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
                print(name,'exit',result.returncode,flush=True)
                assert result.returncode==0, name
                assert read(OUT/f'{name}-result.json')['completed']
                guard=read(OUT/f'{name}-guard.json');assert guard['status']=='finished' and guard['reason'] is None
                completed.append(name)
            verify_pins(plan)
            put(OUT/'acceptance.json',dict(accepted=True,completed=completed,seconds=time.monotonic()-start,
                plan_sha256=expected,artifact=read(OUT/'artifact.json'),no_training=True,
                default_consumer_changed=False,holdout_scored=False))
        except BaseException:
            put(OUT/'failure.json',dict(completed=completed,traceback=traceback.format_exc(),seconds=time.monotonic()-start));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','run','worker'])
    parser.add_argument('plan_sha',nargs='?');parser.add_argument('--phase');parser.add_argument('--backend')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='run':run(args.plan_sha)
    else:worker(args.phase,args.backend,args.plan_sha)
