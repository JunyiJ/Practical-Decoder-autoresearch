"""Once-only cached consumer check on accepted H200 fixtures; no training."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT=BASE/'cache-math001'
TORCH=None
CHUNKS=[(0,32)]+[(p,p+1) for p in range(32,96)]+[(96,128)]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while x:=f.read(1048576):h.update(x)
    return h.hexdigest()
def pins(rows):
    for r in rows:
        p=Path(r['path']);assert p.stat().st_size==r['size'] and sha(p)==r['sha256'],str(p)
def publish(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

class Guard(CpuPrerequisiteGuard):
    def _sample(self):
        sample,reason=super()._sample()
        paths=list(OUT.iterdir())
        if any(p.is_symlink() or not p.is_file() for p in paths):reason='unsafe_output'
        sample['output_bytes']=sum(p.stat().st_size for p in paths)
        if sample['output_bytes']>64*1024**2:reason='output_limit'
        if TORCH is not None:
            sample['mps_driver_bytes']=TORCH.mps.driver_allocated_memory()
            if sample['mps_driver_bytes']>=6*1024**3:reason='mps_limit'
        return sample,reason

def run(plan):
    import gc
    import torch
    from research import byte_input_inference_v1 as inference
    from research import byte_input_training_v2 as training
    from research.byte_input_model_v3 import Spec,build_codebook
    from research.byte_input_model_v3.contract import signature
    from research.byte_input_cache_v1 import Session
    def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    def weights(m):return {n:digest(p) for n,p in m.ordered_parameters()}
    def snapshot(s):return dict(position=s.position,layers=[dict(position=c.position,c=digest(c.c_cache),r=digest(c.r_cache)) for c in s.caches])
    torch.set_num_threads(4);cpu=torch.random.get_rng_state().clone();mps=torch.mps.get_rng_state().clone()
    original=plan['training_plan'];assets=[Path(r['path']).read_bytes() for r in original['codebook_assets']]
    data=json.loads((training.BASE/'quality002-plan.json').read_text())['data'];results=[]
    try:
        for backend in ('cpu','mps'):
            for variant in ('native','raw_bytes','nonlexical_codes'):
                prefix=backend+'-'+variant
                export=next(r for r in plan['exports'] if Path(r['path']).name==prefix+'-h200.pt')
                parent=next(r for r in plan['parents'] if Path(r['path']).name==prefix+'-step4.pt')
                book=None if variant=='native' else build_codebook(*assets,variant,
                    original['roles']['nonlexical_key'] if variant=='nonlexical_codes' else None)
                spec=Spec(variant) if book is None else Spec(variant,original['roles']['composer_initializer'],book.identity)
                identity=dict(kind=training.KIND,recipe=training.recipe(4,.00125),roles=original['roles'],
                    binding=original['binding'],data=data,model=signature(spec))
                model=inference.load(export,identity,parent,book,backend)
                view=training.load_view(original['roles'],backend)
                before=weights(model);rng=(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
                for batch in (1,8):
                    ids=view.train_data[:128].unsqueeze(0).repeat(batch,1).to(backend)
                    with torch.no_grad():full=model.primary_logits(ids).values
                    session=Session(model,batch);assert session.storage_bytes==196608*batch
                    checks=[];first=[];first_state=None
                    for replay in (0,1):
                        if replay:
                            session.reset()
                            assert session.position==0 and all(not bool(t.any()) for t in session._tensors)
                        for i,(lo,hi) in enumerate(CHUNKS):
                            actual=session.logits(ids[:,lo:hi],lo);target=full[:,lo:hi]
                            torch.testing.assert_close(actual,target,atol=3e-5,rtol=3e-4,equal_nan=False)
                            error=(actual-target).abs();ratio=error/(3e-5+3e-4*target.abs())
                            h=digest(actual)
                            if replay:assert h==first[i]
                            else:first.append(h)
                            checks.append(dict(replay=replay,start=lo,end=hi,coordinates=actual.numel(),
                                max_absolute_error=float(error.max()),max_gate_ratio=float(ratio.max()),sha256=h))
                            del actual,target,error,ratio
                        state=snapshot(session)
                        if replay:assert state==first_state
                        else:first_state=state
                    invalid=[];saved=snapshot(session)
                    cases=[('position',ids[:,:1],0),('boolean_position',ids[:,:1],True),
                        ('overflow',ids[:,:1],128),('dtype',ids[:,:1].float(),128),
                        ('id_range',torch.full_like(ids[:,:1],50257),128),
                        ('empty',ids[:,:0],128),('batch',ids[:1,:1] if batch==8 else ids.repeat(8,1)[:,:1],128)]
                    for name,x,pos in cases:
                        try:session.logits(x,pos)
                        except ValueError:invalid.append(name)
                        else:raise AssertionError('Invalid cache input accepted: '+name)
                        assert snapshot(session)==saved
                    model.validate();assert weights(model)==before
                    assert rng==(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
                    row=dict(backend=backend,variant=variant,batch=batch,checks=checks,invalid_rejections=invalid,
                        cache_bytes=session.storage_bytes,reset_exact=True,weights_unchanged=True,rng_preserved=True,
                        full_context=128,atol=3e-5,rtol=3e-4)
                    publish(prefix+'-b'+str(batch)+'.json',row);results.append(row)
                    del full,session,ids,cases;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
                del model,view,book;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
        assert len(results)==12
        publish('result.json',dict(accepted=True,cases=12,chunk_comparisons=sum(len(r['checks']) for r in results),
            coordinates=sum(c['coordinates'] for r in results for c in r['checks']),invalid_transactions=84,
            max_absolute_error=max(c['max_absolute_error'] for r in results for c in r['checks']),
            max_gate_ratio=max(c['max_gate_ratio'] for r in results for c in r['checks']),updates_executed=0,
            scope='Cached/full128 FP32 logits with native inherited tolerance, B1/B8 CPU/MPS, exact reset replay and invalid-position preservation. Existing four-step H200 fixtures only; no timings, quality or worker qualification.'))
    finally:torch.random.set_rng_state(cpu);torch.mps.set_rng_state(mps)

def main(expected):
    global TORCH
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert {p.name for p in OUT.iterdir()}=={'plan.json'} and sha(OUT/'plan.json')==expected
        plan=json.loads((OUT/'plan.json').read_text());stat=os.statvfs(ROOT)
        assert stat.f_bavail*stat.f_frsize>=plan['forecast']['entry_bytes']
        publish('attempt.json',dict(pid=os.getpid(),plan_sha256=expected,no_replay=True))
        try:
            with Guard(OUT/'guard.json',ROOT,timeout_seconds=600,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,poll_seconds=.25):
                assert sys.flags.optimize==0
                pins(plan['sources']);pins(plan['assets']);pins(plan['exports']);pins(plan['parents']);pins(plan['prerequisites'])
                assert json.loads((BASE/'inference-review001/review.json').read_text())['accepted']
                for key,value in plan['environment'].items():assert os.environ.get(key)==value
                import torch
                TORCH=torch
                run(plan)
                pins(plan['sources']);pins(plan['assets']);pins(plan['exports']);pins(plan['parents'])
            publish('STOP',dict(status='complete_original_cache_consumers',no_replay=True))
        except BaseException as exc:
            if not (OUT/'STOP').exists():publish('STOP',dict(status='closed_failed_original_cache_consumers',error=repr(exc),no_replay=True))
            raise

if __name__=='__main__':main(sys.argv[1])
