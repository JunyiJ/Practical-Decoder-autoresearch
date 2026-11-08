"""Exact CPU update/resume comparison and prospective warmed MPS training timing."""
import copy
import fcntl
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1024*1024): h.update(chunk)
    return h.hexdigest()


def read(p): return json.loads(p.read_text())


def publish(name, value):
    with (OUT/name).open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def run():
    assert not (OUT/'guard.json').exists()
    plan = read(OUT/'plan.json'); assert sha(Path(__file__)) == plan['runner_sha256']
    for r in plan['bindings']:
        p=ROOT/r['path']; assert p.stat().st_size==r['size'] and sha(p)==r['sha256']
    mapping=read(OUT.parent/'bulk-hash-integration-source001/source-map.json')
    for pair in mapping['pairs']:
        for r in (pair['original'],pair['candidate']):
            p=ROOT/r['path']; assert p.stat().st_size==r['size'] and sha(p)==r['sha256']
        restored=(ROOT/pair['candidate']['path']).read_text()
        for old,new in reversed(pair['exact_replacements']): restored=restored.replace(new,old)
        assert restored==(ROOT/pair['original']['path']).read_text()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from research.byte_input_quality_recovery_v1.guard import Guard
        with Guard(OUT/'guard.json',ROOT,timeout_seconds=180,rss_limit_bytes=4*1024**3,
                   reserve_bytes=8*1024**3,weight_roots=[ROOT/'research_runs/byte-input-quality-v1-001',
                   ROOT/'research_runs/byte-input-quality-recovery-v1-001']) as guard:
            import torch
            from research import byte_input_checkpoint_consumer_v1 as original
            torch.set_num_threads(4);assert torch.backends.mps.is_available();guard.torch=torch
            cpu_rng=torch.get_rng_state().clone();mps_rng=torch.mps.get_rng_state().clone()
            recipe=read(OUT.parent/'optimizer-state-adoption001/recipe.json')
            implementation=original.read_bound(recipe['implementation'])
            for r in implementation['sources']:original.verify(r)
            prepared=original.prepare(recipe['loader']['plan_record'],recipe['loader']['acceptance_record'])
            binding=dict(source_sha256=sha(OUT/'plan.json'),runtime_sha256=hashlib.sha256(torch.__version__.encode()).hexdigest())
            def tree(value):
                if isinstance(value,torch.Tensor):
                    value=value.detach().cpu()
                    return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
                if type(value) is dict:return {k:tree(v) for k,v in value.items()}
                if type(value) is tuple:return dict(tuple=[tree(v) for v in value])
                if type(value) is list:return [tree(v) for v in value]
                assert value is None or type(value) in (str,int,float,bool)
                return value
            cpu_evidence=None;gpu_rows=[]
            for backend in ('cpu','mps'):
                roles=copy.deepcopy(prepared['identity']['roles']);roles.update(plan['fresh_sampler_roles'][backend])
                horizon=4 if backend=='cpu' else 20
                def create(label):
                    versions=plan['implementations'][label]
                    inference=importlib.import_module('research.'+versions['inference'])
                    trainer=importlib.import_module('research.'+versions['training'])
                    optimizer_api=importlib.import_module('research.'+versions['optimizer'])
                    model=inference.load(prepared['h200'],prepared['identity'],prepared['parent'],prepared['codebook'],backend)
                    optimizer=optimizer_api.build_optimizer(model,.0025)
                    return trainer.Context(model,optimizer,trainer.load_view(roles,backend),horizon,.0025,roles,binding),trainer
                contexts={};trainers={}
                for label in ('reference','batched'):contexts[label],trainers[label]=create(label)
                cached={}
                for label,context in contexts.items():
                    cache=importlib.import_module('research.'+plan['implementations'][label]['cache'])
                    prefix=context.view.train_data[:128].to(backend)
                    cached[label]={}
                    for batch in (1,8):
                        session=cache.Session(context.model,batch)
                        tokens=prefix.unsqueeze(0).repeat(batch,1)
                        cached[label][str(batch)]=[tree(session.logits(tokens[:,:96],0)),tree(session.logits(tokens[:,96:],96))]
                        assert session.position==128
                        session.reset();assert session.position==0
                        del session,tokens
                assert cached['reference']==cached['batched'],('Cached logits differ',backend)
                publish(backend+'-cache.json',dict(exact=True,batches=[1,8],prefix=96,continuation=32,reset=True,fingerprints=cached))
                del cached,prefix,context
                boundaries=[];resume_payload=None
                for step in range(1,(2 if backend=='cpu' else 20)+1):
                    order=['reference','batched'] if step%2 else ['batched','reference']
                    timing={};histories={}
                    for label in order:
                        if backend=='mps':torch.mps.synchronize()
                        started=time.perf_counter();histories[label]=contexts[label].step()
                        if backend=='mps':torch.mps.synchronize()
                        timing[label]=time.perf_counter()-started
                    assert histories['reference']==histories['batched'],('History mismatch',backend,step)
                    assert histories['reference']['gradient_tensors']==105
                    row=dict(backend=backend,step=step,order=order,seconds=timing,exact_history=True,
                             measured=backend=='mps' and step>4,candidate_over_reference=timing['batched']/timing['reference'])
                    publish(f'{backend}-step{step:02}.json',row)
                    if backend=='mps':gpu_rows.append(row)
                    if backend=='cpu' or step==20:
                        endpoints={}
                        for label,context in contexts.items():
                            value=trainers[label].capture(context);trainers[label].validate(context,value)
                            endpoints[label]=tree(value)
                            if backend=='cpu' and step==1 and label=='batched':resume_payload=value
                            else:del value
                            publish(f'{backend}-{label}-state{step:02}.json',endpoints[label])
                        assert endpoints['reference']==endpoints['batched'],('Complete state mismatch',backend,step)
                        boundaries.append(step)
                if backend=='cpu':
                    resumed,trainer=create('batched')
                    assert trainer.restore_pristine(resumed,resume_payload)==1
                    restored=trainer.capture(resumed);trainer.validate(resumed,restored)
                    assert tree(restored)==read(OUT/'cpu-batched-state01.json');del restored
                    row=resumed.step();assert row==contexts['reference'].history[-1]
                    restored=trainer.capture(resumed);trainer.validate(resumed,restored)
                    assert tree(restored)==endpoints['reference']
                    publish('cpu-resume-state02.json',tree(restored))
                    assert resumed.optimizer._steps==2 and resumed.averages.snapshots==1
                    cpu_evidence=dict(exact_boundaries=boundaries,exact_resume_step1_to_step2=True,
                                      numerical_updates=5,gradient_tensors=105,averaging_snapshots=1,roles=roles)
                    del restored,resumed,trainer,resume_payload
                else:
                    assert all(c.optimizer._steps==20 and c.averages.snapshots==8 for c in contexts.values())
                del context,contexts,trainers,endpoints
                gc.collect();torch.mps.empty_cache()
            assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(mps_rng,torch.mps.get_rng_state())
            measured=gpu_rows[4:];ratios=[r['candidate_over_reference'] for r in measured]
            phases={name:statistics.median(r['candidate_over_reference'] for r in measured if lo<=r['step']<=hi)
                    for name,lo,hi in [('before_averaging',5,12),('with_averaging',13,20)]}
            faster=sum(r<1 for r in ratios)
            gates=dict(median_at_most_095=statistics.median(ratios)<=.95,
                       at_least_12_of_16_faster=faster>=12,neither_phase_regresses=all(r<=1 for r in phases.values()))
            result=dict(completed=True,exact_history_and_endpoint=True,cpu=cpu_evidence,
                        useful_training_improvement=all(gates.values()),gates=gates,measured_pairs=16,warmup_pairs=4,
                        median_candidate_over_reference=statistics.median(ratios),paired_faster=faster,
                        phase_median_ratios=phases,measured_seconds={label:sum(r['seconds'][label] for r in measured)
                                                                  for label in ('reference','batched')},
                        total_numerical_updates=45,quality_evaluations=0,adopted=False,
                        model_math_and_optimizer_arithmetic_unchanged=True,exact_cached_logits_and_reset=True,plan_sha256=sha(OUT/'plan.json'),
                        runner_sha256=sha(Path(__file__)),rng_unchanged=True,limits=plan['interpretation_limits'])
        assert read(OUT/'guard.json')['status']=='finished'
        result['guard_sha256']=sha(OUT/'guard.json')
        result['evidence']={p.name:sha(p) for p in sorted(OUT.glob('*.json')) if p.name.startswith(('cpu-','mps-'))}
        publish('result.json',result)
        publish('STOP',dict(status='complete_useful_improvement_pending_adoption' if result['useful_training_improvement'] else 'complete_no_useful_training_gain',no_replay=True))
    print(json.dumps({k:result[k] for k in ('completed','exact_history_and_endpoint','cpu','useful_training_improvement','gates','median_candidate_over_reference','paired_faster','phase_median_ratios','measured_seconds')}))


if __name__=='__main__':
    try:run()
    except Exception:
        publish('failure.json',dict(traceback=traceback.format_exc(),no_automatic_replay=True))
        raise
