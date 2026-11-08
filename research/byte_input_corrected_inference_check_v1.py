"""Bound selected-checkpoint public-inference proof; no quality re-evaluation."""
import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import traceback
from research.byte_context_transform_pilot_v1 import ROOT, sha, put, read, runtime, pins

OUT=ROOT/'research/tokenizer-evidence/public-inference-correction-001'
RECIPE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001/bulk-hash-adoption001/recipe.json'


def record(path):
    path=Path(path)
    return dict(path=str(path),size=path.stat().st_size,sha256=sha(path))


def prepare():
    assert not OUT.exists()
    from research import byte_input_corrected_inference_v1
    from research.byte_input_training_v6 import load_view
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    recipe=read(RECIPE);loader=recipe['loader']
    old=read(loader['plan_record']['path']);impl=read(loader['implementation_record']['path'])
    mapping=read(impl['source_map']['path'])
    sources={}
    def add(path):
        p=Path(path);p=p if p.is_absolute() else ROOT/p
        sources[str(p.relative_to(ROOT))]=sha(p)
    for r in old['sources']+impl['sources']:add(r['path'])
    for pair in mapping['pairs']:
        for side in ('original','candidate'):add(pair[side]['path'])
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).suffix=='.py' and Path(path).is_relative_to(ROOT):
            relative=Path(path).relative_to(ROOT)
            if relative.parts[0] in ('research','src'):add(path)
    add(__file__)
    assets=[RECIPE,ROOT/'research/PUBLIC_INFERENCE_CORRECTION_PROTOCOL.md']
    for folder in ('inference-epsilon003','cpu-reference-recovery001'):
        base=ROOT/'research/tokenizer-evidence/byte-context-transform-001'/folder
        accepted=read(base/'acceptance.json');assert accepted['accepted']
        assert accepted['result_sha256']==sha(base/'result.json')
        assets.extend(base/n for n in ('plan.json','acceptance.json','result.json','guard.json'))
    assets.extend(Path(loader[k]['path']) for k in ('plan_record','implementation_record','acceptance_record'))
    assets.extend(Path(r['path']) for r in old['assets'])
    assets.append(Path(recipe['representative']['h200']['path']))
    plan=dict(kind='public_inference_correction_check_v1',runtime=runtime(),sources=sources,
        assets=[record(p) for p in dict.fromkeys(assets)],recipe=record(RECIPE),loader=loader,
        checkpoint=recipe['representative']['h200'],batches=[1,8],context=128,prefill=96,
        logit_atol=3e-5,logit_rtol=3e-4,fp64_ce_atol=3e-5,mean_ce_atol=3e-5,target_ce_atol=3e-4,
        seconds=600,rss=4*1024**3,mps=6*1024**3,reserve=8*1024**3,extra_headroom=2*1024**3,
        output_cap=16*1024**2,no_training=True,no_validation_scoring=True,no_holdout_scoring=True)
    assert shutil.disk_usage(ROOT).free>=plan['reserve']+plan['extra_headroom']+plan['output_cap']
    OUT.mkdir();put(OUT/'plan.json',plan);print(sha(OUT/'plan.json'),flush=True)


def run(expected):
    import torch
    from torch.nn import functional as F
    from research import byte_input_corrected_inference_v1 as public
    from research import byte_input_inference_v5 as original
    from research.byte_input_training_v6 import load_view
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    assert sha(OUT/'plan.json')==expected
    plan=read(OUT/'plan.json');pins(plan)
    assert not (OUT/'attempt.json').exists()

    class Guard(CpuPrerequisiteGuard):
        def _sample(self):
            sample,reason=super()._sample()
            sample['mps_bytes']=torch.mps.driver_allocated_memory()
            sample['output_bytes']=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
            if sample['mps_bytes']>=plan['mps']:reason='mps_limit'
            if sample['output_bytes']>plan['output_cap']:reason='output_limit'
            return sample,reason

    def digest(t):
        import hashlib
        return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    def rejected(fn):
        try:fn()
        except ValueError:return
        raise AssertionError('Invalid public operation accepted')

    def compare(a,b):
        error=(a-b).abs()
        ratio=float((error/(plan['logit_atol']+plan['logit_rtol']*b.abs())).max())
        row=dict(max_absolute_error=float(error.max()),max_gate_ratio=ratio)
        assert ratio<=1,row
        return row

    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free>=plan['reserve']+plan['extra_headroom']+plan['output_cap']
        put(OUT/'attempt.json',dict(pid=os.getpid(),plan_sha256=expected))
        try:
            with Guard(OUT/'guard.json',ROOT,timeout_seconds=plan['seconds'],rss_limit_bytes=plan['rss'],reserve_bytes=plan['reserve'],poll_seconds=.5):
                assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
                assert not torch.is_inference_mode_enabled()
                torch.set_num_threads(4)
                recipe=read(RECIPE);loader=plan['loader'];paired={};cases=[]
                for backend in ('cpu','mps'):
                    rng=(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
                    handle=public.load(loader['implementation_record'],loader['plan_record'],loader['acceptance_record'],backend)
                    assert handle.artifact==plan['checkpoint']
                    payload=torch.load(handle.artifact['path'],map_location='cpu',weights_only=True)
                    before={n:digest(p) for n,p in handle.model.ordered_parameters()}
                    assert before=={n:digest(p) for n,p in payload['parameters'].items()}
                    del payload
                    semantics=handle.model._semantic_inventory()
                    view=load_view(handle.identity['roles'],'cpu')
                    paired[backend]={}
                    for batch in plan['batches']:
                        inputs=view.train_data[:batch*128].reshape(batch,128).to(backend)
                        targets=view.train_data[1:batch*128+1].reshape(batch,128).to(backend)
                        full=handle.logits(inputs);assert not full.requires_grad
                        with torch.enable_grad():reference=original._logits(handle.model,inputs).detach()
                        equivalence=compare(full,reference);del reference
                        losses=handle.token_losses(inputs,targets);assert not losses.requires_grad
                        fp64=[]
                        for i in range(batch):
                            values=F.cross_entropy(full[i].cpu().double(),targets[i].cpu(),reduction='none')
                            fp64.extend(values.tolist())
                        actual=losses.cpu().double().reshape(-1).tolist()
                        max_fp64=max(abs(x-y) for x,y in zip(actual,fp64,strict=True))
                        assert max_fp64<=plan['fp64_ce_atol']
                        paired[backend][batch]=actual
                        session=handle.session(batch);chunks=[(0,96)]+[(i,i+1) for i in range(96,128)]
                        checks=[];first=[];state=None
                        for replay in (0,1):
                            if replay:
                                session.reset();assert all(not bool(t.any()) for t in session._tensors)
                            for i,(lo,hi) in enumerate(chunks):
                                logits=session.logits(inputs[:,lo:hi],lo);assert not logits.requires_grad
                                row=dict(replay=replay,start=lo,end=hi,**compare(logits,full[:,lo:hi]),sha256=digest(logits))
                                checks.append(row)
                                if replay:assert row['sha256']==first[i]
                                else:first.append(row['sha256'])
                                del logits
                            snapshot=[digest(t) for t in session._tensors]
                            assert session.position==128 and session.storage_bytes==196608*batch
                            if replay:assert snapshot==state
                            else:state=snapshot
                        session.reset();rejected(lambda:session.logits(inputs[:,:1],1));assert session.position==0
                        row=dict(backend=backend,batch=batch,training_target_count=len(actual),full_vs_training=equivalence,
                            max_loss_error_vs_fp64=max_fp64,token_losses=actual,cache_checks=checks,reset_exact=True)
                        put(OUT/f'{backend}-b{batch}.json',row);cases.append(row)
                        del full,losses,session;gc.collect();torch.mps.empty_cache()
                    rejected(lambda:handle.logits(torch.full_like(inputs,-1)))
                    inputs=view.train_data[:128].reshape(1,128).to(backend)
                    targets=view.train_data[1:129].reshape(1,128).to(backend)
                    rejected(lambda:handle.token_losses(inputs,targets.float()))
                    rejected(lambda:handle.token_losses(inputs,targets[:,:127]))
                    rejected(lambda:handle.token_losses(inputs,torch.full_like(targets,50257)))
                    handle.model.train();rejected(lambda:handle.logits(inputs));handle.model.eval()
                    session=handle.session(1)
                    with torch.inference_mode():
                        rejected(lambda:handle.logits(inputs));rejected(lambda:session.logits(inputs[:,:1],0))
                    with torch.autocast(backend,dtype=torch.float16):
                        rejected(lambda:handle.logits(inputs));rejected(lambda:session.logits(inputs[:,:1],0))
                    assert session.position==0
                    handle.check();assert handle.model._semantic_inventory()==semantics
                    assert {n:digest(p) for n,p in handle.model.ordered_parameters()}==before
                    assert all(p.grad is None for _,p in handle.model.ordered_parameters())
                    with torch.no_grad():
                        parameter=next(handle.model.parameters());parameter.copy_(parameter)
                    rejected(lambda:handle.logits(inputs));rejected(lambda:session.logits(inputs[:,:1],0))
                    assert {n:digest(p) for n,p in handle.model.ordered_parameters()}==before
                    assert rng==(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
                    del handle,session,view,parameter,inputs,targets;gc.collect();torch.mps.empty_cache()
                numerical={}
                for batch in plan['batches']:
                    a,b=paired['cpu'][batch],paired['mps'][batch]
                    error=dict(mean_ce_error=abs(math.fsum(a)-math.fsum(b))/len(a),max_target_ce_error=max(abs(x-y) for x,y in zip(a,b,strict=True)))
                    numerical[str(batch)]=error
                    assert error['mean_ce_error']<=plan['mean_ce_atol'] and error['max_target_ce_error']<=plan['target_ce_atol']
                pins(plan)
                assert sha(plan['checkpoint']['path'])==plan['checkpoint']['sha256']
                put(OUT/'result.json',dict(completed=True,adopted=False,plan_sha256=expected,numerical=numerical,
                    cache_cases=len(cases),public_loader_verified=True,original_parameter_values_exact=True,
                    weights_semantics_rng_unchanged=True,no_gradients=True,no_training=True,
                    no_validation_scoring=True,no_holdout_scoring=True,invalid_inputs_rejected=True,
                    value_preserving_version_mutation_rejected=True))
            put(OUT/'acceptance.json',dict(accepted=True,result_sha256=sha(OUT/'result.json'),guard_sha256=sha(OUT/'guard.json')))
        except BaseException:
            put(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));p.add_argument('plan_sha',nargs='?');a=p.parse_args()
    prepare() if a.action=='prepare' else run(a.plan_sha)
