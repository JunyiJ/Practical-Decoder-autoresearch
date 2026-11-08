"""Full-split saved-weight verification of a CPU-only reference correction."""
import argparse,fcntl,json,math,os,shutil,time,traceback
from pathlib import Path
from research.byte_context_transform_pilot_v1 import ROOT,BASE,sha,put,read,runtime,pins

Q=BASE/'quality-recovery001'
OUT=BASE/'cpu-reference-recovery001'


def prepare():
    assert not OUT.exists()
    original=read(Q/'quality-plan.json');pins(original)
    assert read(BASE/'numerical-diagnostic002/acceptance.json')['accepted']
    inputs=[Q/'quality-plan.json',Q/'slot02-failure-preservation.json',
            BASE/'numerical-diagnostic002/plan.json',BASE/'numerical-diagnostic002/result.json']
    checkpoints=[]
    for slot in range(3):
        folder=Q/f'{slot:02d}'
        inputs += [folder/'validation-cpu.json',folder/'validation-mps.json',folder/'numerical.json']
        p=ROOT/f'research_runs/{"byte-context-transform-001" if slot==0 else "byte-context-transform-recovery-001"}/{slot:02d}/h200.pt'
        inputs.append(p);checkpoints.append(dict(slot=slot,path=str(p),sha256=sha(p)))
    plan=dict(kind='byte_context_cpu_reference_recovery_v1',runtime=runtime(),
        sources={str(p.relative_to(ROOT)):sha(p) for f in ('research','src') for p in (ROOT/f).rglob('*.py')},
        assets=original['assets']+[dict(path=str(p),sha256=sha(p)) for p in inputs],
        checkpoints=checkpoints,slots=[0,1,2],targets=34278,
        mean_ce_atol=3e-5,target_ce_atol=3e-4,cpu_reference_fp64_atol=3e-5,
        primary_mps_policy='Use original saved primary reports unchanged; no new MPS quality score',
        native_cpu_policy='Reproduce every original CPU target score exactly on each completed checkpoint',
        seconds=600,rss=4*1024**3,reserve=8*1024**3,output_cap=32*1024**2,
        no_training=True,no_checkpoint_rewrite=True,no_threshold_change=True)
    assert shutil.disk_usage(ROOT).free>=plan['reserve']+plan['output_cap']+2*1024**3
    OUT.mkdir();put(OUT/'plan.json',plan);print(sha(OUT/'plan.json'),flush=True)


def run(expected):
    import torch
    from torch.nn import functional as F
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    from research.byte_context_cpu_reference_v1 import cpu_cross_entropy
    from research.byte_context_transform_v1 import construct,load_weights
    from research.byte_input_model_v3 import build_codebook
    from research.byte_input_training_v6 import load_view
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
        put(OUT/'attempt.json',dict(pid=os.getpid(),plan_sha256=expected))
        try:
            with Guard(OUT/'guard.json',ROOT,timeout_seconds=600,reserve_bytes=plan['reserve']):
                torch.set_num_threads(4)
                original=read(Q/'quality-plan.json')
                book=build_codebook(*(Path(r['path']).read_bytes() for r in original['codebook_assets']),'raw_bytes')
                rows=[]
                for record in plan['checkpoints']:
                    slot=record['slot'];spec=original['specs'][slot]
                    saved=torch.load(record['path'],map_location='cpu',weights_only=True)
                    assert saved['spec']==spec
                    model=construct(spec['arm'],spec['roles'],spec['projection_seed'],book,'cpu')
                    load_weights(model,saved['weights']);model.eval();view=load_view(spec['roles'],'cpu')
                    cpu_rng=torch.random.get_rng_state().clone();sampler=view.generators['train'].get_state().clone()
                    old=read(Q/f'{slot:02d}/validation-cpu.json')['per_target_losses']
                    mps=read(Q/f'{slot:02d}/validation-mps.json')['per_target_losses']
                    losses=[];native=[];maximum=0.;old_maximum=0.;started=time.perf_counter()
                    with torch.enable_grad():
                        for start in range(0,plan['targets'],128):
                            count=min(128,plan['targets']-start)
                            ids=view.val_data[start:start+count].unsqueeze(0)
                            targets=view.val_data[start+1:start+count+1]
                            logits,_=model(ids);x=logits[0].detach()
                            values=cpu_cross_entropy(x,targets)
                            original_values=F.cross_entropy(x,targets,reduction='none')
                            high=x.double();high=high-high.amax(dim=-1,keepdim=True)
                            reference=high.exp().sum(dim=-1).log()-high.gather(-1,targets[:,None]).squeeze(-1)
                            maximum=max(maximum,float((values.double()-reference).abs().max()))
                            old_maximum=max(old_maximum,float((original_values.double()-reference).abs().max()))
                            assert torch.isfinite(values).all() and (values>=0).all()
                            losses.extend(values.double().tolist());native.extend(original_values.double().tolist())
                            del logits,x,values,original_values,high,reference
                    assert len(losses)==plan['targets']
                    numerical=dict(mean_ce_error=abs(math.fsum(losses)-math.fsum(mps))/plan['targets'],
                        max_target_ce_error=max(abs(a-b) for a,b in zip(losses,mps,strict=True)))
                    report=dict(ce=math.fsum(losses)/plan['targets'],per_target_losses=losses,
                        target_count=plan['targets'],seconds=time.perf_counter()-started,holdout_scored=False,
                        device='cpu',scoring_policy='byte_context_cpu_stable_fp32_reference_v1')
                    row=dict(slot=slot,numerical=numerical,fp64_max_error=maximum,native_fp64_max_error=old_maximum,
                        original_cpu_exact=native==old,primary_mps_sha256=sha(Q/f'{slot:02d}/validation-mps.json'))
                    put(OUT/f'{slot:02d}-cpu.json',report);put(OUT/f'{slot:02d}-verification.json',row)
                    assert native==old
                    assert maximum<=plan['cpu_reference_fp64_atol'] and maximum<old_maximum
                    assert numerical['mean_ce_error']<=plan['mean_ce_atol'] and numerical['max_target_ce_error']<=plan['target_ce_atol']
                    assert all(torch.equal(p.detach(),saved['weights'][n]) and p.grad is None for n,p in model.ordered_parameters())
                    assert torch.equal(cpu_rng,torch.random.get_rng_state()) and torch.equal(sampler,view.generators['train'].get_state())
                    model.validate();rows.append(row);del model,view,saved
                    print(json.dumps(row),flush=True)
                pins(plan)
                put(OUT/'result.json',dict(completed=True,rows=rows,primary_scores_unchanged=True,no_training=True,
                    no_checkpoint_rewrite=True,no_threshold_change=True,plan_sha256=expected,adopted=False))
            put(OUT/'acceptance.json',dict(accepted=True,result_sha256=sha(OUT/'result.json'),guard_sha256=sha(OUT/'guard.json')))
        except BaseException:
            put(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));p.add_argument('plan_sha',nargs='?');a=p.parse_args()
    prepare() if a.action=='prepare' else run(a.plan_sha)
