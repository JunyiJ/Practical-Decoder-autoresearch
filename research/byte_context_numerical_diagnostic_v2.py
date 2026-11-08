"""Read-only localization of the new saved tanh-candidate CPU/MPS discrepancy."""
import argparse,fcntl,hashlib,json,os,sys,traceback
from pathlib import Path
from research.byte_context_transform_pilot_v1 import ROOT,BASE,sha,put,read,runtime,pins
Q=BASE/'quality-recovery001'
OUT=BASE/'numerical-diagnostic002'
TORCH=None


def prepare():
    assert not OUT.exists()
    qp=read(Q/'quality-plan.json');pins(qp)
    failure=read(Q/'02/numerical.json')
    assert failure['mean_ce_error']<=3e-5 and failure['max_target_ce_error']>3e-4
    a=read(Q/'02/validation-mps.json')['per_target_losses'];b=read(Q/'02/validation-cpu.json')['per_target_losses']
    worst=max(range(len(a)),key=lambda i:abs(a[i]-b[i]))
    sources={str(p.relative_to(ROOT)):sha(p) for f in ('research','src') for p in (ROOT/f).rglob('*.py')}
    inputs=[Q/'quality-plan.json',Q/'02/failure.json',Q/'02/numerical.json',Q/'02/validation-mps.json',Q/'02/validation-cpu.json',ROOT/'research_runs/byte-context-transform-recovery-001/02/h200.pt']
    plan=dict(kind='byte_context_tanh_numerical_localization_v2',sources=sources,assets=qp['assets']+[dict(path=str(p),sha256=sha(p)) for p in inputs],runtime=runtime(),
              quality_plan_sha256=sha(Q/'quality-plan.json'),weight_sha256=sha(inputs[-1]),
              block_starts=sorted(set([0,(worst//128)*128])),worst_target=worst,
              purpose='Locate propagated/isolated body, context and loss error on fixed first and worst-error blocks; same grad-enabled forward as failed scorer, no backward/training/quality selection.',
              loss_reference_probe='Compare identical FP32 logits under CPU and MPS cross-entropy against CPU64 logsumexp reference. Isolate projection/tanh/readout using common CPU inputs. Does not adopt a new scorer.',
              historical_scope='Uses only new byte-context recovery slot02 weights. Closed native projection and anchor diagnostics remain closed; no replay or use of their scores.',
              limits=dict(seconds=600,rss=4*1024**3,mps=6*1024**3,output=16*1024**2,reserve=8*1024**3),holdout_scored=False)
    OUT.mkdir();put(OUT/'plan.json',plan);print(json.dumps(dict(plan_sha256=sha(OUT/'plan.json'),blocks=plan['block_starts'],worst_target=worst)),flush=True)


def delta(a,b):
    import torch
    a=a.detach().cpu().double();b=b.detach().cpu().double();d=a-b
    return dict(max_abs=float(d.abs().max()),rms=float(d.square().mean().sqrt()),reference_rms=float(a.square().mean().sqrt()))


def run():
    global TORCH
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    plan=read(OUT/'plan.json');pins(plan);qp=read(Q/'quality-plan.json');spec=qp['specs'][2]
    class Guard(CpuPrerequisiteGuard):
        def _sample(self):
            sample,reason=super()._sample()
            if TORCH is not None:
                sample['mps_bytes']=TORCH.mps.driver_allocated_memory()
                if sample['mps_bytes']>=plan['limits']['mps']:reason='mps_limit'
            sample['retained_bytes']=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
            if sample['retained_bytes']>=plan['limits']['output']:reason='output_limit'
            return sample,reason
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        put(OUT/'attempt.json',dict(pid=os.getpid(),plan_sha256=sha(OUT/'plan.json')))
        try:
            with Guard(OUT/'guard.json',ROOT,timeout_seconds=600,poll_seconds=.5):
                import torch
                from torch.nn import functional as F
                from research.byte_context_transform_v1 import construct,load_weights
                from research.byte_input_model_v3 import build_codebook
                from research.byte_input_training_v6 import load_view
                TORCH=torch;torch.set_num_threads(4)
                assert torch.backends.mps.is_available() and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
                book=build_codebook(*(Path(r['path']).read_bytes() for r in qp['codebook_assets']),'raw_bytes')
                payload=torch.load(ROOT/'research_runs/byte-context-transform-recovery-001/02/h200.pt',map_location='cpu',weights_only=True)
                assert payload['plan_sha256']==sha(Q/'quality-plan.json') and payload['spec']==spec and payload['codebook_identity']==book.identity
                cpu=construct('tanh',spec['roles'],spec['projection_seed'],book,'cpu');load_weights(cpu,payload['weights'])
                mps=construct('tanh',spec['roles'],spec['projection_seed'],book,'mps');load_weights(mps,payload['weights'])
                assert payload['identity']==cpu.identity==mps.identity
                view=load_view(spec['roles'],'cpu');reports=[]
                with torch.enable_grad():
                    for start in plan['block_starts']:
                        ids=view.val_data[start:start+128].unsqueeze(0);targets=view.val_data[start+1:start+129]
                        stages={};traces={}
                        for dev,model in [('cpu',cpu),('mps',mps)]:
                            model.eval();base=model.base;x=ids.to(dev);indices=x.reshape(-1)
                            h=base.composer(base.code_ids.index_select(0,indices),base.code_lengths.index_select(0,indices)).reshape(1,128,256)
                            traces[dev]={'composer':h.detach().cpu()}
                            for i,block in enumerate(base.blocks):
                                h,aux=block(h,None,0);assert aux is None;traces[dev][f'block{i}']=h.detach().cpu()
                            z=base.final_norm(h);traces[dev]['final_norm']=z.detach().cpu()
                            projected=F.linear(z,model.context_weight);traces[dev]['projected']=projected.detach().cpu()
                            transformed=torch.tanh(projected);traces[dev]['tanh']=transformed.detach().cpu()
                            logits=base.readout(transformed)/4.;traces[dev]['logits']=logits.detach().cpu()
                            traces[dev]['losses']=F.cross_entropy(logits[0],targets.to(dev),reduction='none').detach().cpu()
                        stages={n:delta(traces['cpu'][n],traces['mps'][n]) for n in traces['cpu']}
                        injected=[]
                        for i,block in enumerate(mps.base.blocks):
                            h=traces['cpu']['composer' if i==0 else f'block{i-1}'].to('mps')
                            got,_=block(h,None,0);injected.append(dict(block=i,error=delta(traces['cpu'][f'block{i}'],got)))
                        isolated=[]
                        for name,x in [('projection',traces['cpu']['final_norm']),('tanh',traces['cpu']['projected']),('readout',traces['cpu']['tanh'])]:
                            values={}
                            for dev,model in [('cpu',cpu),('mps',mps)]:
                                common=x.to(dev).detach().requires_grad_()
                                value=F.linear(common,model.context_weight) if name=='projection' else torch.tanh(common) if name=='tanh' else model.base.readout(common)/4.
                                values[dev]=value.detach().cpu()
                            isolated.append(dict(stage=name,error=delta(values['cpu'],values['mps'])))
                        loss_probes=[]
                        for origin in ('cpu','mps'):
                            common=traces[origin]['logits'][0]
                            high=common.double()
                            expected=torch.logsumexp(high,dim=-1)-high.gather(1,targets[:,None]).squeeze(1)
                            values={}
                            for dev in ('cpu','mps'):
                                incoming=common.to(dev).detach().requires_grad_()
                                values[dev]=F.cross_entropy(incoming,targets.to(dev),reduction='none').detach().cpu()
                            loss_probes.append(dict(logits_origin=origin,cpu_vs_fp64=delta(expected,values['cpu']),
                                mps_vs_fp64=delta(expected,values['mps']),same_input_backend_error=delta(values['cpu'],values['mps']),
                                cpu_losses=values['cpu'].tolist(),mps_losses=values['mps'].tolist(),fp64_losses=expected.tolist()))
                        cpu_ce_from_mps_logits=F.cross_entropy(traces['mps']['logits'][0],targets,reduction='none')
                        reports.append(dict(block_start=start,propagated_stages=stages,isolated_blocks=injected,
                            isolated_context=isolated,isolated_cross_entropy=loss_probes,
                            common_cpu_ce_backend_logits_error=delta(traces['cpu']['losses'],cpu_ce_from_mps_logits)))
                put(OUT/'result.json',dict(completed=True,reports=reports,no_training=True,no_gradients=True,no_adoption=True,
                                         source_failure=read(Q/'02/numerical.json'),quality_evidence_unchanged=True))
                assert all(p.grad is None for model in (cpu,mps) for _,p in model.ordered_parameters())
                pins(plan)
            put(OUT/'acceptance.json',dict(accepted=True,result_sha256=sha(OUT/'result.json'),guard_sha256=sha(OUT/'guard.json')))
            print(json.dumps(read(OUT/'result.json')),flush=True)
        except BaseException:
            put(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));a=p.parse_args()
    prepare() if a.action=='prepare' else run()
