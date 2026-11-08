"""Verify the specific inference-only RMSNorm correction on the failed full split."""
import argparse,fcntl,hashlib,json,os,traceback
from pathlib import Path
from research.byte_context_transform_pilot_v1 import ROOT,BASE,sha,put,read,runtime,pins
OUT=BASE/'scoring-recovery001'


def prepare():
    assert not OUT.exists()
    parent=read(BASE/'rms-diagnostic001/plan.json');pins(parent)
    inputs=[BASE/'rms-diagnostic001/result.json',BASE/'rms-diagnostic001/acceptance.json']
    sources={str(p.relative_to(ROOT)):sha(p) for f in ('research','src') for p in (ROOT/f).rglob('*.py')}
    plan=dict(kind='byte_context_scoring_recovery_v1',sources=sources,assets=parent['assets']+[dict(path=str(p),sha256=sha(p)) for p in inputs],runtime=runtime(),
              weight_sha256=parent['weight_sha256'],mean_ce_atol=3e-5,target_ce_atol=3e-4,
              scope='Same full34278 targets and saved new slot00H200 weights, corrected inference graph path. No training, gradients, seed replacement, tolerance relaxation or historical fixture replay.',
              correction='torch.enable_grad forward with immediate detached losses, no backward; retains intended FP32 epsilon from training path.',
              references=['https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/RMSNorm.mm',
                          'https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/layer_norm.cpp'],
              limits=dict(seconds=600,rss=4*1024**3,mps=6*1024**3,reserve=8*1024**3,output=16*1024**2))
    OUT.mkdir();put(OUT/'plan.json',plan);print(sha(OUT/'plan.json'),flush=True)


def weight_digest(model):
    h=hashlib.sha256()
    for name,p in model.ordered_parameters():h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run():
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    import torch
    from research.byte_context_transform_v1 import construct,load_weights
    from research.byte_context_scoring_v2 import score
    from research.byte_input_model_v3 import build_codebook
    from research.byte_input_training_v6 import load_view
    plan=read(OUT/'plan.json');pins(plan);qp=read(BASE/'quality-plan.json');spec=qp['specs'][0]
    class Guard(CpuPrerequisiteGuard):
        def _sample(self):
            sample,reason=super()._sample();sample['mps_bytes']=torch.mps.driver_allocated_memory()
            if sample['mps_bytes']>=6*1024**3:reason='mps_limit'
            sample['retained_bytes']=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
            if sample['retained_bytes']>=16*1024**2:reason='output_limit'
            return sample,reason
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        put(OUT/'attempt.json',dict(pid=os.getpid(),plan_sha256=sha(OUT/'plan.json')))
        try:
            with Guard(OUT/'guard.json',ROOT,timeout_seconds=600,poll_seconds=.5):
                torch.set_num_threads(4);assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
                book=build_codebook(*(Path(r['path']).read_bytes() for r in qp['codebook_assets']),'raw_bytes')
                path=ROOT/'research_runs/byte-context-transform-001/00/h200.pt';assert sha(path)==plan['weight_sha256']
                saved=torch.load(path,map_location='cpu',weights_only=True)
                model=construct('byte',spec['roles'],spec['projection_seed'],book,'mps');load_weights(model,saved['weights'])
                assert saved['identity']==model.identity and saved['spec']==spec and saved['plan_sha256']==sha(BASE/'quality-plan.json')
                view=load_view(spec['roles'],'mps');before=weight_digest(model);cpu_rng=torch.get_rng_state().clone();mps_rng=torch.mps.get_rng_state().clone()
                mps=score(model,view,'mps');put(OUT/'validation-mps.json',mps)
                model.to('cpu');cpu=score(model,view,'cpu');put(OUT/'validation-cpu.json',cpu)
                original=read(BASE/'00/validation-cpu.json')
                result=dict(completed=True,mean_ce_error=abs(mps['ce']-cpu['ce']),
                    max_target_ce_error=max(abs(a-b) for a,b in zip(mps['per_target_losses'],cpu['per_target_losses'])),
                    cpu_original_exact=cpu['per_target_losses']==original['per_target_losses'],
                    weights_unchanged=weight_digest(model)==before,gradients_absent=all(p.grad is None for _,p in model.ordered_parameters()),
                    cpu_rng_unchanged=torch.equal(torch.get_rng_state(),cpu_rng),mps_rng_unchanged=torch.equal(torch.mps.get_rng_state(),mps_rng),
                    corrected_mps_ce=mps['ce'],corrected_cpu_ce=cpu['ce'],training_updates=0,
                    original_failure=read(BASE/'00/numerical.json'),quality_panel_accepted=False)
                put(OUT/'result.json',result)
                assert result['mean_ce_error']<=plan['mean_ce_atol'] and result['max_target_ce_error']<=plan['target_ce_atol']
                assert all(result[k] for k in ['cpu_original_exact','weights_unchanged','gradients_absent','cpu_rng_unchanged','mps_rng_unchanged'])
                pins(plan)
            put(OUT/'acceptance.json',dict(accepted=True,result_sha256=sha(OUT/'result.json'),guard_sha256=sha(OUT/'guard.json')))
            print(json.dumps(read(OUT/'result.json')),flush=True)
        except BaseException:
            put(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));a=p.parse_args()
    prepare() if a.action=='prepare' else run()
