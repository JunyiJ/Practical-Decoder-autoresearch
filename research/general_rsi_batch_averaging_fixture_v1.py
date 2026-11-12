"""Qualify only the new one-step accumulation integration."""
import fcntl,gc,json,os,shutil,signal,time,traceback
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,check_stop,batch_hash
from research.general_rsi_batch_averaging_v1 import update_once
OUT=ROOT/'research/tokenizer-evidence/general-rsi-batch-averaging-fixture-001'

def execute(p,binding):
    import torch
    from research.general_rsi_continuation_decay_v1 import Context
    from research.general_rsi_qualification_v1 import equal
    assert torch.__version__==p['torch_version'] and os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0';torch.set_num_threads(4)
    def context():return Context('none',1200*10**9,60*10**9,p['roles'],p['pseudo_seed'],p['parent'],binding)
    c=context();canonical=c.update();state=c.capture();del c;gc.collect();torch.mps.empty_cache()
    c=context();one=update_once(c,1);new=c.capture()
    for k in ['parameters','children','samplers','cpu_rng','mps_rng','step','ema','snapshots']:equal(state[k],new[k])
    assert one['microbatch_sha256']==[canonical['batch_sha256']] and one['clip']==canonical['clip'] and one['loss']==canonical['loss']
    del c,state,new;gc.collect();torch.mps.empty_cache()
    observed={}
    def capture(model):
        for n,v in model.ordered_parameters():observed[n]=v.grad.detach().cpu().clone()
    c=context();eight=update_once(c,8,capture);c.optimizer.validate();peak=torch.mps.driver_allocated_memory();del c;gc.collect();torch.mps.empty_cache()
    c=context();oracle={};hashes=[]
    for i in range(8):
        check_stop();c.optimizer.zero_grad();x,y=c.data.batch();hashes.append(batch_hash(x,y));loss=c.model.forward_train(x,y).loss;loss.backward();del loss
        for n,v in c.model.ordered_parameters():
            value=v.grad.detach().cpu()
            if i==0:oracle[n]=value.clone()
            else:oracle[n].add_(value)
    checks=[]
    for n,g in observed.items():
        want=oracle[n]/8;error=(g-want).abs();assert torch.isfinite(g).all() and torch.allclose(g,want,atol=p['gradient_atol'],rtol=p['gradient_rtol'])
        checks.append(dict(name=n,max_abs_error=float(error.max()),reference_max_abs=float(want.abs().max())))
    assert hashes==eight['microbatch_sha256'] and len(checks)==88 and c.step==0 and c.pseudo.draws==0
    peak=max(peak,torch.mps.driver_allocated_memory());assert peak<6*2**30
    return dict(plan_sha256=binding,accepted=True,single_batch_parameter_and_full_child_state_exact=True,single_batch_sampler_rng_exact=True,independent_gradient_checks=checks,gradient_atol=p['gradient_atol'],gradient_rtol=p['gradient_rtol'],one=one,eight=eight,eight_update_seconds=(eight['end_ns']-eight['begin_ns'])/1e9,peak_mps_bytes=peak,optimizer_updates=3,oracle_backward_passes=8,total_forward_backward_passes=18,new_checkpoints=0,test_opened=False,validation_scored=False)

def main():
    lock=open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop()
    p=json.loads((OUT/'plan.json').read_text());binding=sha(OUT/'plan.json');assert all(sha(ROOT/f)==h for f,h in p['sources'].items());assert shutil.disk_usage(ROOT).free>=p['required_free_bytes']
    a=OUT/'attempt-01';a.mkdir();put(a/'launch.json',dict(pid=os.getpid(),plan_sha256=binding,started_unix=time.time()))
    def stop(*_):
        import research.general_rsi_pseudo_update_diagnostic_v1 as shared
        shared.STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    try:
        with CpuPrerequisiteGuard(a/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4*2**30,reserve_bytes=10*2**30):
            result=execute(p,binding);assert all(sha(ROOT/f)==h for f,h in p['sources'].items());put(a/'result.json',result);print(json.dumps(dict(passed=True,eight_seconds=result['eight_update_seconds'])),flush=True)
    except BaseException:
        put(a/'failure.json',dict(traceback=traceback.format_exc(),plan_sha256=binding));raise
if __name__=='__main__':main()
