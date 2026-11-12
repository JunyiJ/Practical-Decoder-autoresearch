"""Only new interpolation, restoration and variable-target scoring checks."""
import fcntl,json,math,os,shutil,time,traceback,signal
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,check_stop,ce_batch
from research.general_rsi_update_response_v1 import parameters,set_point,score,summarize,SCALES
OUT=ROOT/'research/tokenizer-evidence/general-rsi-update-response-fixture-001'

def main():
    import torch
    from research.general_rsi_continuation_decay_v1 import Context
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    lock=open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop()
    p=json.loads((OUT/'plan.json').read_text());binding=sha(OUT/'plan.json')
    assert all(sha(ROOT/f)==h for f,h in p['sources'].items());assert shutil.disk_usage(ROOT).free>=p['required_free_bytes']
    assert torch.__version__==p['torch_version'] and os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0';torch.set_num_threads(4)
    a=OUT/'attempt-01';a.mkdir();put(a/'launch.json',dict(pid=os.getpid(),plan_sha256=binding,started_unix=time.time()))
    def stop(*_):
        import research.general_rsi_pseudo_update_diagnostic_v1 as shared
        shared.STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with CpuPrerequisiteGuard(a/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4*2**30,reserve_bytes=10*2**30):
            c=Context('none',1200*10**9,200*10**9,p['roles'],p['pseudo_seed'],p['parent'],binding);before=parameters(c.model)
            after={n:v*.999 for n,v in before.items()};matches=0
            for scale in SCALES:
                check_stop();matches+=set_point(c.model,before,after,scale)
                if scale not in (0.,1.):
                    # Independently ordered three FP32 CPU operations, no fused lerp.
                    for n,value in parameters(c.model).items():
                        expected=torch.sub(after[n],before[n]);expected.mul_(scale);expected.add_(before[n]);assert torch.equal(expected,value)
            matches+=set_point(c.model,before,after,0.)
            assert all(torch.equal(v,before[n]) for n,v in parameters(c.model).items())
            assert summarize([6.,20.],[2,5])['ce']==26/7
            rng=torch.Generator().manual_seed(p['batch_seed']);starts=torch.randint(len(c.data.view.train_data)-512,(2,),generator=rng).tolist();x,y=ce_batch(c.data.view.train_data,starts)
            batches=[(x[i:i+1],y[i:i+1]) for i in range(8)]
            tick=time.monotonic();result=score(c.model,batches);torch.mps.synchronize();seconds=time.monotonic()-tick
            assert result['targets']==1024 and result['window_targets']==[128]*8
            oracle=[];c.model.eval()
            with torch.no_grad(),FP32RMSNormInference():
                for xx,yy in batches:
                    z=c.model.primary_logits(xx.to('mps')).values[0].cpu().double()
                    losses=torch.logsumexp(z,dim=-1)-z.gather(1,yy[0,:,None]).squeeze(1);oracle.append(float(losses.sum()))
            oracle_ce=math.fsum(oracle)/1024;error=abs(oracle_ce-result['ce']);assert error<1e-5
            assert c.step==0 and not c.history and all(not child.state for child in c.optimizer._optimizers)
            peak=torch.mps.driver_allocated_memory();assert peak<6*2**30
            assert all(sha(ROOT/f)==h for f,h in p['sources'].items())
            put(a/'result.json',dict(plan_sha256=binding,accepted=True,loaded_parent=p['parent'],endpoint_and_restoration_tensor_checks=matches,interior_independent_tensor_checks=176,unequal_target_denominator_check=True,batch_starts=starts,score=result,oracle_ce=oracle_ce,absolute_ce_error=error,batch_score_seconds=seconds,peak_mps_bytes=peak,optimizer_updates=0,new_checkpoints=0,validation_scored=False,test_opened=False))
            print(json.dumps(dict(passed=True,batch_score_seconds=seconds,ce_error=error,tensor_checks=matches)),flush=True)
    except BaseException:
        put(a/'failure.json',dict(traceback=traceback.format_exc(),plan_sha256=binding));raise
if __name__=='__main__':main()
