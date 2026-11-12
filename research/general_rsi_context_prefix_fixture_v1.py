"""Check only new target slicing and its reference logits; no optimizer updates."""
import fcntl,json,os,shutil,signal,time,traceback,math
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,check_stop
from research.general_rsi_context_prefix_v1 import case,score,CONTEXTS
OUT=ROOT/'research/tokenizer-evidence/general-rsi-context-prefix-fixture-001'

def execute(p,binding):
    import torch
    from torch.nn import functional as F
    from research.general_rsi_continuation_decay_v1 import Context
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    assert torch.__version__==p['torch_version'] and os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0';torch.set_num_threads(4)
    values=torch.arange(512);checks=0
    for doc,t in [(dict(offset=0,tokens=128),96),(dict(offset=11,tokens=256),107),(dict(offset=11,tokens=256),235)]:
        for c in CONTEXTS:
            x,y,s=case(values,doc,t,c);assert x.tolist()==[list(range(t-c,t+31))] and y.tolist()==list(range(t,t+32))
            assert [int(v)+1 for v in x[0,s]]==y.tolist();checks+=1
    rejected=0
    for t in [95,97]:
        try:case(values,dict(offset=0,tokens=128),t,96)
        except AssertionError:rejected+=1
    assert rejected==2
    ctx=Context('none',1200*10**9,60*10**9,p['roles'],p['pseudo_seed'],p['parent'],binding);d=ctx.data.indexes['train'][p['document']];t=d['offset']+96;values=ctx.data.view.train_data
    scores=[];timing=[]
    for c in CONTEXTS:
        tick=time.monotonic();s=score(ctx.model,values,d,t,c);torch.mps.synchronize();timing.append(time.monotonic()-tick);scores.append(s)
    # Direct full-input reference, with explicit last32 selection independent of case().
    ctx.model.eval()
    with torch.no_grad(),FP32RMSNormInference():
        z=ctx.model.primary_logits(values[d['offset']:d['offset']+127][None,:].to('mps')).values[0]
        chosen=z[-32:];y=values[d['offset']+96:d['offset']+128].to('mps')
        ref=F.cross_entropy(chosen,y,reduction='none').cpu().double().tolist()
        # Prefix helper uses the same long input; compare complete selected logits exactly.
        x,yy,sel=case(values,d,t,96);z2=ctx.model.primary_logits(x.to('mps')).values[0,sel,:]
        assert torch.equal(chosen,z2) and ref==scores[-1]['token_losses'] and yy.tolist()==scores[-1]['target_ids']
    assert ctx.step==0 and not ctx.history and ctx.pseudo.draws==0 and all(not child.state for child in ctx.optimizer._optimizers)
    peak=torch.mps.driver_allocated_memory();assert peak<6*2**30
    return dict(plan_sha256=binding,accepted=True,synthetic_alignment_checks=checks,boundary_rejections=rejected,longest_logits_exact=True,longest_token_losses_exact=True,all_context_targets_identical=all(s['target_ids']==scores[0]['target_ids'] for s in scores),scores=scores,seconds_by_context=dict(zip(map(str,CONTEXTS),timing)),peak_mps_bytes=peak,optimizer_updates=0,test_opened=False,validation_scored=False,new_checkpoints=0)

def main():
    lock=open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop()
    p=json.loads((OUT/'plan.json').read_text());binding=sha(OUT/'plan.json');assert all(sha(ROOT/f)==h for f,h in p['sources'].items());assert shutil.disk_usage(ROOT).free>=p['required_free_bytes']
    a=OUT/'attempt-01';a.mkdir();put(a/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=binding))
    def stop(*_):
        import research.general_rsi_pseudo_update_diagnostic_v1 as shared
        shared.STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    try:
        with CpuPrerequisiteGuard(a/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4*2**30,reserve_bytes=10*2**30):
            result=execute(p,binding);assert all(sha(ROOT/f)==h for f,h in p['sources'].items());put(a/'result.json',result);print(json.dumps(dict(passed=True,timing=result['seconds_by_context'])),flush=True)
    except BaseException:
        put(a/'failure.json',dict(traceback=traceback.format_exc(),plan_sha256=binding));raise
if __name__=='__main__':main()
