"""New-context integration and measured early-step admission fixture."""
import fcntl,json,os,shutil,signal,time,traceback,gc
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,check_stop,digest
OUT=ROOT/'research/tokenizer-evidence/general-rsi-context-length-fixture-001'

def execute(p,binding,a):
    import torch
    from research.byte_input_model_v6 import ByteInputModel,Spec as OldSpec,build_optimizer
    from research.general_rsi_context_length_v1.model import ContextModel
    from research.general_rsi_context_length_v1.contract import Spec
    from research.general_rsi_context_length_v1.optimizer import ContextOptimizer
    from research.general_rsi_context_length_v1.data import TrainData
    from research.gradient_clip_model_v1.clipping import clip_gradients
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    assert torch.__version__==p['torch_version'] and os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0';torch.set_num_threads(4)
    def construct(length,old=False):
        check_stop();torch.manual_seed(p['roles']['initializer'])
        m=(ByteInputModel(OldSpec('native')) if old else ContextModel(Spec(length))).to('mps')
        return m,(build_optimizer(m,.0025) if old else ContextOptimizer(m,.0025))
    def weights(m):return {n:x.detach().cpu().clone() for n,x in m.ordered_parameters()}
    def logits(m,x):
        m.eval()
        with torch.no_grad(),FP32RMSNormInference():return m.primary_logits(x).values.cpu()
    def update(m,o,x,y):
        check_stop();torch.mps.synchronize();start=time.monotonic();m.train();o.zero_grad();out=m.forward_train(x,y);loss=float(out.loss.detach());out.loss.backward();clip=clip_gradients((q for _,q in m.ordered_parameters()),1.);o.step();o.zero_grad();o.validate();torch.mps.synchronize()
        return dict(seconds=time.monotonic()-start,loss=loss,clip=clip)
    d128=TrainData(p['roles']['sampler'],128);d256=TrainData(p['roles']['sampler'],256)
    pairs=[]
    for i in range(25):
        x,y=d128.batch();u,v=d256.batch();assert torch.equal(x.flatten(),u.flatten()) and torch.equal(y.flatten(),v.flatten());pairs.append(sha_bytes(x,y))
        if i==0:first=(x,y,u,v)
    assert torch.equal(d128.view.generators['train'].get_state(),d256.view.generators['train'].get_state())
    x,y,u,v=first
    old,oo=construct(128,True);initial=weights(old);zold=logits(old,x[:1]);old_step=update(old,oo,x,y);wold=weights(old);statesold=digest([child.state_dict() for child in oo._optimizers]);del old,oo;gc.collect();torch.mps.empty_cache()
    new,no=construct(128);assert all(torch.equal(initial[n],q.cpu()) for n,q in new.ordered_parameters());assert torch.equal(zold,logits(new,x[:1]));new_step=update(new,no,x,y)
    assert all(torch.equal(wold[n],q.cpu()) for n,q in new.ordered_parameters()) and statesold==digest([child.state_dict() for child in no._optimizers]);del new,no,wold;gc.collect();torch.mps.empty_cache()
    long,lo=construct(256);assert all(torch.equal(initial[n],q.cpu()) for n,q in long.ordered_parameters());assert len(initial)==88
    zlong=logits(long,x[:1]);maxerr=float((zlong-zold).abs().max());torch.testing.assert_close(zlong,zold,atol=p['atol'],rtol=p['rtol'])
    # Full256 tail perturbation checks causal independence before midpoint.
    z=logits(long,u[:1]);modified=u[:1].clone();modified[:,128:]=(modified[:,128:]+1)%50257
    zz=logits(long,modified);causalerr=float((z[:,:128]-zz[:,:128]).abs().max());torch.testing.assert_close(z[:,:128],zz[:,:128],atol=p['atol'],rtol=p['rtol'])
    masks=[b for n,b in long.named_buffers() if n.endswith('causal_mask')];assert len(masks)==4 and all(tuple(b.shape)==(256,256) for b in masks)
    long_step=update(long,lo,u,v)
    tensor_bytes=sum(q.numel()*q.element_size() for q in initial.values())
    state_bytes=sum(t.numel()*t.element_size() for child in lo._optimizers for fields in child.state.values() for t in fields.values())
    peak=torch.mps.driver_allocated_memory();assert peak<6*2**30
    return dict(plan_sha256=binding,accepted=True,parameter_tensors_exact=176,new128_logits_exact=True,new128_update_exact=True,new128_children_exact=True,short_prefix_max_abs=maxerr,causality_max_abs=causalerr,paired_draws_exact=25,paired_batch_sha256=pairs,steps={'old128':old_step,'new128':new_step,'new256':long_step},parameter_bytes=tensor_bytes,optimizer_tensor_bytes=state_bytes,peak_mps_bytes=peak,optimizer_updates=3,validation_opened=False,test_opened=False)

def sha_bytes(x,y):
    import hashlib
    return hashlib.sha256(x.cpu().numpy().tobytes()+y.cpu().numpy().tobytes()).hexdigest()

def main():
    with open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop()
        p=json.loads((OUT/'plan.json').read_text());binding=sha(OUT/'plan.json');assert all(sha(ROOT/f)==h for f,h in p['sources'].items());assert shutil.disk_usage(ROOT).free>=p['required_free_bytes']
        a=OUT/'attempt-01';a.mkdir();put(a/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=binding))
        def stop(*_):
            import research.general_rsi_pseudo_update_diagnostic_v1 as shared
            shared.STOP=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
        try:
            with CpuPrerequisiteGuard(a/'guard.json',ROOT,timeout_seconds=180,rss_limit_bytes=4*2**30,reserve_bytes=10*2**30):
                result=execute(p,binding,a);assert all(sha(ROOT/f)==h for f,h in p['sources'].items());put(a/'result.json',result);print(json.dumps(result),flush=True)
        except BaseException:
            put(a/'failure.json',dict(traceback=traceback.format_exc(),plan_sha256=binding));raise
if __name__=='__main__':main()
