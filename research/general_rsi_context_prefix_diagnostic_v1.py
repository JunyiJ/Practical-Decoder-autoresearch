"""Frozen source-balanced common-target context truncation on retained models."""
import fcntl,gc,json,os,shutil,signal,time,traceback
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,digest,check_stop
from research.general_rsi_pseudo_accumulated_v1 import append
from research.general_rsi_update_response_v1 import parameters
from research.general_rsi_context_prefix_v1 import score,inventory
OUT=ROOT/'research/tokenizer-evidence/general-rsi-context-prefix-001'

def execute(p,binding,a):
    import torch
    from research.general_rsi_continuation_decay_v1 import Context
    assert torch.__version__==p['torch_version'] and os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0';torch.set_num_threads(4)
    peak=0;sequences=0;targets=0;tokens=0;reports=[]
    def boundary():
        nonlocal peak
        check_stop();torch.mps.synchronize();peak=max(peak,torch.mps.driver_allocated_memory());assert peak<p['mps_limit_bytes']
        assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<p['metadata_cap_bytes']
    for parent in p['parents']:
        boundary();tick=time.monotonic();ctx=Context('none',1200*10**9,60*10**9,parent['roles'],parent['pseudo_seed'],parent['parent'],binding)
        assert inventory(ctx.data,p['selection_seed'])==p['windows'];initial=digest(parameters(ctx.model));slot=parent['parent']['slot'];sums={str(c):[] for c in p['contexts']}
        for w in p['windows']:
            d=ctx.data.indexes['train'][w['document']]
            order=p['contexts'][w['window']%3:]+p['contexts'][:w['window']%3]
            for c in order:
                boundary();s=score(ctx.model,ctx.data.view.train_data,d,w['target_start'],c)
                assert s['target_ids']==w['target_ids'] and s['input_sha256']==w['input_sha256'][str(c)]
                append(a/f'p{slot:02d}-scores.jsonl',dict(parent=slot,window=w['window'],document=w['document'],source=w['source'],target_start=w['target_start'],**s))
                sequences+=1;targets+=s['targets'];tokens+=s['input_tokens'];sums[str(c)].append(s['loss_sum'])
            if (w['window']+1)%32==0:print(json.dumps(dict(parent=slot,completed_windows=w['window']+1)),flush=True)
        assert ctx.step==0 and not ctx.history and all(not child.state for child in ctx.optimizer._optimizers)
        assert digest(parameters(ctx.model))==initial and sha(parent['parent']['path'])==parent['parent']['sha256'];boundary()
        report=dict(parent=slot,parameter_sha256=initial,parameters_unchanged=True,checkpoint_unchanged=True,sequences=384,seconds=time.monotonic()-tick)
        put(a/f'p{slot:02d}-review.json',report);reports.append(report);del ctx;gc.collect();torch.mps.empty_cache()
    assert sequences==1152 and targets==36864 and tokens==97152
    return dict(plan_sha256=binding,reports=reports,sequences=sequences,scored_targets=targets,input_tokens=tokens,unique_corpus_targets=4096,peak_mps_bytes=peak,optimizer_updates=0,new_checkpoints=0,test_opened=False,validation_scored=False)

def main():
    lock=open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop();tick=time.monotonic()
    p=json.loads((OUT/'plan.json').read_text());binding=sha(OUT/'plan.json');assert json.loads((OUT/'admission.json').read_text())['plan_sha256']==binding
    assert all(sha(ROOT/f)==h for f,h in p['sources'].items());assert shutil.disk_usage(ROOT).free>=p['required_free_bytes']
    a=OUT/'attempt-01';a.mkdir();put(a/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=binding))
    def stop(*_):
        import research.general_rsi_pseudo_update_diagnostic_v1 as shared
        shared.STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    try:
        with CpuPrerequisiteGuard(a/'guard.json',ROOT,timeout_seconds=p['total_seconds'],rss_limit_bytes=p['rss_limit_bytes'],reserve_bytes=p['reserve_bytes']):
            result=execute(p,binding,a);assert all(sha(ROOT/f)==h for f,h in p['sources'].items());result['total_seconds']=time.monotonic()-tick;put(a/'result.json',result)
    except BaseException:
        put(a/'failure.json',dict(traceback=traceback.format_exc(),plan_sha256=binding));raise
if __name__=='__main__':main()
