"""Only qualify new grouped decay, canonical equivalence and policy-bound full restart."""
import copy,fcntl,gc,json,os,shutil,signal,time,traceback
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,check_stop
OUT=ROOT/'research/tokenizer-evidence/general-rsi-continuation-decay-integration-001'
WEIGHTS=ROOT/'research_runs/general-rsi-continuation-decay-integration-001'

def main():
    tick=time.monotonic();os.chdir(ROOT);lock=open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop()
    plan=json.loads((OUT/'plan.json').read_text());binding=sha(OUT/'plan.json')
    assert shutil.disk_usage(ROOT).free>=plan['required_free_bytes']
    assert all(sha(ROOT/p)==h for p,h in plan['sources'].items())
    attempt=OUT/'attempt-01';attempt.mkdir();WEIGHTS.mkdir()
    put(attempt/'launch.json',dict(pid=os.getpid(),plan_sha256=binding,started_unix=time.time()))
    def stop(*_):
        import research.general_rsi_pseudo_update_diagnostic_v1 as shared
        shared.STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    try:
        with CpuPrerequisiteGuard(attempt/'guard.json',ROOT,timeout_seconds=600,rss_limit_bytes=4*1024**3,reserve_bytes=10*1024**3):
            import torch
            from research.general_rsi_continuation_decay_v1 import Context,GroupDecayOptimizer,POLICIES,KIND
            from research.general_rsi_pseudo_training_v1 import Context as Original
            from research.general_rsi_qualification_v1 import equal
            from research.general_rsi_training_v2 import write_checkpoint,read_checkpoint
            assert torch.__version__==plan['torch_version'] and torch.backends.mps.is_available()
            assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0';torch.set_num_threads(4);peak=0
            def boundary():
                nonlocal peak
                check_stop();torch.mps.synchronize();peak=max(peak,torch.mps.driver_allocated_memory());assert peak<plan['mps_limit_bytes']
            def new(policy,original=False):
                cls=Original if original else Context
                return cls('native_ce' if original else policy,plan['budget_ns'],plan['half_life_ns'],plan['roles'],plan['pseudo_seed'],plan['parent'],binding)
            # Same complete initial state, canonical three-update reference including
            # actual clipped gradients, child moments and exact recorded clock replay.
            ctx=new('all',True);initial=ctx.capture()
            for _ in range(3):boundary();ctx.update()
            expected=ctx.capture();del ctx;gc.collect();torch.mps.empty_cache()
            ctx=new('all');adapted=dict(initial,kind=KIND,config=ctx.config,decay_policy=POLICIES['all']);ctx.restore(adapted)
            for row in expected['history']:
                ticks=iter((0,row['end_ns']-row['begin_ns']));ctx.update(now_ns=lambda:next(ticks))
            actual=ctx.capture();equal({k:v for k,v in actual.items() if k not in ['kind','config','decay_policy']},{k:v for k,v in expected.items() if k not in ['kind','config']})
            put(attempt/'canonical-equivalence.json',dict(passed=True,updates=3,complete_state_exact=True))
            del ctx,initial,adapted,expected,actual;gc.collect();torch.mps.empty_cache()
            reports=[];policy_checks=[]
            for policy in plan['policies']:
                boundary();ctx=new(policy);initial=ctx.capture();ctx.optimizer.apply_decay();count=0
                for n,p in ctx.model.ordered_parameters():
                    d=POLICIES[policy][0 if n=='lm_head.weight' else 1]
                    assert torch.equal(p.detach().cpu(),initial['parameters'][n]*(1.-d));count+=1
                ctx.restore(initial);equal(ctx.capture(),initial);del initial
                policy_checks.append(dict(policy=policy,tensors=count,exact_group_decay=True,restored_exactly=True))
                receipt=None
                while not ctx.clock.done:
                    boundary();ctx.update()
                    if ctx.snapshots and not ctx.clock.done and receipt is None:
                        receipt=write_checkpoint(WEIGHTS/f'{policy}-restart.pt',ctx.capture())
                        put(attempt/f'{policy}-receipt.json',receipt)
                assert receipt and ctx.snapshots>=2
                expected=ctx.capture();del ctx;gc.collect();torch.mps.empty_cache()
                ctx=new(policy);saved=read_checkpoint(receipt);bad=copy.copy(saved);bad['decay_policy']=(-1.,-1.)
                try:ctx.restore(bad)
                except AssertionError:pass
                else:raise AssertionError('Corrupt grouped decay accepted')
                assert ctx.step==0 and not ctx.history
                bad=copy.deepcopy(saved);bad['config']['decay_policy']=(-1.,-1.)
                try:ctx.restore(bad)
                except AssertionError:pass
                else:raise AssertionError('Corrupt saved configuration accepted')
                assert ctx.step==0 and not ctx.history
                ctx.restore(saved);equal(ctx.capture(),saved);del saved,bad;replayed=0
                while not ctx.clock.done:
                    boundary();row=expected['history'][ctx.step];ticks=iter((0,row['end_ns']-row['begin_ns']))
                    ctx.update(now_ns=lambda:next(ticks));replayed+=1
                equal(ctx.capture(),expected);boundary()
                report=dict(policy=policy,steps=ctx.step,snapshots=ctx.snapshots,checkpoint=receipt,training_ns=ctx.clock.spent_ns,
                    mean_update_seconds=ctx.clock.spent_ns/1e9/ctx.step,replayed_updates=replayed,full_restart_exact=True,
                    corrupt_policy_rejected_before_mutation=True,history=expected['history'])
                put(attempt/f'{policy}-result.json',report);reports.append(report)
                print(json.dumps({k:v for k,v in report.items() if k not in ['history','checkpoint']}),flush=True)
                del ctx,expected;gc.collect();torch.mps.empty_cache()
            streams=[[r['batch_sha256'] for r in v['history']] for v in reports];n=min(map(len,streams));assert all(s[:n]==streams[0][:n] for s in streams)
            put(attempt/'group-policy-oracle.json',dict(passed=True,rows=policy_checks))
            retained=sum(p.stat().st_size for p in WEIGHTS.iterdir());assert retained<=plan['retained_cap_bytes']
            assert all(sha(ROOT/p)==h for p,h in plan['sources'].items())
            put(attempt/'result.json',dict(qualified=True,plan_sha256=binding,paired_natural_prefix=n,retained_bytes=retained,
                peak_mps_bytes=peak,total_seconds=time.monotonic()-tick,canonical_equivalence_exact=True,
                reports=[{k:v for k,v in r.items() if k!='history'} for r in reports],test_opened=False,validation_scored=False,quality_result=False))
    except BaseException:
        put(attempt/'failure.json',dict(traceback=traceback.format_exc(),plan_sha256=binding));raise
if __name__=='__main__':main()
