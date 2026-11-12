"""Only qualify the new weighted accumulation, component accounting and restart path."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,batch_hash,check_stop

OUT=ROOT/'research/tokenizer-evidence/general-rsi-joint-integration-001'
WEIGHTS=ROOT/'research_runs/general-rsi-joint-integration-001'


def run():
    tick=time.monotonic();os.chdir(ROOT)
    lock=(ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop()
    attempt=OUT/'attempt-01';attempt.mkdir(exist_ok=False);WEIGHTS.mkdir(exist_ok=False)
    plan=json.loads((OUT/'plan.json').read_text());binding=sha(OUT/'plan.json')
    assert shutil.disk_usage(ROOT).free>=plan['required_free_bytes']
    def stop(*_):
        import research.general_rsi_pseudo_update_diagnostic_v1 as shared
        shared.STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    put(attempt/'launch.json',dict(pid=os.getpid(),plan_sha256=binding,started_unix=time.time()))
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with CpuPrerequisiteGuard(attempt/'guard.json',ROOT,timeout_seconds=plan['total_seconds'],
                             rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):
        try:
            import torch
            from research.general_rsi_joint_training_v1 import Context
            from research.general_rsi_pseudo_spans_v1 import repetition_loss
            from research.general_rsi_training_v2 import write_checkpoint,read_checkpoint
            from research.general_rsi_qualification_v1 import equal
            for path,h in plan['sources'].items():assert sha(path)==h,path
            assert torch.__version__==plan['torch_version'] and torch.backends.mps.is_available()
            assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0';torch.set_num_threads(4)
            peak=0
            def check():
                nonlocal peak
                check_stop();torch.mps.synchronize()
                peak=max(peak,torch.mps.driver_allocated_memory());assert peak<plan['mps_limit_bytes']
            def new(arm):
                return Context(arm,plan['budget_ns'],plan['half_life_ns'],plan['roles'],plan['pseudo_seed'],plan['parent'],binding)
            # New accumulation oracle: independently differentiate each unweighted
            # component at the exact same state, then form its frozen weighted sum.
            ctx=new('ce_pseudo_mean');initial=ctx.capture();ctx.model.train();ctx.optimizer.zero_grad()
            x,y=ctx.data.batch();px,py,previous,mask,details=ctx.pseudo.batch('mps')
            ctx.model.forward_train(x,y).loss.backward()
            ce_grad={n:p.grad.detach().cpu().clone() for n,p in ctx.model.ordered_parameters()}
            ctx.optimizer.zero_grad()
            pseudo,info=repetition_loss(ctx.model.primary_logits(px).values,py,previous,mask)
            pseudo.backward();del pseudo
            pseudo_grad={n:p.grad.detach().cpu().clone() for n,p in ctx.model.ordered_parameters()}
            ctx.optimizer.zero_grad();ctx.restore(initial);equal(ctx.capture(),initial)
            oracle=[]
            for arm in plan['arms']:
                if ctx.config['quality_arm']!=arm:
                    del ctx;gc.collect();torch.mps.empty_cache();ctx=new(arm)
                torch.random.set_rng_state(initial['cpu_rng']);torch.mps.set_rng_state(initial['mps_rng'])
                ctx.model.train();ctx.optimizer.zero_grad();row=ctx.backward_components()
                assert row['batch_sha256']==batch_hash(x,y)
                if arm=='ce_pseudo_mean':assert row['pseudo_sha256']==batch_hash(px,py,previous,mask)
                norms=0.;errors=0.;max_abs=0.;tensor_count=0
                for name,p in ctx.model.ordered_parameters():
                    a,b=ctx.config['loss_weights'];expected=a*ce_grad[name]+b*pseudo_grad[name]
                    actual=p.grad.detach().cpu()
                    try:
                        assert bool(torch.isfinite(actual).all())
                        torch.testing.assert_close(actual,expected,rtol=plan['gradient_rtol'],atol=plan['gradient_atol'])
                    except (AssertionError,RuntimeError):
                        failure=write_checkpoint(WEIGHTS/'first-gradient-failure.pt',
                            dict(name=name,arm=arm,actual=actual,expected=expected,ce_gradient=ce_grad[name],
                                 pseudo_gradient=pseudo_grad[name],plan_sha256=binding),cap=256*1024**2)
                        put(attempt/'gradient-failure-receipt.json',failure)
                        raise
                    delta=(actual-expected).double();norms+=float(expected.double().square().sum())
                    errors+=float(delta.square().sum());max_abs=max(max_abs,float(delta.abs().max()));tensor_count+=1
                assert ctx.step==0 and not ctx.history
                oracle.append(dict(arm=arm,tensors=tensor_count,max_abs=max_abs,relative_l2=(errors/norms)**.5,
                                   accumulation_before_optimizer=True,ce_hash=row['batch_sha256'],pseudo_hash=row['pseudo_sha256']))
                ctx.optimizer.zero_grad();check()
            put(attempt/'gradient-oracle.json',dict(passed=True,rtol=plan['gradient_rtol'],atol=plan['gradient_atol'],rows=oracle,
                scope='New gradient-accumulation integration only; not arithmetic quality selection or an old FP32 qualification retry.'))
            del ctx,initial,ce_grad,pseudo_grad,expected,actual,delta,x,y,px,py,previous,mask
            gc.collect();torch.mps.empty_cache()
            reports=[]
            for arm in plan['arms']:
                check();ctx=new(arm);receipt=None
                while not ctx.clock.done:
                    check();ctx.update()
                    if ctx.snapshots and not ctx.clock.done and receipt is None:
                        receipt=write_checkpoint(WEIGHTS/f'{arm}-restart.pt',ctx.capture())
                assert receipt and ctx.snapshots>=2
                expected=ctx.capture();del ctx;gc.collect();torch.mps.empty_cache()
                ctx=new(arm);saved=read_checkpoint(receipt)
                bad=dict(saved,component_counts=dict(ce=-1,pseudo=-1))
                try:ctx.restore(bad)
                except AssertionError:pass
                else:raise AssertionError('Bad new component accounting accepted')
                assert ctx.step==0 and not ctx.history and ctx.pseudo.draws==0
                ctx.restore(saved);equal(ctx.capture(),saved);del saved,bad
                replayed=0
                while not ctx.clock.done:
                    check();record=expected['history'][ctx.step];ticks=iter((0,record['end_ns']-record['begin_ns']))
                    ctx.update(now_ns=lambda:next(ticks));replayed+=1
                equal(ctx.capture(),expected);check()
                report=dict(arm=arm,checkpoint=receipt,steps=ctx.step,component_counts=expected['component_counts'],
                            snapshots=ctx.snapshots,training_ns=ctx.clock.spent_ns,mean_update_seconds=ctx.clock.spent_ns/1e9/ctx.step,
                            exact_full_restore_and_replay=True,replayed_updates=replayed,bad_component_counts_rejected=True,
                            history=expected['history'],saturated_targets=sum(r['saturated_targets'] for r in expected['history']))
                put(attempt/f'{arm}-result.json',report);reports.append(report)
                print(json.dumps(dict(arm=arm,steps=ctx.step,mean_update_seconds=report['mean_update_seconds'],restart_exact=True)),flush=True)
                del ctx,expected;gc.collect();torch.mps.empty_cache()
            streams=[[r['batch_sha256'] for r in report['history']] for report in reports]
            n=min(map(len,streams));assert all(s[:n]==streams[0][:n] for s in streams)
            retained=sum(p.stat().st_size for p in WEIGHTS.iterdir());assert retained<=plan['retained_cap_bytes']
            put(attempt/'result.json',dict(qualified=True,plan_sha256=binding,paired_natural_prefix=n,
                full_restart_exact=True,gradient_accumulation_checked=True,retained_bytes=retained,
                peak_mps_boundary_bytes=peak,total_seconds=time.monotonic()-tick,test_opened=False,quality_result=False,
                validation_scored=False,reports=[{k:v for k,v in r.items() if k!='history'} for r in reports]))
        except BaseException:
            put(attempt/'failure.json',dict(traceback=traceback.format_exc(),seconds=time.monotonic()-tick));raise


if __name__=='__main__':run()
