"""Wait for the sole qualifier, measure complete admission, then execute serially."""
import datetime,fcntl,os,shutil,sys,time,traceback
from research.general_rsi_train_serve_context_v1 import ROOT,OUT,read,put,sha,base
from research.general_rsi_rolling_context_supervisor_v1 import state_update


def main():
    qualifier=int(sys.argv[1]);started=time.monotonic()
    with (OUT/'admit.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        put(OUT/'admit-start.json',dict(pid=os.getpid(),qualifier_pid=qualifier,source_sha256=sha(__file__)))
        try:
            while True:
                base.check_stop();assert time.monotonic()-started<1000
                try:os.kill(qualifier,0)
                except ProcessLookupError:break
                time.sleep(10)
            qpath=OUT/'qualification-02/result.json';q=read(qpath)
            assert q['accepted'] and read(OUT/'qualification-02/guard.json')['status']=='finished'
            plan=read(OUT/'plan.json');assert q['plan_sha256']==sha(OUT/'plan.json')
            while True:
                base.check_stop();assert time.monotonic()-started<7200
                h=base.host();free=shutil.disk_usage(ROOT).free
                if int(h['pressure'])==1 and h['reclaimable_proxy_bytes']>=1.25*2**30 and free>=plan['reserve_bytes']+plan['allowance_bytes']:break
                state_update(status='TRAIN_SERVE_ADMISSION_WAIT',active_worker_pid=None,active_session=None,
                    next_action='Numerical qualification passed. Admission controller waits for normal host pressure, >=1.25GiB host proxy and12GiB disk reserve plus512MiB. No model worker is active.')
                base.append(OUT/'admission-wait.jsonl',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=h,free_bytes=free));time.sleep(30)
            with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research/general_rsi_train_serve_context_v2.py',ROOT/'research/general_rsi_train_serve_supervisor_v2.py',ROOT/'research/general_rsi_train_serve_analysis_v1.py',OUT/'repair-02.json',qpath,PathSelf()]}
                for p,digest in read(OUT/'repair-02.json')['sources'].items():assert sha(ROOT/p)==digest
                put(OUT/'admission.json',dict(accepted=True,plan_sha256=sha(OUT/'plan.json'),qualification_sha256=sha(qpath),forecast_seconds=q['forecast_seconds'],
                    free_bytes=free,required_bytes=plan['reserve_bytes']+plan['allowance_bytes'],host=h,sources=sources,
                    stored_weight_copies=0,optimizer_bytes=0,ema_bytes=0,output_atomic_metadata_failure_allowance_bytes=plan['allowance_bytes']))
            # exec preserves one supervisor and avoids a second queued model worker.
            os.execv(sys.executable,[sys.executable,'-m','research.general_rsi_train_serve_supervisor_v2'])
        except BaseException as exc:
            put(OUT/'admit-failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
            state_update(status='TRAIN_SERVE_ADMISSION_FAILED',active_worker_pid=None,active_session=None,
                next_action='Inspect preserved qualifier/admission failure. Do not start a partial panel or change scientific rules. Repair only demonstrated execution failures in distinct attempts.')
            raise


def PathSelf():
    from pathlib import Path
    return Path(__file__).resolve()


if __name__=='__main__':main()
