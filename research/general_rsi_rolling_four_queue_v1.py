"""Wait for the unchanged B/C job, then qualify, admit, score and analyze D."""
import datetime,fcntl,os,shutil,signal,subprocess,sys,time,traceback
from research.general_rsi_rolling_four_v1 import ROOT,OUT,BC,read,put,sha,bindings,base
from research.general_rsi_rolling_context_supervisor_v1 import state_update


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def run_stage(mode,status):
    with (OUT/f'{mode}.log').open('x') as log:
        if mode=='analysis':
            cmd=[sys.executable,'-c',"from research.general_rsi_rolling_four_v1 import OUT,ROOT; from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard; from research.general_rsi_rolling_four_analysis_v1 import main;\nwith CpuPrerequisiteGuard(OUT/'analysis-guard.json',ROOT,rss_limit_bytes=4*2**30,reserve_bytes=12*2**30,timeout_seconds=300): main()"]
        else:cmd=[sys.executable,'-m','research.general_rsi_rolling_four_v1',mode]
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        put(OUT/f'{mode}-start.json',dict(pid=p.pid,at=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        state_update(status=status,active_worker_pid=p.pid,active_session='general-rsi-rolling-four-001',next_action=f'Four-condition diagnostic {mode} is active. Preserve frozen sources and this serial worker. No training.')
        code=p.wait()
    put(OUT/f'{mode}-exit.json',dict(returncode=code));assert code==0,(mode,code)


def main():
    signal.signal(signal.SIGTERM,base.on_stop);signal.signal(signal.SIGINT,base.on_stop)
    with (OUT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(OUT/'plan.json');put(OUT/'queue-start.json',dict(pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(OUT/'plan.json')))
        started=time.monotonic();launched=False
        try:
            while True:
                base.check_stop();assert time.monotonic()-started<43800,'B/C wait ceiling'
                if (BC/'scoring-exit.json').exists():assert read(BC/'scoring-exit.json')['returncode']==0,'B/C scoring failed'
                if (BC/'analysis-exit.json').exists():
                    assert read(BC/'analysis-exit.json')['returncode']==0,'B/C analysis failed'
                    original=read(BC/'supervisor-start.json')
                    if not alive(original['pid']) and not alive(original['worker_pid']):break
                time.sleep(15)
            assert read(BC/'scoring-01/guard.json')['status']=='finished' and read(BC/'analysis-guard.json')['status']=='finished'
            assert read(ROOT/'research_state/general-rsi-state.json').get('active_worker_pid') is None
            bindings(plan);base.check_stop()
            # No lock is held during the wait. Children acquire the shared lock nonblockingly.
            launched=True;run_stage('qualify','ROLLING_FOUR_QUALIFYING')
            q=read(OUT/'qualification-01/result.json');g=read(OUT/'qualification-01/guard.json')
            assert q['accepted'] and q['plan_sha256']==sha(OUT/'plan.json') and g['status']=='finished'
            rows=[__import__('json').loads(x) for x in (OUT/'qualification-01/memory.jsonl').read_text().splitlines()]
            free=shutil.disk_usage(ROOT).free;required=plan['reserve_bytes']+plan['combined_allowance_bytes'];assert free>=required
            put(OUT/'admission.json',dict(accepted=True,plan_sha256=sha(OUT/'plan.json'),qualification_sha256=sha(OUT/'qualification-01/result.json'),
                 forecast_seconds=q['forecast_seconds'],free_bytes=free,required_free_bytes=required,margin_bytes=free-required,
                 peak_rss_bytes=g['observation']['peak_rss_bytes'],peak_mps_bytes=max(r['mps_driver_bytes'] for r in rows),
                 minimum_host_reclaimable_proxy_bytes=min(r['host']['reclaimable_proxy_bytes'] for r in rows),no_training=True,test_opened=False))
            bindings(plan);run_stage('score','ROLLING_FOUR_SCORING');bindings(plan);run_stage('analysis','ROLLING_FOUR_ANALYZING')
            state_update(status='ROLLING_FOUR_AWAITING_REVIEW',active_worker_pid=None,active_session=None,
                         next_action='Review completed four-condition analysis, matched targets, permutation/control checks, hashes and guards; report results and reconcile source/evidence. Do not rerun completed scoring or launch training.')
            put(OUT/'queue-result.json',dict(completed=True))
        except BaseException as exc:
            put(OUT/'queue-failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),launched=launched))
            if launched:state_update(status='ROLLING_FOUR_EXECUTION_FAILED',active_worker_pid=None,active_session=None,next_action='Inspect preserved four-condition failure; repair only demonstrated implementation/resource failure in a separate attempt. Preserve all first scores and original B/C results.')
            raise
if __name__=='__main__':main()
