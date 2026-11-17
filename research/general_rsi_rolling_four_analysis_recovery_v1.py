"""Run the corrected input-path analysis after the preserved scorer exits."""
import datetime,fcntl,os,signal,time,traceback
from research.general_rsi_rolling_four_v1 import ROOT,OUT,read,put,sha,base
from research.general_rsi_rolling_context_supervisor_v1 import state_update
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard


def main():
    signal.signal(signal.SIGTERM,base.on_stop);signal.signal(signal.SIGINT,base.on_stop)
    with (OUT/'analysis-recovery.lock').open('a') as qlock:
        fcntl.flock(qlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        repair=read(OUT/'analysis-path-repair-03.json');started=time.monotonic()
        put(OUT/'analysis-recovery-start.json',dict(pid=os.getpid(),repair_sha256=sha(OUT/'analysis-path-repair-03.json')))
        while True:
            base.check_stop();assert time.monotonic()-started<43800
            try:os.kill(repair['scoring_worker_pid'],0)
            except ProcessLookupError:break
            time.sleep(15)
        assert read(OUT/'score-02-exit.json')['returncode']==0 and read(OUT/'scoring-02/result.json')['completed']
        assert read(OUT/'scoring-02/guard.json')['status']=='finished'
        failure=read(OUT/'scoring-02/recovery-failure.json')
        assert 'FileNotFoundError' in failure['error'] and 'scoring-01/result.json' in failure['traceback']
        for name,digest in repair['sources'].items():assert sha(ROOT/name)==digest
        state_update(status='ROLLING_FOUR_ANALYZING',active_worker_pid=os.getpid(),active_session='rolling-four-analysis-path-repair-03',next_action='Bound v3 analysis repairs only stale D input paths after preserved v2 path failure. All D scores remain from scoring-02, with unchanged scientific contrasts and guards.')
        try:
            from research.general_rsi_rolling_four_analysis_v3 import main as analyze
            with CpuPrerequisiteGuard(OUT/'analysis-03-guard.json',ROOT,rss_limit_bytes=4*2**30,reserve_bytes=12*2**30,timeout_seconds=300):analyze()
            put(OUT/'analysis-03-exit.json',dict(returncode=0,original_path_failure_sha256=sha(OUT/'scoring-02/recovery-failure.json')))
            state_update(status='ROLLING_FOUR_AWAITING_REVIEW',active_worker_pid=None,active_session=None,rolling_four_condition_status='complete_awaiting_review',next_action='Review integrated four-condition analysis, original host-admission and path-wrapper failures, all score/endpoint/source hashes, controls and guards. Report results, commit/push evidence and close monitor.')
        except BaseException as exc:
            put(OUT/'analysis-03-failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
            state_update(status='ROLLING_FOUR_ANALYSIS_FAILED',active_worker_pid=None,active_session=None,next_action='Inspect preserved analysis03 failure without rerunning scoring or changing scientific rules.')
            raise
if __name__=='__main__':main()
