"""Distinct launch after pre-model host admission failure; unchanged D scoring."""
import datetime,fcntl,os,signal,time,traceback
from research.general_rsi_rolling_four_v1 import ROOT,OUT,BC,read,put,sha,bindings,base,score
from research.general_rsi_rolling_context_supervisor_v1 import state_update
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard


def main():
    signal.signal(signal.SIGTERM,base.on_stop);signal.signal(signal.SIGINT,base.on_stop)
    plan=read(OUT/'plan.json');recovery=read(OUT/'recovery-02.json')
    for name,digest in recovery['sources'].items():assert sha(ROOT/name)==digest
    out=OUT/'scoring-02';out.mkdir()
    try:
        with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with CpuPrerequisiteGuard(out/'guard.json',ROOT,rss_limit_bytes=4*2**30,reserve_bytes=12*2**30,timeout_seconds=plan['max_seconds'],poll_seconds=.2):
                base.check_stop();bindings(plan)
                h=base.host()
                # Log the complete observation BEFORE testing the unchanged admission.
                put(out/'launch.json',dict(pid=os.getpid(),host=h,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(OUT/'plan.json'),recovery_sha256=sha(OUT/'recovery-02.json')))
                assert int(h['pressure'])==1 and h['reclaimable_proxy_bytes']>=2**30+256*2**20,h
                assert __import__('shutil').disk_usage(ROOT).free>=plan['reserve_bytes']+plan['combined_allowance_bytes']
                import torch
                assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0' and torch.backends.mps.is_available()
                assert torch.__version__==read(BC/'plan.json')['torch_version'];torch.set_num_threads(4)
                state_update(status='ROLLING_FOUR_SCORING',active_worker_pid=os.getpid(),active_session='general-rsi-rolling-four-001/scoring-02',rolling_four_condition_status='scoring_D_attempt02',
                  next_action='D attempt02 scores with unchanged frozen scorer, weights and permutations after attempt01 pre-model host-memory admission failure. Preserve both attempts. Bound v2 analysis changes only the D input directory to scoring-02.')
                score(plan,out)
        put(OUT/'score-02-exit.json',dict(returncode=0))
        for name,digest in recovery['sources'].items():assert sha(ROOT/name)==digest
        state_update(status='ROLLING_FOUR_ANALYZING',next_action='D attempt02 completed. Same frozen four-condition analysis reads scoring-02 through the separately bound v2 path repair.')
        from research.general_rsi_rolling_four_analysis_v2 import main as analyze
        with CpuPrerequisiteGuard(OUT/'analysis-02-guard.json',ROOT,rss_limit_bytes=4*2**30,reserve_bytes=12*2**30,timeout_seconds=300):analyze()
        put(OUT/'analysis-02-exit.json',dict(returncode=0))
        state_update(status='ROLLING_FOUR_AWAITING_REVIEW',active_worker_pid=None,active_session=None,rolling_four_condition_status='complete_awaiting_review',
          next_action='Review complete A/B/C/D analysis and first-attempt host-admission failure, all source/endpoint hashes, target/permutation controls and guards. Report results, reconcile state, commit/push evidence, then close the monitor.')
    except BaseException as exc:
        put(out/'recovery-failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        state_update(status='ROLLING_FOUR_EXECUTION_FAILED',active_worker_pid=None,active_session=None,next_action='Inspect distinct scoring-02 recovery failure. Preserve all artifacts and frozen scientific rules; do not duplicate work.')
        raise
if __name__=='__main__':main()
