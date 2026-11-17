"""Serial saved-model scorer then bounded analysis; records failures without retries."""
import datetime, json, os, subprocess, sys
from research.general_rsi_rolling_context_v1 import ROOT, OUT, read, put, sha


def state_update(**changes):
    p=ROOT/'research_state/general-rsi-state.json';s=read(p);s.update(changes)
    s['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    temp=p.with_suffix('.rolling.tmp')
    with temp.open('w') as f:json.dump(s,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,p)


def main():
    admission=read(OUT/'admission.json');assert admission['accepted']
    for name,digest in admission['execution_sources'].items():assert sha(ROOT/name)==digest
    with (OUT/'scoring.log').open('x') as log:
        worker=subprocess.Popen([sys.executable,'-m','research.general_rsi_rolling_context_v1','score'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        put(OUT/'supervisor-start.json',dict(pid=os.getpid(),worker_pid=worker.pid,admission_sha256=sha(OUT/'admission.json')))
        state_update(status='ROLLING_CONTEXT_SCORING',phase='saved_model_rolling_history_diagnostic',active_worker_pid=worker.pid,
                     active_session='general-rsi-rolling-context-001',rolling_context_plan='research/tokenizer-evidence/general-rsi-rolling-context-001/plan.json',
                     next_action='Serial saved-model rolling128/256 evaluation is active. Preserve worker and bound sources. Supervisor runs the frozen analysis after all six endpoints finish. Review results and reconcile state on completion. No training.')
        code=worker.wait()
    put(OUT/'scoring-exit.json',dict(returncode=code))
    if code:
        state_update(status='ROLLING_CONTEXT_EXECUTION_FAILED',active_worker_pid=None,active_session=None,next_action='Inspect rolling-context scoring failure and preserve first artifacts. Repair only a demonstrated implementation/resource failure in a separately identified attempt; no scientific retry or training.')
        return code
    for name,digest in admission['execution_sources'].items():assert sha(ROOT/name)==digest
    with (OUT/'analysis.log').open('x') as log:
        worker=subprocess.Popen([sys.executable,'-c',
            "from research.general_rsi_rolling_context_v1 import OUT, ROOT; from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard; from research.general_rsi_rolling_context_analysis_v1 import main; guard=CpuPrerequisiteGuard(OUT/'analysis-guard.json', ROOT, rss_limit_bytes=4*2**30, reserve_bytes=12*2**30, timeout_seconds=300);\nwith guard: main()"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        state_update(status='ROLLING_CONTEXT_ANALYZING',active_worker_pid=worker.pid,next_action='All rolling scores are complete. Bound bootstrap and target/hash verification are active; review result after supervisor completion.')
        code=worker.wait()
    put(OUT/'analysis-exit.json',dict(returncode=code))
    state_update(status='ROLLING_CONTEXT_AWAITING_REVIEW' if code==0 else 'ROLLING_CONTEXT_ANALYSIS_FAILED',active_worker_pid=None,active_session=None,
                 next_action='Review rolling-context scoring/analysis artifacts, verify all six matched target inventories and resource guards, write results and reconcile evidence. No new training or scoring retry.')
    return code
if __name__=='__main__':sys.exit(main())
