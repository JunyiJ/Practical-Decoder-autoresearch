"""One serial scorer followed by the bound CPU analysis, no automatic retries."""
import os,subprocess,sys
from research.general_rsi_train_serve_context_v1 import ROOT,OUT,read,put,sha
from research.general_rsi_rolling_context_supervisor_v1 import state_update


def main():
    admission=read(OUT/'admission.json');assert admission['accepted']
    for p,h in admission['sources'].items():assert sha(ROOT/p)==h
    for stage in ['score','analysis']:
        command=[sys.executable,'-m','research.general_rsi_train_serve_context_v1','score'] if stage=='score' else [sys.executable,'-c',
            "from research.general_rsi_train_serve_context_v1 import ROOT,OUT; from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard; from research.general_rsi_train_serve_analysis_v1 import main;\nwith CpuPrerequisiteGuard(OUT/'analysis-guard.json',ROOT,rss_limit_bytes=4*2**30,reserve_bytes=12*2**30,timeout_seconds=600): main()"]
        for p,h in admission['sources'].items():assert sha(ROOT/p)==h
        with (OUT/f'{stage}.log').open('x') as log:
            worker=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            put(OUT/f'{stage}-start.json',dict(supervisor_pid=os.getpid(),worker_pid=worker.pid,admission_sha256=sha(OUT/'admission.json')))
            state_update(status='TRAIN_SERVE_SCORING' if stage=='score' else 'TRAIN_SERVE_ANALYZING',active_worker_pid=worker.pid,
                active_session='general-rsi-train-serve-context-001',train_serve_plan=str((OUT/'plan.json').relative_to(ROOT)),
                next_action='Preserve serial train/serve diagnostic worker and all bound sources. Supervisor runs the frozen analysis; review after both stages exit. No training.')
            code=worker.wait()
        put(OUT/f'{stage}-exit.json',dict(returncode=code))
        if code:
            state_update(status='TRAIN_SERVE_EXECUTION_FAILED',active_worker_pid=None,active_session=None,next_action=f'Inspect preserved train/serve {stage} failure. Repair only demonstrated implementation/resource failure in distinct attempt. No scientific retry.')
            return code
    state_update(status='TRAIN_SERVE_AWAITING_REVIEW',active_worker_pid=None,active_session=None,
        next_action='Independently review all matched train/serve cells, reuse/endpoint/source hashes, native and extension qualification, bootstrap and guards. Report training-long/serving-short versus training-short/serving-long; publish authorized evidence and close monitor.')
    return 0

if __name__=='__main__':sys.exit(main())
