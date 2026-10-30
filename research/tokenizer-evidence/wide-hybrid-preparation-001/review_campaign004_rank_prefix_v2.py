"""Read-only live rank prefix audit; no training, selection, cleanup or retry."""
import argparse,base64,json,signal,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_quality_dispatch_v2 import _prior
from src.research.tokenizer_wide_hybrid_quality_rank_startup_v2 import _audit
HERE=Path(__file__).resolve().parent
RUN=ROOT/'research_runs/tokenizer-wide-hybrid-004'
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def stopped(signum,frame):raise KeyboardInterrupt('Review signal '+str(signum))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--completed',type=int,required=True);a=ap.parse_args()
    assert 10<=a.completed<=14
    stem=f'campaign004-rank-prefix-{a.completed:02d}-v2'
    for suffix in ('-guard.json','-review.json','-failure.json'):
        assert not (HERE/(stem+suffix)).exists()
    freeze=json.loads((HERE/'campaign004-freeze-v2-review.json').read_text());sha=freeze['plan_sha256']
    boundary=json.loads((HERE/'campaign004-lr-cleanup-v2-review.json').read_text())['reference']
    terminal=json.loads((HERE/f'campaign004-rank-{a.completed-1:02d}-v2-terminal.json').read_text())
    assert terminal['status']=='quality_trial_accepted' and terminal['slot']==a.completed-1
    projection=dict(path='meta/projection.json',sha256=file_digest(RUN/'evidence/meta/projection.json'))
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stopped)
    try:
        with session_lock(ROOT/'research_state/tokenizer-worker-lock') as g,session_lock(RUN) as c, CpuPrerequisiteGuard(HERE/(stem+'-guard.json'),ROOT,timeout_seconds=1800):
            assert not (RUN/f'evidence/quality/{a.completed:02d}').exists()
            writer=MetadataWriter(RUN,(g,c))
            checked=_audit(writer,a.completed,sha,projection,boundary)
            refs=_prior(writer,a.completed,sha,projection,checked['history']['pilot_completion_ref'],lr_completion_ref=boundary)
            assert refs[-1]==terminal['reference']
            files={};metadata={}
            for p in sorted(RUN.rglob('*')):
                assert not p.is_symlink()
                if not p.is_file():continue
                n=str(p.relative_to(RUN));files[n]=dict(size=p.stat().st_size,sha256=file_digest(p))
                if not n.endswith('.pt'):metadata[n]=base64.b64encode(p.read_bytes()).decode()
            assert all((RUN/n).stat().st_size==v['size'] and file_digest(RUN/n)==v['sha256'] for n,v in files.items())
            report=dict(status='independently_reviewed_rank_prefix',completed=a.completed,plan_sha256=sha,
                reviewer_sha256=file_digest(Path(__file__)),lr_completion_ref=boundary,dispatch_references=refs,
                records=checked['prefix']['records'],checkpoint_pairs=checked['checkpoint_pairs'],storage=checked['storage'],
                results=[json.loads((RUN/f'evidence/quality/{i:02d}/result.json').read_text()) for i in range(9,a.completed)],
                rates=checked['history']['rates'],selection=None,adopted=False,holdout_consumed=False,
                files=files,noncheckpoint_bytes_base64=metadata)
            save(HERE/(stem+'-review.json'),report)
            print(json.dumps(dict(status=report['status'],completed=a.completed,storage=report['storage'])),flush=True)
    except BaseException as error:
        save(HERE/(stem+'-failure.json'),dict(status='failed_closed',error=repr(error),retry_allowed=False))
        raise
if __name__=='__main__':main()
