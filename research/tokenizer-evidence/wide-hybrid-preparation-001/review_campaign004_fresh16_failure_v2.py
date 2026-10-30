"""Read-only failed-startup preservation and frozen request diagnostic; no retry."""
import base64,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_first_confirmation_startup_v2 import _frozen
HERE=Path(__file__).resolve().parent
RUN=ROOT/'research_runs/tokenizer-wide-hybrid-004'
def main():
    stem='campaign004-fresh16-failure-review-v2'
    assert not (HERE/(stem+'.json')).exists()
    prior=json.loads((HERE/'campaign004-fresh-prefix-16-v2-review.json').read_text())
    terminal=json.loads((HERE/'campaign004-fresh-16-v2-terminal.json').read_text())
    assert terminal['status']=='failed_closed'
    with session_lock(ROOT/'research_state/tokenizer-worker-lock') as g,session_lock(RUN) as c, CpuPrerequisiteGuard(HERE/(stem+'-guard.json'),ROOT,timeout_seconds=1800):
        writer=MetadataWriter(RUN,(g,c));root=RUN/'evidence'
        for n,v in prior['files'].items():
            assert (RUN/n).stat().st_size==v['size'] and file_digest(RUN/n)==v['sha256']
        for n,v in prior['noncheckpoint_bytes_base64'].items():
            assert (RUN/n).read_bytes()==base64.b64decode(v)
        assert not (RUN/'control/process-quality-16.json').exists()
        assert not (root/'meta/quality-confirm-startup-16.json').exists()
        assert not (root/'meta/quality-readiness-16.json').exists()
        assert sorted(p.name for p in (root/'quality/16').iterdir())==['request.json']
        projection=dict(path='meta/projection.json',sha256=file_digest(root/'meta/projection.json'))
        bound,requests=_frozen(writer,16,prior['plan_sha256'],projection,prior['rates'],prior['selection'])
        pending=json.loads((root/'quality/16/request.json').read_text())
        seal=json.loads((root/'meta/quality-readiness-15.json').read_text())
        assert digest(pending)==digest(requests[-1])
        assert bound['files']==seal['frozen_files'] and bound['context']==seal['context']
        files={};metadata={}
        for p in sorted(RUN.rglob('*')):
            assert not p.is_symlink()
            if not p.is_file():continue
            n=str(p.relative_to(RUN));files[n]=dict(size=p.stat().st_size,sha256=file_digest(p))
            if not n.endswith('.pt'):metadata[n]=base64.b64encode(p.read_bytes()).decode()
        assert all(file_digest(RUN/n)==v['sha256'] for n,v in files.items())
        report=dict(status='failed_startup_preserved',terminal=terminal,plan_sha256=prior['plan_sha256'],
            reviewer_sha256=file_digest(Path(__file__)),prior_files_unchanged=len(prior['files']),
            prior_noncheckpoint_bytes_unchanged=len(prior['noncheckpoint_bytes_base64']),
            appended_paths=sorted(set(files)-set(prior['files'])),child_launched=False,
            pending_request_matches_frozen_choice=True,frozen_files_and_context_unchanged=True,
            diagnosis='Startup compares the full bind_campaign result before and after publishing the request and finishing its guard. That result includes mutable metadata_bytes; the comparison treats intentional metadata growth as frozen identity drift. Original intermediate bound values were not recorded.',
            current_metadata_bytes=bound['metadata_bytes'],retry_allowed=False,adopted=False,holdout_consumed=False,
            files=files,noncheckpoint_bytes_base64=metadata)
        with (HERE/(stem+'.json')).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        print(json.dumps({k:v for k,v in report.items() if k not in ('files','noncheckpoint_bytes_base64')}),flush=True)
if __name__=='__main__':main()
