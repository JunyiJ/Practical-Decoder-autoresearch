"""One explicit rank boundary phase using frozen production APIs; never train/retry."""
import argparse
import base64
import json
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.research.protocol import file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_rank_selection_v2 import save_rank_selection, verify_rank_selection
from src.research.tokenizer_wide_hybrid_rank_boundary_v2 import cleanup_rank_boundary, verify_rank_boundary

HERE = Path(__file__).resolve().parent
RUN = ROOT / 'research_runs/tokenizer-wide-hybrid-004'


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def archive():
    files = {}
    metadata = {}
    for path in sorted(RUN.rglob('*')):
        assert not path.is_symlink(), path
        if path.is_file():
            name = str(path.relative_to(RUN))
            files[name] = dict(size=path.stat().st_size, sha256=file_digest(path))
            if not name.endswith('.pt'):
                metadata[name] = base64.b64encode(path.read_bytes()).decode()
    return dict(files=files, noncheckpoint_bytes_base64=metadata)


def stopped(signum, frame):
    raise KeyboardInterrupt('Boundary signal ' + str(signum))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['select', 'cleanup'])
    args = parser.parse_args()
    stem = 'campaign004-rank-' + args.phase + '-v2'
    for suffix in ('-guard.json', '-review.json', '-failure.json', '-before.json'):
        assert not (HERE / (stem + suffix)).exists(), 'No boundary retry'
    sha = json.loads((HERE / 'campaign004-freeze-v2-review.json').read_text())['plan_sha256']
    assert file_digest(RUN / 'evidence/meta/plan.json') == sha
    terminal = json.loads((HERE / 'campaign004-rank-14-v2-terminal.json').read_text())
    assert terminal['status'] == 'quality_trial_accepted' and terminal['slot'] == 14
    previous = None
    if args.phase == 'cleanup':
        previous = json.loads((HERE / 'campaign004-rank-select-v2-review.json').read_text())
        guard = json.loads((HERE / 'campaign004-rank-select-v2-guard.json').read_text())
        assert previous['review']['status'] == 'rank_selection_verified_pending_cleanup'
        assert previous['plan_sha256'] == sha
        assert guard['status'] == 'finished' and guard['reason'] is None
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stopped)
    try:
        with session_lock(ROOT / 'research_state/tokenizer-worker-lock') as g, session_lock(RUN) as c, CpuPrerequisiteGuard(HERE / (stem + '-guard.json'), ROOT, timeout_seconds=1800):
            writer = MetadataWriter(RUN, (g, c))
            before = archive()
            if previous is not None:
                assert before == previous['archive'], 'Evidence changed since selection review'
            save(HERE / (stem + '-before.json'), before)
            if args.phase == 'select':
                lr_boundary = json.loads((HERE / 'campaign004-lr-cleanup-v2-review.json').read_text())['reference']
                projection = dict(path='meta/projection.json', sha256=file_digest(RUN / 'evidence/meta/projection.json'))
                ref = save_rank_selection(writer, sha, projection_ref=projection, lr_completion_ref=lr_boundary)
                review = verify_rank_selection(writer, sha, ref)
                deleted = set()
            else:
                ref = cleanup_rank_boundary(writer, sha, previous['reference'])
                review = verify_rank_boundary(writer, sha, ref)
                deleted = {'evidence/' + row['path'] for row in previous['review']['audit']['stage']['manifest']['delete']}
            after = archive()
            assert set(before['files']) - set(after['files']) == deleted
            assert all(after['files'][n] == binding for n, binding in before['files'].items() if n not in deleted)
            assert all(after['noncheckpoint_bytes_base64'][n] == value for n, value in before['noncheckpoint_bytes_base64'].items())
            save(HERE / (stem + '-review.json'), dict(plan_sha256=sha, reference=ref,
                reviewer_sha256=file_digest(Path(__file__)), review=review, archive=after,
                deleted_paths=sorted(deleted), adopted=False, holdout_consumed=False, training_launched=False))
        print(json.dumps(dict(phase=args.phase, reference=ref, status=review['status'])), flush=True)
    except BaseException as error:
        save(HERE / (stem + '-failure.json'), dict(error=repr(error), retry_allowed=False))
        raise


if __name__ == '__main__':
    main()
