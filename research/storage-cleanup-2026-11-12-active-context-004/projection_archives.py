"""Exact audit/retirement of nine obsolete generated projection proof archives."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'storage-cleanup-2026-11-09'))
from common import ROOT, opens, protected_current, remove_entries, row, save, sha

OUT = Path(__file__).resolve().parent
NAME = 'tokenizer-readout-projection-resume-v1-001'
ARCHIVES = ROOT / 'research_runs' / NAME
EVIDENCE = ROOT / 'research/tokenizer-evidence/post-readout-boundary-001'
REVIEW = EVIDENCE / 'projection-resume001-review.json'
CLOSURE = EVIDENCE / 'projection-resume001-archive-closure.json'
FAILURE = ROOT / 'research/tokenizer-evidence/projection-quality-preparation-001/failure-closure001/review.json'
ACTIVE = ROOT / 'research/tokenizer-evidence/general-rsi-context-length-quality-001/plan.json'


def dependencies():
    for p in [ACTIVE, ROOT / 'research/GENERAL_RSI_PLAN.md', ROOT / 'research/GENERAL_RSI_NEXT_CAMPAIGNS.md']:
        assert NAME not in p.read_text(), p
    closure = json.loads(CLOSURE.read_text())
    assert closure['status'] == 'complete_accepted_generated_serialized_projection_resume_fixture'
    assert sha(REVIEW) == closure['review']['sha256']
    failure = json.loads(FAILURE.read_text())
    assert failure['adoption'] is False and failure['fresh_confirmation'] is False
    assert json.loads((ROOT / 'research_runs/tokenizer-readout-projection-quality-v1-001/STOP').read_text())['no_retry']


def prepare():
    dependencies()
    review = json.loads(REVIEW.read_text())
    assert review['all_original_commands_closed_no_retry_no_replay']
    entries = []
    for original in review['retained_model_archives']:
        p = Path(original['path'])
        assert p.parent == ARCHIVES and p.suffix == '.pt' and not original['non_tensor_fixture']
        e = row(p)
        assert e['sha256'] == original['sha256'] and e['size'] == original['bytes']
        entries.append(e)
    assert len(entries) == 9 and sum(e['size'] for e in entries) == 3016484885
    protected = protected_current()
    excluded = {e['path'] for e in entries}
    roots = [ARCHIVES, EVIDENCE / 'projection-resume001',
             ROOT / 'research/tokenizer-evidence/projection-quality-preparation-001/failure-closure001',
             ROOT / 'research_runs/tokenizer-readout-projection-quality-v1-001']
    for root in roots:
        for p in root.rglob('*'):
            if p.is_file() and not p.is_symlink() and str(p) not in excluded:
                protected[str(p)] = sha(p)
    for p in [REVIEW, CLOSURE, FAILURE, ACTIVE]:
        protected[str(p)] = sha(p)
    for study in ['general-rsi-native-horizon-001', 'general-rsi-composer-fp32-quality-001', 'general-rsi-context-length-quality-001']:
        for p in (ROOT / 'research_runs' / study).glob('slot-*/*.pt'):
            if p.name in ('raw.pt', 'time_ema.pt', 'h200.pt'):
                protected[str(p)] = sha(p)
    r = subprocess.run(['rg', '-l', NAME, 'research', 'src', 'research_state', '--glob', '*.py', '--glob', '*.json', '--glob', '*.md'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode in (0, 1)
    save(OUT / 'projection-manifest.json', dict(entries=entries, protected_sha256=protected,
         source_sha256={str(Path(__file__).resolve()): sha(__file__), str(ROOT / 'research/storage-cleanup-2026-11-09/common.py'): sha(ROOT / 'research/storage-cleanup-2026-11-09/common.py')},
         consumers=r.stdout.splitlines(), projected_bytes=sum(e['allocated'] for e in entries),
         policy_exception='Latest human AGENTS.md supersedes older retain-all proof-archive wording for these exact obsolete generated engineering payloads. Preserve their original proof receipts, source, logs, IO fixtures, all scientific/failure payloads and selected/fresh models.',
         consumer_review='References are the closed generated proof, its preparation/acceptance receipts, historical inventories and this audit. The downstream projection quality study is terminal no-retry after numerical failure. No current plan consumes these generated archives. No scientific quality inference or model replay is performed.',
         status='audited_pending_shared_research_lock'))
    print(json.dumps({'audited_files': len(entries), 'projected_bytes': sum(e['allocated'] for e in entries), 'protected_files': len(protected)}))


def execute():
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ps = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True)
        workers = [line for line in ps.splitlines() if 'python' in line.lower() and int(line.split()[0]) != os.getpid() and ('general_rsi' in line or NAME in line)]
        assert not workers, workers
        dependencies()
        m = json.loads((OUT / 'projection-manifest.json').read_text())
        for p, h in {**m['source_sha256'], **m['protected_sha256']}.items():
            assert sha(p) == h, p
        opened = opens()
        for e in m['entries']:
            assert e['path'] not in opened and row(e['path']) == e
        removed, skipped = remove_entries(m['entries'], OUT / 'projection-deletion-journal.jsonl')
        for p, h in m['protected_sha256'].items():
            assert sha(p) == h, p
        import shutil
        save(OUT / 'projection-completed.json', dict(files_removed=len(removed), allocated_bytes_removed=sum(e['allocated'] for e in removed), skipped=skipped, protected_files_verified=len(m['protected_sha256']), free_bytes=shutil.disk_usage(ROOT).free, manifest_sha256=sha(OUT / 'projection-manifest.json')))
        print((OUT / 'projection-completed.json').read_text())


if __name__ == '__main__':
    {'prepare': prepare, 'execute': execute}[sys.argv[1]]()
