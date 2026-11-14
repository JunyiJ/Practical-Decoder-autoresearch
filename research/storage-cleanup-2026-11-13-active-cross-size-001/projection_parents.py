"""Audit three obsolete accepted screen parents; retire only under the idle lock."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'storage-cleanup-2026-11-09'))
from common import ROOT, opens, remove_entries, row, save, sha
OUT = Path(__file__).resolve().parent
NAME = 'tokenizer-readout-projection-quality-v1-001'
RUN = ROOT / 'research_runs' / NAME
CLOSURE = ROOT / 'research/tokenizer-evidence/projection-quality-preparation-001/failure-closure001/review.json'
PANEL = ROOT / 'research/tokenizer-evidence/general-rsi-cross-size-quality-001/plan.json'
SLOTS = (1, 2, 4)

def dependencies():
    j = json.loads(CLOSURE.read_text())
    assert j['status'] == 'CLOSED_FAILED_INCOMPLETE_CPU_MPS_RAW_ACCEPTANCE'
    assert not any(j[k] for k in ('adoption', 'fresh_confirmation', 'scientific_selection'))
    assert j['no_retry_no_stop_clear_no_source_or_tolerance_rescue']
    assert json.loads((RUN / 'STOP').read_text())['no_retry']
    assert all(j['rows'][i]['independently_accepted'] for i in SLOTS)
    for p in (PANEL, ROOT / 'research/GENERAL_RSI_PLAN.md', ROOT / 'research/GENERAL_RSI_NEXT_CAMPAIGNS.md'):
        assert NAME not in p.read_text(), p
    for receipt in j['original_metadata']:
        p = Path(receipt['path'])
        assert p.stat().st_size == receipt['bytes'] and sha(p) == receipt['sha256'], p
    return j

def prepare():
    j = dependencies()
    entries = []
    for original in j['retained_archives']:
        if original['kind'] == 'parent' and original['slot'] in SLOTS:
            p = RUN / f"evidence/quality/{original['slot']:02d}/checkpoints/parent.pt"
            assert str(p) == original['path']
            e = row(p)
            assert e['sha256'] == original['sha256'] and e['size'] == original['size']
            entries.append(e)
    assert len(entries) == 3 and sum(e['size'] for e in entries) == 1353498915
    excluded = {e['path'] for e in entries}
    protected = json.loads((OUT / 'cache-manifest.json').read_text())['protected_sha256']
    for root in (RUN, CLOSURE.parent):
        for p in root.rglob('*'):
            if p.is_file() and not p.is_symlink() and str(p) not in excluded:
                protected[str(p)] = sha(p)
    protected[str(PANEL)] = sha(PANEL)
    opened = opens()
    assert not excluded.intersection(opened)
    r = subprocess.run(['rg', '-l', NAME, 'research', 'src', 'research_state', '--glob', '*.py', '--glob', '*.json', '--glob', '*.md'], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode in (0, 1)
    save(OUT / 'projection-parent-manifest.json', dict(
        status='audited_pending_slot08_completion_and_shared_research_lock', entries=entries,
        projected_allocated_bytes=sum(e['allocated'] for e in entries), protected_sha256=protected,
        source_sha256={str(Path(__file__).resolve()): sha(__file__), str(ROOT / 'research/storage-cleanup-2026-11-09/common.py'): sha(ROOT / 'research/storage-cleanup-2026-11-09/common.py')},
        consumers=r.stdout.splitlines(),
        consumer_review='Only historical screen contract/protocol, closure/acceptance receipts, inventories and cleanup audits reference this study. Its incomplete numerical failure is terminal no-retry, with no selection, adoption or fresh confirmation. No active plan consumes these parents. The three accepted screening slots are obsolete; all H200 payloads, native parents, failed slot07 parent and original source/log/config/metric/receipt evidence remain preserved.',
        policy_exception='Latest human AGENTS.md permits obsolete concluded-study checkpoint retirement after dependency closure. This narrowly supersedes the historical retained_archive retention labels and closure_creates_no_retirement_eligibility=true policy. Historical evidence is unchanged. No scientific negative is inferred from numerical failure.',
        original_metadata_verified=len(j['original_metadata'])))
    print(json.dumps(dict(files=len(entries), allocated_bytes=sum(e['allocated'] for e in entries), protected_files=len(protected))))

def execute():
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ps = subprocess.check_output(['ps', '-axo', 'pid=,args='], text=True)
        workers=[]
        for line in ps.splitlines():
            parts=line.strip().split(None, 2)
            if len(parts) >= 2 and int(parts[0]) != os.getpid() and Path(parts[1]).name.lower().startswith('python'):
                workers.append(line)
        assert not workers, workers
        dependencies()
        m=json.loads((OUT / 'projection-parent-manifest.json').read_text())
        for p,h in {**m['source_sha256'], **m['protected_sha256']}.items():
            assert sha(p) == h, p
        opened=opens()
        for e in m['entries']:
            assert row(e['path']) == e and e['path'] not in opened
        before=shutil.disk_usage(ROOT).free
        removed,skipped=remove_entries(m['entries'], OUT / 'projection-parent-deletion-journal.jsonl')
        for p,h in m['protected_sha256'].items():
            assert sha(p) == h, p
        plan=json.loads(PANEL.read_text())
        retained=sum(p.stat().st_size for p in (ROOT/'research_runs/general-rsi-cross-size-quality-001').glob('slot-*/*.pt') if p.name in ('raw.pt','h200.pt'))
        required=plan['reserve_bytes']+max(0,plan['incremental_peak_bytes']-retained)
        free=shutil.disk_usage(ROOT).free
        result=dict(files_removed=len(removed), allocated_bytes_removed=sum(e['allocated'] for e in removed), skipped=skipped,
                    protected_files_verified=len(m['protected_sha256']), free_bytes_before=before, free_bytes=free,
                    required_free_bytes=required, margin_bytes=free-required, reserve_includes_10GiB_plus_2GiB=True,
                    manifest_sha256=sha(OUT/'projection-parent-manifest.json'))
        save(OUT/'projection-parent-completed.json',result)
        print(json.dumps(result))

if __name__ == '__main__':
    {'prepare': prepare, 'execute': execute}[sys.argv[1]]()
