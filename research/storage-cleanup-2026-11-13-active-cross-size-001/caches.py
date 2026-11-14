"""Evict verified unused HTTP/code/package cache objects under external lock."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'storage-cleanup-2026-11-09'))
from common import ROOT, opens, protected_current as previous_protected, remove_entries, row, save, sha
OUT=Path(__file__).resolve().parent
PANEL=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-quality-001/plan.json'
def capacity():
    import shutil
    plan=json.loads(PANEL.read_text())
    runs=ROOT/'research_runs/general-rsi-cross-size-quality-001'
    retained=sum(f.stat().st_size for f in runs.glob('slot-*/*.pt') if f.name in ('raw.pt','h200.pt'))
    state=json.loads((ROOT/'research_state/general-rsi-state.json').read_text())
    active=runs/f"slot-{state['active_slot']:02d}"
    # Credit the currently retained restart once against its already-budgeted
    # storage. The second overlapping serialization slot remains reserved.
    active_restart=sum(f.stat().st_size for f in active.glob('restart-*.pt'))
    required=plan['reserve_bytes']+max(0,plan['incremental_peak_bytes']-retained-active_restart)
    free=shutil.disk_usage(ROOT).free
    return dict(free_bytes=free,required_free_bytes=required,margin_bytes=free-required,deficit_bytes=max(0,required-free),retained_endpoint_bytes=retained,active_restart_bytes=active_restart,reserve_includes_10GiB_plus_2GiB=True,panel=str(PANEL.relative_to(ROOT)))
def protected_current():
    protected=previous_protected()
    paths=[PANEL,ROOT/'research/GENERAL_RSI_CROSS_SIZE_QUALITY_DESIGN.md']
    paths += [p for p in (ROOT/'research_runs/general-rsi-native-horizon-001').glob('slot-*/*.pt') if p.name in ('raw.pt','time_ema.pt')]
    paths += [p for p in (ROOT/'research_runs/general-rsi-context-length-quality-001').glob('slot-*/*.pt') if p.name in ('raw.pt','time_ema.pt')]
    paths += [p for p in (ROOT/'research_runs/general-rsi-cross-size-quality-001').glob('slot-*/*.pt') if p.name in ('raw.pt','h200.pt')]
    protected.update({str(p):sha(p) for p in paths})
    return protected

BASES = [Path('/Users/junyi/Library/Caches/Google/Chrome'),
         Path('/Users/junyi/Library/Caches/Codex'),
         Path('/Users/junyi/Library/Application Support/Code/CachedData')]
NPM = Path('/Users/junyi/.npm/_cacache/content-v2/sha512')

with (ROOT/'research_state/external-cache-maintenance.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    protected = protected_current()
    opened = opens()
    processes = subprocess.check_output(['ps','-axo','pid=,comm=,args='],text=True)
    package_busy = [line for line in processes.splitlines() if re.search(r'\b(npm|npx|pnpm|yarn)(?:\s|$)',line)]
    candidates=[]
    for base in BASES:
        for p in base.rglob('*'):
            if p.is_symlink() or not p.is_file() or str(p) in opened or not re.fullmatch(r'[0-9a-f]{16}_0',p.name):
                continue
            try:
                with p.open('rb') as f: magic=f.read(8)
                if magic==bytes.fromhex('305c72a71b6dfbfc'): candidates.append(row(p))
            except FileNotFoundError:
                continue
    if not package_busy:
        for p in NPM.glob('*/*/*'):
            if p.is_symlink() or not p.is_file() or str(p) in opened: continue
            if p.stat().st_uid != os.getuid() or p.stat().st_nlink != 1 or p.resolve() != p: continue
            digest=''.join(p.relative_to(NPM).parts)
            if not re.fullmatch('[0-9a-f]{128}',digest): continue
            with p.open('rb') as f: actual=hashlib.file_digest(f,'sha512').hexdigest()
            if digest==actual: candidates.append(row(p))
    before=capacity()
    save(OUT/'cache-manifest.json',dict(reason='Standing-authorized recovery after a new active cross-size panel headroom deficit. Only self-identifying unopened Chromium cache entries and integrity-verified npm content objects without package-manager consumers. Indexes, profiles, settings, credentials, runtime assets, installed packages and pending updaters excluded.',
         before=before, entries=candidates, protected_sha256=protected, package_busy=package_busy))
    removed,skipped=remove_entries(candidates, OUT/'cache-deletion-journal.jsonl')
    for p,h in protected.items(): assert sha(p)==h,p
    result=dict(status='completed',files_removed=len(removed),allocated_bytes_removed=sum(e['allocated'] for e in removed),
                before=before,after=capacity(),skipped_changed_or_open=skipped,protected_hashes_verified=len(protected))
    save(OUT/'cache-completed.json',result)
    print(json.dumps(result))
