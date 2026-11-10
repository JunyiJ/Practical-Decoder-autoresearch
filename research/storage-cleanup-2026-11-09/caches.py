"""Evict verified unused HTTP/code/package cache objects under external lock."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from common import OUT, ROOT, capacity, opens, protected_current, remove_entries, row, save, sha

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
    save(OUT/'cache-manifest.json',dict(reason='Direct human request. Only self-identifying unopened Chromium cache entries and integrity-verified npm content objects without package-manager consumers. Indexes, profiles, settings, credentials, runtime assets, installed packages and pending updaters excluded.',
         before=before, entries=candidates, protected_sha256=protected, package_busy=package_busy))
    removed,skipped=remove_entries(candidates, OUT/'cache-deletion-journal.jsonl')
    for p,h in protected.items(): assert sha(p)==h,p
    result=dict(status='completed',files_removed=len(removed),allocated_bytes_removed=sum(e['allocated'] for e in removed),
                before=before,after=capacity(),skipped_changed_or_open=skipped,protected_hashes_verified=len(protected))
    save(OUT/'cache-completed.json',result)
    print(json.dumps(result))
