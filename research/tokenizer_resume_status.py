"""Read-only tokenizer handoff/status. Does not acquire worker locks or launch work."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / 'research/TOKENIZER_RESUME_STATE.json'

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

def main():
    index = read(INDEX)
    checks = {}
    for name, expected in index['evidence_sha256'].items():
        path = ROOT / name
        checks[name] = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    root = ROOT / index['campaign_root']
    controls = []
    for path in sorted((root / 'control').glob('process-*.json')):
        value = read(path)
        controls.append(dict(path=str(path.relative_to(ROOT)), **{
            key: value.get(key) for key in
            ('status', 'pid', 'parent_pid', 'elapsed_seconds', 'returncode', 'error')}))
    live = read(ROOT / 'research_state/tokenizer-quality-progress.json')
    terminal = read(ROOT / index['terminal_receipt'])
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(),
        saved_observation_at=index['observed_at'], branch=index['branch'],
        campaign_root=index['campaign_root'], immutable_evidence_checks=checks,
        local_progress=None if live is None else {k: live.get(k) for k in
            ('updated_at', 'status', 'active_run', 'next_action')},
        controller_terminal=terminal, process_records=controls,
        independent_final_review=(read(ROOT / index['independent_final_review'])
            if index.get('independent_final_review') else None),
        independent_final_guard=(read(ROOT / index['independent_final_guard'])
            if index.get('independent_final_guard') else None),
        quality_result_files=[str(p.relative_to(ROOT)) for p in sorted(
            (root / 'evidence/quality').glob('*/result.json'))],
        inference_preparation=read(root / 'control/inference-preparation.json'),
        inference_seed_receipts={str(seed): {
            'review': read(root / f'evidence/inference/{seed}/review.json'),
            'controller': read(root / f'evidence/inference/{seed}/controller.json')}
            for seed in (3511, 3512, 3513)},
        free_bytes=shutil.disk_usage(ROOT).free,
        caution='Process records and PIDs can be stale. Inspect actual processes before acting. Result files alone do not prove parent acceptance. Missing local files do not authorize reruns. This command never launches or resumes anything.')
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1

if __name__ == '__main__':
    raise SystemExit(main())
