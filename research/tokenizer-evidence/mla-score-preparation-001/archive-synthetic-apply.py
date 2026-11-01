"""Typed-reference correction identified before archive apply; original plan preserved."""
import importlib.util,json
from pathlib import Path
P=Path(__file__).with_name('archive-synthetic.py')
spec=importlib.util.spec_from_file_location('mla_score_synthetic_archive',P)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
original_bound=a.bound

def bound(reference):
    path=Path(reference['path'])
    if path.suffix in ('.py','.md'):
        assert a.file_digest(path)==reference['sha256']
        return path
    return original_bound(reference)

a.bound=bound
from research.ffn_quality_v1.support import idle
with a.session_lock(a.ROOT/'research_state/tokenizer-worker-lock'):
    idle()
    with a.CpuPrerequisiteGuard(a.E/'apply-guard.json',a.ROOT,timeout_seconds=1800):
        a.publish(a.E/'apply-runner.json',dict(runner=a.ref(Path(__file__)),planned_source=a.ref(P),
            correction='Static pre-apply review found JSON parsing incorrectly applied to hash-only Python/Markdown references. Verify their exact hashes without parsing; JSON evidence binding, archive algorithm and allowlist unchanged. Original plan/source preserved; no failed archive attempt or deletion preceded this correction.'))
        a.apply()
