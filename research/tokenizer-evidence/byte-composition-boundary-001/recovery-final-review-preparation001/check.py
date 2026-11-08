"""New immutable-metadata reader checks; no model, weights or statistics run."""
import ast
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch

from research import byte_input_quality_recovery_final_review_v1 as reviewer
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

OUT = Path(__file__).resolve().parent
started = time.perf_counter()
with CpuPrerequisiteGuard(OUT/'guard.json', reviewer.ROOT, timeout_seconds=60,
                         rss_limit_bytes=256*1024**2, reserve_bytes=8*1024**3):
    source = Path(reviewer.__file__)
    tree = ast.parse(source.read_text())
    imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(module and ('torch' in module or 'byte_input_model' in module) for module in imported)
    assert reviewer.sha(Path(reviewer.prior.__file__)) == reviewer.PRIOR_SHA
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and isinstance(n.func.value, ast.Name) and n.func.value.id == 'prior'
               and n.func.attr == 'contrast' for n in ast.walk(tree))
    result = reviewer.lineage()
    assert result['verified_metadata_only'] and result['accepts_no_panel']
    real_read = reviewer.read
    rejected = []
    cases = [
        (reviewer.BASE/'slots/14/lineage.json', 'new_training_updates', 1),
        (reviewer.ORIGINAL/'slots/14/guard.json', 'status', 'finished'),
        (reviewer.BASE/'slots/14/lineage.json', 'original_guard_remains_failed', False),
    ]
    for target, key, value in cases:
        def changed_read(path):
            data = real_read(path)
            if path == target:
                data = copy.deepcopy(data); data[key] = value
            return data
        try:
            with patch.object(reviewer, 'read', changed_read):
                reviewer.lineage()
        except AssertionError:
            rejected.append(dict(path=str(target), field=key, rejected_value=value))
        else:
            raise AssertionError(f'Invalid recovery metadata accepted: {key}')
    assert len(rejected) == 3 and reviewer.lineage() == result
    report = dict(accepted_preparation=True, full_panel_accepted=False, no_model_or_weight_loading=True,
                  no_new_bootstrap_draws=True, immutable_inputs_unchanged=True,
                  arithmetic_reused_sha256=reviewer.PRIOR_SHA,
                  reviewer_sha256=reviewer.sha(source), check_source_sha256=reviewer.sha(Path(__file__)),
                  checked_lineage=result, rejected_mutations=rejected,
                  elapsed_seconds=time.perf_counter()-started)
assert reviewer.read(OUT/'guard.json')['status'] == 'finished'
with (OUT/'result.json').open('x') as stream:
    json.dump(report, stream, indent=2, sort_keys=True); stream.write('\n')
print(json.dumps(report, indent=2))
