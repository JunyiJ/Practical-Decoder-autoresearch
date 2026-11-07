"""Read existing CPU metadata and prepare a future source-only norm candidate."""
import ast
import hashlib
import json
import os
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
B = Path(__file__).resolve().parent
OUT = B/'forward-sensitivity002-analysis001'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def put(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+len(raw) < 2*1024**2
    with (OUT/name).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def main():
    assert not (OUT/'attempt.json').exists()
    plan = json.loads((OUT/'plan.json').read_text())
    put('attempt.json', dict(plan_sha256=sha(OUT/'plan.json'), pid=os.getpid()))
    with CpuPrerequisiteGuard(OUT/'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
        for name, digest in plan['sources'].items():
            assert sha(ROOT/name) == digest, name
        for name in plan['source_only_ast_review']:
            ast.parse((ROOT/name).read_text())
        files = sorted((B/'forward-sensitivity002').glob('case-*.json'))
        rows = [json.loads(p.read_text()) for p in files]
        assert len(rows) == 54 and all(r['device'] == 'cpu' for r in rows)
        assert not (B/'forward-sensitivity002/result.json').exists()
        summaries = {}
        for dtype in ('torch.float64', 'torch.float32'):
            group = [r for r in rows if r['dtype'] == dtype]
            assert len(group) == 27
            summaries[dtype] = {stage: max(r['stage_max_absolute_errors'][stage] for r in group)
                                for stage in ('features', 'projected', 'transformed', 'logits', 'nll')}
        paired = []
        for row in rows:
            if row['dtype'] == 'torch.float32' and row['case'].endswith('shifted'):
                center = next(r for r in rows if r['dtype'] == row['dtype'] and r['arm'] == row['arm']
                              and r['case'] == row['case'].replace('shifted', 'centered'))
                assert row['scalar_ce'] == center['scalar_ce']
                paired.append(dict(center=center['case'], shifted=row['case'], arm=row['arm'],
                                   observed_cpu32_mean_ce_shift=row['ce']-center['ce'],
                                   scalar_mean_ce_shift=0.))
        assert len(paired) == 9
        result = dict(status='complete_existing_CPU_metadata_analysis_and_untested_source_preparation',
            original_diagnostic='CLOSED_FAILED_REPORTING_DTYPE_MPS_INCOMPLETE', cpu_rows=54, mps_rows=0,
            stage_maxima_vs_independent_scalar=summaries, cpu32_translation_pairs=paired,
            sources_preserved=len(plan['sources']), candidate_runtime_tested=False,
            input_metadata=[dict(path=str(p), sha256=sha(p)) for p in files],
            zero_torch_tensor_model_optimizer_checkpoint_corpus_rng_scoring_calls=True,
            diagnostic_complete_or_MPS_numeric_or_failed_checkpoint_cause=False,
            no_closed_command_retry_or_source_edit=True)
        put('result.json', result)
    put('STOP', dict(status='complete_read_only_CPU_analysis_and_source_only_preparation', no_replay=True))
    print(json.dumps(dict(status=result['status'], cpu_rows=54, mps_rows=0, bindings=result['sources_preserved'])))


if __name__ == '__main__':
    main()
