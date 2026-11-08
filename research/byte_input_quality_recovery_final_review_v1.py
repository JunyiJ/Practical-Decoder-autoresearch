"""Independent recovered-panel reader; reuse the already checked statistics.

No model imports or execution. The original failed guard is never rewritten or
treated as a terminal pass. Inspect its immutable lineage and the separately
accepted saved-checkpoint recovery before applying the unchanged quality gates.
"""
import ast
import fcntl
import json
import math
import os
from pathlib import Path
import sys

from research import byte_input_quality_final_review_v1 as prior

ROOT, ORIGINAL, PLAN_SHA = prior.ROOT, prior.BASE, prior.PLAN_SHA
BASE = ORIGINAL.parent/'quality-recovery001'
RECOVERY_SHA = 'd41ab762478c90a353ac6319b0e3801d9b562eab5869f5a54687e86de26c3f29'
PRIOR_SHA = '088039e6783a322f9baf7ebaa00f617233199405975e7dcbb18ffb80a628c58a'
read, sha, verify_file = prior.read, prior.sha, prior.verify_file


def slot_dir(index):
    return (ORIGINAL if index < 14 else BASE)/'slots'/f'{index:02}'


def check_guard(guard, timeout, rss=4*1024**3, outputs=True):
    assert guard['status'] == 'finished' and guard['reason'] is None
    assert guard['limits'] == dict(timeout_seconds=timeout, rss_limit_bytes=rss, reserve_bytes=8*1024**3)
    observation = guard['observation']
    assert 0 <= guard['elapsed_seconds'] < timeout
    assert observation['peak_rss_bytes'] < rss and observation['free_bytes'] >= 8*1024**3
    if outputs:
        assert observation['metadata_allowance_bytes'] == 512*1024**2
        assert observation['charged_output_bytes'] == observation['retained_weight_bytes']+512*1024**2
        assert observation['charged_output_bytes'] <= 10*1024**3
        assert observation['mps_driver_bytes'] < 6*1024**3


def lineage():
    """Read only immutable metadata; this function alone accepts no panel."""
    assert sha(Path(prior.__file__)) == PRIOR_SHA
    assert sha(ORIGINAL/'plan.json') == PLAN_SHA and sha(BASE/'plan.json') == RECOVERY_SHA
    plan, recovery = read(ORIGINAL/'plan.json'), read(BASE/'plan.json')
    assert recovery['original_plan_sha256'] == PLAN_SHA and recovery['scientific_changes'] == []
    assert recovery['design'] == plan['design'] and recovery['selection'] == read(ORIGINAL/'selection.json')
    assert recovery['slots'] == [14, 15, 16, 17] and recovery['new_training_slots'] == [15, 16, 17]
    assert recovery['postprocessing_only_slot'] == 14 and recovery['training_replay'] is False
    assert recovery['original_failed_records_immutable'] and recovery['independent_final_acceptance_required']
    assert recovery['adopted'] is False
    for record in recovery['sources']+recovery['prerequisites']:
        verify_file(record)
    verify_file(recovery['original_stop'])
    original_stop = read(ORIGINAL/'STOP')
    assert original_stop['status'] == 'closed_failed_original_quality_controller' and original_stop['no_replay']
    assert original_stop['error'] == "AssertionError(('Scientific slot failed; preserve and diagnose, no automatic retry', 14, 91))"
    failed_out = ORIGINAL/'slots/14'; failed_guard = read(failed_out/'guard.json')
    assert failed_guard['status'] == 'guard_error' and failed_guard['reason'] == "FileNotFoundError(2, 'No such file or directory')"
    assert failed_guard['observation'] is None  # Never invent a recovered terminal sample.
    assert failed_guard['pid'] == read(failed_out/'attempt.json')['pid']
    assert failed_guard['limits'] == dict(timeout_seconds=7200, rss_limit_bytes=4*1024**3, reserve_bytes=8*1024**3)
    assert 0 < failed_guard['elapsed_seconds'] == recovery['failed_guard_elapsed_seconds'] < 7200
    assert not (failed_out/'acceptance.json').exists() and not (failed_out/'STOP').exists()
    failure = read(Path(recovery['failure_record']))
    assert failure['worker_exit_code'] == 91 and failure['controller_exit_code'] == 1
    assert failure['original_closed'] and failure['no_training_replay']
    expected = {str(p.resolve()) for p in [ORIGINAL/'STOP', failed_out/'attempt.json', failed_out/'guard.json',
        ROOT/'research/byte_input_quality_v1/entry.py', ROOT/'research/byte_input_training_v2.py',
        ROOT/'src/research/cpu_prerequisite_guard.py']}
    expected.update(str((failed_out/f'step{i:04}.json').resolve()) for i in range(1, 1201))
    expected.update(str((failed_out/f'diagnostic{i:04}.json').resolve()) for i in (1, 301, 601, 901, 1200))
    records = failure['original_evidence']
    assert len(records) == len(expected) == 1211 and {r['path'] for r in records} == expected
    for record in records:
        verify_file(record)
    audit_root = Path(recovery['failure_record']).parent
    audit = read(audit_root/'raw-audit.json')
    check_guard(read(audit_root/'raw-audit-guard.json'), 120, rss=2*1024**3, outputs=False)
    assert audit['accepted_read_only_integrity'] and audit['original_scientific_slot_accepted'] is False
    assert audit['no_training_or_model_execution'] and audit['plan_sha256'] == PLAN_SHA
    assert audit['completed_steps'] == audit['matched_step_records'] == 1200 and audit['snapshots'] == 41
    assert audit['parameter_tensors'] == 88 and audit['parameter_elements'] == 28492416
    assert audit['raw_receipt'] == failure['raw_receipt'] == recovery['raw_receipt']
    assert audit['source_sha256'] == sha(audit_root/'inspect_raw.py')
    reproduction = read(audit_root/'reproduction.json')
    assert reproduction['accepted'] and reproduction['no_model_execution'] and reproduction['limits_unchanged']
    assert reproduction['cases']['original_race']['type'] == 'FileNotFoundError'
    assert Path(reproduction['cases']['original_race']['filename']).name == 'raw.pt.tmp'
    for path, digest in reproduction['source_sha256'].items():
        assert sha(Path(path)) == digest
    original_driver = (ROOT/'research/byte_input_quality_v1/driver.py').read_text()
    recovery_driver = (ROOT/'research/byte_input_quality_recovery_v1/driver.py').read_text()
    def suffix(source):
        start = source.index('  averaged=inference.export(c,parent)')
        return source[start:source.index('\n finally:', start)]
    assert suffix(original_driver) == suffix(recovery_driver)
    import hashlib
    assert hashlib.sha256(suffix(recovery_driver).encode()).hexdigest() == recovery['exact_original_posttraining_suffix_sha256']
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'step'
                   for n in ast.walk(ast.parse(recovery_driver)))
    out = slot_dir(14); link = read(out/'lineage.json')
    assert link['raw_receipt'] == failure['raw_receipt'] == read(out/'review.json')['parent']
    assert link['new_training_updates'] == 0 and link['original_guard_remains_failed']
    assert link['original_failed_slot'] == str(failed_out)
    assert link['original_guard_elapsed_seconds'] == failed_guard['elapsed_seconds']
    for name in [f'step{i:04}.json' for i in range(1, 1201)]+[f'diagnostic{i:04}.json' for i in (1, 301, 601, 901, 1200)]:
        assert sha(out/name) == sha(failed_out/name)
    check_guard(read(out/'guard.json'), 7200-failed_guard['elapsed_seconds'])
    assert read(out/'guard.json')['elapsed_seconds']+failed_guard['elapsed_seconds'] < 7200
    assert read(out/'review.json')['sampler_hashes'] == audit['sampler_hashes']
    assert all(not (ORIGINAL/'slots'/f'{i:02}'/'attempt.json').exists() for i in (15, 16, 17))
    forecast = recovery['forecast']
    assert forecast['original_complete_upper_bytes'] == plan['forecast']['complete_upper_bytes']
    assert forecast['original_wall_budget_seconds'] == 130800
    assert forecast['free_bytes_now'] >= forecast['remaining_entry_floor_bytes']
    assert forecast['remaining_entry_floor_bytes'] == 20*1024**3-forecast['retained_bytes_now']
    return dict(verified_metadata_only=True, accepts_no_panel=True, original_failed_guard_preserved=True,
        original_training_final_rss_observation_unavailable=True, recovered_training_updates=0,
        original_guard_elapsed_seconds=failed_guard['elapsed_seconds'],
        recovery_guard_elapsed_seconds=read(out/'guard.json')['elapsed_seconds'],
        original_failure_manifest_sha256=sha(Path(recovery['failure_record'])), recovery_plan_sha256=RECOVERY_SHA)


def run():
    recovery_lineage = lineage(); plan, recovery = read(ORIGINAL/'plan.json'), read(BASE/'plan.json')
    assert read(BASE/'STOP')['status'] == 'complete_recovered_quality_panel_pending_independent_review'
    check_guard(read(BASE/'statistics-guard.json'), 300, outputs=False)
    final = read(BASE/'final.json')
    assert final['accepted_complete_panel'] and final['trajectories'] == 18 and final['updates'] == 21600
    assert final['design'] == plan['design'] and final['bootstrap_seed'] == plan['bootstrap_seed']
    assert final['adopted'] is False and final['independent_final_acceptance_required'] is True
    assert final['execution_recovery'] == recovery and final['original_failed_controller_not_reclassified']
    original_started = read(ORIGINAL/'controller-attempt.json')['started_unix']
    recovery_attempt = read(BASE/'controller-attempt.json')
    assert recovery_attempt['recovery_plan_sha256'] == RECOVERY_SHA
    assert recovery_attempt['original_plan_sha256'] == PLAN_SHA
    assert original_started < recovery_attempt['started_unix'] <= (BASE/'STOP').stat().st_mtime
    assert (BASE/'STOP').stat().st_mtime-original_started < 130800
    for record in plan['sources']+plan['assets']+plan['prerequisites']:
        verify_file(record)
    rows, losses, receipts = [], [], []
    for index, spec in enumerate(plan['slots']):
        out = slot_dir(index); receipt = read(out/'acceptance.json')
        assert receipt['accepted'] and receipt['slot'] == index and receipt['no_adoption']
        assert set(receipt['evidence']) == prior.EVIDENCE | ({'lineage.json'} if index >= 14 else set())
        for name, digest in receipt['evidence'].items():
            assert sha(out/name) == digest
        stop = read(out/'STOP')
        assert stop['status'] == ('complete_original_scientific_slot' if index < 14 else 'complete_recovery_scientific_slot')
        check_guard(read(out/'guard.json'), 7200-recovery['failed_guard_elapsed_seconds'] if index == 14 else 7200)
        row = read(out/'review.json'); attempt = read(out/'attempt.json')
        assert row['accepted'] and row['completed_steps'] == 1200 and row['targets_seen'] == 1228800
        assert row['averaging_snapshots'] == 41 and row['validation_targets'] == 34278 and not row['quality_selected']
        for key in ('slot', 'stage', 'bundle', 'variant'):
            assert row[key] == spec[key]
        assert attempt['plan_sha256'] == PLAN_SHA and attempt['spec'] == {k: row[k] for k in spec}
        if index >= 14:
            assert attempt['recovery_plan_sha256'] == RECOVERY_SHA
            assert receipt['original_failed_slot_not_reclassified'] is (index == 14)
            assert stop['original_failed_slot_not_reclassified'] is (index == 14)
            link = read(out/'lineage.json')
            assert link['original_plan_sha256'] == PLAN_SHA and link['recovery_plan_sha256'] == RECOVERY_SHA
            assert link['original_failed_controller'] == recovery['original_stop'] and link['original_scientific_rules_unchanged']
            assert link['new_training_updates'] == (0 if index == 14 else 1200)
        for key in ('parent', 'h200'):
            verify_file(row[key]); assert Path(row[key]['path']).stat().st_nlink == 1
        saved = read(out/'validation.json'); report, loss = saved['report'], saved['per_target_losses']
        assert report['target'] == plan['target'] and report['binding'] == row['h200']
        assert len(loss) == 34278 and all(type(x) is float and math.isfinite(x) and x >= 0 for x in loss)
        assert abs(math.fsum(loss)-report['total_loss']) < 1e-8
        assert row['ce'] == report['ce'] == report['total_loss']/34278
        rows.append(row); losses.append(loss)
        receipts.append(dict(slot=index, path=str(out/'acceptance.json'), sha256=sha(out/'acceptance.json')))
    assert len(rows) == 18 and final['all_slot_acceptances'] == receipts
    choices = {}
    for arm in prior.ARMS:
        candidates = [r for r in rows[:9] if r['variant'] == arm]
        assert sorted(r['learning_rate'] for r in candidates) == [.000625, .00125, .0025]
        choices[arm] = min(candidates, key=lambda r: (r['ce'], r['learning_rate']))['learning_rate']
    selection = dict(learning_rates=choices, screens=[dict(slot=r['slot'], variant=r['variant'],
        learning_rate=r['learning_rate'], ce=r['ce'], review_sha256=sha(slot_dir(r['slot'])/'review.json')) for r in rows[:9]])
    assert selection == read(ORIGINAL/'selection.json') == final['selection']
    for start, length in ((0, 9), (9, 3), (12, 3), (15, 3)):
        assert all(r['sampler_hashes'] == rows[start]['sampler_hashes'] for r in rows[start:start+length])
    for row, spec in zip(rows, plan['slots']):
        assert row['learning_rate'] == (spec['learning_rate'] if row['stage'] == 'screen' else choices[row['variant']])
    pairs = [{rows[i]['variant']: losses[i] for i in range(start, start+3)} for start in (9, 12, 15)]
    independent = {}
    for control, threshold in (('native', -.02), ('nonlexical_codes', 0.)):
        rebuilt = prior.contrast(pairs, control, plan['bootstrap_seed'], threshold)
        stored = final['contrasts'][control]
        for a, b in zip(rebuilt['deltas'], stored['deltas'], strict=True):
            assert abs(a-b) < 1e-12
        assert abs(rebuilt['mean_delta']-stored['mean_delta']) < 1e-12
        for a, b in zip(rebuilt['bounds'], stored['bounds'], strict=True):
            assert a['block_tokens'] == b['block_tokens'] and abs(a['upper']-b['upper']) < 1e-12
        for key in ('threshold', 'gates', 'passes'):
            assert rebuilt[key] == stored[key]
        independent[control] = rebuilt
    assert final['primary_passes'] == independent['native']['passes']
    assert final['lexical_content_supported'] == independent['nonlexical_codes']['passes']
    return dict(accepted=True, original_plan_sha256=PLAN_SHA, recovery_plan_sha256=RECOVERY_SHA,
        final_sha256=sha(BASE/'final.json'), reviewer_sha256=sha(Path(__file__)), arithmetic_reviewer_sha256=PRIOR_SHA,
        recovery_lineage=recovery_lineage, selection=choices, contrasts=independent,
        practical_eligible=independent['native']['passes'], lexical_content_supported=independent['nonlexical_codes']['passes'],
        adopted=False, interpretation=plan['design']['uncertainty'],
        scope='Independent complete-panel selection, target-loss statistics, retained evidence/weight hashes and explicit execution-recovery lineage; reuses accepted per-run numerical checks. Original failed guard is preserved, not passed retroactively.')


if __name__ == '__main__':
    assert len(sys.argv) == 2
    output = Path(sys.argv[1]).resolve()
    assert not output.exists() and not output.is_relative_to(ORIGINAL) and not output.is_relative_to(BASE)
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
        with CpuPrerequisiteGuard(output.with_suffix('.guard.json'), ROOT, timeout_seconds=300,
                                 rss_limit_bytes=1024**3, reserve_bytes=8*1024**3):
            result = run()
        assert read(output.with_suffix('.guard.json'))['status'] == 'finished'
        with output.open('x') as stream:
            json.dump(result, stream, sort_keys=True, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
