"""New literal metadata-only proof of future paired-report publication."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.projection_validation_pair_v1 import publish_then_require, POLICY

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_suffix('')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) + len(raw) < 2*1024**2
    with (OUT/name).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def main(plan_sha):
    assert digest(OUT/'plan.json') == plan_sha
    plan = json.loads((OUT/'plan.json').read_text())
    for path, sha in plan['sources'].items():
        assert digest(ROOT/path) == sha
    assert not (OUT/'attempt.json').exists() and not (OUT/'STOP').exists()
    put('attempt.json', dict(pid=os.getpid(), plan_sha256=plan_sha))
    with CpuPrerequisiteGuard(OUT/'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
        target = dict(policy=POLICY, targets=34278, target_bytes=111536, context=128,
                      first_target_offset=1, tail_retained=True, holdout_scored=False,
                      literal_fixture='new_metadata_only_not_physical_codec_validation')
        provenance = dict(scope='new_literal_report_fixture', no_model_or_corpus=True)
        def report(ce=0.):
            return dict(ce=ce, target=copy.deepcopy(target), literal_values=[.125, .375])
        rows = []
        def case(name, cpu, mps, passes, saved, prov=None, suffix='.json'):
            path = OUT/(name+suffix)
            before = json.dumps([cpu, mps, provenance if prov is None else prov], sort_keys=True)
            try:
                result = publish_then_require(path, cpu, mps, provenance if prov is None else prov)
            except (ValueError, FileExistsError):
                assert not passes
            else:
                assert passes and result['accepted']
            assert path.exists() is saved
            if saved:
                value = json.loads(path.read_text())
                assert value['cpu'] == cpu and value['mps'] == mps
                assert value['accepted'] is passes and value['absolute_ce_tolerance'] == 3e-5
            assert json.dumps([cpu, mps, provenance if prov is None else prov], sort_keys=True) == before
            rows.append(dict(case=name, passed=True, pair_saved=saved, gate_accepted=passes))
        case('equal', report(), report(), True, True)
        case('within', report(), report(2e-5), True, True)
        case('boundary', report(), report(3e-5), True, True)
        case('above', report(), report(.000030001), False, True)
        mps = report(); mps['target']['literal_fixture'] = 'distinct_new_literal_target'
        case('target_mismatch', report(), mps, False, True)
        cpu = report(); cpu['target']['policy'] = 'foreign'
        case('equal_foreign_policy', cpu, copy.deepcopy(cpu), False, True)
        case('nonfinite', report(float('nan')), report(), False, False)
        case('integer_ce', report(0), report(), False, False)
        cpu = report(); cpu['oversized_literal'] = 'x'*(512*1024)
        case('metadata_overflow', cpu, report(), False, False)
        case('wrong_suffix', report(), report(), False, False, suffix='.txt')
        case('empty_provenance', report(), report(), False, False, prov={})
        original = (OUT/'equal.json').read_bytes()
        try:
            publish_then_require(OUT/'equal.json', report(), report(2e-5), provenance)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing paired report overwritten')
        assert (OUT/'equal.json').read_bytes() == original
        rows.append(dict(case='overwrite', passed=True, existing_bytes_unchanged=True))
        assert len(rows) == 12
        put('result.json', dict(status='accepted_new_literal_pair_reporting_only', cases=rows,
            zero_torch_models_tensors_corpus_checkpoints_random_roles=True,
            original_campaign_replayed=False, runtime_integration_or_numerical_root_cause=False))
    put('STOP', dict(status='complete_literal_pair_reporting_fixture', no_retry_or_replay=True))
    print(json.dumps(dict(status='accepted_new_literal_pair_reporting_only', cases=12)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan-sha', required=True)
    main(p.parse_args().plan_sha)
