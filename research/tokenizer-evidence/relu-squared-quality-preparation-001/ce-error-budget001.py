"""Existing scalar evidence + conditional source arithmetic; no tensor replay."""
import argparse
import fcntl
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def checked(ref):
    assert sha(ref['path']) == ref['sha256'], ref['path']
    return json.loads(Path(ref['path']).read_text())


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw) <= 2*1024**2
    with (R/name).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan-sha', required=True); args = ap.parse_args()
    assert sha(R/'plan.json') == args.plan_sha and not (R/'attempt.json').exists()
    plan = json.loads((R/'plan.json').read_text()); assert sha(__file__) == plan['new_source']['sha256']
    assert __import__('shutil').disk_usage(ROOT).free >= 10739515392
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish('attempt.json', dict(pid=os.getpid(), plan_sha256=args.plan_sha))
        try:
            with CpuPrerequisiteGuard(R/'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
                cpu = checked(plan['CPU32_report'])['rows'][2]
                mps = checked(plan['MPS32_report'])['rows'][2]
                original = checked(plan['original_result'])
                assert cpu['target'] == mps['target'] == 32003 and not cpu['gradient_elementwise_pass']
                assert original['status'] == 'fixed_numerical_predicate_failed'
                for ref in plan['local_headers']:
                    assert sha(ref['path']) == ref['sha256']
                u = Fraction(1, 2**24); n = 50257; width = 4
                terms = [(n+width-1-i)//width for i in range(width)]
                additions = [v-1 for v in terms]
                longest_path = max(additions)+2
                assert terms == [12565,12564,12564,12564] and longest_path == 12566
                def gamma(k):
                    assert k*u < 1
                    exact = (k*u)/(1-k*u)
                    return dict(k=k, exact_rational=str(exact), approximate=float(exact),
                                log_normalizer_addition_only_abs_allowance=-math.log1p(-float(exact)))
                budgets = dict(conditional_NEON4=gamma(longest_path), serial_general=gamma(n-1),
                    hypothetical_balanced_tree_not_executed_or_proposed=gamma((n-1).bit_length()))
                saved = []
                for label, row in [('cpu32',cpu),('mps32',mps)]:
                    delta = row['ce']-row['oracle_ce']
                    # These are scalar interpretations of published CE, NOT recovered
                    # intermediate sums/probabilities or missing gradient coordinates.
                    effective_ratio = math.expm1(delta)
                    uniform_only_bound = abs(math.expm1(-delta))
                    saved.append(dict(backend=label, saved_CE=row['ce'], saved_oracle_CE=row['oracle_ce'],
                        saved_signed_CE_error=delta, relative_error_of_effective_exp_CE_scalar=effective_ratio,
                        uniform_log_normalizer_only_gradient_abs_upper=uniform_only_bound,
                        recorded_gradient_abs_max=row['gradient_abs_max'],
                        recorded_max_fits_hypothetical_bound=row['gradient_abs_max'] <= uniform_only_bound,
                        actual_gradient_coordinates_or_intermediate_sum_reconstructed=False))
                publish('result.json', dict(status='completed_existing_scalar_conditional_source_error_budget',
                    unit_roundoff_exact=str(u),vocabulary=n,conditional_vector_width=width,
                    conditional_lane_terms=terms,conditional_lane_additions=additions,
                    conditional_horizontal_additions_per_path=2,conditional_gamma_budgets=budgets,
                    mapped_exp_error_symbol='eta: relative error INCLUDING shifted-input subtraction and exp; not measured or certified',
                    relative_normalizer_symbol='rho=(1+eta)*(1+gamma_k)-1',
                    ideal_CE_normalizer_error_symbol='abs(log(1+relative_normalizer_error)) <= -log(1-rho),rho<1',
                    additional_error_terms=['actual log implementation rounding','target-minus-maximum subtraction rounding',
                        'final subtraction rounding','backward exp/subtraction rounding','actual binary dispatch/build flags'],
                    saved_scalar_interpretations=saved,
                    arithmetic_scope='Exact rational gamma plus approximate scalar log/expm1; no directed-rounding claim, no full CE/gradient certificate.',
                    conclusions=dict(no_universal_3e_minus5_certificate=True,
                        long_FP32_reduction_consistent_with_but_not_identified_as_literal_cause=True,
                        ordinary_exp_is_not_fexp_u20_in_read_logsoftmax_source=True,
                        NEON_or_SLEEF_actual_binary_selection_not_measured=True,
                        literal_and_trained_model_failures_remain_closed_not_qualified=True),
                    no_torch_numpy_model_optimizer_tensor_oracle_backend_corpus_seed_rng_calls=True,
                    no_predecessor_weight_or_source_audit_repetition=True,
                    metadata_inputs=plan['CPU32_report'],local_header_pins_unchanged=True))
                for ref in plan['local_headers']:assert sha(ref['path']) == ref['sha256']
                assert sha(__file__) == plan['new_source']['sha256']
            publish('STOP', dict(status='complete_existing_scalar_source_analysis',no_replay=True,no_retry=True))
        except BaseException as exc:
            if not (R/'STOP').exists():publish('STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
            raise

if __name__ == '__main__':
    main()
