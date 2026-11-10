"""Synthetic gate and weighting checks; no model scoring or training."""
import copy
import json
import time
import numpy as np
from research.general_rsi_native_horizon_analysis_v1 import PANEL, BINDING, ROOT, read, sha, decide, analyze
from research.general_rsi_equal_time_analysis_v1 import bootstrap
from research.general_rsi_native_horizon_worker_v1 import put


def main():
    start = time.monotonic()
    assert sha(PANEL/'plan.json') == BINDING
    contract = read(PANEL/'plan.json')['acceptance']
    counts = np.array([1, 1, 2])
    sources = ['fineweb_edu', 'fineweb_edu', 'wikipedia']
    checks = {}
    def constant(ces): return np.asarray(ces)[:, None]*counts[None, :]
    passed = decide(constant([-.04]*3), counts, sources, contract)
    assert passed['longer_recipe_accepted'] and passed['retained_recipe'] == 'long'
    checks['clear_gain_passes'] = passed
    boundary = decide(constant([-.02]*3), counts, sources, contract)
    assert boundary['gates']['mean_threshold'] and boundary['longer_recipe_accepted']
    checks['inclusive_mean_threshold'] = boundary
    shallow = decide(constant([-.01]*3), counts, sources, contract)
    assert not shallow['gates']['mean_threshold'] and not shallow['longer_recipe_accepted']
    assert all(v for k,v in shallow['gates'].items() if k != 'mean_threshold')
    checks['small_gain_rejected'] = shallow
    mixed = decide(constant([-.1, -.1, 0]), counts, sources, contract)
    assert not mixed['gates']['all_three_deltas_negative'] and mixed['retained_recipe'] == 'short'
    checks['zero_pair_rejected'] = mixed
    zero = decide(np.zeros((3,3)), counts, sources, contract)
    assert zero['bootstrap_upper_99'] == 0 and not zero['gates']['bootstrap_upper_negative']
    assert zero['retained_recipe'] == 'short'
    checks['zero_bootstrap_bound_rejected'] = zero
    regression = decide(constant([.04]*3), counts, sources, contract)
    assert regression['retained_recipe'] == 'short' and not regression['longer_recipe_accepted']
    checks['positive_sign_rejected'] = regression
    source_bad = decide(np.tile([-10., -10., .06], (3,1)), [100,100,1], sources, contract)
    assert not source_bad['gates']['source_guard'] and not source_bad['longer_recipe_accepted']
    assert abs(source_bad['source_deltas']['wikipedia']-.06) < 1e-12
    checks['source_regression_rejected'] = source_bad
    noisy = decide(np.tile([-.2,-.2,.1], (3,1)), [1,1,1], sources, contract)
    assert noisy['mean_delta_long_minus_short'] < -.02 and noisy['bootstrap_upper_99'] > 0
    assert not noisy['longer_recipe_accepted']
    checks['uncertain_document_bootstrap_rejected'] = noisy
    # A long document changes the correct global estimand. Do not average document CE.
    weighted = np.tile([-1., -1., -6.], (3,1))
    mean, draws = bootstrap(weighted, [1,1,100], 773, 32)
    assert abs(mean-(-8/102)) < 1e-12
    assert abs(mean-np.mean([-1.,-1.,-.06])) > .5
    rng = np.random.Generator(np.random.PCG64(773))
    oracle = []
    for _ in range(32):
        numerator = denominator = 0
        for i in rng.integers(0, 3, size=3):
            numerator += [-1., -1., -6.][i]
            denominator += [1,1,100][i]
        oracle.append(numerator/denominator)
    assert np.array_equal(draws, np.array(oracle))
    checks['unequal_count_bootstrap_oracle'] = dict(passed=True, mean=mean, resamples=32)
    rejected = 0
    for invalid, invalid_counts in ((np.full((3,3), np.nan), counts), (np.zeros((3,3)), [1,0,1])):
        try: decide(invalid, invalid_counts, sources, contract)
        except AssertionError: rejected += 1
    assert rejected == 2
    checks['nonfinite_and_zero_count_rejected'] = True
    assert not (PANEL/'slot-05/review.json').exists()
    try: analyze()
    except AssertionError as error:
        assert str(error) == 'Six accepted endpoints required'
        checks['incomplete_panel_rejected_before_scores'] = True
    else: raise AssertionError('Incomplete panel accepted')
    out = PANEL/'analysis-check.json'
    put(out, dict(qualified=True, plan_sha256=BINDING, seconds=time.monotonic()-start,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [
            ROOT/'research/general_rsi_native_horizon_analysis_v1.py',
            ROOT/'research/general_rsi_native_horizon_analysis_check_v1.py',
            ROOT/'research/general_rsi_equal_time_analysis_v1.py']},
        checks=checks, scientific_endpoint_scores_read=False, test_opened=False))
    print(json.dumps(dict(qualified=True, checks=len(checks), seconds=time.monotonic()-start)))


if __name__ == '__main__': main()
