"""Same-agent artifact audit and independent arithmetic for the closed horizon round."""
import json
import shutil
import statistics
from pathlib import Path
import numpy as np
from src.research.protocol import digest, file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_study import runtime_identity
from src.research.tokenizer_horizon_study import RECIPE, PROTOCOL, config

run = Path('research_runs/tokenizer-horizon-001')
out = Path('research/tokenizer-evidence/horizon-001')
plan = json.loads((run/'plan.json').read_text())
state = json.loads((run/'state.json').read_text())
assert state['status'] == 'completed' and len(state['records']) == 9
assert state['plan_sha256'] == digest(plan)
assert source_fingerprint() == plan['source'] and runtime_identity() == plan['runtime']
for path, key in [(RECIPE, 'recipe_sha256'), (PROTOCOL, 'protocol_sha256'),
                  (Path(plan['config']['data']['path']), 'corpus_sha256'),
                  (Path('research/tokenizer-evidence/horizon-pilot-001/resource-review.json'), 'resource_review_sha256')]:
    assert file_digest(path) == plan[key]
expected = [('screen', 2801, n) for n in (1200, 1800, 3000)]
expected += [('confirm', s, n) for s in (2811, 2812, 2813) for n in (1200, 3000)]
assert sorted((r['stage'], r['seed'], r['steps']) for r in state['records']) == sorted(expected)
for r in state['records']:
    trial = run/r['id']
    rev = json.loads((out/r['id']/'review.json').read_text())
    req = json.loads((trial/'request.json').read_text())
    assert r == json.loads((trial/'result.json').read_text()) and r['status'] == 'completed'
    assert not r['holdout_consumed'] and rev['status'] == 'passed'
    assert rev['result_sha256'] == file_digest(trial/'result.json')
    assert r['request_sha256'] == rev['request_sha256'] == file_digest(trial/'request.json')
    assert req['config'] == config(trial.resolve(), r['seed'], r['steps'])
    for arm in ('raw', 'h200'):
        assert file_digest(Path(r[arm]['checkpoint'])) == r[arm]['checkpoint_sha256']
        assert r[arm]['target_count'] == 34278 and r[arm]['target_bytes'] == 111536
screen = {r['steps']: r['h200']['mean_ce'] for r in state['records'] if r['stage'] == 'screen'}
eligible = [n for n in (1200, 1800) if screen[3000]-screen[n] > .02]
selected = min(eligible, key=lambda n: (screen[n], n)) if eligible else 3000
assert selected == state['selection'] == 1200
assert json.loads((run/'selection.json').read_text())['selected_steps'] == selected
arms = {n: {r['seed']: r['h200'] for r in state['records'] if r['stage'] == 'confirm' and r['steps'] == n} for n in (1200, 3000)}
for a in arms.values():
    assert sorted(a) == [2811, 2812, 2813]
deltas = {s: arms[1200][s]['mean_ce']-arms[3000][s]['mean_ce'] for s in (2811, 2812, 2813)}
intervals = []
for saved in state['conclusion']['intervals']:
    counts = arms[3000][2811]['block_target_counts']
    step = saved['text_block_tokens']//2048
    for a in arms.values():
        for v in a.values():
            assert v['block_target_counts'] == counts and v['block_tokens'] == 2048
    avg = [statistics.mean(arms[1200][s]['block_loss_sums'][i]-arms[3000][s]['block_loss_sums'][i] for s in (2811, 2812, 2813)) for i in range(len(counts))]
    sums = [sum(avg[i:i+step]) for i in range(0, len(counts), step)]
    weights = [sum(counts[i:i+step]) for i in range(0, len(counts), step)]
    draws = np.random.default_rng(20261024).integers(len(weights), size=(2000, len(weights)))
    estimates = [sum(sums[i] for i in d)/sum(weights[i] for i in d) for d in draws]
    bounds = np.quantile(estimates, [.025, .975]).tolist()
    assert np.allclose(bounds, saved['conditional_text_bootstrap_95pct'], atol=1e-12, rtol=0)
    assert abs(statistics.mean(deltas.values())-saved['mean_delta']) < 1e-12
    intervals.append(dict(text_block_tokens=saved['text_block_tokens'], conditional_text_bootstrap_95pct=bounds))
assert [i['text_block_tokens'] for i in intervals] == [2048, 4096, 8192]
adopt = statistics.mean(deltas.values()) < -.02 and all(d < 0 for d in deltas.values()) and all(i['conditional_text_bootstrap_95pct'][1] < 0 for i in intervals)
assert adopt and state['conclusion']['quality_eligible']
times = {n: [r['training']['training_time_sec'] for r in state['records'] if r['stage'] == 'confirm' and r['steps'] == n] for n in (1200, 3000)}
mean_times = {n: statistics.mean(v) for n, v in times.items()}
review = dict(status='completed', adopted=True, selected_steps=1200,
    mean_h200_ce={n: statistics.mean(v['mean_ce'] for v in a.values()) for n, a in arms.items()},
    paired_seed_deltas=deltas, mean_delta=statistics.mean(deltas.values()), intervals=intervals,
    mean_training_seconds=mean_times, training_time_reduction=1-mean_times[1200]/mean_times[3000],
    sampled_targets={1200: 1228800, 3000: 3072000}, target_reduction=.6,
    all_nine_artifact_reviews_passed=True, source_runtime_protocol_recipe_corpus_resource_unchanged=True,
    total_training_seconds=sum(r['training']['training_time_sec'] for r in state['records']),
    scope='Adopt the combined 1200-update/cooldown-start800/H200-tail-start800 recipe only for width256 GQA LR0.00125. General MHA and width64 recipes unchanged. Same architecture and inference cache; no new inference-cost claim.',
    limitations='Same-agent audit with separately recomputed interval arithmetic. Adaptive validation, three fresh seeds, conditional text intervals are not seed-population uncertainty or independent test evidence. Does not isolate duration, cooldown or EMA window. Test closed.')
(out/'final-review.json').write_text(json.dumps(review, indent=2)+'\n')
shutil.copyfile(run/'state.json', out/'final-state.json')
r = next(r for r in state['records'] if r['stage']=='confirm' and r['seed']==2811 and r['steps']==1200)
cfg = json.loads((run/r['id']/'request.json').read_text())['config']
(out/'adopted-width256-gqa-recipe.json').write_text(json.dumps(dict(config=cfg, scope=review['scope'], evidence='final-review.json', checkpoints={str(s): arms[1200][s]['checkpoint'] for s in arms[1200]}), indent=2)+'\n')
print(json.dumps(review, indent=2))
