"""Same-agent MLA artifact audit with separately recomputed quality/cost arithmetic."""
import json
import math
import shutil
import statistics as st
from pathlib import Path

import numpy as np

from src.research.protocol import digest, file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_mla_study import RECIPE, PROTOCOL, RESOURCE, PILOT, config
from src.research.tokenizer_study import runtime_identity

run = Path('research_runs/tokenizer-mla-001')
out = Path('research/tokenizer-evidence/mla-001')
plan = json.loads((run/'plan.json').read_text())
state = json.loads((run/'state.json').read_text())
assert state['status'] == 'completed' and len(state['records']) == 10
assert state['plan_sha256'] == digest(plan)
assert source_fingerprint() == plan['source'] and runtime_identity() == plan['runtime']
for path, key in [(RECIPE, 'recipe_sha256'), (PROTOCOL, 'protocol_sha256'),
                  (RESOURCE, 'resource_review_sha256'), (PILOT/'review.json', 'pilot_review_sha256'),
                  (Path(plan['config']['data']['path']), 'corpus_sha256')]:
    assert file_digest(path) == plan[key]
seeds = (2911, 2912, 2913)
families = ('mha', 'mla')
selection = {}
for family in families:
    rows = [r for r in state['records'] if r['stage'] == 'screen' and r['family'] == family]
    assert len(rows) == 2 and {r['seed'] for r in rows} == {2901}
    assert {r['lr'] for r in rows} == {.00125, .0025}
    selection[family] = min(rows, key=lambda r: (r['h200']['mean_ce'], r['lr']))['lr']
assert selection == state['selection'] == json.loads((run/'selection.json').read_text())
expected = [('screen', 2901, f, lr) for f in families for lr in (.00125, .0025)]
expected += [('confirm', s, f, selection[f]) for s in seeds for f in families]
assert sorted((r['stage'], r['seed'], r['family'], r['lr']) for r in state['records']) == sorted(expected)
for r in state['records']:
    trial = run/r['id']
    rev = json.loads((out/r['id']/'review.json').read_text())
    req = json.loads((trial/'request.json').read_text())
    assert r == json.loads((trial/'result.json').read_text()) and r['status'] == 'completed'
    assert not r['holdout_consumed'] and rev['status'] == 'passed'
    assert rev['result_sha256'] == file_digest(trial/'result.json')
    assert r['request_sha256'] == rev['request_sha256'] == file_digest(trial/'request.json')
    assert req['config'] == config(trial.resolve(), r['family'], r['seed'], r['lr'])
    for arm in ('raw', 'h200'):
        assert file_digest(Path(r[arm]['checkpoint'])) == r[arm]['checkpoint_sha256']
        assert r[arm]['target_count'] == 34278 and r[arm]['target_bytes'] == 111536
arms = {f: {r['seed']: r['h200'] for r in state['records'] if r['stage'] == 'confirm' and r['family'] == f} for f in families}
deltas = {s: arms['mla'][s]['mean_ce']-arms['mha'][s]['mean_ce'] for s in seeds}
counts = arms['mha'][2911]['block_target_counts']
for a in arms.values():
    assert sorted(a) == list(seeds)
    for v in a.values():
        assert v['block_target_counts'] == counts and v['block_tokens'] == 2048
intervals = []
assert [v['text_block_tokens'] for v in state['conclusion']['intervals']] == [2048, 4096, 8192]
for saved in state['conclusion']['intervals']:
    step = saved['text_block_tokens']//2048
    avg = [st.mean(arms['mla'][s]['block_loss_sums'][i]-arms['mha'][s]['block_loss_sums'][i] for s in seeds) for i in range(len(counts))]
    sums = [sum(avg[i:i+step]) for i in range(0, len(counts), step)]
    weights = [sum(counts[i:i+step]) for i in range(0, len(counts), step)]
    draws = np.random.default_rng(20261024).integers(len(weights), size=(2000, len(weights)))
    estimates = [sum(sums[i] for i in d)/sum(weights[i] for i in d) for d in draws]
    bounds = np.quantile(estimates, [.025, .975]).tolist()
    assert np.allclose(bounds, saved['conditional_text_bootstrap_95pct'], atol=1e-12, rtol=0)
    assert abs(st.mean(deltas.values())-saved['mean_delta']) < 1e-12
    intervals.append(dict(text_block_tokens=saved['text_block_tokens'], conditional_text_bootstrap_95pct=bounds))
quality = st.mean(deltas.values()) <= .01 and all(d <= .01 for d in deltas.values()) and all(v['conditional_text_bootstrap_95pct'][1] <= .01 for v in intervals)
assert quality == state['conclusion']['quality_eligible'] and not state['conclusion']['adopted']
completion = json.loads((run/'final-inference/completion.json').read_text())
assert completion == dict(status='completed', pairs=3, source_unchanged=True)
data = []
pair_costs = []
for seed in seeds:
    p = run/'final-inference'/f'w256-s{seed}.json'
    d = json.loads(p.read_text())
    assert d['status'] == 'completed' and d['device'] == 'mps' and d['dtype'] == 'float32'
    assert (d['prompt_tokens'], d['decode_tokens'], d['warmups'], d['repeats']) == (32, 64, 2, 8)
    assert len(d['fixed_train_token_ids']) == 96
    if data:
        assert d['fixed_train_token_ids'] == data[0]['fixed_train_token_ids']
    assert len(d['records']) == 32 and len(d['correctness']) == 4
    assert {(c['batch'], c['family']) for c in d['correctness']} == {(b, f) for b in (1, 8) for f in families}
    assert all(math.isfinite(c['max_absolute_error']) for c in d['correctness'])
    expected_order = [(b, i, f) for b in (1, 8) for i in range(8) for f in (families if i % 2 == 0 else families[::-1])]
    assert [(r['batch'], r['repeat'], r['family']) for r in d['records']] == expected_order
    for f in families:
        assert d['metadata'][f]['sha256'] == arms[f][seed]['checkpoint_sha256']
        assert d['metadata'][f]['width'] == 256
    for b in (1, 8):
        kv = {}
        for f in families:
            cells = [r for r in d['records'] if r['batch'] == b and r['family'] == f]
            values = {r['kv_bytes'] for r in cells}
            assert len(values) == 1 and len(cells) == 8
            kv[f] = values.pop()
            assert kv[f] == b*4*128*(512 if f == 'mha' else 96)*4
            assert all(r['decode_seconds'] > 0 and r['prefill_seconds'] > 0 for r in cells)
        pair_costs.append(dict(seed=seed, batch=b, kv_bytes=kv, reduction=1-kv['mla']/kv['mha']))
    data.append(d)
costs = []
for b in (1, 8):
    summary = {}
    for f in families:
        rows = [r for d in data for r in d['records'] if r['family'] == f and r['batch'] == b]
        summary[f] = dict(kv_bytes=rows[0]['kv_bytes'], parameter_bytes=data[0]['metadata'][f]['parameter_bytes'],
                          median_prefill_seconds=st.median(r['prefill_seconds'] for r in rows),
                          median_decode_seconds=st.median(r['decode_seconds'] for r in rows),
                          median_decode_tokens_per_second=st.median(r['decode_tokens_per_second'] for r in rows),
                          decode_seconds_range=[min(r['decode_seconds'] for r in rows), max(r['decode_seconds'] for r in rows)],
                          observed_mps_current_bytes_range=[min(r['observed_mps_current_bytes'] for r in rows), max(r['observed_mps_current_bytes'] for r in rows)])
    costs.append(dict(batch=b, arms=summary, decode_median_ratio=summary['mla']['median_decode_seconds']/summary['mha']['median_decode_seconds']))
memory = all(v['reduction'] >= .6 for v in pair_costs)
adopt = quality and memory
review = dict(status='completed', adopted=adopt, quality_eligible=quality, memory_eligible=memory,
              mean_h200_ce={f: st.mean(v['mean_ce'] for v in a.values()) for f, a in arms.items()},
              paired_seed_deltas=deltas, mean_delta=st.mean(deltas.values()), intervals=intervals,
              pair_cache_measurements=pair_costs, inference=costs, selected_learning_rates=selection,
              all_ten_artifact_reviews_passed=True, source_runtime_protocol_recipe_corpus_resource_unchanged=True,
              scope='Scoped width256 quarter-latent RoPE MLA LR0.00125, 1200 updates/cooldown and H200 start800, versus equally tuned fresh MHA. Does not replace or compare against adopted GQA; general MHA and width64 remain separate.',
              limitations='Same-agent audit with separately recomputed conditional text intervals. Adaptive validation, three fresh seeds, not independent test or seed-population uncertainty. FP32 MPS context128, training-text teacher forcing, two models resident. KV savings are not total memory or cloud-price savings. Report latency separately; test closed.')
(out/'final-review.json').write_text(json.dumps(review, indent=2)+'\n')
shutil.copyfile(run/'state.json', out/'final-state.json')
shutil.copytree(run/'final-inference', out/'final-inference', dirs_exist_ok=False)
if adopt:
    r = next(r for r in state['records'] if r['stage'] == 'confirm' and r['family'] == 'mla' and r['seed'] == 2911)
    cfg = json.loads((run/r['id']/'request.json').read_text())['config']
    (out/'adopted-width256-mla-recipe.json').write_text(json.dumps(dict(config=cfg, scope=review['scope'], evidence='final-review.json', checkpoints={str(s): arms['mla'][s]['checkpoint'] for s in seeds}), indent=2)+'\n')
print(json.dumps(review, indent=2))
