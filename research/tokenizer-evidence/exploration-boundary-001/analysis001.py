"""Read accepted metadata only; no training, checkpoint loading or holdout access."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'research/tokenizer-evidence/final-ln-scale-preparation-001'
OUT = Path(__file__).with_suffix('.json')


def reference(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


plan_path = BASE / 'quality002-plan.json'
plan = json.loads(plan_path.read_text())
rows = []
for slot in range(6):
    anchor_path = BASE / f'quality002-accepted-{slot:02d}.json'
    anchor = json.loads(anchor_path.read_text())
    binding = next(x for x in anchor['files'] if x['path'].endswith('/metrics.jsonl'))
    path = Path(binding['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == binding['sha256']
    metrics = [json.loads(line) for line in path.read_text().splitlines()]
    assert [x['step'] for x in metrics] == [0, 300, 600, 900, 1199]
    spec = plan['fresh'][slot]
    rows.append(dict(slot=slot, arm=spec['arm'], seed=spec['seed'],
                     anchor=reference(anchor_path), metrics=reference(path),
                     observations=[dict(step=x['step'], train_ce=x['train_ce_loss'],
                                        sampled_validation_ce=x['val_ce_loss']) for x in metrics],
                     late_validation_change=metrics[-1]['val_ce_loss']-metrics[-2]['val_ce_loss']))

result = dict(
    status='readonly_existing_data_diagnostic', source=reference(Path(__file__)),
    plan=reference(plan_path), observations=rows,
    train_tokens=plan['data']['train_tokens'], validation_targets=plan['data']['validation_targets'],
    sampled_targets_per_trajectory=1228800,
    sampled_targets_to_train_token_count=1228800 / plan['data']['train_tokens'],
    untied_vocabulary_table_parameters=2 * 50257 * 256,
    incumbent_total_parameters=28492416,
    vocabulary_table_parameter_fraction=(2 * 50257 * 256) / 28492416,
    algebraic_parameter_scope='Two untied 50257-by-256 tables, derived from the bound model geometry; not a measured FLOP fraction.',
    all_six_sampled_validation_curves_improve_at_every_saved_step=all(
        all(y['sampled_validation_ce'] < x['sampled_validation_ce']
            for x, y in zip(row['observations'], row['observations'][1:])) for row in rows),
    learned_mean_late_validation_change=mean(x['late_validation_change'] for x in rows if x['arm']=='learned'),
    unit_mean_late_validation_change=mean(x['late_validation_change'] for x in rows if x['arm']=='unit'),
    limitations=[
        'Five sampled online observations per trajectory; no intermediate checkpoint selection or new quality scoring.',
        'Training-batch CE and sampled validation CE use different observations; their gap is not a matched generalization estimate.',
        'Sampled target exposure divided by train-token count is not literal epochs for the random-window sampler.',
        'Current trajectories share one depth and width; these curves cannot identify an optimal shape or a causal bottleneck.',
        'Historical character results are separate evidence, never pooled into subword CE or fresh seed confirmation.',
    ],
    training_launched=False, checkpoint_loads=0, holdout_scored=False,
)
with OUT.open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps({key: value for key, value in result.items() if key not in ('observations', 'limitations')}))
