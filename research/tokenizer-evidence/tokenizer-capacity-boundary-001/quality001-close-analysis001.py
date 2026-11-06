"""Read closed evidence once; no model loading, training, scoring or bootstrap."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent
R = BASE / 'research_runs/tokenizer-gpt2-prefix-quality-v1-001'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ref(p):
    p = Path(p).resolve()
    return dict(path=str(p), sha256=sha(p), size=p.stat().st_size)


def verify_refs(d):
    if isinstance(d, dict):
        if 'path' in d and 'sha256' in d:
            p = Path(d['path'])
            if not p.is_absolute():
                p = BASE / p
            assert p.is_file() and sha(p) == d['sha256'], str(p)
            if 'size' in d:
                assert p.stat().st_size == d['size'], str(p)
        for v in d.values():
            verify_refs(v)
    elif isinstance(d, list):
        for v in d:
            verify_refs(v)


def write_new(p, d):
    with Path(p).open('x') as f:
        json.dump(d, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def main():
    # Terminal evidence must exist before any output directory is created.
    out = E / 'quality001-closure001.json'
    assert not out.exists()
    stop = read(R / 'STOP')
    assert stop['status'] == 'complete_finite_scientific_panel'
    assert stop['never_rerun_training_inference_review'] is True
    verify_refs(stop)
    plan = read(E / 'quality001-plan.json')
    assert len(plan['sources']) == 1004
    for p, digest in plan['sources'].items():
        assert sha(BASE / p) == digest, p
    final = read(E / 'quality001-final-review.json')
    verify_refs(final)
    assert final['adopted'] is False
    assert not any(final['statistics']['gates'].values())
    independent = read(E / 'quality001-independent-final-review.json')
    assert independent['result']['status'] == 'independent_complete_panel_statistics_agreed'
    for name in ['quality001-fresh-inference-guard.json', 'quality001-independent-review-guard.json']:
        g = read(E / name)
        assert g['status'] == 'finished' and g['reason'] is None

    rows, weights, acceptances, inputs = [], [], [], []
    for i in range(12):
        slot = R / f'evidence/quality/{i:02}'
        anchor = read(E / f'quality001-accepted-{i:02}.json')
        verify_refs(anchor)
        a = read(slot / 'accepted.json')
        verify_refs(a)
        w = read(slot / 'worker.json')
        verify_refs(w)
        req = read(slot / 'request.json')
        verify_refs(req)
        cfg = read(slot / 'config.json')
        assert cfg == req['spec']['config'] == w['config']
        ac = a['acceptance']
        assert ac['status'] == 'accepted'
        assert ac['step'] == w['step'] == 1200
        assert ac['targets'] == w['targets'] == 1228800
        assert ac['ema_snapshots'] == w['ema_snapshots'] == 41
        assert ac['complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked']
        assert ac['holdout_scored'] is w['holdout_scored'] is False
        for key in ['worker_guard', 'parent_guard']:
            g = read(a[key]['path'])
            assert g['status'] == 'finished' and g['reason'] is None
        proc = read(a['process']['path'])
        assert proc['status'] == 'completed' and proc['returncode'] == 0
        for kind in ['parent', 'h200']:
            weights.append(dict(slot=i, arm=a['arm'], kind=kind, **ref(ac['binding'][kind]['path'])))
            for device in ['cpu', 'mps']:
                v = ac['full_validation'][kind if kind == 'h200' else 'raw'][device]
                assert v['target_bytes'] == 111475 and not v['holdout_scored']
                assert v['target_sha256'] == plan['design']['common_target_sha256']
        exposure = ac['raw_target_byte_exposure']
        assert len(exposure['per_update_target_bytes']) == 1200
        assert sum(exposure['per_update_target_bytes']) == exposure['sampled_target_bytes']
        curves = [json.loads(s) for s in (slot / 'metrics.jsonl').read_text().splitlines()]
        assert [v['step'] for v in curves] == [1, 301, 601, 901, 1200]
        h = ac['full_validation']['h200']['cpu']
        rows.append(dict(slot=i, arm=a['arm'], seed=cfg['training']['seed'],
                         lr=cfg['training']['learning_rate'], h200_bpb=h['bits_per_byte'],
                         raw_bpb=ac['full_validation']['raw']['cpu']['bits_per_byte'],
                         train_seconds=w['training_seconds'], process_seconds=proc['elapsed_seconds'],
                         parent_seconds=read(a['parent_guard']['path'])['elapsed_seconds'],
                         clipped_updates=ac['clipped_updates'],
                         sampled_target_bytes=exposure['sampled_target_bytes'],
                         mean_bytes_per_sampled_token=exposure['mean_bytes_per_sampled_token'],
                         online_sampled_curves=curves))
        acceptances.append(ref(slot / 'accepted.json'))
        inputs.extend([ref(slot / n) for n in ['accepted.json', 'metrics.jsonl', 'worker.json']])

    assert len(weights) == 24 and len({w['sha256'] for w in weights}) == 24
    pairs = []
    for seed in plan['seeds']['fresh']:
        a = next(v for v in rows[6:] if v['seed'] == seed and v['arm'] == 'native')
        b = next(v for v in rows[6:] if v['seed'] == seed and v['arm'] == 'prefix8192')
        pairs.append(dict(seed=seed, native_bpb=a['h200_bpb'], prefix_bpb=b['h200_bpb'],
                          delta=b['h200_bpb'] - a['h200_bpb'],
                          sampled_byte_ratio=b['sampled_target_bytes'] / a['sampled_target_bytes'],
                          training_time_ratio=b['train_seconds'] / a['train_seconds'],
                          process_time_ratio=b['process_seconds'] / a['process_seconds']))
    assert all(abs(a['delta'] - b) < 1e-12 for a, b in zip(pairs, final['statistics']['candidate_minus_native_bpb']))

    block_rows = {}
    for width in ['4096', '8192', '16384']:
        seed_values = []
        for seed in plan['seeds']['fresh']:
            pair = {}
            for row in rows[6:]:
                if row['seed'] == seed:
                    ac = read(R / f"evidence/quality/{row['slot']:02}/accepted.json")['acceptance']
                    pair[row['arm']] = ac['full_validation']['h200']['cpu']['raw_byte_blocks'][width]
            a, b = pair['native'], pair['prefix8192']
            assert a['byte_counts'] == b['byte_counts']
            seed_values.append([(v-u)/(n*math.log(2)) for u,v,n in zip(a['loss_sums'], b['loss_sums'], a['byte_counts'])])
        means = [statistics.mean(t) for t in zip(*seed_values)]
        block_rows[width] = dict(mean_delta_per_block=means,
                                 positive_blocks=sum(v > 0 for v in means), total_blocks=len(means),
                                 all_three_seeds_positive_blocks=sum(all(v > 0 for v in t) for t in zip(*seed_values)),
                                 scope='Descriptive existing block losses only;no new confidence calculation or selection.')

    serving = read(E / 'quality001-fresh-inference.json')
    assert serving['all_six_fresh'] and not serving['holdout_scored'] and len(serving['rows']) == 6
    inference = {}
    for batch in [1, 8]:
        ratios = {'full128': [], 'decode32': []}
        for seed in plan['seeds']['fresh']:
            pair = {}
            for v in serving['rows']:
                if v['seed'] == seed:
                    pair[v['arm']] = next(m for m in v['measurements'] if m['batch'] == batch)
            for name, key in [('full128', 'median_full128_seconds'), ('decode32', 'median_decode32_seconds')]:
                ratios[name].append(pair['prefix8192'][key] / pair['native'][key])
        inference[str(batch)] = ratios

    # Portable metadata only; all scientific weights stay in original paths.
    mirror = E / 'quality001-complete-metadata-mirror001'
    mirror.mkdir()
    files = [R / 'STOP']
    for i in range(6, 12):
        files.extend(p for p in (R / f'evidence/quality/{i:02}').rglob('*') if p.is_file() and 'checkpoints' not in p.parts)
        files.extend(R / 'control' / f'{prefix}-quality-{i:02}{suffix}' for prefix,suffix in [('worker','-guard.json'),('parent','-guard.json'),('process','.json')])
    copied = []
    for p in sorted(files):
        assert p.is_file() and not p.is_symlink() and p.suffix not in ['.pt', '.tmp']
        target = mirror / p.relative_to(R)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(p.read_bytes())
        assert sha(p) == sha(target)
        copied.append(dict(source=ref(p), copy=ref(target)))
    write_new(E / 'quality001-complete-metadata-mirror001.json', dict(files=copied, count=len(copied), weight_copies=0))
    analysis = dict(status='closed_existing_evidence_analysis_complete', inputs=inputs, rows=rows,
                    fresh_pairs=pairs, descriptive_byte_blocks=block_rows, serving_ratios=inference,
                    numerical_training=False, model_loading=False, new_scoring=False, new_bootstrap=False,
                    holdout_opened=False, new_scientific_seed=False,
                    limitations=['Five sparse online sampled-validation points are not full checkpoint curves or a horizon optimum.',
                                 'Cross-codec CE/token cannot be compared;only fixed-byte H200BPB is the primary quality result.',
                                 'Different segmentation changes vocabulary capacity,normalization,raw context and exposure together;no isolated cause.',
                                 'Timing ratios are descriptive local measurements;not equal raw-text work or a cost-only adoption gate.'])
    write_new(E / 'quality001-existing-analysis001.json', analysis)
    closure = dict(status='CLOSED_NOT_ADOPTED', observed_at=datetime.now(timezone.utc).isoformat(),
                   stop=ref(R / 'STOP'), final_review=ref(E / 'quality001-final-review.json'),
                   independent_review=ref(E / 'quality001-independent-final-review.json'),
                   source_count=len(plan['sources']), source_bindings_unchanged=True,
                   accepted_endpoints=12, acceptances=acceptances, all_payloads=weights,
                   retained_payload_count=len(weights), retained_weight_bytes=sum(w['size'] for w in weights),
                   metadata_mirror=ref(E / 'quality001-complete-metadata-mirror001.json'),
                   existing_analysis=ref(E / 'quality001-existing-analysis001.json'),
                   actual_free_bytes=os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize,
                   adopted=False, keep_incumbent=True, no_cleanup_or_retirement=True,
                   next_action='Review this existing-data analysis and primary evidence to qualify the next mechanism;no automatic vocabulary/horizon rescue or new seed assignment.')
    write_new(out, closure)
    print(json.dumps(dict(closure=str(out), payloads=len(weights), weight_bytes=closure['retained_weight_bytes'],
                          metadata_files=len(copied), fresh_pairs=pairs, positive_blocks={k:v['positive_blocks'] for k,v in block_rows.items()},
                          serving_ratios=inference), indent=2))


if __name__ == '__main__':
    main()
