"""Independent read-only review of the finished shared-table panel.

No model construction, seed consumption, training, validation rescoring or retry.
Bootstrap arithmetic is independently rebuilt with the established scalar reviewer.
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/tokenizer-evidence/output-anchored-tying-001'
PLAN_SHA = 'cc34e2b43349f2281b4a13303080b689bb34fb0ebe4060934a573dfc105921b6'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2): h.update(block)
    return h.hexdigest()


def run():
    import torch
    from research.byte_input_quality_final_review_v1 import contrast
    torch.set_num_threads(4)
    assert sha(BASE/'plan.json') == PLAN_SHA
    plan = read(BASE/'plan.json'); final = read(BASE/'final.json')
    assert final['completed'] and final['trajectories'] == 9 and final['updates'] == 10800
    assert not final['adopted'] and not final['holdout_scored'] and final['byte_incumbent_unchanged']
    for path, digest in plan['sources'].items(): assert sha(ROOT/path) == digest, path
    receipts = []; rows = []; scores = []; cost = []; buffer_hashes = None
    buffers = {f'transformer.h.{i}.freq_{part}' for i in range(4) for part in ('cos','sin')}
    for index, spec in enumerate(plan['specs']):
        out = BASE/f'{index:02d}'
        receipt = read(out/'complete.json'); row = read(out/'result.json')
        assert receipt['accepted'] and receipt['result_sha256'] == sha(out/'result.json')
        assert receipt['guard_sha256'] == sha(out/'guard.json')
        guard = read(out/'guard.json')
        assert guard['status'] == 'finished' and guard['reason'] is None
        assert guard['observation']['peak_rss_bytes'] < 4*1024**3
        assert guard['observation']['mps_driver_bytes'] < 6*1024**3
        assert guard['observation']['free_bytes'] >= 8*1024**3
        assert row['completed'] and row['spec'] == spec and row['steps'] == 1200
        assert row['averaging_snapshots'] == 41 and row['targets_seen'] == 1228800
        assert not row['holdout_scored'] and not row['adopted']
        count = 87 if spec['arm'] == 'tied' else 88
        elements = 15626624 if spec['arm'] == 'tied' else 28492416
        assert row['parameter_count'] == elements
        history = [json.loads(s) for s in (out/'training.jsonl').read_text().splitlines()]
        assert len(history) == 1200
        for step, h in enumerate(history, 1):
            assert h['step'] == step and h['gradient_tensors'] == count and h['threshold'] == 1.
            assert all(math.isfinite(h[k]) for k in ('loss','seconds','pre_norm','post_norm','coefficient'))
            assert h['loss'] >= 0 and h['seconds'] > 0 and 0 <= h['post_norm'] <= 1.00001
            assert abs(h['coefficient']-min(1., 1./(h['pre_norm']+1e-6))) < 2e-5
            assert abs(h['post_norm']-h['pre_norm']*h['coefficient']) <= max(1e-5,h['post_norm']*2e-5)
        assert row['curves'] == [read(out/f'curve-{step}.json') for step in (300,600,900,1200)]
        saved = []
        for key, kind in (('raw','final_raw_weights_not_resume'),('h200','h200_inference_weights')):
            record = row[key]; path = Path(record['path'])
            assert path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
            assert path.stat().st_size == record['size'] and sha(path) == record['sha256']
            payload = torch.load(path, map_location='cpu', weights_only=True)
            assert payload['kind'] == kind and payload['spec'] == spec and payload['plan_sha256'] == PLAN_SHA
            tensors = payload['weights']
            assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in tensors.values())
            assert len({t.untyped_storage().data_ptr() for t in tensors.values()}) == len(tensors)
            assert tensors['transformer.wte.weight'].shape == (50257,256)
            assert ('lm_head.weight' in tensors) == (spec['arm'] != 'tied')
            # State includes eight persistent RoPE buffers, not trainable parameters.
            assert buffers <= tensors.keys() and len(tensors)-len(buffers) == count
            assert sum(t.numel() for n,t in tensors.items() if n not in buffers) == elements
            assert all(tensors[n].shape == (128,16) for n in buffers)
            observed_buffers = {n:hashlib.sha256(tensors[n].numpy().tobytes()).hexdigest() for n in sorted(buffers)}
            if buffer_hashes is None: buffer_hashes = observed_buffers
            assert observed_buffers == buffer_hashes
            saved.append(tensors)
        assert saved[0].keys() == saved[1].keys()
        assert all(saved[0][n].shape == t.shape for n,t in saved[1].items())
        assert any(not torch.equal(saved[0][n],t) for n,t in saved[1].items())
        del saved, tensors, payload
        score = read(out/'validation.json'); losses = score['per_target_losses']
        assert len(losses) == score['target_count'] == 34278 and not score['holdout_scored']
        assert all(type(x) is float and math.isfinite(x) and x >= 0 for x in losses)
        assert math.fsum(losses)/34278 == score['ce'] == row['ce']
        receipts.append(dict(slot=index, acceptance_sha256=sha(out/'complete.json'),
                             validation_sha256=sha(out/'validation.json'), training_sha256=sha(out/'training.jsonl')))
        rows.append(row); scores.append(losses)
        cost.append(dict(slot=index, arm=spec['arm'], training_seconds=math.fsum(h['seconds'] for h in history),
                         clipped_fraction=sum(h['coefficient'] < 1 for h in history)/1200,
                         peak_rss_bytes=guard['observation']['peak_rss_bytes']))
    pairs = []
    for start in (0,3,6):
        assert len({r['train_sampler_sha256'] for r in rows[start:start+3]}) == 1
        assert all(r['spec']['roles'] == rows[start]['spec']['roles'] for r in rows[start:start+3])
        pairs.append({rows[i]['spec']['arm']:scores[i] for i in range(start,start+3)})
    rebuilt = {}
    for candidate, control in (('tied','native'),('tied','matched'),('matched','native')):
        # The old helper's names are labels only; it is a pure saved-loss calculation.
        mapped = [{'raw_bytes':p[candidate], 'control':p[control]} for p in pairs]
        result = contrast(mapped, 'control', plan['bootstrap_seed'], -.02)
        stored = final['comparisons'][candidate+'_vs_'+control]
        assert all(abs(a-b) < 1e-12 for a,b in zip(result['deltas'],stored['deltas'],strict=True))
        assert abs(result['mean_delta']-stored['mean']) < 1e-12
        assert all(a['block_tokens'] == b['block_tokens'] and abs(a['upper']-b['upper']) < 1e-12
                   for a,b in zip(result['bounds'],stored['bounds'],strict=True))
        assert result['passes'] == stored['passes']
        rebuilt[candidate+'_vs_'+control] = result
    assert final['quality_success'] == all(rebuilt['tied_vs_'+a]['passes'] for a in ('native','matched'))
    return dict(accepted=True, plan_sha256=PLAN_SHA, final_sha256=sha(BASE/'final.json'),
                reviewer_sha256=sha(Path(__file__)), receipts=receipts, comparisons=rebuilt,
                costs=cost, quality_success=final['quality_success'], adopted=False,
                scope='Saved weights, completion/guard/history/coverage/pairing checks and independent scalar statistics; no new model execution. Final weights do not provide optimizer resume or independent reconstruction of all EMA snapshots.')


if __name__ == '__main__':
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with CpuPrerequisiteGuard(BASE/'final-review-guard.json', ROOT, timeout_seconds=300):
            result = run()
        with (BASE/'final-review.json').open('x') as stream:
            json.dump(result,stream,sort_keys=True,allow_nan=False);stream.write('\n')
