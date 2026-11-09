"""Read-only Adam denominator analysis of the preselected incumbent RAW."""
import collections
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import traceback

from research.output_anchored_tying_frequency_v1 import bin_for, digest, read, sha

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT/'research/tokenizer-evidence/public-inference-correction-001/adoption/recipe.json'
PRIOR = ROOT/'research/tokenizer-evidence/byte-sampler-gap-001/result.json'
OUT = ROOT/'research/tokenizer-evidence/byte-output-moment-001'
CAP = 1024**2


def run():
    import numpy as np
    import torch
    import tiktoken
    torch.set_num_threads(1)
    rng = torch.random.get_rng_state().clone()
    assert sha(RECIPE) == '1a5029445e36f2701fa3e4b9d5bcf6af29ce21fbb07c54b0366bb027aa1d3823'
    receipt = read(RECIPE)['representative']['parent']; path = Path(receipt['path'])
    assert receipt['size'] == path.stat().st_size == 252854065
    assert receipt['sha256'] == sha(path) == 'c086a81207231fac943c2c68b8b7a9dcf88e485d9add058d84d58e7d6d55b853'
    raw = torch.load(path, map_location='cpu', weights_only=True)
    assert raw['kind'] == receipt['kind'] == 'byte_input_training_context_v2'
    num = raw['numerical']; step = num['completed_steps']
    assert num['kind'] == 'byte_input_numerical_checkpoint_v2' and step == 1200
    assert num['identity'] == raw['identity']['model']
    assert [child['kind'] for child in num['children']] == ['Muon', 'AdamW']
    child = num['children'][1]; name = 'lm_head.weight'
    groups = [g for g in child['groups'] if g['name'] == name]; assert len(groups) == 1
    assert all(g['name'] != name for g in num['children'][0]['groups'])
    options = groups[0]['options']; beta = options['betas'][1]; eps = options['eps']
    assert tuple(options['betas']) == (.9, .999) and eps == 1e-6 and not options['amsgrad']
    assert options['weight_decay'] == 0 and not options['maximize']
    state = child['state'][name]
    assert set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and state['step'].item() == step
    for t in (num['parameters'][name], state['exp_avg'], state['exp_avg_sq']):
        assert tuple(t.shape) == (50257, 256) and t.dtype == torch.float32
        assert t.device.type == 'cpu' and bool(torch.isfinite(t).all())
    v = state['exp_avg_sq']; assert bool((v >= 0).all())
    correction = 1-beta**step
    toy_v = np.array([0., 1., 4.])*eps**2*correction
    toy_s = np.sqrt(toy_v/correction)
    assert np.allclose(toy_s/eps, [0, 1, 2], rtol=0, atol=1e-12)
    assert np.allclose(eps/(toy_s+eps), [1, .5, 1/3], rtol=0, atol=1e-12)
    data = raw['identity']['data']; prior = read(PRIOR)
    for key, value in prior['data_identity'].items(): assert data[key] == value
    assert importlib.metadata.version('tiktoken') == data['tokenizer']['version']
    enc = tiktoken.get_encoding('gpt2')
    vocabulary = json.dumps(sorted((t.hex(), r) for t, r in enc._mergeable_ranks.items())).encode()
    assert hashlib.sha256(vocabulary).hexdigest() == data['tokenizer']['vocabulary_sha256']
    corpus = Path(data['path']); assert sha(corpus) == data['corpus_sha256']
    text = corpus.read_text(); assert len(text) == data['raw_characters']
    train = enc.encode(text[:data['raw_train_end']], disallowed_special=())
    val = enc.encode(text[data['raw_train_end']:data['raw_val_end']], disallowed_special=())
    assert digest(train) == data['train_ids_sha256'] and digest(val) == data['validation_ids_sha256']
    assert digest(val[1:]) == data['validation_target_ids_sha256']
    freq, vf = collections.Counter(train), collections.Counter(val[1:])
    rows = []
    for old in prior['rows']:
        ids = [i for i in range(50257) if bin_for(freq[i]) == old['bin']]
        ratio = np.sqrt(v.numpy()[ids].astype(np.float64)/correction)/eps
        share = 1/(1+ratio); weights = np.array([vf[i] for i in ids], dtype=np.float64)
        assert len(ids) == old['types'] and weights.sum() == old['validation_targets']
        rows.append(dict(bin=old['bin'], types=len(ids), coordinates=int(ratio.size),
            validation_targets=int(weights.sum()),
            root_over_epsilon_quantiles=np.quantile(ratio, [0, .25, .5, .75, 1]).tolist(),
            epsilon_share_quantiles=np.quantile(share, [0, .25, .5, .75, 1]).tolist(),
            epsilon_share_mean=float(share.mean()),
            fraction_root_le_epsilon=float((ratio <= 1).mean()),
            validation_weighted_epsilon_share=float(np.dot(share.mean(axis=1), weights)/weights.sum())))
    assert sum(r['coordinates'] for r in rows) == 12865792
    assert sum(r['validation_targets'] for r in rows) == 34278
    assert sha(path) == receipt['sha256'] and sha(corpus) == data['corpus_sha256']
    assert torch.equal(rng, torch.random.get_rng_state())
    return dict(completed=True, raw_receipt=receipt, steps=step, beta2=beta, epsilon=eps,
        bias_correction=correction, parameter=name, rows=rows,
        data_identity=prior['data_identity'], cpu_rng_unchanged=True, holdout_encoded=False,
        no_model_execution=True, no_new_losses=True, no_parameter_updates=True,
        scope='One preselected incumbent RAW endpoint; denominator arithmetic, not causal or quality evidence.')


def put(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    assert len(raw)+sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) < CAP
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


if __name__ == '__main__':
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free >= 10*1024**3+CAP
        OUT.mkdir()
        paths = [Path(__file__), ROOT/'research/BYTE_OUTPUT_MOMENT_PROTOCOL.md', RECIPE, PRIOR,
            ROOT/'research/output_anchored_tying_frequency_v1.py',
            ROOT/'src/research/cpu_prerequisite_guard.py',
            ROOT/'research/byte_input_state_v6.py', ROOT/'research/byte_input_training_v6.py',
            ROOT/'research/byte_input_model_v6/optimizer.py']
        pins = {str(p.relative_to(ROOT)): sha(p) for p in paths}
        put(OUT/'attempt.json', dict(input_sources=pins, output_cap_bytes=CAP))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json', ROOT, timeout_seconds=120,
                    rss_limit_bytes=2*1024**3, reserve_bytes=10*1024**3):
                result = run()
                assert all(sha(ROOT/path) == value for path, value in pins.items())
            result.update(attempt_sha256=sha(OUT/'attempt.json'), guard_sha256=sha(OUT/'guard.json'))
            put(OUT/'result.json', result)
            print(json.dumps(result, sort_keys=True), flush=True)
        except BaseException:
            put(OUT/'failure.json', dict(traceback=traceback.format_exc())); raise
