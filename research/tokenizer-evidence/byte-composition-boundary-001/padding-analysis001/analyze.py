"""Exact training-token and paired-window length counts; no model or RNG."""
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'research/tokenizer-evidence/byte-composition-boundary-001'
PLAN = BASE / 'quality001/plan.json'

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def receipt(path):
    return dict(path=str(path), size=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    plan = json.loads(PLAN.read_text())
    expected = plan['data']
    for asset in plan['codebook_assets']:
        path = Path(asset['path'])
        assert path.stat().st_size == asset['size']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset['sha256']
    # Use only the already pinned local cache; both GPT2 assets are verified above.
    os.environ['TIKTOKEN_CACHE_DIR'] = str(Path(plan['codebook_assets'][0]['path']).parent)
    import tiktoken
    encoder = tiktoken.get_encoding('gpt2')
    assert importlib.metadata.version('tiktoken') == expected['tokenizer']['version']
    vocabulary_hash = hashlib.sha256(json.dumps(sorted((raw.hex(), rank)
        for raw, rank in encoder._mergeable_ranks.items())).encode()).hexdigest()
    assert vocabulary_hash == expected['tokenizer']['vocabulary_sha256']
    path = Path(expected['path'])
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected['corpus_sha256']
    text = raw.decode('utf8')
    assert len(text) == expected['raw_characters']
    # No validation or holdout encoding; exact established training split only.
    ids = encoder.encode(text[:expected['raw_train_end']], disallowed_special=())
    assert len(ids) == expected['train_tokens'] and digest(ids) == expected['train_ids_sha256']
    by_id = {rank:len(raw)+2 for raw, rank in encoder._mergeable_ranks.items()}
    by_id[50256] = 3  # explicit special between begin/end, as in the frozen builder
    lengths = [by_id[token] for token in ids]
    counts = collections.Counter(lengths)
    # All admissible first-512-token blocks, exactly matching randint(N-512).
    queue = collections.deque()
    maxima = collections.Counter()
    for index, length in enumerate(lengths):
        while queue and lengths[queue[-1]] <= length:
            queue.pop()
        queue.append(index)
        while queue and queue[0] <= index-512:
            queue.popleft()
        if 511 <= index <= len(lengths)-2:
            maxima[max(6, lengths[queue[0]])] += 1
    starts = len(ids)-512
    assert sum(maxima.values()) == starts
    # Two independent uniformly drawn block starts: P(max<=L)=P(blockmax<=L)^2.
    cumulative = 0
    pair_counts = {}
    for length, count in sorted(maxima.items()):
        pair_counts[length] = (cumulative+count)**2-cumulative**2
        cumulative += count
    pairs = starts**2
    assert sum(pair_counts.values()) == pairs
    channels = (25,50,75,100,125,150)
    def filter_macs(length):
        return 1024*sum((length-width+1)*15*width*channel
                        for width,channel in enumerate(channels,1))
    expected_width = sum(length*count for length,count in pair_counts.items())/pairs
    expected_filter_macs = sum(filter_macs(length)*count
                               for length,count in pair_counts.items())/pairs
    result = dict(kind='exact_training_byte_length_and_paired_sampler_analysis',
        no_model_execution=True, no_random_draws=True, validation_encoded=False,
        holdout_encoded=False, no_quality_scores_used=True,
        inputs=[receipt(PLAN),receipt(path),receipt(ROOT/'research/byte_composer_v2.py'),
                receipt(ROOT/'research/byte_input_model_v3/codebook.py'),
                receipt(ROOT/'research/block_norm_v1/runtime/data/loaders.py')],
        train_ids_sha256=digest(ids), training_tokens=len(ids),
        code_length_includes_boundaries=True, code_length_counts=dict(sorted(counts.items())),
        weighted_mean_code_length=sum(lengths)/len(lengths), max_training_code_length=max(lengths),
        padded_width=130, admissible_block_starts=starts,
        block_max_counts=dict(sorted(maxima.items())), paired_max_counts=pair_counts,
        ordered_start_pairs=pairs, expected_paired_batch_width=expected_width,
        fixed_width_filter_forward_macs=filter_macs(130),
        expected_batch_trimmed_filter_forward_macs=expected_filter_macs,
        trimmed_to_fixed_filter_mac_ratio=expected_filter_macs/filter_macs(130),
        limits=['Exact arithmetic work counts, not timings, FLOP-equivalent hardware utilization or quality predictions.',
                'Filter linear products only; excludes activations, pooling, highway, projection, decoder, backward and validation.',
                'Expectation over frozen sampler support, not a replay or observation of scientific seed sequences.',
                'Cropping is not implemented or qualified; no running source or scientific contract changed.'])
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
