"""Selected-rate saved-loss comparison using the unchanged diagnostic001 bins."""
import collections
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'research/tokenizer-evidence/byte-composition-boundary-001'
Q = BASE / 'quality001'

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def frequency_bin(count):
    return '0' if count == 0 else '1-9' if count < 10 else '10-99' if count < 100 else '100-999' if count < 1000 else '1000+'

def length_bin(length):
    return '1' if length == 1 else '2-3' if length <= 3 else '4-7' if length <= 7 else '8+'

def main():
    # Reuse diagnostic001 bins unchanged; all comparisons remain exploratory.
    plan = json.loads((Q/'plan.json').read_text())
    expected = plan['data']
    inputs = [Q/'plan.json', Q/'selection.json', BASE/'screen-diagnostic001/analyze.py', BASE/'screen-diagnostic002/analyze.py', BASE/'screen-diagnostic002/result.json', Path(__file__).resolve()]
    selected = json.loads((Q/'selection.json').read_text())
    for screen in selected['screens']:
        review_path = Q/'slots'/str(screen['slot']).zfill(2)/'review.json'
        assert sha(review_path) == screen['review_sha256']
        row = json.loads(review_path.read_text())
        assert all(row[k] == screen[k] for k in ('slot','variant','learning_rate','ce'))
        inputs.append(review_path)
    assert len(selected['screens']) == 9
    for arm in ('native','raw_bytes','nonlexical_codes'):
        candidates = [r for r in selected['screens'] if r['variant'] == arm]
        assert sorted(r['learning_rate'] for r in candidates) == [.000625,.00125,.0025]
        assert selected['learning_rates'][arm] == min(candidates,key=lambda r:(r['ce'],r['learning_rate']))['learning_rate']
    saved = []
    reviews = []
    for slot in ('05','08','06'):
        folder = Q/'slots'/slot
        acceptance = json.loads((folder/'acceptance.json').read_text())
        assert acceptance['accepted'] and acceptance['slot'] == int(slot)
        for name, bound in acceptance['evidence'].items():
            assert sha(folder/name) == bound
        assert json.loads((folder/'guard.json').read_text())['status'] == 'finished'
        assert json.loads((folder/'STOP').read_text())['status'] == 'complete_original_scientific_slot'
        reviews.append(json.loads((folder/'review.json').read_text()))
        saved.append(json.loads((folder/'validation.json').read_text()))
        inputs += [folder/'acceptance.json', folder/'review.json', folder/'validation.json']
    assert [r['variant'] for r in reviews] == ['native','raw_bytes','nonlexical_codes']
    assert all(r['learning_rate'] == selected['learning_rates'][r['variant']] and r['bundle'] == 'screen' and not r['quality_selected'] for r in reviews)
    assert all(r['sampler_hashes'] == reviews[0]['sampler_hashes'] for r in reviews)
    assert all(s['report']['target'] == saved[0]['report']['target'] for s in saved)
    for asset in plan['codebook_assets']:
        assert Path(asset['path']).stat().st_size == asset['size'] and sha(Path(asset['path'])) == asset['sha256']
    os.environ['TIKTOKEN_CACHE_DIR'] = str(Path(plan['codebook_assets'][0]['path']).parent)
    import tiktoken
    assert importlib.metadata.version('tiktoken') == expected['tokenizer']['version']
    enc = tiktoken.get_encoding('gpt2')
    assert hashlib.sha256(json.dumps(sorted((raw.hex(), rank) for raw,rank in enc._mergeable_ranks.items())).encode()).hexdigest() == expected['tokenizer']['vocabulary_sha256']
    corpus = Path(expected['path']);inputs.append(corpus)
    assert sha(corpus) == expected['corpus_sha256']
    text = corpus.read_text()
    train = enc.encode(text[:expected['raw_train_end']], disallowed_special=())
    validation = enc.encode(text[expected['raw_train_end']:expected['raw_val_end']], disallowed_special=())
    assert digest(train) == expected['train_ids_sha256'] and digest(validation) == expected['validation_ids_sha256']
    assert digest(validation[1:]) == saved[0]['report']['target']['target_ids_sha256']
    frequencies = collections.Counter(train)
    byte_lengths = {rank:len(raw) for raw,rank in enc._mergeable_ranks.items()}
    assert all(token in byte_lengths for token in validation)
    a,b,c = [s['per_target_losses'] for s in saved]
    assert len(a) == len(b) == len(c) == len(validation)-1 == 34278
    assert all(math.isfinite(x) and x >= 0 for x in a+b+c)
    for losses, score in zip((a,b,c),saved):
        assert abs(math.fsum(losses)-score['report']['total_loss']) < 1e-8
    contrasts = dict(raw_minus_native=[y-x for x,y in zip(a,b)],
        nonlexical_minus_native=[z-x for x,z in zip(a,c)],
        raw_minus_nonlexical=[y-z for y,z in zip(b,c)])
    global_deltas = {name:math.fsum(values)/len(values) for name,values in contrasts.items()}
    original = json.loads((BASE/'screen-diagnostic002/result.json').read_text())
    for name,left,right in [('raw_minus_native',1,0),('nonlexical_minus_native',2,0),('raw_minus_nonlexical',1,2)]:
        assert abs(global_deltas[name]-(saved[left]['report']['ce']-saved[right]['report']['ce'])) < 1e-12
    def stratify(ids, classifier):
        groups = collections.defaultdict(list)
        for index, token in enumerate(ids):groups[classifier(token)].append(index)
        rows = []
        for label, indices in groups.items():
            row = dict(bin=label, targets=len(indices),
                native_ce=math.fsum(a[i] for i in indices)/len(indices),
                raw_byte_ce=math.fsum(b[i] for i in indices)/len(indices),
                nonlexical_ce=math.fsum(c[i] for i in indices)/len(indices), contrasts={})
            for name, delta in contrasts.items():
                total = math.fsum(delta[i] for i in indices)
                row['contrasts'][name] = dict(mean_ce_delta=total/len(indices),
                    contribution_to_global_delta=total/len(delta),
                    first_arm_lower_loss_fraction=sum(delta[i]<0 for i in indices)/len(indices))
            rows.append(row)
        assert sum(r['targets'] for r in rows)==len(a)
        for name in contrasts:
            assert abs(math.fsum(r['contrasts'][name]['contribution_to_global_delta'] for r in rows)-global_deltas[name])<1e-12
        return rows
    result = dict(kind='exploratory_selected_rate_three_arm_fixed_bin_saved_loss_comparison',
        selected_slots={r['variant']:r['slot'] for r in reviews}, learning_rates=selected['learning_rates'], paired_screen_bundle=True, global_ce={r['variant']:s['report']['ce'] for r,s in zip(reviews,saved)},
        global_ce_deltas=global_deltas,
        inputs=[dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for path in inputs],
        target_frequency=stratify(validation[1:],lambda token:frequency_bin(frequencies[token])),
        preceding_input_frequency=stratify(validation[:-1],lambda token:frequency_bin(frequencies[token])),
        target_byte_length=stratify(validation[1:],lambda token:length_bin(byte_lengths[token])),
        preceding_input_byte_length=stratify(validation[:-1],lambda token:length_bin(byte_lengths[token])),
        no_model_execution=True, no_new_losses=True, holdout_encoded=False, holdout_scored=False,
        limitations=['Independently selected rates on one paired screening bundle; selection-biased exploratory results, no fresh confirmation.',
                    'Exploratory after aggregate result was known; no confirmatory subgroup claim or confidence interval.',
                    'Frequency and length correlate with difficulty, context and token identity; no causal attribution.',
                    'Preceding input is the last token of the model context, not the whole context.',
                    'Keep the complete frozen scientific panel and adoption criteria unchanged.'])
    # Preserve every original group definition and count; selected models have new losses.
    for partition in ('target_frequency','preceding_input_frequency','target_byte_length','preceding_input_byte_length'):
        for before,after in zip(original[partition],result[partition],strict=True):
            assert before['bin']==after['bin'] and before['targets']==after['targets']
            for name in contrasts:
                after['contrasts'][name]['previous_first_rate_mean_ce_delta'] = before['contrasts'][name]['mean_ce_delta']
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == '__main__':main()
