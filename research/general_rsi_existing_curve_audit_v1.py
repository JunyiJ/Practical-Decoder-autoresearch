"""Matched-document learning curves from existing accepted equal-time evidence."""
import json
import math
import statistics
import time
from research.general_rsi_transfer_data_v1 import ROOT, read, put, sha

P = ROOT/'research/tokenizer-evidence/general-rsi-equal-time-001'
OUT = ROOT/'research/tokenizer-evidence/general-rsi-next-exposure-analysis-001'


def aggregate(rows):
    count = sum(r['count'] for r in rows)
    return dict(count=count, ce=math.fsum(r['loss_sum'] for r in rows)/count)


def main():
    start = time.monotonic()
    plan = read(P/'plan.json')
    assert sha(P/'plan.json') == '4f51d54569d8b649c9609d25d42ba5bbaefdfd390db99b3283d5efc1c2c1165d'
    documents = plan['probe_documents']
    records, bindings = [], {}
    for slot in range(4, 10):
        folder = P/f'slot-{slot:02d}'
        paths = [folder/x for x in ['probe-1.json', 'probe-2.json', 'raw-validation.json', 'time_ema-validation.json', 'review.json', 'steps-01.jsonl']]
        bindings.update({str(p.relative_to(ROOT)): sha(p) for p in paths})
        assert read(folder/'review.json')['accepted_execution']
        probes = [read(folder/f'probe-{i}.json') for i in [1, 2]]
        final = read(folder/'raw-validation.json')
        selected = [r for r in final['documents'] if r['document'] in documents]
        assert [r['document'] for r in selected] == documents
        stages = [p['documents'] for p in probes]+[selected]
        inventory = [(r['document'], r['source'], r['count']) for r in selected]
        assert all([(r['document'], r['source'], r['count']) for r in stage] == inventory for stage in stages)
        curves = {}
        for source in ['all', 'fineweb_edu', 'wikipedia']:
            vals = [aggregate([r for r in stage if source == 'all' or r['source'] == source]) for stage in stages]
            curves[source] = dict(stages=vals, second_minus_first=vals[1]['ce']-vals[0]['ce'], final_minus_second=vals[2]['ce']-vals[1]['ce'])
        with (folder/'steps-01.jsonl').open() as f:
            steps = [json.loads(line) for line in f]
        assert [s['step'] for s in steps] == list(range(1, len(steps)+1))
        bins = []
        for lo, hi in [(0, 2400), (2400, 4800), (4800, 7202)]:
            rows = [x for x in steps if lo*1e9 <= x['begin_ns'] < hi*1e9]
            bins.append(dict(begin_seconds=lo, end_seconds=hi, updates=len(rows), mean_train_loss=statistics.mean(x['loss'] for x in rows),
                clipped_fraction=statistics.mean(x['clip']['coefficient'] < 1 for x in rows),
                median_update_seconds=statistics.median((x['end_ns']-x['begin_ns'])/1e9 for x in rows)))
        records.append(dict(slot=slot, arm=plan['slots'][slot]['arm'], bundle=plan['slots'][slot]['bundle'],
            matched_probe=curves, time_bins=bins, final_full_raw_ce=final['ce'],
            final_full_time_ema_ce=read(folder/'time_ema-validation.json')['ce']))
    means = {}
    for arm in ['native','raw_bytes']:
        rows = [r for r in records if r['arm'] == arm]
        means[arm] = {source: dict(stage_mean_ce=[statistics.mean(r['matched_probe'][source]['stages'][i]['ce'] for r in rows) for i in range(3)],
            mean_final_minus_second=statistics.mean(r['matched_probe'][source]['final_minus_second'] for r in rows),
            all_pairs_improve_final=all(r['matched_probe'][source]['final_minus_second'] < 0 for r in rows)) for source in ['all','fineweb_edu','wikipedia']}
    put(OUT/'curves.json', dict(source_sha256=sha(__file__), input_sha256=bindings, documents=documents,
        stages_training_seconds=[2400,4800,7200], records=records, means=means,
        elapsed_seconds=time.monotonic()-start, new_scores=0, test_opened=False,
        limitations='Eight fixed documents only; final raw scores restricted to identical8586 targets. Last-third progress combines further updates with cooldown; cannot attribute to exposure alone or extrapolate beyond7200s. Training means use changing random samples and are not validation estimates. No new significance test or acceptance gate.'))
    print(json.dumps(means, indent=2))


if __name__ == '__main__':
    OUT.mkdir(exist_ok=False)
    main()
