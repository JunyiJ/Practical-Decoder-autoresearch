"""Posthoc attribution of retained cross-size scores. No model or corpus access."""
import hashlib
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'research/tokenizer-evidence/general-rsi-cross-size-quality-001'
OUT = ROOT / 'research/tokenizer-evidence/general-rsi-cross-size-attribution-001'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    inputs = {}
    def read(p):
        inputs[str(p.relative_to(ROOT))] = sha(p)
        return json.loads(p.read_text())
    plan = read(P / 'plan.json')
    assert read(P / 'completion-review.json')['accepted_completion']
    decision = read(P / 'decision.json')
    scores, steps, probes = {}, {}, {}
    for s in plan['slots']:
        d = P / f"slot-{s['slot']:02d}"
        assert read(d / 'review.json')['accepted_execution']
        key = s['depth'], s['bundle'], s['arm']
        for endpoint in ('raw', 'h200'):
            scores[key + (endpoint,)] = read(d / f'{endpoint}-validation.json')
        q = d / 'steps-01.jsonl'
        inputs[str(q.relative_to(ROOT))] = sha(q)
        steps[key] = [json.loads(x) for x in q.read_text().splitlines()]
        assert [x['step'] for x in steps[key]] == list(range(1, 10241))
        for n in plan['probe_steps']:
            probes[key + (n,)] = read(d / f'probe-{n:05d}.json')['ce']
    ref = scores[4, 1, 'native', 'h200']['documents']
    inventory = [(d['document'], d['source'], d['count']) for d in ref]
    total = sum(d['count'] for d in ref)
    assert len(ref) == 108 and total == 100416
    for score in scores.values():
        assert inventory == [(d['document'], d['source'], d['count']) for d in score['documents']]
        assert abs(sum(d['loss_sum'] for d in score['documents']) / total - score['ce']) < 1e-12
    for b in (1, 2, 3):
        baseline = [x['batch_sha256'] for x in steps[4, b, 'native']]
        for depth in (4, 8, 16):
            for arm in ('native', 'raw_bytes'):
                assert baseline == [x['batch_sha256'] for x in steps[depth, b, arm]]
    documents, groups, sensitivity = [], [], []
    for endpoint in ('raw', 'h200'):
        rows = []
        for i, doc in enumerate(ref):
            deltas = {str(depth): [(scores[depth, b, 'raw_bytes', endpoint]['documents'][i]['loss_sum'] - scores[depth, b, 'native', endpoint]['documents'][i]['loss_sum']) / doc['count'] for b in (1, 2, 3)] for depth in (4, 8, 16)}
            interactions = [x-y for x, y in zip(deltas['16'], deltas['4'])]
            row = dict(endpoint=endpoint, document=doc['document'], source=doc['source'], targets=doc['count'], paired_deltas=deltas, mean_deltas={d: st.mean(v) for d, v in deltas.items()}, paired_interactions=interactions, mean_interaction=st.mean(interactions), contribution=st.mean(interactions)*doc['count']/total)
            rows.append(row)
        documents.extend(rows)
        for source in ('all', *sorted({d['source'] for d in ref})):
            selected = [r for r in rows if source == 'all' or r['source'] == source]
            count = sum(r['targets'] for r in selected)
            paired = [sum(r['paired_interactions'][b]*r['targets'] for r in selected)/count for b in range(3)]
            groups.append(dict(endpoint=endpoint, source=source, documents=len(selected), targets=count, mean_deltas={str(d):sum(r['mean_deltas'][str(d)]*r['targets'] for r in selected)/count for d in (4,8,16)}, paired_interactions=paired, mean_interaction=st.mean(paired), positive_mean_documents=sum(r['mean_interaction']>0 for r in selected), all_three_positive_documents=sum(all(x>0 for x in r['paired_interactions']) for r in selected), positive_mean_target_fraction=sum(r['targets'] for r in selected if r['mean_interaction']>0)/count, contribution_to_overall=sum(r['contribution'] for r in selected)))
        interaction = sum(r['contribution'] for r in rows)
        loo = [(interaction*total-r['mean_interaction']*r['targets'])/(total-r['targets']) for r in rows]
        ordered = sorted(rows, key=lambda r:r['contribution'], reverse=True)
        removals = []
        for k in (1, 5, 10):
            removed = ordered[:k]
            count = total-sum(r['targets'] for r in removed)
            value = (interaction*total-sum(r['mean_interaction']*r['targets'] for r in removed))/count
            removals.append(dict(k=k, removed_ids=[r['document'] for r in removed], remaining_targets=count, remaining_interaction=value, removed_net_contribution_fraction=sum(r['contribution'] for r in removed)/interaction))
        sensitivity.append(dict(endpoint=endpoint, leave_one_out_range=[min(loo),max(loo)], remove_largest_positive_contributors=removals, top_positive_documents=[r['document'] for r in ordered[:10]], top_negative_documents=[r['document'] for r in ordered[-10:]]))
    curve_rows, training_rows = [], []
    for n in plan['probe_steps']:
        deltas = {str(d):[probes[d,b,'raw_bytes',n]-probes[d,b,'native',n] for b in (1,2,3)] for d in (4,8,16)}
        interaction = [x-y for x,y in zip(deltas['16'],deltas['4'])]
        curve_rows.append(dict(step=n, phase='before_cooldown' if n<6826 else 'during_cooldown', paired_deltas=deltas, mean_deltas={d:st.mean(v) for d,v in deltas.items()}, paired_interactions=interaction, mean_interaction=st.mean(interaction)))
    for (depth,bundle,arm), ss in sorted(steps.items()):
        for lo, hi in ((1,1024),(1025,2048),(2049,3072),(3073,4096),(4097,5120),(5121,6144),(6145,6826),(6827,7168),(7169,8192),(8193,9216),(9217,10240)):
            chunk=ss[lo-1:hi]
            training_rows.append(dict(depth=depth,bundle=bundle,arm=arm,first_step=lo,last_step=hi,mean_loss=st.mean(x['loss'] for x in chunk),clip_fraction=st.mean(x['clip']['coefficient']<1 for x in chunk),mean_gradient_norm=st.mean(x['clip']['pre_norm'] for x in chunk),max_gradient_norm=max(x['clip']['pre_norm'] for x in chunk),lr_ratio_range=[min(x['lr_ratio'] for x in chunk),max(x['lr_ratio'] for x in chunk)]))
    result=dict(source_sha256=sha(Path(__file__)),input_sha256=inputs,posthoc=True,policy='Existing scores only. Token-weighted paired seed means. Leave-out and document counts are descriptive sensitivity, not new tests or gates. Probe uses two FineWeb documents (4503 targets) and online weights; final H200 is distinct. No causal attribution or population-seed inference.',test_opened=False,targets=total,documents=documents,source_summary=groups,sensitivity=sensitivity,probe_interaction=curve_rows,training_blocks=training_rows,paired_batch_hashes_equal=True)
    OUT.mkdir(exist_ok=True)
    with (OUT/'analysis.json').open('x') as f:
        json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps({k:result[k] for k in ('source_summary','sensitivity','probe_interaction')},indent=2))

if __name__ == '__main__':
    run()
