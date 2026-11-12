"""Paired descriptive context-order analysis, separate natural and loop cases."""
import json
import math
from pathlib import Path
import statistics
from research.general_rsi_repetition_probability_v1 import sha,read,put
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/tokenizer-evidence/general-rsi-context-order-001'


def main():
    plan=read(P/'plan.json');dest=P/'attempt-01';complete=read(dest/'complete.json')
    assert complete['plan_sha256']==sha(P/'plan.json') and complete['rows_sha256']==sha(dest/'rows.jsonl')
    assert read(dest/'guard.json')['status']=='finished'
    rows=[json.loads(l) for l in (dest/'rows.jsonl').read_text().splitlines()]
    expected={(r['slot'],r['kind'],r['id']):r for r in plan['cases']}
    assert len(rows)==len(expected)==281 and len({(r['slot'],r['kind'],r['id']) for r in rows})==281
    cases=[]
    for r in rows:
        spec=expected[r['slot'],r['kind'],r['id']]
        assert len(r['original'])==len(r['permuted'])==len(spec['target_positions'])
        delta=[];argmax_changes=0;lost_top1=0;gained_top1=0;rankdiff=[]
        for a,b,pos in zip(r['original'],r['permuted'],spec['target_positions'],strict=True):
            assert a['token']==b['token']==spec['original'][pos]
            assert all(math.isfinite(x['log_probability']) and x['log_probability']<=0 for x in [a,b])
            delta.append(b['log_probability']-a['log_probability'])
            argmax_changes+=a['argmax']!=b['argmax']
            lost_top1+=a['argmax']==a['token'] and b['argmax']!=b['token']
            gained_top1+=a['argmax']!=a['token'] and b['argmax']==b['token']
            rankdiff.append(b['rank']-a['rank'])
        cases.append(dict(slot=r['slot'],kind=r['kind'],id=r['id'],targets=len(delta),changed_positions=r['changed_positions'],
            mean_log_probability_delta=statistics.mean(delta),per_token_delta=delta,mean_rank_change=statistics.mean(rankdiff),
            argmax_changes=argmax_changes,lost_top1=lost_top1,gained_top1=gained_top1))
    groups={}
    for kind in ['loop','natural']:
        groups[kind]={}
        for slot in [1,3,5]:
            subset=[r for r in cases if r['kind']==kind and r['slot']==slot];values=[r['mean_log_probability_delta'] for r in subset]
            groups[kind][str(slot)]=dict(cases=len(subset),targets=sum(r['targets'] for r in subset),
                reduced_probability=sum(v<0 for v in values),increased_probability=sum(v>0 for v in values),
                mean_case_log_probability_delta=statistics.mean(values),median_case_log_probability_delta=statistics.median(values),
                min_case_delta=min(values),max_case_delta=max(values),
                argmax_changes=sum(r['argmax_changes'] for r in subset),lost_top1=sum(r['lost_top1'] for r in subset),gained_top1=sum(r['gained_top1'] for r in subset),
                unchanged_input_cases=sum(r['changed_positions']==0 for r in subset))
    result=dict(plan_sha256=sha(P/'plan.json'),rows_sha256=sha(dest/'rows.jsonl'),analysis_source_sha256=sha(__file__),
        groups=groups,cases=cases,delta_direction='permuted minus intact; negative=original order supported target probability',
        limits=['Length, positions, prefix multiset and immediate two-token context controlled.',
                'Earlier sequence order, syntax and semantic coherence change together.',
                'Natural and loop cohorts differ and are not matched causal populations.',
                'Selected retained examples; no fresh quality or adoption outcome.'],new_generations=0,test_opened=False)
    put(P/'analysis.json',result);print(json.dumps(groups,indent=2))


if __name__=='__main__':main()
