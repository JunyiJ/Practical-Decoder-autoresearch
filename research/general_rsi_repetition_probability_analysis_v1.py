"""Descriptive aligned-copy analysis of the frozen repetition replay."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/tokenizer-evidence/general-rsi-repetition-probability-001'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(x):return statistics.mean(x) if x else None


def summarize(rows):
    values=[r['third_minus_first_mean_log_probability'] for r in rows]
    return dict(loops=len(rows),positive=sum(x>0 for x in values),negative=sum(x<0 for x in values),
                mean=mean(values),median=statistics.median(values) if values else None,
                monotone_copy_means=sum(r['copy_mean_log_probabilities'][0]<r['copy_mean_log_probabilities'][1]<r['copy_mean_log_probabilities'][2] for r in rows),
                positive_aligned_tokens=sum(r['positive_aligned_tokens'] for r in rows),aligned_tokens=sum(r['period'] for r in rows))


def main():
    plan=read(P/'plan.json');dest=P/'attempt-01';complete=read(dest/'complete.json')
    assert sha(P/'plan.json')==complete['plan_sha256'] and sha(dest/'rows.jsonl')==complete['rows_sha256']
    assert read(dest/'guard.json')['status']=='finished'
    rows=[json.loads(x) for x in (dest/'rows.jsonl').read_text().splitlines()]
    assert len(rows)==264 and len({(r['slot'],r['kind'],r.get('id',r.get('document'))) for r in rows})==264
    specs={(r['slot'],r['id']):r for r in plan['generations']}
    windows={w['document']:w for w in plan['training_windows']}
    loops=[];natural=[];greedy_targets=0
    for r in rows:
        stats=r['tokens'];assert all(math.isfinite(x['log_probability']) and x['log_probability']<=0 and 1<=x['rank']<=50257 for x in stats)
        if r['kind']=='generation':
            spec=specs[r['slot'],r['id']];assert [x['token'] for x in stats]==spec['tokens']
            if r['decoder']=='greedy':
                assert all(x['token']==x['argmax'] for x in stats);greedy_targets+=len(stats)
            if not spec['terminal_loop_phrase_lengths']:continue
            period=min(spec['terminal_loop_phrase_lengths']);n=len(stats)-(spec['tokens'][-1]==50256)
            copies=[stats[n-3*period+i*period:n-2*period+i*period] for i in range(3)]
            assert all([x['token'] for x in copies[0]]==[x['token'] for x in c] for c in copies)
            means=[mean([x['log_probability'] for x in c]) for c in copies]
            delta=[b['log_probability']-a['log_probability'] for a,b in zip(copies[0],copies[2],strict=True)]
            loops.append(dict(slot=r['slot'],id=r['id'],period=period,copy_mean_log_probabilities=means,
                third_minus_first_mean_log_probability=mean(delta),positive_aligned_tokens=sum(d>0 for d in delta),
                third_minus_first_per_token=delta))
        else:
            w=windows[r['document']];assert [x['token'] for x in stats]==w['tokens'][1:]
            seen={}
            for end in range(2,len(w['tokens'])):
                tri=tuple(w['tokens'][end-2:end+1])
                if tri in seen:
                    first=seen[tri]
                    natural.append(dict(slot=r['slot'],document=r['document'],source=r['source'],first_endpoint=first,
                        repeat_endpoint=end,log_probability_delta=stats[end-1]['log_probability']-stats[first-1]['log_probability'],
                        rank_first=stats[first-1]['rank'],rank_repeat=stats[end-1]['rank']))
                else:seen[tri]=end
    assert len(loops)==23 and greedy_targets==2304
    groups={str(s):summarize([r for r in loops if r['slot']==s]) for s in [1,3,5]}
    refs={str(s):dict(repeated_trigram_endpoints=len(v:=[r for r in natural if r['slot']==s]),
         mean_log_probability_delta=mean([r['log_probability_delta'] for r in v]),positive=sum(r['log_probability_delta']>0 for r in v)) for s in [1,3,5]}
    result=dict(plan_sha256=sha(P/'plan.json'),completion_sha256=sha(dest/'complete.json'),rows_sha256=sha(dest/'rows.jsonl'),
        analysis_source_sha256=sha(Path(__file__)),per_seed=groups,all_loops_descriptive=summarize(loops),
        natural_reference=refs,loop_rows=loops,natural_rows=natural,all_original_greedy_targets_match=greedy_targets,
        excluded_generation_rows_without_terminal_loop=72-len(loops),new_generations=0,test_opened=False,
        interpretation_limits=['Descriptive conditional replay, not fresh quality comparison.',
            'Different positions and prefix lengths confound any causal effect of repetition.',
            'Terminal-loop conditioning selects successes of repetition; early onset is not directly tested.',
            'Natural windows differ from generated prefixes; no causal paired control or statistical population claim.'])
    with (P/'analysis.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['loop_rows','natural_rows']},indent=2))


if __name__=='__main__':main()
