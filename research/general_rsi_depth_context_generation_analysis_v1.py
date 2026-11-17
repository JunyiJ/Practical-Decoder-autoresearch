"""Blind and summarize the288 retained first completions; never generate or reroll."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
from research.general_rsi_generation_metrics_v1 import score
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/tokenizer-evidence/general-rsi-depth-context-quality-001'
OUT=P/'generation-evaluation'
BINDING='b99578af06d57549cee6f9ffad19acc0cdeeb89d5f9c5c3a03fcf8ee26dafe1c'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,value):
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def prepare():
    assert not OUT.exists()
    assert sha(P/'plan.json')==BINDING
    plan=read(P/'plan.json')
    for slot in range(12):
        assert (P/f'slot-{slot:02d}'/'review.json').exists(), 'Twelve accepted execution reviews required'
    rows=[]; bindings={}
    for slot in range(12):
        d=P/f'slot-{slot:02d}';r=read(d/'result.json');v=read(d/'review.json');g=read(d/'generation/result.json')
        assert v['accepted_execution'] and v['result_sha256']==sha(d/'result.json')
        assert r['binding']==v['binding']==g['binding']==BINDING and r['slot']==plan['slots'][slot]
        expected={(p['id'],decoder) for p in plan['generation_protocol']['prompts'] for decoder in plan['generation_protocol']['decoders']}
        assert {(row['prompt']['id'],row['decoder']) for row in g['rows']}==expected
        assert len(g['rows'])==24 and g['checkpoint']['sha256']==r['endpoints'][1]['weights']['sha256']
        bindings[str((d/'generation/result.json').relative_to(ROOT))]=sha(d/'generation/result.json')
        for row in g['rows']:
            assert row['checkpoint_sha256']==g['checkpoint']['sha256']
            assert row['metrics']==score(row['generated_token_ids'])
            rows.append((slot,row))
    assert len(rows)==288
    random.Random(plan['blind_order_seed']).shuffle(rows)
    blinded=[];mapping=[];metrics=[]
    for i,(slot,row) in enumerate(rows):
        key=f'b{i:03d}'
        blinded.append(dict(id=key,prompt=row['prompt'],continuation=row['continuation']))
        mapping.append(dict(id=key,slot=slot,depth=plan['slots'][slot]['depth'],arm=plan['slots'][slot]['arm'],bundle=plan['slots'][slot]['bundle'],original_id=row['id'],decoder=row['decoder']))
        metrics.append(dict(id=key,**score(row['generated_token_ids'])))
    OUT.mkdir()
    put(OUT/'blind.json',blinded);put(OUT/'mapping.json',mapping);put(OUT/'metrics.json',metrics)
    put(OUT/'preparation.json',dict(plan_sha256=BINDING,source_sha256=sha(Path(__file__)),metrics_sha256=sha(OUT/'metrics.json'),input_sha256=bindings,shuffle_seed=plan['blind_order_seed'],count=288,blind_sha256=sha(OUT/'blind.json'),mapping_sha256=sha(OUT/'mapping.json'),test_opened=False))
    print('Prepared288 blind rows; mapping kept separate.')
def summary(rows):
    result={'count':len(rows)}
    for field in ['relevance','consistency']:
        c=Counter(str(x[field]) for x in rows);assess=[x[field] for x in rows if type(x[field]) is int]
        byprompt=defaultdict(list)
        for x in rows:
            if type(x[field]) is int:byprompt[x['prompt']['id']].append(x[field])
        result[field]=dict(ordinal_counts={k:c[k] for k in ['0','1','2','unassessable']},unassessable_fraction=c['unassessable']/len(rows),assessable_mean=statistics.mean(assess) if assess else None,equal_weight_prompt_mean=statistics.mean(statistics.mean(v) for v in byprompt.values()) if byprompt else None,assessable_prompt_count=len(byprompt))
    for field in ['content_tokens','repeated_trigram_fraction','terminal_loop','stopped_eos','hit_token_cap','unexpected_early_stop']:
        values=[x[field] for x in rows if x[field] is not None]
        result[field]=dict(mean=statistics.mean(values) if values else None,assessable_count=len(values))
    return result
def aggregate():
    prep=read(OUT/'preparation.json');assert sha(OUT/'blind.json')==prep['blind_sha256'] and sha(OUT/'mapping.json')==prep['mapping_sha256']
    assert sha(P/'plan.json')==prep['plan_sha256']==BINDING and sha(OUT/'metrics.json')==prep['metrics_sha256']
    for path,digest in prep['input_sha256'].items():assert sha(ROOT/path)==digest
    blind=read(OUT/'blind.json'); annotations=read(OUT/'annotations.json')
    assert len(annotations)==288 and [x['id'] for x in annotations]==[x['id'] for x in blind]
    metrics={x['id']:x for x in read(OUT/'metrics.json')};mapping={x['id']:x for x in read(OUT/'mapping.json')}
    joined=[]
    for a,b in zip(annotations,blind):
        assert a['relevance'] in [0,1,2,'unassessable'] and a['consistency'] in [0,1,2,'unassessable']
        assert a['reason'] and a['annotator_identity'] and a['evidence_quote'] in b['continuation']
        joined.append(dict(**mapping[a['id']],prompt=b['prompt'],**{k:v for k,v in metrics[a['id']].items() if k!='id'},**{k:v for k,v in a.items() if k!='id'}))
    groups={}
    for depth in [4,16]:
        groups[str(depth)]={}
        for arm in ['context128','context256']:
            groups[str(depth)][arm]={}
            for decoder in ['greedy','temperature_0.8_top_k_40']:
                rows=[r for r in joined if r['depth']==depth and r['arm']==arm and r['decoder']==decoder];assert len(rows)==36
                groups[str(depth)][arm][decoder]=dict(all_seeds=summary(rows),per_seed={str(i):summary([r for r in rows if r['bundle']==i]) for i in [1,2,3]})
    put(OUT/'summary.json',dict(groups=groups,count=288,analysis_source_sha256=sha(Path(__file__)),annotations_sha256=sha(OUT/'annotations.json'),preparation_sha256=sha(OUT/'preparation.json'),limits='Exploratory agent ratings, not independent human validation. Consistency is not exhaustive factual accuracy. No composite with CE or adoption gate.',test_opened=False))
    print('Summarized288 fixed first outputs by depth,context,decoder and bundle.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','aggregate']);a=p.parse_args();prepare() if a.action=='prepare' else aggregate()
