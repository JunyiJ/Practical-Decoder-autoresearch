"""Verify a completed exposure slot without running or selecting models."""
import argparse
import json
import math
from pathlib import Path
from research.general_rsi_training_v1 import sha
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/tokenizer-evidence/general-rsi-exposure-001'

def read(p):return json.loads(p.read_text())
def review(slot,attempt):
    plan=read(P/'plan.json');d=P/f'slot-{slot:02d}'
    r,g=read(d/'result.json'),read(d/f'guard-{attempt:02d}.json')
    assert r['status']=='completed' and r['plan_sha256']==sha(P/'plan.json')
    config=plan['slots'][slot];h=config['horizon']
    assert r['slot']==config and r['steps']==h and not r['test_opened']
    assert r['ema_snapshots']==(h-math.ceil((2*h/3)/10)*10)//10+1
    assert g['status']=='finished' and g['reason'] is None
    assert g['elapsed_seconds']<plan['slot_timeout_seconds']
    assert g['observation']['peak_rss_bytes']<plan['rss_limit_bytes']
    assert read(d/'latest.json')['status']=='retired_after_completion'
    for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
    rows=[json.loads(x) for x in (d/'steps.jsonl').read_text().splitlines()]
    assert [x['step'] for x in rows]==list(range(1,h+1)) and rows[-1]['lr_ratio']==.1
    for x in rows:
        assert math.isfinite(x['loss']) and math.isfinite(x['clip']['pre_norm'])
        assert math.isfinite(x['clip']['post_norm']) and x['clip']['post_norm']<=1.00001
        assert x['clip']['gradient_tensors']==105 and x['mps_driver_bytes']<plan['mps_limit_bytes']
    base=read(ROOT/'research/tokenizer-evidence/general-rsi-panel-001/slot-09/h200-validation.json')['documents']
    inv=lambda docs:[(x['document'],x['source'],x['count']) for x in docs]
    for e in r['endpoints']:
        w=e['weights'];assert Path(w['path']).stat().st_size==w['size']==65441133 and sha(w['path'])==w['sha256']
        v=read(d/f"{e['endpoint']}-validation.json");docs=v['documents']
        assert inv(docs)==inv(base) and len(docs)==108
        assert sum(x['count'] for x in docs)==v['count']==e['count']==100416
        assert abs(sum(x['loss_sum'] for x in docs)/v['count']-e['ce'])<1e-12 and v['ce']==e['ce']
        for source,count in [('fineweb_edu',80354),('wikipedia',20062)]:
            group=[x for x in docs if x['source']==source]
            assert sum(x['count'] for x in group)==count
            assert abs(sum(x['loss_sum'] for x in group)/count-e['sources'][source]['ce'])<1e-12
    generation={}
    if config['generation']:
        for name in ['initial','raw','h200']:
            samples=read(d/f'generation-{name}.json');assert len(samples)==16
            assert [x['prompt'] for x in samples]==plan['prompts']*2
            assert [x['decoding'] for x in samples]==['greedy']*8+['temperature_0.8_top_k_40']*8
            for x in samples:
                assert x['seed']==42 and 0<len(x['tokens'])<=64
                assert x['stop_reason']==('eos' if x['tokens'][-1]==50256 else 'length_limit')
            generation[name]={'records':len(samples),'length_limit':sum(x['stop_reason']=='length_limit' for x in samples),'sha256':sha(d/f'generation-{name}.json')}
    pair=None
    if slot%2:
        prior=P/f'slot-{slot-1:02d}'
        short=[json.loads(x) for x in (prior/'steps.jsonl').read_text().splitlines()]
        assert [x['batch_sha256'] for x in rows[:10240]]==[x['batch_sha256'] for x in short]
        p=read(prior/'result.json')
        pair={'short_slot':slot-1,'batch_prefix_matches':True,'h200_delta':r['endpoints'][1]['ce']-p['endpoints'][1]['ce']}
        if config['generation']:assert sha(d/'generation-initial.json')==sha(prior/'generation-initial.json')
    return dict(slot=slot,attempt=attempt,accepted_execution=True,result_sha256=sha(d/'result.json'),guard_sha256=sha(d/f'guard-{attempt:02d}.json'),review_source_sha256=sha(__file__),source_bindings_verified=len(plan['sources']),steps=h,ema_snapshots=r['ema_snapshots'],finite_gradients_and_clipping_verified=True,retained_endpoint_hashes_verified=True,document_coverage_verified=True,generation=generation,paired_comparison=pair,primary_h200_ce=r['endpoints'][1]['ce'],secondary_raw_ce=r['endpoints'][0]['ce'],horizon_selected=False,test_opened=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('slot',type=int);ap.add_argument('attempt',type=int);a=ap.parse_args()
    result=review(a.slot,a.attempt)
    with (P/f'slot-{a.slot:02d}/review.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))
