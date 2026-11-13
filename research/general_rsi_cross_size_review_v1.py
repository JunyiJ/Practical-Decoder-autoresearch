"""Execution review of a frozen depth slot, without scientific winner selection."""
import argparse,json,math
from pathlib import Path
from research.general_rsi_training_v2 import sha
from research.general_rsi_context_length_worker_v1 import put
from research.general_rsi_generation_metrics_v1 import score as generation_metrics
ROOT=Path(__file__).resolve().parents[1]
PANEL=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-quality-001'
def read(p):return json.loads(p.read_text())
def review(slot):
    plan=read(PANEL/'plan.json');binding=sha(PANEL/'plan.json');out=PANEL/f'slot-{slot:02d}'
    assert all(sha(ROOT/n)==h for n,h in plan['sources'].items())
    r=read(out/'result.json');assert r['status']=='completed' and r['binding']==binding and r['slot']==plan['slots'][slot]
    assert r['scientific_training'] and not r['test_opened'] and r['steps']==10240 and r['snapshots']==342
    rows=[json.loads(l) for p in sorted(out.glob('steps-*.jsonl')) for l in p.read_text().splitlines()]
    history=read(out/'training-history.json');assert len(history)==len(rows)==10240
    assert [x['step'] for x in rows]==list(range(1,10241))
    for a,b in zip(rows,history):
        assert all(a[k]==v for k,v in b.items())
        assert math.isfinite(a['loss']) and a['seconds']>0 and a['mps_bytes']<plan['mps_limit_bytes']
        expected=1 if a['step']-1<6826 else max(.1,1-.9*((a['step']-1-6826)/(10239-6826)))
        assert abs(a['lr_ratio']-expected)<1e-12
    assert r['executed_targets']==10485760 and abs(sum(x['seconds'] for x in rows)-r['training_seconds'])<1e-6
    # Every previously completed arm/size in this bundle must share exact batches.
    for i in range(slot):
        if plan['slots'][i]['bundle']==r['slot']['bundle']:
            other=read(PANEL/f'slot-{i:02d}/training-history.json')
            assert [x['batch_sha256'] for x in history]==[x['batch_sha256'] for x in other]
    guards=[read(p) for p in out.glob('guard-*.json')];assert guards and guards[-1]['status']=='finished'
    assert sum(g['elapsed_seconds'] for g in guards)<=r['slot']['timeout_seconds']
    assert all(g['observation']['peak_rss_bytes']<plan['rss_limit_bytes'] for g in guards)
    assert sum(read(p)['elapsed_seconds'] for p in PANEL.glob('slot-*/guard-*.json'))<=plan['tranche_seconds']
    for e in r['endpoints']:
        w=e['weights'];p=Path(w['path']);assert p.stat().st_size==w['size'] and sha(p)==w['sha256']
        for label,count,docs in [('validation',100416,108),('broader',32768,16)]:
            s=read(out/f"{e['endpoint']}-{label}.json");assert s['count']==count and len(s['documents'])==docs
            assert [x['document'] for x in s['documents']]==list(range(docs))
            assert sum(x['count'] for x in s['documents'])==count
            assert abs(math.fsum(x['loss_sum'] for x in s['documents'])/count-s['ce'])<1e-12
            assert s['ce']==e['ce' if label=='validation' else 'broader_ce']
    assert [e['endpoint'] for e in r['endpoints']]==['raw','h200']
    for step in plan['probe_steps']:
        p=read(out/f'probe-{step:05d}.json');assert [x['document'] for x in p['documents']]==plan['probe_documents']
    gen=read(out/'generation/result.json');assert len(gen['rows'])==24
    assert len({x['id'] for x in gen['rows']})==24
    endpoint=r['endpoints'][1]['weights'];assert gen['checkpoint']==endpoint and gen['binding']==binding
    for x in gen['rows']:
        assert x['checkpoint_sha256']==endpoint['sha256'] and x['metrics']==generation_metrics(x['generated_token_ids'])
        lines=[json.loads(l) for l in (out/'generation'/f"{x['id']}.jsonl").read_text().splitlines()]
        assert lines[0]['event']=='start' and lines[-1]==dict(event='complete',row=x)
        assert [t['token'] for t in lines if t['event']=='token']==x['generated_token_ids']
        assert x['prompt'] in plan['generation_protocol']['prompts'] and x['decoder'] in plan['generation_protocol']['decoders'] and x['seed']==42
    journals=[json.loads(l) for l in (out/'deletion-journal.jsonl').read_text().splitlines()]
    assert len(journals)%2==0
    for a,b in zip(journals[::2],journals[1::2]):
        assert a['event']=='before_unlink' and b['event']=='unlinked'
        assert {k:v for k,v in a.items() if k!='event'}=={k:v for k,v in b.items() if k!='event'}
        assert not Path(a['receipt']['path']).exists()
    assert read(out/'latest.json')['status']=='retired_after_completion'
    evidence={str(p.relative_to(ROOT)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.log')}
    value=dict(accepted_execution=True,binding=binding,result_sha256=sha(out/'result.json'),slot=slot,steps=10240,snapshots=342,first_outputs=24,
        training_seconds=r['training_seconds'],guard_seconds=sum(g['elapsed_seconds'] for g in guards),peak_mps_bytes=max(x['mps_bytes'] for x in rows),peak_rss_bytes=max(g['observation']['peak_rss_bytes'] for g in guards),evidence_sha256=evidence,quality_decision=False,test_opened=False)
    put(out/'review.json',value);print(json.dumps({k:v for k,v in value.items() if k!='evidence_sha256'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('slot',type=int);review(p.parse_args().slot)
