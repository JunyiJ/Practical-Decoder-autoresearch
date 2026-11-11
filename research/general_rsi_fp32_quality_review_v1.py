"""Review exact first-attempt matched-update endpoints; recovery needs explicit audit."""
import argparse
import json
from pathlib import Path
from research.general_rsi_training_v2 import sha
from research.general_rsi_time_journal_v1 import accounted_intervals
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/tokenizer-evidence/general-rsi-composer-fp32-quality-001'
def read(p):return json.loads(p.read_text())
def main():
    a=argparse.ArgumentParser();a.add_argument('slot',type=int);args=a.parse_args();slot=args.slot
    plan=read(P/'plan.json');binding=sha(P/'plan.json');out=P/f'slot-{slot:02d}'
    r=read(out/'result.json');assert r['binding']==binding and r['slot']==plan['slots'][slot]
    assert r['status']=='completed' and r['steps']==10240 and r['snapshots']==342 and not r['test_opened']
    assert len(list(out.glob('attempt-*.json')))==1,'Recovery requires separately recorded complete replay review'
    exit=read(out/'supervisor-01-exit.json');guard=read(out/'guard-01.json')
    assert exit['returncode']==0 and exit['binding']==binding and guard['status']=='finished' and guard['reason'] is None
    rows=[json.loads(x) for x in (out/'steps-01.jsonl').read_text().splitlines()]
    assert [x['step'] for x in rows]==list(range(1,10241)) and rows[-1]['lr_ratio']==.1
    intervals=accounted_intervals([json.loads(x) for x in (out/'intervals-01.jsonl').read_text().splitlines()])
    assert len(intervals)==10240 and sum(x['duration_ns'] for x in intervals)==r['training_ns'] and r['lost_training_ns']==0
    assert all(a['duration_ns']==b['duration_ns'] for a,b in zip(rows,intervals))
    for e in r['endpoints']:
        w=e['weights'];assert Path(w['path']).stat().st_size==w['size'] and sha(w['path'])==w['sha256']
        v=read(out/(e['endpoint']+'-validation.json'))
        assert len(v['documents'])==108 and v['count']==100416 and v['ce']==e['ce']
        assert [d['document'] for d in v['documents']]==list(range(108))
    g=read(out/'generation/result.json');assert len(g['rows'])==24 and g['binding']==binding and g['checkpoint']==r['endpoints'][1]['weights']
    paired=[s for s in plan['slots'] if s['bundle']==r['slot']['bundle'] and s['slot']!=slot][0]['slot']
    compared=False
    if paired<slot:
        other=P/f'slot-{paired:02d}';assert read(other/'review.json')['accepted_execution']
        before=[json.loads(x) for x in (other/'steps-01.jsonl').read_text().splitlines()]
        assert [x['batch_sha256'] for x in before]==[x['batch_sha256'] for x in rows];compared=True
    for path,h in plan['sources'].items():assert sha(ROOT/path)==h,path
    review=dict(accepted_execution=True,slot=slot,binding=binding,result_sha256=sha(out/'result.json'),steps=10240,all_108_documents_100416_targets=True,first_generations=24,paired_batch_stream_equal=compared,training_ns=r['training_ns'],worker_seconds=guard['elapsed_seconds'],test_opened=False)
    with (out/'review.json').open('x') as f:json.dump(review,f,indent=2);f.write('\n')
    print(json.dumps(review))
if __name__=='__main__':main()
