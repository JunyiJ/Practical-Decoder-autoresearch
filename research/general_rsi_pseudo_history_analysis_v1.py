"""Read-only analysis of completed pseudo-repetition logs and fixed probe documents."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/tokenizer-evidence/general-rsi-pseudo-repetition-001'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=read(P/'plan.json');decision=read(P/'decision.json');assert sha(P/'plan.json')==decision['plan_sha256']
    inputs={};rows=[]
    for i in range(6):
        d=P/f'slot-{i:02d}';r=read(d/'result.json');review=read(d/'review.json')
        assert review['accepted_execution'] and review['result_sha256']==sha(d/'result.json')
        steps=[json.loads(x) for x in (d/'steps-01.jsonl').read_text().splitlines()]
        assert len(steps)==r['steps'] and not r['recovery_charges']
        probe=[read(d/f'probe-{j}.json') for j in [1,2]]
        assert all([x['document'] for x in q['documents']]==plan['probe_documents'] for q in probe)
        raw=read(d/'raw-validation.json');subset=[x for x in raw['documents'] if x['document'] in plan['probe_documents']]
        losses={}
        for objective in ['ce','pseudo']:
            z=[x for x in steps if x['objective']==objective]
            losses[objective]=dict(updates=len(z),targets=sum(x['valid_loss_targets'] for x in z),first_100_mean=statistics.mean(x['loss'] for x in z[:100]) if z else None,last_100_mean=statistics.mean(x['loss'] for x in z[-100:]) if z else None,clipped_fraction=statistics.mean(x['clip']['coefficient']<1 for x in z) if z else None)
        rows.append(dict(slot=i,recipe=r['slot']['recipe'],training_seconds=r['training_ns']/1e9,updates=r['steps'],objective=losses,raw_probe_ce_at_thirds=[q['ce'] for q in probe]+[sum(x['loss_sum'] for x in subset)/sum(x['count'] for x in subset)],probe_targets=probe[0]['count'],saturated_targets=review['saturated_targets']))
        for n in ['result.json','review.json','steps-01.jsonl','probe-1.json','probe-2.json','raw-validation.json']:inputs[str((d/n).relative_to(ROOT))]=sha(d/n)
    pairs=[dict(bundle=i+1,raw_probe_delta_pseudo_minus_ce=[b-a for a,b in zip(rows[2*i]['raw_probe_ce_at_thirds'],rows[2*i+1]['raw_probe_ce_at_thirds'])],natural_target_ratio=rows[2*i+1]['objective']['ce']['targets']/rows[2*i]['objective']['ce']['targets']) for i in range(3)]
    result=dict(plan_sha256=sha(P/'plan.json'),source_sha256=sha(Path(__file__)),input_sha256=inputs,rows=rows,pairs=pairs,total_training_seconds=sum(x['training_seconds'] for x in rows),scope='Post hoc descriptive logs and the unchanged eight fixed probe documents. RAW probes at training thirds and RAW final subset; no checkpoint selection, gate change, extra scoring or causal attribution.',test_opened=False)
    with (P/'history-analysis.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['pairs','total_training_seconds']},indent=2))
if __name__=='__main__':main()
