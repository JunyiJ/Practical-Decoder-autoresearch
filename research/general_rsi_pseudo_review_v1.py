"""Independent execution review of one pseudo-repetition slot; no scientific reselection."""
import argparse,json,math
from pathlib import Path
from research.general_rsi_training_v2 import sha
from research.general_rsi_time_journal_v1 import accounted_intervals
ROOT=Path(__file__).resolve().parents[1]
PANEL=ROOT/'research/tokenizer-evidence/general-rsi-pseudo-repetition-001'


def review(slot):
    binding=sha(PANEL/'plan.json');plan=json.loads((PANEL/'plan.json').read_text())
    for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
    out=PANEL/f'slot-{slot:02d}';result=json.loads((out/'result.json').read_text())
    expected=dict(plan['slots'][slot])
    assert result['status']=='completed' and result['binding']==binding and result['slot']==expected
    attempts=sorted(out.glob('attempt-*.json'));assert 1<=len(attempts)<=plan['max_attempts_per_slot']
    total=0;conservative=0;durations=[];steps={};guard_seconds=0
    for path in attempts:
        n=int(path.stem.split('-')[-1])
        exit_record=json.loads((out/f'supervisor-{n:02d}-exit.json').read_text())
        begin=json.loads((out/f'supervisor-{n:02d}-start.json').read_text())
        assert begin['boot']==exit_record['boot'] and begin['binding']==exit_record['binding']==binding
        jp=out/f'intervals-{n:02d}.jsonl'
        rows=[json.loads(l) for l in jp.read_text().splitlines()] if jp.exists() else []
        for interval in accounted_intervals(rows,exit_ns=exit_record['exit_ns']):
            assert interval['spent_before_ns']==total and total<expected['budget_ns']
            total+=interval['duration_ns'];durations.append(interval['duration_ns'])
            conservative+=interval['duration_ns'] if interval['conservative'] else 0
        sp=out/f'steps-{n:02d}.jsonl'
        if sp.exists():
            for line in sp.read_text().splitlines():
                r=json.loads(line);assert math.isfinite(r['loss']) and r['mps_driver_bytes']<plan['mps_limit_bytes'];steps[r['step']]=r
        guard=json.loads((out/f'guard-{n:02d}.json').read_text());guard_seconds+=guard['elapsed_seconds']
        if n==len(attempts):assert guard['status']=='finished' and guard['reason'] is None and exit_record['returncode']==0
    assert total==result['training_ns'] and total>=expected['budget_ns']
    assert result['overshoot_ns']==total-expected['budget_ns']<durations[-1]
    assert guard_seconds<=expected['timeout_seconds']
    assert sum(json.loads(p.read_text())['elapsed_seconds'] for p in PANEL.glob('slot-*/guard-*.json'))<=plan['tranche_seconds']
    canonical=[steps[i] for i in range(1,result['steps']+1)]
    assert all(r['step']==i for i,r in enumerate(canonical,1))
    for endpoint in result['endpoints']:
        receipt=endpoint['weights'];p=Path(receipt['path'])
        assert p.stat().st_size==receipt['size'] and sha(p)==receipt['sha256']
        scored=json.loads((out/f"{endpoint['endpoint']}-validation.json").read_text())
        assert endpoint['ce']==scored['ce'] and endpoint['count']==scored['count']==100416
        assert len(scored['documents'])==108 and [r['document'] for r in scored['documents']]==list(range(108))
        assert math.isclose(sum(r['loss_sum'] for r in scored['documents'])/100416,scored['ce'],rel_tol=0,abs_tol=1e-12)
        assert all(math.isfinite(r['loss_sum']) for r in scored['documents'])
    assert [e['endpoint'] for e in result['endpoints']]==['raw','time_ema']
    natural=[r for r in canonical if r['objective']=='ce']
    for r in canonical:
        objective='pseudo' if expected['recipe']=='native_pseudo_repetition' and r['step']%2==0 else 'ce'
        assert r['objective']==objective
        assert 0<r['valid_loss_targets']<=1024
        if objective=='ce':assert r['valid_loss_targets']==1024
    if slot%2:
        paired=PANEL/f'slot-{slot-1:02d}'
        previous=json.loads((paired/'review.json').read_text())['batch_prefix']
        n=min(len(previous),len(natural));assert previous[:n]==[r['batch_sha256'] for r in natural[:n]]
    if expected['phase']=='confirmation':
        gen=json.loads((out/'generation/result.json').read_text());assert len(gen['rows'])==24
        assert gen['checkpoint']['sha256']==result['endpoints'][1]['weights']['sha256']
        for row in gen['rows']:
            log=[json.loads(l) for l in (out/'generation'/f"{row['id']}.jsonl").read_text().splitlines()]
            assert log[0]['event']=='start' and log[-1]==dict(event='complete',row=row)
            assert [r['token'] for r in log[1:-1]]==row['generated_token_ids']
            assert all(r['position']==i for i,r in enumerate(log[1:-1]))
    return dict(accepted_execution=True,result_sha256=sha(out/'result.json'),binding=binding,training_ns=total,
        guard_seconds_all_attempts=guard_seconds,conservative_failure_charge_ns=conservative,
        time_ema_ce=result['endpoints'][1]['ce'],batch_prefix=[r['batch_sha256'] for r in natural],
        objective_counts={k:sum(r['objective']==k for r in canonical) for k in ['ce','pseudo']},
        valid_loss_targets={k:sum(r['valid_loss_targets'] for r in canonical if r['objective']==k) for k in ['ce','pseudo']},
        saturated_targets=sum(r['saturated_targets'] for r in canonical),test_opened=False)


def main():
    from research.general_rsi_pseudo_worker_v1 import put
    p=argparse.ArgumentParser();p.add_argument('slot',type=int);a=p.parse_args();r=review(a.slot)
    put(PANEL/f'slot-{a.slot:02d}'/'review.json',r)
    print(json.dumps({k:v for k,v in r.items() if k!='batch_prefix'}))
if __name__=='__main__':main()
