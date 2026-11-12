"""Complete-case descriptive analysis for the frozen accumulated-state diagnostic."""
import json
import math
from pathlib import Path
from research.general_rsi_pseudo_accumulated_v1 import OUT, ROOT
from research.general_rsi_pseudo_update_diagnostic_v1 import sha, put


def stats(values):
    assert values and all(math.isfinite(v) for v in values)
    return dict(mean=math.fsum(values)/len(values), minimum=min(values),maximum=max(values),
                positive=sum(v>0 for v in values),negative=sum(v<0 for v in values),zero=sum(v==0 for v in values),values=values)


def main():
    attempt=OUT/'attempt-01';receipts={}
    def read(path):
        receipts[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text())
    plan=read(OUT/'plan.json');admission=read(OUT/'admission.json')
    result=read(attempt/'result.json');guard=read(attempt/'guard.json')
    assert guard['status']=='finished' and result['plan_sha256']==admission['plan_sha256']==sha(OUT/'plan.json')
    assert result['total_seconds']<plan['timeout_seconds']
    assert result['peak_mps_boundary_bytes']<plan['mps_limit_bytes']
    assert guard['observation']['peak_rss_bytes']<plan['rss_limit_bytes']
    assert guard['observation']['free_bytes']>=plan['reserve_bytes']
    parents=[];all_updates=0;seconds=0.;counts={k:0 for k in ('ce','zero','pseudo')}
    for config in plan['parents']:
        slot=config['parent']['slot'];initial=read(attempt/f'p{slot:02d}-initial.json')
        arms={};first_ce_loss=None;first_probe=None;first_optimizer=None
        for arm in config['arm_order']:
            endpoint=read(attempt/f'p{slot:02d}-{arm}-result.json')
            assert endpoint['restored_state_sha256']==initial['common_sha256']
            path=attempt/f'p{slot:02d}-{arm}-steps.jsonl';receipts[str(path.relative_to(ROOT))]=sha(path)
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            expected=[]
            for n in range(1,101):
                expected.append((n,'ce'))
                if n<100 and arm!='ce':expected.append((n,'zero' if arm=='ce_zero' else 'pseudo'))
            assert [(r['natural_update'],r['objective']) for r in rows]==expected
            for index,r in enumerate(rows,1):
                assert r['optimizer_steps']==index and r['seconds']>0
                n=r['natural_update'];objective=r['objective'];counts[objective]+=1
                if objective=='ce':assert r['input_sha256']==config['inputs']['natural'][n-1]['sha256']
                if objective=='pseudo':assert r['input_sha256']==config['inputs']['synthetic'][n-1]['sha256']
                if objective=='zero':assert r['input_sha256'] is None and r['loss'] is None and r['clip'] is None
                else:
                    assert math.isfinite(r['loss']) and r['clip']['post_norm']<=1.00001
            if first_ce_loss is None:first_ce_loss=rows[0]['loss']
            else:assert rows[0]['loss']==first_ce_loss
            assert len(rows)==endpoint['optimizer_steps']==(100 if arm=='ce' else 199)
            assert math.isclose(math.fsum(r['seconds'] for r in rows),endpoint['synchronized_update_seconds'],abs_tol=1e-8)
            all_updates+=len(rows);seconds+=endpoint['synchronized_update_seconds']
            probes={}
            for n in plan['probe_natural_updates']:
                probe=read(attempt/f'p{slot:02d}-{arm}-n{n:03d}.json')
                assert (probe['parent_slot'],probe['arm'],probe['natural_updates'])==(slot,arm,n)
                assert probe['optimizer_steps']==(n if arm=='ce' else 2*n-1)
                assert probe['evaluation']['targets']==8192 and len(probe['evaluation']['window_loss_sums'])==64
                assert probe['delta_from_parent']==probe['evaluation']['ce']-initial['evaluation']['ce']
                if n==1:
                    if first_probe is None:first_probe=probe['evaluation'];first_optimizer=probe['optimizer']
                    else:assert probe['evaluation']==first_probe and probe['optimizer']==first_optimizer
                probes[n]=probe
            assert probes[100]['evaluation']['ce']==endpoint['final_ce']
            natural_losses=[r['loss'] for r in rows if r['objective']=='ce']
            arms[arm]=dict(endpoint=endpoint,probes=probes,natural_losses=natural_losses,
                clipped_updates={o:sum(r['clip']['coefficient']<1 for r in rows if r['objective']==o) for o in ('ce','pseudo')},
                saturated_pseudo_targets=sum(r['pseudo_info']['saturated'] for r in rows if r['pseudo_info']))
        curves=[]
        for n in plan['probe_natural_updates']:
            scores={arm:a['probes'][n]['evaluation']['ce'] for arm,a in arms.items()}
            curves.append(dict(natural_updates=n,ce=scores,
                contrasts={c:scores[c.split('-')[0]]-scores[c.split('-')[1]] for c in plan['contrasts']}))
        loss_differences={}
        for c in plan['contrasts']:
            left,right=c.split('-')
            delta=[a-b for a,b in zip(arms[left]['natural_losses'],arms[right]['natural_losses'])]
            loss_differences[c]=dict(all_100=stats(delta),blocks=[dict(first=i+1,last=i+10,**stats(delta[i:i+10])) for i in range(0,100,10)])
        parents.append(dict(parent_slot=slot,initial_ce=initial['evaluation']['ce'],curves=curves,
                            natural_training_loss_contrasts=loss_differences,arms=arms))
    assert all_updates==result['updates']==1494 and counts==dict(ce=900,zero=297,pseudo=297)
    assert math.isclose(seconds,result['synchronized_update_seconds'],abs_tol=1e-8)
    pooled=[]
    for j,n in enumerate(plan['probe_natural_updates']):
        pooled.append(dict(natural_updates=n,contrasts={c:stats([p['curves'][j]['contrasts'][c] for p in parents]) for c in plan['contrasts']}))
    summary=dict(kind='pseudo_accumulated_state_analysis_v1',sources=receipts,source_sha256=sha(__file__),
        complete=True,parents=parents,descriptive_pooled_curves=pooled,counts=counts,
        total_seconds=result['total_seconds'],synchronized_update_seconds=seconds,
        peak_mps_boundary_bytes=result['peak_mps_boundary_bytes'],guard=guard,
        validation_scored=False,test_opened=False,quality_decision_changed=False,
        limitations='Three fresh paired prefixes of selected parents, fixed training windows. No quality confirmation, confidence interval, coefficient selection or model adoption. Zero contrast combines momentum/decay and their trajectory interactions; no universal causal share.')
    put(OUT/'analysis.json',summary)
    print(json.dumps(dict(curves=pooled,total_seconds=summary['total_seconds'],update_seconds=seconds),indent=2))


if __name__=='__main__':main()
