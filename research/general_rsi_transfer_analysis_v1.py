"""Frozen symmetric two-domain transfer gates; document-paired bootstrap."""
import argparse
import datetime
import json
from pathlib import Path
import numpy as np
from research.general_rsi_transfer_data_v1 import ROOT,P,BINDING,read,put,sha,contract

def draws_for_domains(deltas,counts,seed,replicates):
    rng=np.random.Generator(np.random.PCG64(seed));out={}
    for domain in ['pg19_books','python_library_rst']:
        sums=np.mean(np.asarray(deltas[domain],dtype=np.float64),axis=0)
        n=np.asarray(counts[domain]);assert len(sums)==len(n) and np.all(n>0)
        ids=rng.integers(0,len(n),size=(replicates,len(n)))
        out[domain]=sums[ids].sum(axis=1)/n[ids].sum(axis=1)
    return out

def qualify():
    ds={d:[[-2.,-6.]]*3 for d in ['pg19_books','python_library_rst']};ns={d:[1,3] for d in ds}
    r=draws_for_domains(ds,ns,62111000999,20000)
    assert all(np.all(x==-2.) for x in r.values())
    r=draws_for_domains({d:-np.array(x) for d,x in ds.items()},ns,62111000999,20000)
    assert all(np.all(x==2.) and np.quantile(x,.9975,method='linear')==2. for x in r.values())
    put(P/'analysis-qualification.json',dict(passed=True,analysis_sha256=sha(__file__),checks=['two-domain unequal-count constant -2/+2 per-token fixtures;20000 replicates','adjusted upper0.9975 quantile preserves sign'],test_opened=False))

def analyze(attempt):
    plan=contract();dataset=read(P/'dataset.json');out=P/f'evaluation-attempt-{attempt:02d}'
    complete=read(out/'complete.json');assert complete['protocol_sha256']==BINDING and complete['slots']==[4,5,6,7,8,9]
    assert read(P/'analysis-qualification.json')['analysis_sha256']==sha(__file__)
    admission=read(P/'scoring-admission.json');assert admission['analysis_sha256']==sha(__file__)
    assert admission['dataset_sha256']==sha(P/'dataset.json')
    results={};bindings={}
    inventory=[(i,x['source'],x['tokens']-1) for i,x in enumerate(dataset['documents'])]
    for rec in plan['models']:
        f=out/f"slot-{rec['slot']:02d}.json";r=read(f);bindings[str(f.relative_to(ROOT))]=sha(f)
        assert r['checkpoint']==rec and sha(rec['path'])==rec['sha256']
        assert r['protocol_sha256']==BINDING and r['dataset_sha256']==sha(P/'dataset.json') and r['worker_sha256']==admission['worker_sha256'] and not r['test_opened']
        score=r['score'];assert [(x['document'],x['source'],x['count']) for x in score['documents']]==inventory
        assert score['count']==204800 and abs(sum(x['loss_sum'] for x in score['documents'])/204800-score['ce'])<1e-12
        for domain in plan['domains']:
            rows=[x for x in score['documents'] if x['source']==domain]
            assert len(rows)==32 and sum(x['count'] for x in rows)==102400
            assert abs(sum(x['loss_sum'] for x in rows)/102400-score['sources'][domain]['ce'])<1e-12
        results[rec['slot']]=score
    deltas={d:[] for d in plan['domains']};counts={d:[] for d in plan['domains']};pairs=[]
    for n,b in [(4,5),(6,7),(8,9)]:
        row={'bundle':(n-2)//2,'native_slot':n,'byte_slot':b,'domain_deltas_native_minus_byte':{}}
        for domain in plan['domains']:
            nr=[x for x in results[n]['documents'] if x['source']==domain];br=[x for x in results[b]['documents'] if x['source']==domain]
            deltas[domain].append([x['loss_sum']-y['loss_sum'] for x,y in zip(nr,br)])
            counts[domain]=[x['count'] for x in nr]
            row['domain_deltas_native_minus_byte'][domain]=results[n]['sources'][domain]['ce']-results[b]['sources'][domain]['ce']
        row['macro_delta_native_minus_byte']=float(np.mean(list(row['domain_deltas_native_minus_byte'].values())));pairs.append(row)
    bc=plan['analysis']['bootstrap'];draws=draws_for_domains(deltas,counts,bc['seed'],bc['replicates'])
    means={d:float(np.mean([r['domain_deltas_native_minus_byte'][d] for r in pairs])) for d in plan['domains']}
    directions={}
    for arm,sign in [('native',1),('raw_bytes',-1)]:
        domains={}
        for d in plan['domains']:
            upper=float(np.quantile(sign*draws[d],bc['domain_upper_quantile'],method='linear'))
            gates={'all_three_pairs_favor':all(sign*r['domain_deltas_native_minus_byte'][d]<0 for r in pairs),'mean_at_most_minus_0_02':sign*means[d]<=-.02,'adjusted_upper_negative':upper<0}
            domains[d]={'mean_delta':sign*means[d],'one_sided_adjusted_upper':upper,'gates':gates,'accepted':all(gates.values())}
        macro=sign*(draws[plan['domains'][0]]+draws[plan['domains'][1]])/2
        directions[arm]={'domains':domains,'accepted_both_domains':all(x['accepted'] for x in domains.values()),'macro_mean_delta':sign*float(np.mean(list(means.values()))),'macro_upper_99_descriptive':float(np.quantile(macro,.99,method='linear'))}
    winners=[a for a,d in directions.items() if d['accepted_both_domains']];assert len(winners)<=1
    guards={f.name:read(f) for f in P.glob('*-guard-*.json')};spent=sum(x['elapsed_seconds'] for x in guards.values());assert spent<=7200
    result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=BINDING,dataset_sha256=sha(P/'dataset.json'),analysis_sha256=sha(__file__),attempt=attempt,pairs=pairs,
        mean_ce={arm:{d:float(np.mean([results[i]['sources'][d]['ce'] for i in slots])) for d in plan['domains']} for arm,slots in [('native',[4,6,8]),('raw_bytes',[5,7,9])]},directions=directions,accepted_transfer_arm=winners[0] if winners else None,guard_seconds=spent,guard_statuses={k:v['status'] for k,v in guards.items()},bootstrap=bc,evidence_sha256=bindings,test_opened=False,scope='Two selected excerpt domains, fixed six checkpoints and three seed pairs. No retraining, full-source benchmark, population-seed confidence or general generation claim.')
    put(P/'decision.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['evidence_sha256','bootstrap']},indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['qualify','analyze']);a.add_argument('--attempt',type=int,default=1);args=a.parse_args();qualify() if args.action=='qualify' else analyze(args.attempt)
