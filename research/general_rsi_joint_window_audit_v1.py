"""Exploratory localization of saved endpoint losses; no new model scoring."""
import json,math,statistics
from pathlib import Path
import numpy as np
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put
from research.general_rsi_joint_diagnostic_v1 import OUT,ARMS

def main():
    p=json.loads((OUT/'plan.json').read_text());A=OUT/'attempt-01';assert (A/'result.json').exists()
    data=ROOT/'data/raw/rsi-general-10m-v1';manifest=json.loads((data/'manifest.json').read_text())
    assert sha(data/'train.bin')==manifest['files']['train.bin']['sha256'];values=np.fromfile(data/'train.bin',dtype='<u2')
    rows=[]
    for c in p['parents']:
        slot=c['parent']['slot'];initial=json.loads((A/f'p{slot:02d}-initial.json').read_text())['scores']['natural']['window_loss_sums']
        scores={arm:json.loads((A/f'p{slot:02d}-{arm}-n100.json').read_text())['scores']['natural']['window_loss_sums'] for arm in ARMS}
        for i,w in enumerate(c['inputs']['evaluation_windows']):
            seq=values[w['start']:w['start']+129].tolist();seen=set();repeated=0
            for end in range(2,129):
                tri=tuple(seq[end-2:end+1]);repeated+=tri in seen;seen.add(tri)
            rows.append(dict(parent_slot=slot,window=i,document=w['document'],source=w['source'],start=w['start'],
                repeated_trigram_endpoints=repeated,contains_repeated_trigram=bool(repeated),
                initial_ce=initial[i]/128,final_ce={a:scores[a][i]/128 for a in ARMS},
                joint_minus_full=(scores['ce_pseudo_mean'][i]-scores['ce_full'][i])/128,
                joint_minus_half=(scores['ce_pseudo_mean'][i]-scores['ce_half'][i])/128,
                full_drift=(scores['ce_full'][i]-initial[i])/128))
    groups=[]
    for slot in [1,3,5]:
        local=[r for r in rows if r['parent_slot']==slot]
        for field in ['source','contains_repeated_trigram']:
            for value in sorted({r[field] for r in local},key=str):
                selected=[r for r in local if r[field]==value]
                groups.append(dict(parent_slot=slot,field=field,value=value,windows=len(selected),targets=128*len(selected),
                    means={k:statistics.mean(r[k] for r in selected) for k in ['joint_minus_full','joint_minus_half','full_drift']},
                    joint_minus_full_positive_windows=sum(r['joint_minus_full']>0 for r in selected)))
    put(OUT/'exploratory-window-audit.json',dict(kind='posthoc_saved_window_loss_localization',source_sha256=sha(__file__),
        plan_sha256=sha(OUT/'plan.json'),result_sha256=sha(A/'result.json'),train_sha256=sha(data/'train.bin'),rows=rows,groups=groups,
        question='Does joint-gradient endpoint prediction damage occur only in windows with an internally repeated trigram or also elsewhere, and in both sampled sources?',
        limitations='Posthoc descriptive groups on192 prior-training windows. Exact token trigram repetition is not a semantic redundancy or necessary-repetition label. Window losses cannot attribute individual token effects. Unequal small source groups are not population estimates. Frozen mechanism criterion unchanged.',
        model_scores_computed=0,test_opened=False,quality_claim=False))
    print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
