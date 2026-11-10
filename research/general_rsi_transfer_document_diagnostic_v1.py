"""Post-result descriptive associations only; no new model evaluations or gates."""
import datetime
import numpy as np
from research.general_rsi_transfer_data_v1 import ROOT,P,D,read,put,sha,contract

def main():
    contract();dataset=read(P/'dataset.json');decision=read(P/'decision.json')
    train=ROOT/'data/raw/rsi-general-10m-v1/train.bin';manifest=read(train.parent/'manifest.json')
    assert sha(train)==manifest['files']['train.bin']['sha256']
    assert sha(D/'excerpts.bin')==dataset['payloads'][0]['sha256']
    counts=np.bincount(np.fromfile(train,dtype='<u2'),minlength=50257)
    tokens=np.fromfile(D/'excerpts.bin',dtype='<u2')
    results={}
    for slot in range(4,10):
        f=P/f'evaluation-attempt-01/slot-{slot:02d}.json'
        assert sha(f)==decision['evidence_sha256'][str(f.relative_to(ROOT))]
        results[slot]=read(f)['score']['documents']
    rows=[]
    for i,doc in enumerate(dataset['documents']):
        target=tokens[doc['offset']+1:doc['offset']+doc['tokens']];seen=counts[target]
        ce={}
        for arm,slots in [('native',[4,6,8]),('raw_bytes',[5,7,9])]:
            scores=[]
            for s in slots:
                r=results[s][i];assert r['document']==i and r['source']==doc['source'] and r['count']==len(target)
                scores.append(r['loss_sum']/r['count'])
            ce[arm]=float(np.mean(scores))
        rows.append(dict(document=i,source=doc['source'],name=doc['name'],count=len(target),unseen_fraction=float(np.mean(seen==0)),below10_fraction=float(np.mean(seen<10)),mean_ce=ce,native_minus_byte=ce['native']-ce['raw_bytes']))
    summary={}
    for domain in ['pg19_books','python_library_rst']:
        rr=[r for r in rows if r['source']==domain];delta=np.array([r['native_minus_byte'] for r in rr]);unseen=np.array([r['unseen_fraction'] for r in rr])
        assert len(rr)==32 and abs(delta.mean()-decision['directions']['native']['domains'][domain]['mean_delta'])<1e-12
        summary[domain]={'parents':32,'parents_favor_native':int((delta<0).sum()),'parents_favor_byte':int((delta>0).sum()),'ties':int((delta==0).sum()),'delta_min':float(delta.min()),'delta_max':float(delta.max()),'pearson_unseen_fraction_vs_native_minus_byte':float(np.corrcoef(unseen,delta)[0,1]) if np.std(unseen)>0 and np.std(delta)>0 else None}
    put(P/'document-diagnostic.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),decision_sha256=sha(P/'decision.json'),dataset_sha256=sha(P/'dataset.json'),train_sha256=sha(train),summary=summary,documents=rows,scope='Post-result descriptive association, no hypothesis test, causal attribution or acceptance gate. Correlations use 32 selected parents per domain and average the same three model pairs; topic, markup, token identity and dependence may confound them. Full-corpus counts are not actual sampled training exposure. No new model evaluation; test unopened.',test_opened=False))
    print(summary)
if __name__=='__main__':main()
