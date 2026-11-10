"""Descriptive token-frequency coverage of frozen transfer excerpts; no loss gates."""
import datetime
import numpy as np
from research.general_rsi_transfer_data_v1 import ROOT,P,D,read,put,sha,contract

def main():
    contract();dataset=read(P/'dataset.json');old=ROOT/'data/raw/rsi-general-10m-v1';manifest=read(old/'manifest.json')
    assert sha(old/'train.bin')==manifest['files']['train.bin']['sha256']
    assert sha(D/'excerpts.bin')==dataset['payloads'][0]['sha256']
    counts=np.bincount(np.fromfile(old/'train.bin',dtype='<u2'),minlength=50257)
    ids=np.fromfile(D/'excerpts.bin',dtype='<u2');domains={}
    for domain in ['pg19_books','python_library_rst']:
        targets=np.concatenate([ids[r['offset']+1:r['offset']+r['tokens']] for r in dataset['documents'] if r['source']==domain]);seen=counts[targets]
        buckets={'unseen':seen==0,'seen_1_to_9':(seen>=1)&(seen<=9),'seen_10_to_99':(seen>=10)&(seen<=99),'seen_at_least100':seen>=100}
        domains[domain]={'targets':len(targets),'distinct_target_types':len(np.unique(targets)),'buckets':{k:{'count':int(m.sum()),'fraction':float(m.mean())} for k,m in buckets.items()}}
        assert sum(v['count'] for v in domains[domain]['buckets'].values())==102400
    out={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(__file__),'dataset_sha256':sha(P/'dataset.json'),'training_tokens_sha256':sha(old/'train.bin'),'domains':domains,'scope':'Post-launch descriptive diagnostic only. Full-corpus token counts do not measure actual sampled training exposure. Subword frequency is not semantic novelty or document contamination. Does not modify frozen transfer gates.','new_model_evaluations':0,'test_opened':False}
    put(P/'token-coverage.json',out);print(out['domains'])
if __name__=='__main__':main()
