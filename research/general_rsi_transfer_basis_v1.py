"""Summarize closed equal-time evidence to choose the next question; no inference."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/tokenizer-evidence/general-rsi-equal-time-001'
OUT=ROOT/'research/tokenizer-evidence/general-rsi-transfer-preparation-001/local-basis.json'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    bindings={};runs=[];docs={}
    for i in range(4,10):
        d=P/f'slot-{i:02d}'
        files=[d/'result.json',d/'review.json',d/'steps-01.jsonl',d/'probe-1.json',d/'probe-2.json',d/'raw-validation.json',d/'time_ema-validation.json']
        bindings.update({str(f.relative_to(ROOT)):sha(f) for f in files})
        result=read(d/'result.json');review=read(d/'review.json')
        assert review['accepted_execution'] and review['result_sha256']==sha(d/'result.json')
        steps=[json.loads(line) for line in (d/'steps-01.jsonl').read_text().splitlines()]
        assert len(steps)==result['steps'] and all(math.isfinite(x['loss']) and math.isfinite(x['clip']['pre_norm']) for x in steps)
        probes=[read(d/f'probe-{j}.json') for j in [1,2]]
        raw=read(d/'raw-validation.json');ema=read(d/'time_ema-validation.json');docs[i]=ema['documents']
        ids={x['document'] for x in probes[0]['documents']}
        final=[x for x in raw['documents'] if x['document'] in ids]
        assert sum(x['count'] for x in final)==probes[0]['count']==probes[1]['count']
        runs.append(dict(slot=i,arm=result['slot']['arm'],bundle=result['slot']['bundle'],steps=len(steps),
            clipping_fraction=statistics.mean(x['clip']['coefficient']<1 for x in steps),
            pre_norm_median=statistics.median(x['clip']['pre_norm'] for x in steps),pre_norm_p95=float(np.quantile([x['clip']['pre_norm'] for x in steps],.95)),
            sampled_train_loss_first256_mean=statistics.mean(x['loss'] for x in steps[:256]),sampled_train_loss_last256_mean=statistics.mean(x['loss'] for x in steps[-256:]),
            raw_fixed_eight_doc_probes=[x['ce'] for x in probes]+[sum(x['loss_sum'] for x in final)/sum(x['count'] for x in final)],
            primary_eval_seconds=result['endpoints'][1]['score_seconds'],primary_eval_targets=result['endpoints'][1]['count']))
    inventory=[(x['document'],x['source'],x['count']) for x in docs[4]]
    assert all([(x['document'],x['source'],x['count']) for x in v]==inventory for v in docs.values())
    deltas=np.mean([[a['loss_sum']/a['count']-b['loss_sum']/b['count'] for a,b in zip(docs[n],docs[n+1])] for n in [4,6,8]],axis=0)
    sources={}
    for source in ['fineweb_edu','wikipedia']:
        selected=[float(delta) for inv,delta in zip(inventory,deltas) if inv[1]==source]
        sources[source]=dict(documents=len(selected),native_lower_fraction=statistics.mean(x<0 for x in selected),unweighted_doc_delta_mean=statistics.mean(selected),unweighted_doc_delta_min=min(selected),unweighted_doc_delta_max=max(selected))
    seconds_per_target={arm:max(x['primary_eval_seconds']/x['primary_eval_targets'] for x in runs if x['arm']==arm) for arm in ['native','raw_bytes']}
    out=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind='existing_data_campaign_boundary_analysis',runs=runs,source_document_heterogeneity=sources,
        prospective_inference_cost=dict(targets_per_model=307200,models_per_arm=3,seconds_per_target_upper_observed=seconds_per_target,estimated_scoring_seconds=sum(seconds_per_target.values())*3*307200,scope='Linear extrapolation from same context128 scorer; excludes load, preprocessing, deduplication, storage and retries. Not admission.'),
        limits='Fixed8-document RAW probes are diagnostics. Train means use different sampled prefixes, not comparable controlled quality endpoints. Document means are unweighted diagnostics, not the frozen token-weighted acceptance statistic.',evidence_sha256=bindings,analysis_sha256=sha(Path(__file__)),test_opened=False,new_model_evaluations=0)
    with OUT.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['evidence_sha256']},indent=2))
if __name__=='__main__':main()
