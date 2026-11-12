"""Post-hoc mechanism diagnostic using retained logs only; no new selection gate."""
import datetime
import json
import numpy as np
from research.general_rsi_fp32_quality_analysis_v1 import PANEL, ROOT, BINDING, PAIRS, read, sha
from research.general_rsi_fp32_quality_generation_analysis_v1 import put


def main():
    assert sha(PANEL/'plan.json')==BINDING
    rows={}; evidence={}; summaries={}
    for slot in range(6):
        p=PANEL/f'slot-{slot:02d}';r=read(p/'result.json');v=read(p/'review.json')
        assert v['accepted_execution'] and v['result_sha256']==sha(p/'result.json')
        log=p/'steps-01.jsonl';rows[slot]=[json.loads(x) for x in log.read_text().splitlines()]
        assert len(rows[slot])==10240 and [x['step'] for x in rows[slot]]==list(range(1,10241))
        evidence[str(log.relative_to(ROOT))]=sha(log)
        norms=np.array([x['clip']['pre_norm'] for x in rows[slot]]);coeff=np.array([x['clip']['coefficient'] for x in rows[slot]])
        assert np.isfinite(norms).all() and np.isfinite(coeff).all()
        summaries[slot]=dict(clipped_fraction=float(np.mean(coeff<1)),pre_norm_median=float(np.median(norms)),pre_norm_p99=float(np.quantile(norms,.99)),mean_update_seconds=float(np.mean([x['duration_ns'] for x in rows[slot]]))/1e9)
    pairs=[]
    for bundle,(a,b) in enumerate(PAIRS,1):
        assert all(x['batch_sha256']==y['batch_sha256'] for x,y in zip(rows[a],rows[b]))
        d=np.array([y['loss']-x['loss'] for x,y in zip(rows[a],rows[b])])
        chunks=[]
        for lo,hi in [(1,3413),(3414,6826),(6827,10240)]:
            x=d[lo-1:hi];chunks.append(dict(first_step=lo,last_step=hi,mean_train_loss_delta=float(x.mean())))
        probes={}
        for step in [3413,6826]:
            pa=PANEL/f'slot-{a:02d}/probe-{step}.json';pb=PANEL/f'slot-{b:02d}/probe-{step}.json'
            x,y=read(pa),read(pb)
            assert [(z['document'],z['source'],z['count']) for z in x['documents']]==[(z['document'],z['source'],z['count']) for z in y['documents']]
            probes[step]=dict(targets=x['count'],existing_ce=x['ce'],fp32_ce=y['ce'],delta=y['ce']-x['ce'])
            for p in [pa,pb]:evidence[str(p.relative_to(ROOT))]=sha(p)
        averaging={}
        for label,slot in [('existing_byte',a),('byte_composer_fp32',b)]:
            ep={x['endpoint']:x for x in read(PANEL/f'slot-{slot:02d}/result.json')['endpoints']}
            averaging[label]=ep['raw']['ce']-ep['h200']['ce']
        pairs.append(dict(bundle=bundle,training_loss_delta_windows=chunks,raw_eight_document_probes=probes,raw_minus_h200_ce=averaging))
    out=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=BINDING,source_sha256=sha(__file__),post_hoc=True,scientific_decision_unchanged=True,pairs=pairs,slots=summaries,evidence_sha256=evidence,limits='Training losses are correlated optimization diagnostics. Eight-document intermediate RAW probes are not full validation endpoints and cannot be compared directly to full H200 CE. No new gate, seed selection or statistical significance claim.',test_opened=False)
    put(PANEL/'history-analysis.json',out);print(json.dumps({k:v for k,v in out.items() if k in ['pairs','slots']},indent=2))

if __name__=='__main__':main()
