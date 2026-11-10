"""Training-batch duplicate census on exact historical sampler prefixes; no model."""
import datetime
import hashlib
import itertools
import json
import time
import numpy as np
from research.general_rsi_data_v1 import GeneralData
from research.general_rsi_transfer_data_v1 import ROOT,read,put,sha
OUT=ROOT/'research/tokenizer-evidence/general-rsi-composer-efficiency-preparation-001'
P=ROOT/'research/tokenizer-evidence/general-rsi-equal-time-001'

def main():
    start=time.monotonic();plan=read(P/'plan.json');assert sha(P/'plan.json')=='4f51d54569d8b649c9609d25d42ba5bbaefdfd390db99b3283d5efc1c2c1165d'
    records=[];bindings={}
    for slot in [4,6,8]:
        rec=plan['slots'][slot];roles=rec['roles'];data=GeneralData(roles['train_sampler'],roles['validation_sampler'],'cpu')
        logs=[]
        for i in [slot,slot+1]:
            path=P/f'slot-{i:02d}/steps-01.jsonl';bindings[str(path.relative_to(ROOT))]=sha(path)
            with path.open() as f:logs.append([json.loads(x) for x in itertools.islice(f,128)])
        for step in range(128):
            x,y=data.batch();h=hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest()
            assert all(log[step]['batch_sha256']==h and log[step]['step']==step+1 for log in logs)
            unique=np.unique(x.numpy());assert x.numel()==1024
            records.append(dict(bundle=rec['bundle'],step=step+1,batch_sha256=h,positions=1024,unique_ids=len(unique),redundant_composer_rows=1024-len(unique)))
    u=np.array([r['unique_ids'] for r in records]);elapsed=time.monotonic()-start;assert elapsed<60
    result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),sampler_sha256=sha(ROOT/'research/general_rsi_data_v1.py'),model_sha256=sha(ROOT/'research/byte_input_model_v6/model.py'),byte_embedding_sha256=sha(ROOT/'research/byte_embedding_v1.py'),plan_sha256=sha(P/'plan.json'),input_logs_sha256=bindings,summary=dict(batches=len(records),positions_per_batch=1024,unique_min=int(u.min()),unique_max=int(u.max()),unique_mean=float(u.mean()),unique_median=float(np.median(u)),mean_redundant_fraction=float(1-u.mean()/1024)),records=records,elapsed_seconds=elapsed,scope='Exact first128 historical paired batches per bundle, hash-checked against both arms. Existing data analysis, not fresh seeds or model training. Redundant composer-row count is not a measured speedup or cost model for transformer/readout/backward. No implementation changed; no GPU work or test access.',test_opened=False)
    put(OUT/'batch-reuse.json',result);print(result['summary'])
if __name__=='__main__':main()
