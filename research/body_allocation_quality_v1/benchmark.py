"""Complete fresh-panel serving measurements; no seed/endpoint selection."""
import gc,statistics,time
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import require
from research.body_allocation_model_v1.model import GPT,load_weights
from research.body_allocation_worker_v1.validation import verify_artifact
from research.body_allocation_worker_v1.execution import checked

def measure(accepted):
    rows=[]
    for row in accepted:
        worker=checked(row['worker']);cfg=OmegaConf.create(worker['config']);verify_artifact(worker['h200'])
        with torch.random.fork_rng(devices=[]):
            model=GPT(cfg.model).to('mps').eval();h=torch.load(worker['h200']['path'],map_location='cpu',weights_only=True);load_weights(model,h);del h
            cache_bytes=[];measures=[]
            with torch.no_grad():
                for batch in (1,8):
                    x=(torch.arange(batch*128).reshape(batch,128)%400+1).to('mps');c=model.caches(batch)
                    bytes_item=sum(t.numel()*t.element_size() for item in c for t in item.buffers())//batch
                    full_times=[];decode_times=[]
                    for repeat in range(7):
                        for item in c:item.reset()
                        torch.mps.synchronize();start=time.perf_counter();full=model(x)[0];torch.mps.synchronize();full_time=time.perf_counter()-start
                        prefill=model(x[:,:96],cache=c)[0];del prefill;torch.mps.synchronize();start=time.perf_counter()
                        for index in range(96,128):
                            last=model(x[:,index:index+1],cache=c,start_pos=index)[0]
                            if repeat==0:torch.testing.assert_close(full[:,index:index+1],last,rtol=3e-4,atol=3e-5)
                        torch.mps.synchronize();decode_time=time.perf_counter()-start
                        if repeat>=2:full_times.append(full_time);decode_times.append(decode_time)
                        del full,last
                    measures.append(dict(batch=batch,cache_bytes_per_item=bytes_item,full128_seconds=full_times,decode32_after_prefill96_seconds=decode_times,
                        median_full128_seconds=statistics.median(full_times),median_decode32_seconds=statistics.median(decode_times)))
                    cache_bytes.append(bytes_item);del c,x
            rows.append(dict(arm=row['arm'],seed=cfg.training.seed,worker=row['worker'],h200=worker['h200'],measurements=measures,
                timing_scope='One resident model, 2 warm repeats and 5 measured repeats; init/loading/cache allocation excluded. Descriptive local timing, not a quality/adoption gate.'))
            del model;gc.collect();torch.mps.empty_cache();verify_artifact(worker['h200'])
    require(len(rows)==9,'Complete three fresh triplets required');return rows
