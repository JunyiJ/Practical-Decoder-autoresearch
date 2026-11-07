"""Complete fresh-panel serving measurements; no seed/endpoint selection."""
import gc,statistics,time
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import require,plain
from research.readout_mixture_model_v1.model import GPT
from research.readout_mixture_resume_v1.checkpoint import load_export
from research.readout_mixture_resume_v1.data import model_spec
from .validation import verify_artifact,cache_check
from .contract import checked

def measure(accepted):
    rows=[];saved_mps_rng=torch.mps.get_rng_state().clone()
    for row in accepted:
        worker=checked(row['worker']);cfg=OmegaConf.create(worker['config']);verify_artifact(worker['h200'])
        with torch.random.fork_rng(devices=[]):
            model=GPT(model_spec(cfg),initializer_seed=cfg.initializer_seed).to('mps').eval();h=torch.load(worker['h200']['path'],map_location='cpu',weights_only=True);load_export(model,h,cfg);del h
            cache_bytes=[];measures=[];cache_checks=cache_check(model)
            with torch.no_grad():
                for batch in (1,8):
                    x=(torch.arange(batch*128).reshape(batch,128)%400+1).to('mps');c=model.caches(batch)
                    bytes_item=sum(t.numel()*t.element_size() for item in c.layers for t in item.buffers())//batch
                    full_times=[];decode_times=[]
                    for repeat in range(7):
                        c.reset()
                        torch.mps.synchronize();start=time.perf_counter();full=model(x).log_probs();torch.mps.synchronize();full_time=time.perf_counter()-start
                        prefill=model(x[:,:96],cache=c).log_probs();del prefill;torch.mps.synchronize();start=time.perf_counter()
                        for index in range(96,128):
                            last=model(x[:,index:index+1],cache=c,start_pos=index).log_probs()
                            if repeat==0:torch.testing.assert_close(full[:,index:index+1],last,rtol=3e-4,atol=3e-5)
                        torch.mps.synchronize();decode_time=time.perf_counter()-start
                        if repeat>=2:full_times.append(full_time);decode_times.append(decode_time)
                        del full,last
                    measures.append(dict(batch=batch,cache_bytes_per_item=bytes_item,full128_seconds=full_times,decode32_after_prefill96_seconds=decode_times,
                        median_full128_seconds=statistics.median(full_times),median_decode32_seconds=statistics.median(decode_times)))
                    cache_bytes.append(bytes_item);del c,x
            rows.append(dict(arm=row['arm'],seed=cfg.training.seed,worker=row['worker'],h200=worker['h200'],measurements=measures,cache_reset_checks=cache_checks,
                timing_scope='One resident model, 2 warm repeats and 5 measured repeats; init/loading/cache allocation excluded. Descriptive local timing, not a quality/adoption gate.'))
            del model;gc.collect();torch.mps.empty_cache();verify_artifact(worker['h200'])
    torch.mps.set_rng_state(saved_mps_rng)
    require(len(rows)==9,'Complete three fresh triples required');return rows
