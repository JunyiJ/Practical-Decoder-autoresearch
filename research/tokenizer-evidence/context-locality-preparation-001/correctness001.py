"""New locality-policy oracle/cache proof; no training or closed-command replay."""
import argparse,gc,json
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_worker_v1.execution import checked,sources
from research.body_allocation_worker_v1.validation import verify_artifact
from research.mla_context_extension_v1.model import GPT,config,import_original_h200
from research.mla_context_locality_v1.session import Session
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;B=ROOT/'research/tokenizer-evidence/body-allocation-preparation-001'
p=argparse.ArgumentParser();p.add_argument('device',choices=['cpu','mps']);a=p.parse_args();torch.set_num_threads(4)
OUT=E/f'correctness001-{a.device}.json';GUARD=E/f'correctness001-{a.device}-guard.json'
def inventory():
    result=sources()
    for ns in ('mla_context_extension_v1','mla_context_locality_v1'):
        for f in sorted((ROOT/'research'/ns).glob('*.py')):result[str(f.relative_to(ROOT))]=reference(f)['sha256']
    return result
def rejected(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('Invalid locality condition accepted')
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();require(not OUT.exists() and not GUARD.exists(),'Closed proof cannot retry')
    closure=checked(reference(B/'quality001-closure001.json'));row=next(r for r in closure['trajectories'] if r['slot']==9)
    worker=checked(row['worker']);ref=worker['h200'];verify_artifact(ref);pinned=inventory()
    guard=StrictMpsGuard(GUARD,ROOT) if a.device=='mps' else CpuPrerequisiteGuard(GUARD,ROOT,timeout_seconds=120,poll_seconds=.25)
    with guard,torch.random.fork_rng(devices=[]):
        m=GPT(config()).to(a.device);saved=torch.load(ref['path'],map_location='cpu',weights_only=True)
        import_original_h200(m,saved,ref['sha256']);del saved;s=Session(m)
        stamp=m.stamp();buffers=m.buffer_stamp();rejected(lambda:s(torch.ones(1,1,dtype=torch.int64,device=a.device)))
        records=[]
        with torch.no_grad():
            # Independent ordinary-causal attention on literal cropped windows.
            block=m.transformer.h[0];xx=(torch.arange(256*256,device=a.device).reshape(1,256,256)%101).float()/101-.5
            y=s.attention(block.attn,xx,block.freq_cos,block.freq_sin,None,0);oracle_errors=[]
            for i in (0,127,128,255):
                lo=max(0,i-127);oracle=block.attn(xx[:,lo:i+1],freq_cos=block.freq_cos[lo:i+1],freq_sin=block.freq_sin[lo:i+1])[0][:,-1]
                torch.testing.assert_close(y[:,i],oracle,rtol=3e-4,atol=3e-5);oracle_errors.append(float((y[:,i]-oracle).abs().max()))
            expected=torch.tensor([[0<=i-j<128 for j in range(256)] for i in range(256)],device=a.device)
            require(torch.equal(s.mask,expected),'Literal trained-range causal window differs');del xx,y,oracle,expected
            for batch in (1,8):
                x=(torch.arange(batch*256).reshape(batch,256)%400+1).to(a.device)
                native=s(x[:,:128]);oracle=m(x[:,:128]);torch.testing.assert_close(native,oracle,rtol=3e-4,atol=3e-5)
                native_error=float((native-oracle).abs().max());del native,oracle
                full=s(x);cache=s.caches(batch)
                replay=torch.cat([s(x[:,:127],cache=cache),s(x[:,127:128],cache=cache,start_pos=127),s(x[:,128:],cache=cache,start_pos=128)],1)
                torch.testing.assert_close(full,replay,rtol=3e-4,atol=3e-5);cache_error=float((full-replay).abs().max());del replay
                for c in cache:c.reset()
                replay=s(x,cache=cache);torch.testing.assert_close(full,replay,rtol=3e-4,atol=3e-5);del replay
                changed=x.clone();changed[:,128:]=(changed[:,128:]+7)%50257;future=s(changed)
                torch.testing.assert_close(full[:,:128],future[:,:128],rtol=3e-4,atol=3e-5);del changed,future
                for c in cache:c.reset()
                rejected(lambda:m(x,cache=cache));require(all(c.position==0 for c in cache),'Foreign rejection must precede updates')
                foreign=m.caches(batch);rejected(lambda:s(x,cache=foreign));require(all(c.position==0 for c in foreign),'Global cache mutated')
                cache[-1].signature['checkpoint_sha256']='0'*64;rejected(lambda:s(x,cache=cache));require(all(c.position==0 for c in cache),'Last-layer forgery mutated prefix')
                cache=s.caches(batch);rejected(lambda:s(torch.zeros(batch,257,dtype=torch.int64,device=a.device),cache=cache));require(all(c.position==0 for c in cache),'Overflow mutated cache')
                cache[0].c_cache=next(m.parameters()).detach().reshape(-1)[:batch*256*64].reshape(batch,256,64)
                rejected(lambda:s(x,cache=cache));require(all(c.position==0 for c in cache),'Alias mutated cache')
                records.append(dict(batch=batch,native128_max_difference=native_error,local256_cache_max_difference=cache_error,
                    full_cache_reset_causality=True,foreign_policy_checkpoint_overflow_alias_rejection=True,logical_cache_bytes_per_item=393216))
                del full,cache,foreign,x;gc.collect()
                if a.device=='mps':torch.mps.empty_cache()
            require(m.stamp()==stamp and m.buffer_stamp()==buffers,'Original model mutated')
            s.mask[255,0]=True;rejected(lambda:s.validate());require(m.stamp()==stamp and m.buffer_stamp()==buffers,'Mask proof mutated model')
        del s,m;gc.collect();require(inventory()==pinned,'Source changed during proof');verify_artifact(ref)
    result=dict(status='passed',device=a.device,sources=pinned,source=reference(__file__),original_h200=ref,records=records,
        cropped_window_attention_oracle_max_differences=oracle_errors,original_weights_and_buffers_unchanged=True,mask_mutation_rejected=True,
        new_scientific_seed=False,training=False,adoption_permitted=False,holdout_scored=False,guard=reference(GUARD))
    payload=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode();require(len(payload)<=512*1024,'Bounded output')
    with OUT.open('xb') as f:f.write(payload)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','source','original_h200')}))
