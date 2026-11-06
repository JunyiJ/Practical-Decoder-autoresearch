"""Separate MPS oracle proof plus actual device-local parameter-cache alias rejection; model sources unchanged."""
import argparse,copy,gc,json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.body_allocation_model_v1.model import GPT as Original,load_weights
from research.body_allocation_worker_v1.execution import checked,sources
from research.body_allocation_worker_v1.validation import verify_artifact
from research.mla_context_extension_v1.model import GPT,config,import_original_h200
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;B=ROOT/'research/tokenizer-evidence/body-allocation-preparation-001'
p=argparse.ArgumentParser();p.add_argument('device',choices=['cpu','mps']);a=p.parse_args();torch.set_num_threads(4)
OUT=E/f'correctness002-{a.device}.json';GUARD=E/f'correctness002-{a.device}-guard.json'
def inventory():
    result=sources()
    for path in sorted((ROOT/'research/mla_context_extension_v1').glob('*.py')):result[str(path.relative_to(ROOT))]=reference(path)['sha256']
    return result
def rejected(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('Invalid context policy accepted')
with session_lock(GLOBAL_LOCK.parent):
    idle();require(not OUT.exists() and not GUARD.exists(),'Closed correctness proof cannot retry')
    closure=checked(reference(B/'quality001-closure001.json'));r=next(r for r in closure['trajectories'] if r['slot']==9);worker=checked(r['worker']);ref=worker['h200'];verify_artifact(ref)
    pinned=inventory();guard=StrictMpsGuard(GUARD,ROOT) if a.device=='mps' else CpuPrerequisiteGuard(GUARD,ROOT,timeout_seconds=120,poll_seconds=.25)
    with guard,torch.random.fork_rng(devices=[]):
        saved=torch.load(ref['path'],map_location='cpu',weights_only=True);m=GPT(config()).to(a.device);old=Original(OmegaConf.create(saved['cfg']['model'])).to(a.device).eval()
        load_weights(old,saved);initial=[(n,p.detach().cpu().clone()) for n,p in m.named_parameters()]
        rejected(lambda:import_original_h200(m,dict(saved,kind='raw_body_allocation_v1'),ref['sha256']))
        bad=copy.copy(saved);bad['architecture']=dict(saved['architecture'],arm='deeper');rejected(lambda:import_original_h200(m,bad,ref['sha256']))
        bad=copy.copy(saved);bad['averaged_weights']=dict(saved['averaged_weights'],snapshots=40);rejected(lambda:import_original_h200(m,bad,ref['sha256']))
        assert all(torch.equal(p.detach().cpu(),t) for (_,p),(n,t) in zip(m.named_parameters(),initial));del initial,bad
        import_original_h200(m,saved,ref['sha256']);rejected(lambda:import_original_h200(m,saved,ref['sha256']));del saved
        require(all(n==on and torch.equal(p.detach().cpu(),op.detach().cpu()) for (n,p),(on,op) in zip(m.named_parameters(),old.named_parameters())),'Learned weights changed during analytic extension')
        rejected(lambda:m.train(True));x=torch.arange(128,dtype=torch.int64).reshape(1,128).to(a.device)
        rejected(lambda:m(x))
        records=[]
        with torch.no_grad():
            for batch in (1,8):
                x=(torch.arange(batch*256).reshape(batch,256)%400+1).to(a.device)
                oracle=old(x[:,:128])[0];native=m(x[:,:128]);torch.testing.assert_close(native,oracle,rtol=3e-4,atol=3e-5)
                maximum=float((native-oracle).abs().max());del native,oracle
                full=m(x);cache=m.caches(batch)
                replay=torch.cat([m(x[:,:32],cache=cache),m(x[:,32:128],cache=cache,start_pos=32),m(x[:,128:],cache=cache,start_pos=128)],1)
                torch.testing.assert_close(full,replay,rtol=3e-4,atol=3e-5);cache_error=float((full-replay).abs().max());del replay
                for c in cache:c.reset()
                replay=m(x,cache=cache);torch.testing.assert_close(full,replay,rtol=3e-4,atol=3e-5);del replay
                changed=x.clone();changed[:,128:]=(changed[:,128:]+7)%50257;future=m(changed)
                torch.testing.assert_close(full[:,:128],future[:,:128],rtol=3e-4,atol=3e-5);del future,changed
                foreign=old.caches(batch);rejected(lambda:m(x,cache=foreign));assert all(c.position==0 for c in foreign)
                for c in cache:c.reset()
                cache[0].signature['checkpoint_sha256']='0'*64;rejected(lambda:m(x,cache=cache));assert all(c.position==0 and not bool(c.c_cache.any()) and not bool(c.r_cache.any()) for c in cache)
                cache=m.caches(batch);rejected(lambda:m(torch.zeros((batch,257),dtype=torch.int64,device=a.device),cache=cache));assert all(c.position==0 for c in cache)
                if a.device=='cpu' and batch==1:
                    cache[0].c_cache=m.transformer.h[0].attn.causal_mask.view(torch.float32).reshape(1,256,64)
                    rejected(lambda:m(x,cache=cache));assert all(c.position==0 for c in cache)
                if a.device=='mps':
                    cache=m.caches(batch);cache[0].c_cache=next(m.parameters()).detach().reshape(-1)[:batch*256*64].reshape(batch,256,64)
                    rejected(lambda:m(x,cache=cache));assert all(c.position==0 for c in cache)
                records.append(dict(batch=batch,native128_max_logit_difference=maximum,extended256_cache_max_logit_difference=cache_error,
                    extended256_full_cache_reset_causality=True,foreign_checkpoint_cache_overflow_and_alias_rejected=True,cache_bytes_per_item=393216))
                del full,cache,foreign,x;gc.collect()
                if a.device=='mps':torch.mps.empty_cache()
        with torch.no_grad():next(m.parameters()).add_(1e-7)
        rejected(lambda:m.validate());del m,old;gc.collect()
        require(inventory()==pinned,'Source changed during proof');verify_artifact(ref)
    result=dict(status='passed',device=a.device,sources=pinned,source=reference(__file__),original_h200=ref,records=records,
        parameters=28492416,parameter_tensors=88,learned_weights_unchanged=True,analytic_rope_prefix_exact=True,
        training_and_raw_resume_forbidden=True,loaded_weight_mutation_rejected=True,scientific_seed_assigned=False,
        quality_gain_claim=False,holdout_scored=False,guard=reference(GUARD))
    with OUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','source','original_h200')}))
