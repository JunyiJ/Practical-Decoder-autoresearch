"""New generated-token dual-EMA prerequisite, not a historical fixture retry."""
import copy,json,math
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.ema_horizon_v1 import contract as c
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.block_norm_v1.runtime.train.optimizers import build_optimizer
from research.block_norm_v1.runtime.models.gpt import GPT
from research.block_norm_v1.runtime.models.block_norm import contract,validate_model
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock

B=c.ROOT/'research/tokenizer-evidence/ema-horizon-preparation-001'
STEM='shadow-prerequisite-001'
SEEDS=dict(pilot=375901,screen=375902,fresh=[375911,375912,375913],evaluation=375903,bootstrap=375904)
torch.set_num_threads(4)
def clone(x):
    if isinstance(x,torch.Tensor):return x.detach().clone()
    if isinstance(x,dict):return type(x)((k,clone(v)) for k,v in x.items())
    if isinstance(x,list):return [clone(v) for v in x]
    if isinstance(x,tuple):return tuple(clone(v) for v in x)
    return copy.deepcopy(x)
def equal(x,y):
    if isinstance(x,torch.Tensor):return x.dtype==y.dtype and x.shape==y.shape and torch.equal(x,y)
    if type(x) is not type(y):return False
    if isinstance(x,dict):return x.keys()==y.keys() and all(equal(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return len(x)==len(y) and all(equal(a,b) for a,b in zip(x,y))
    return x==y
def rejects(f):
    try:f()
    except (ValueError,TypeError):return
    raise AssertionError('Mutation was accepted')

with session_lock(GLOBAL_LOCK.parent):
    idle()
    assert not (B/(STEM+'-result.json')).exists()
    with CpuPrerequisiteGuard(B/(STEM+'-cpu-guard.json'),c.ROOT,timeout_seconds=120,poll_seconds=.25):
        sources=c.source_inventory();checks=0
        for stage,count in [('pilot',1),('quality',3)]:
            for slot in range(count):
                output=c.ROOT/f'research_runs/tokenizer-ema-horizon-v1-fixture/evidence/{"pilots" if stage=="pilot" else "quality"}/{slot:02d}'
                cfg=c.config(output,stage,slot,SEEDS);c.validate_config(cfg,output,stage,slot,SEEDS);checks+=1
                for key,value in [('learning_rate',.0025),('max_iters',21),('device','cpu'),('keep_best_checkpoint',True)]:
                    bad=copy.deepcopy(cfg);bad['training'][key]=value
                    rejects(lambda:c.validate_config(bad,output,stage,slot,SEEDS));checks+=1
                bad=copy.deepcopy(cfg);bad['training']['weight_averaging']['half_lives']=[200]
                rejects(lambda:c.validate_config(bad,output,stage,slot,SEEDS));checks+=1
        rejects(lambda:c.spec('quality',3,SEEDS));rejects(lambda:c.spec('pilot',True,SEEDS));checks+=2
        assert c.RAW_LIMIT>558903984>512*1024**2 and c.EXPORT_LIMIT==128*1024**2;checks+=1
        # Full 41-snapshot schedule, independent float64 scalar recurrence.
        model=torch.nn.Linear(4,3,bias=True)
        training=OmegaConf.create(dict(max_iters=1200,weight_averaging=dict(kind='spaced_tail_ema',total_steps=1200,start_step=800,interval=10,half_lives=[100,200])))
        tracker=TailWeightAverages(model,training);reference={};max_error=0.
        for step in range(1,1201):
            with torch.no_grad():
                for k,p in enumerate(model.parameters()):p.fill_(math.sin(step*.01+k)*.1)
            before=clone(model.state_dict());rng=torch.get_rng_state().clone();tracker.update(step)
            assert equal(before,model.state_dict()) and torch.equal(rng,torch.get_rng_state())
            if step==800:
                reference={str(h):{n:p.detach().double().clone() for n,p in model.named_parameters()} for h in (100,200)}
            elif step>800 and (step-800)%10==0:
                for h in (100,200):
                    beta=2**(-10/h)
                    for n,p in model.named_parameters():reference[str(h)][n]=beta*reference[str(h)][n]+(1-beta)*p.detach().double()
            if step==1010:
                resumed=TailWeightAverages(model,training);resumed.load_state_dict(clone(tracker.state_dict()),step)
            elif step>1010:resumed.update(step)
        assert tracker.snapshots==41 and equal(tracker.state_dict(),resumed.state_dict());checks+=1
        for h in (100,200):
            for n,p in tracker.averages[str(h)].items():max_error=max(max_error,float((p.double()-reference[str(h)][n]).abs().max()))
        assert max_error<1e-7;checks+=1
        cpu=dict(config_checks=checks,online_and_rng_checks=1200,snapshots=41,max_float64_recurrence_error=max_error,resume_exact=True)
        print(json.dumps(dict(stage='cpu_passed',result=cpu)),flush=True)
        del model,tracker,resumed,reference,before
    with StrictMpsGuard(B/(STEM+'-mps-guard.json'),c.ROOT):
        c.inherited.environment();assert c.source_inventory()==sources
        output=c.ROOT/'research_runs/tokenizer-ema-horizon-v1-fixture/evidence/pilots/00'
        cfg=OmegaConf.create(c.config(output,'pilot',0,SEEDS))
        torch.manual_seed(SEEDS['pilot']);torch.mps.manual_seed(SEEDS['pilot'])
        model=GPT(cfg.model).to('mps');validate_model(model,contract(cfg.model))
        optimizer=build_optimizer(model,cfg.training)
        dual=TailWeightAverages(model,cfg.training)
        single_cfg=copy.deepcopy(cfg.training);single_cfg.weight_averaging.half_lives=[200]
        single=TailWeightAverages(model,single_cfg)
        generator=torch.Generator(device='cpu').manual_seed(SEEDS['pilot']+100)
        tokens=torch.randint(0,50257,(2,5),generator=generator).to('mps')
        losses=[]
        for step in range(1,21):
            optimizer.zero_grad(set_to_none=True)
            logits,loss=model(tokens[:,:4].contiguous(),targets=tokens[:,1:].contiguous());assert loss is not None and torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            parameters=clone(model.state_dict());optim=clone(optimizer.state_dict())
            grads=[p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
            cpu_rng=torch.get_rng_state().clone();mps_rng=torch.mps.get_rng_state().clone()
            single.update(step);dual.update(step);torch.mps.synchronize()
            assert equal(parameters,model.state_dict()) and equal(optim,optimizer.state_dict())
            assert equal(grads,[p.grad for p in model.parameters()])
            assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(mps_rng,torch.mps.get_rng_state())
            if step>=13:assert equal(single.averages['200'],dual.averages['200'])
            if step==16:
                resumed=TailWeightAverages(model,cfg.training);resumed.load_state_dict(clone(dual.state_dict()),step)
            elif step>16:resumed.update(step)
            losses.append(float(loss.detach().cpu()))
            del parameters,optim,grads,logits,loss
        assert dual.snapshots==single.snapshots==8 and equal(dual.state_dict(),resumed.state_dict())
        mps=dict(generated_token_shape=[2,5],optimizer_steps=20,parameters=28492416,shadow_online_optimizer_gradient_rng_checks=20,
            h200_exact_checks=8,dual_resume_exact=True,snapshots=8,losses=losses,
            scope='One shared realized optimizer trajectory isolates shadow-tracker effects; not independent-run determinism or a full batch8/context128 resource pilot.')
        del model,optimizer,dual,single,resumed,tokens
        import gc
        gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
        assert c.source_inventory()==sources
    result=dict(status='passed',cpu=cpu,mps=mps,fixture_seeds=SEEDS,scientific_seeds_reserved=False,
        sources=sources,launch_permitted=False,scientific_training_launched=False,generated_prerequisite_optimizer_steps=20,
        checkpoints_written=False,holdout_read=False,deletion_performed=False)
    with (B/(STEM+'-result.json')).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='passed',cpu=cpu,mps=mps)),flush=True)
