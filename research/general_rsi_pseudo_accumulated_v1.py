"""Bounded accumulated-state mechanism controls; no model selection or checkpoint output."""
import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

from research.general_rsi_pseudo_update_diagnostic_v1 import (
    ROOT, sha, put, digest, batch_hash, ce_batch, evaluate, displacement, check_stop)

OUT = ROOT / 'research/tokenizer-evidence/general-rsi-pseudo-accumulated-001'


def inputs(config, data):
    import torch
    from research.general_rsi_pseudo_spans_v1 import Sampler
    values = data.view.train_data
    natural_rng = torch.Generator().manual_seed(config['roles']['train_sampler'])
    pseudo = Sampler(config['pseudo_seed'])
    natural, synthetic, excluded = [], [], []
    for n in range(1, 101):
        starts = torch.randint(len(values)-512, (2,), generator=natural_rng).tolist()
        natural.append(dict(natural_update=n, starts=starts, sha256=batch_hash(*ce_batch(values, starts))))
        excluded.extend((s,s+513) for s in starts)
        if n < 100:
            x,y,previous,mask,details = pseudo.batch()
            synthetic.append(dict(after_natural_update=n, sha256=batch_hash(x,y,previous,mask), rows=details))
            for row in details:
                _,start,length,prefix = map(int,pseudo.spans[row['inventory_index']])
                excluded.append((start-prefix,start+length))
    rng = torch.Generator().manual_seed(config['evaluation_seed'])
    eligible = [i for i,r in enumerate(data.indexes['train']) if r['tokens'] >= 129]
    windows = []; proposals = 0
    while len(windows) < 64:
        proposals += 1; assert proposals < 10000
        doc = eligible[int(torch.randint(len(eligible),(),generator=rng))]
        row = data.indexes['train'][doc]
        start = row['offset'] + int(torch.randint(row['tokens']-128,(),generator=rng))
        if any(start < b and a < start+129 for a,b in excluded): continue
        windows.append(dict(document=doc,source=row['source'],start=start,targets=128))
        excluded.append((start,start+129))
    evaluation = [(values[w['start']:w['start']+128][None,:],values[w['start']+1:w['start']+129][None,:]) for w in windows]
    inventory = dict(natural=natural,synthetic=synthetic,evaluation_windows=windows,
                     evaluation_sha256=batch_hash(*(t for pair in evaluation for t in pair)),
                     targets=8192,proposals=proposals,disjoint_evaluation_positions=True)
    return inventory,evaluation


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def update(ctx, objective, expected):
    import torch
    from research.general_rsi_pseudo_spans_v1 import repetition_loss
    from research.gradient_clip_model_v1.clipping import clip_gradients
    check_stop();torch.mps.synchronize();tick=time.monotonic()
    ctx.model.train();ctx.optimizer.zero_grad()
    value=None;clip=None;info=None;hashed=None
    if objective == 'zero':
        for _,p in ctx.model.ordered_parameters(): p.grad=torch.zeros_like(p)
    else:
        if objective == 'ce':
            x,y=ctx.data.batch();hashed=batch_hash(x,y)
            loss=ctx.model.forward_train(x,y).loss
        else:
            assert objective == 'pseudo'
            x,y,previous,mask,details=ctx.pseudo.batch('mps')
            hashed=batch_hash(x,y,previous,mask);assert details==expected['rows']
            loss,info=repetition_loss(ctx.model.primary_logits(x).values,y,previous,mask)
        assert hashed==expected['sha256']
        value=float(loss.detach());assert math.isfinite(value)
        loss.backward();clip=clip_gradients((p for _,p in ctx.model.ordered_parameters()),1.0)
    ctx.optimizer.step();ctx.optimizer.zero_grad();ctx.optimizer.validate()
    torch.mps.synchronize()
    return dict(objective=objective,loss=value,clip=clip,pseudo_info=info,input_sha256=hashed,
                optimizer_steps=ctx.optimizer._steps,seconds=time.monotonic()-tick)


def optimizer_summary(ctx):
    """FP64 CPU moment norms plus a complete child-state digest, without persistence."""
    import torch
    results={}
    names={id(p):n for n,p in ctx.model.ordered_parameters()}
    for child in ctx.optimizer._optimizers:
        sums={};hashes={}
        for parameter,fields in child.state.items():
            copied={k:v.detach().cpu() for k,v in fields.items()}
            hashes[names[id(parameter)]]=digest(copied)
            for k,v in copied.items():
                if k!='step':sums[k]=sums.get(k,0.)+float(v.double().square().sum())
        results[type(child).__name__]=dict(field_norms={k:math.sqrt(v) for k,v in sums.items()},
                                         state_sha256=digest(hashes),parameter_states=len(hashes))
    return results


def execute(plan,binding,attempt):
    import torch
    from research.general_rsi_pseudo_training_v1 import Context
    from research.byte_input_model_v6 import build_optimizer
    assert torch.__version__==plan['torch_version'] and torch.backends.mps.is_available()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0';torch.set_num_threads(4)
    peak=0;updates=0;train_seconds=0.;endpoints=[]
    def boundary():
        nonlocal peak
        check_stop();torch.mps.synchronize()
        peak=max(peak,torch.mps.driver_allocated_memory());assert peak<plan['mps_limit_bytes']
    for config in plan['parents']:
        boundary();slot=config['parent']['slot']
        ctx=Context('native_ce',1200*10**9,plan['ema_half_life_ns'],config['roles'],config['pseudo_seed'],config['parent'],binding)
        inventory,evaluation=inputs(config,ctx.data);assert inventory==config['inputs']
        common=ctx.capture();common_hash=digest(common)
        baseline=evaluate(ctx.model,evaluation)
        put(attempt/f'p{slot:02d}-initial.json',dict(parent=config['parent'],common_sha256=common_hash,evaluation=baseline))
        first_probe=None
        for arm in config['arm_order']:
            boundary();arm_tick=time.monotonic()
            ctx.optimizer.zero_grad();ctx.optimizer=build_optimizer(ctx.model,.00025);ctx.history=[]
            ctx.restore(common);assert digest(ctx.capture())==common_hash
            arm_steps=0;arm_seconds=0.;probes=[]
            for n in range(1,101):
                r=update(ctx,'ce',inventory['natural'][n-1]);boundary()
                arm_steps+=1;updates+=1;train_seconds+=r['seconds'];arm_seconds+=r['seconds']
                append(attempt/f'p{slot:02d}-{arm}-steps.jsonl',dict(natural_update=n,**r))
                if n in plan['probe_natural_updates']:
                    score=evaluate(ctx.model,evaluation)
                    if n==1:
                        if first_probe is None:first_probe=score
                        else:assert score==first_probe
                    expected_steps=n if arm=='ce' else 2*n-1
                    assert ctx.optimizer._steps==expected_steps==arm_steps
                    probe=dict(parent_slot=slot,arm=arm,natural_updates=n,extra_updates=expected_steps-n,
                               optimizer_steps=expected_steps,evaluation=score,delta_from_parent=score['ce']-baseline['ce'],
                               displacement_from_parent=displacement(ctx.model,common,None),optimizer=optimizer_summary(ctx))
                    put(attempt/f'p{slot:02d}-{arm}-n{n:03d}.json',probe);probes.append(probe)
                    print(json.dumps(dict(parent_slot=slot,arm=arm,natural_updates=n,ce=score['ce'],delta_from_parent=probe['delta_from_parent'])),flush=True)
                if n<100 and arm!='ce':
                    objective='zero' if arm=='ce_zero' else 'pseudo'
                    r=update(ctx,objective,inventory['synthetic'][n-1]);boundary()
                    arm_steps+=1;updates+=1;train_seconds+=r['seconds'];arm_seconds+=r['seconds']
                    append(attempt/f'p{slot:02d}-{arm}-steps.jsonl',dict(natural_update=n,**r))
            assert arm_steps==(100 if arm=='ce' else 199)
            row=dict(parent_slot=slot,arm=arm,natural_updates=100,optimizer_steps=arm_steps,
                     synchronized_update_seconds=arm_seconds,total_arm_seconds=time.monotonic()-arm_tick,
                     final_ce=probes[-1]['evaluation']['ce'],restored_state_sha256=common_hash)
            put(attempt/f'p{slot:02d}-{arm}-result.json',row);endpoints.append(row)
        del ctx,common,evaluation
        gc.collect();torch.mps.empty_cache()
    assert updates==1494 and len(endpoints)==9
    return dict(kind='pseudo_accumulated_state_diagnostic_v1',plan_sha256=binding,updates=updates,
                natural_updates=900,zero_updates=297,pseudo_updates=297,endpoints=endpoints,
                synchronized_update_seconds=train_seconds,peak_mps_boundary_bytes=peak,
                exact_initial_state_restores=9,exact_first_ce_probe_pairs=6,
                test_opened=False,validation_scored=False,new_saved_models=0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,required=True);args=parser.parse_args()
    tick=time.monotonic();os.chdir(ROOT)
    lock=(ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop()
    attempt=OUT/f'attempt-{args.attempt:02d}';attempt.mkdir(exist_ok=False)
    def stop(signum,frame):
        import research.general_rsi_pseudo_update_diagnostic_v1 as shared
        shared.STOP=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    plan=json.loads((OUT/'plan.json').read_text());binding=sha(OUT/'plan.json')
    put(attempt/'launch.json',dict(pid=os.getpid(),plan_sha256=binding,started_unix=time.time()))
    try:
        with CpuPrerequisiteGuard(attempt/'guard.json',ROOT,rss_limit_bytes=plan['rss_limit_bytes'],
                                 reserve_bytes=plan['reserve_bytes'],timeout_seconds=plan['timeout_seconds']):
            for path,h in plan['sources'].items():assert sha(path)==h,path
            assert json.loads((OUT/'admission.json').read_text())['plan_sha256']==binding
            result=execute(plan,binding,attempt);result['total_seconds']=time.monotonic()-tick
            put(attempt/'result.json',result)
    except BaseException:
        put(attempt/'failure.json',dict(traceback=traceback.format_exc(),seconds=time.monotonic()-tick));raise


if __name__=='__main__':main()
