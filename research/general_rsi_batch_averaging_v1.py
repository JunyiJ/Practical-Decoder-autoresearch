"""One fresh optimizer update after averaging independent natural minibatches."""
import math,time
from research.general_rsi_pseudo_update_diagnostic_v1 import check_stop,batch_hash

def update_once(ctx,count,gradient_observer=None):
    import torch
    from research.gradient_clip_model_v1.clipping import clip_gradients
    assert count in (1,8) and ctx.step==0 and not ctx.history and ctx.clock.spent_ns==0
    assert ctx.optimizer.decay==(0.,0.) and not ctx.ema and ctx.snapshots==0
    losses=[];hashes=[];ctx.model.train();begin=ctx.clock.spent_ns
    with ctx.clock.update(torch.mps.synchronize,time.perf_counter_ns):
        ctx.optimizer.zero_grad()
        for _ in range(count):
            check_stop();x,y=ctx.data.batch();assert y.numel()==1024
            loss=ctx.model.forward_train(x,y).loss;value=float(loss.detach());assert math.isfinite(value)
            losses.append(value);hashes.append(batch_hash(x,y));(loss/count).backward();del loss
        if gradient_observer is not None:gradient_observer(ctx.model)
        clip=clip_gradients((p for _,p in ctx.model.ordered_parameters()),1.)
        ctx.optimizer.step();ctx.optimizer.zero_grad();ctx.optimizer.validate()
    assert ctx.step==1
    row=dict(step=1,objective='ce',loss=math.fsum(losses)/count,microbatch_losses=losses,microbatch_sha256=hashes,microbatches=count,valid_loss_targets=1024*count,clip=clip,lr_ratio=1.,begin_ns=begin,end_ns=ctx.clock.spent_ns)
    ctx.history.append(row);return row

def inputs(config,data):
    import torch
    from research.general_rsi_pseudo_update_diagnostic_v1 import ce_batch
    values=data.view.train_data;cases=[];excluded=[]
    for c in config['cases']:
        rng=torch.Generator().manual_seed(c['roles']['train_sampler']);batches=[]
        for _ in range(8):
            starts=torch.randint(len(values)-512,(2,),generator=rng).tolist();batches.append(dict(starts=starts,sha256=batch_hash(*ce_batch(values,starts))));excluded.extend((s,s+513) for s in starts)
        cases.append(dict(case=c['case'],batches=batches))
    rng=torch.Generator().manual_seed(config['evaluation_seed']);eligible=[i for i,r in enumerate(data.indexes['train']) if r['tokens']>=129];windows=[];proposals=0
    while len(windows)<64:
        proposals+=1;assert proposals<10000
        doc=eligible[int(torch.randint(len(eligible),(),generator=rng))];d=data.indexes['train'][doc];start=d['offset']+int(torch.randint(d['tokens']-128,(),generator=rng))
        if any(start<b and a<start+129 for a,b in excluded):continue
        windows.append(dict(document=doc,source=d['source'],start=start,targets=128));excluded.append((start,start+129))
    batches=[(values[w['start']:w['start']+128][None,:],values[w['start']+1:w['start']+129][None,:]) for w in windows]
    return dict(cases=cases,evaluation_windows=windows,evaluation_sha256=batch_hash(*(t for b in batches for t in b)),targets=8192,probes_disjoint_from_all_update_source_positions=True),batches
