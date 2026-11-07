"""NEW generated future-token model proof; one closed attempt, no corpus/tensor I/O."""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from research.future_token_model_v1.contract import ARMS, model_config
from research.future_token_model_v1.model import GPT, Distribution, load_weights
from research.future_token_model_v1.optimizer import build_optimizer
from research.future_token_model_v1.backward import trunk, head, joint_reference, sequential_backward
from research.body_allocation_model_v1.model import GPT as Native
from research.body_allocation_model_v1.optimizer import build_optimizer as native_optimizer
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')
CAP = 8*1024**2
TRAINING = SimpleNamespace(optimizer='muon', decay_mode='independent', weight_decay=.00006,
    learning_rate=.00125, adamw_beta2=.999, adamw_epsilon=1e-6,
    adamw_beta2_policy='mla4_adamw_beta2_fixed_v1', adamw_epsilon_policy='mla4_adamw_epsilon_fixed_v1')
ROWS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw) <= CAP
    with (R/name).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def checked(label, measurements):
    # Record every scalar, every coordinate predicate and every expected rejection
    # before enforcing the numeric/behavior predicate. No max-only acceptance.
    publish(label+'.json', measurements)
    ROWS.append(label)
    assert all(row['passed'] for row in measurements), label


def compare(name, actual, expected, atol, rtol):
    a, e = actual.detach(), expected.detach().to(actual.device)
    valid = a.shape == e.shape and a.dtype == e.dtype and bool(torch.isfinite(a).all()) and bool(torch.isfinite(e).all())
    error = float((a-e).abs().max().cpu()) if valid and a.numel() else 0.
    passed = valid and bool(((a-e).abs() <= atol+rtol*e.abs()).all())
    return dict(name=name, passed=passed, maximum_absolute_error=error, atol=atol, rtol=rtol,
                elements=a.numel(), shape=list(a.shape))


def fact(name, passed, **values):
    return dict(name=name, passed=bool(passed), **values)


def scalar(name, actual, expected):
    return fact(name, abs(actual-expected) <= 3e-5, actual=actual, expected=expected,
                absolute_error=abs(actual-expected), atol=3e-5, rtol=0.)


def ids(batch, time, device):
    # NEW saved finite formula, no random generator/data loader/tokenizer.
    # seq[b,j]=(211+173*b+37*j+19*j*j) mod50257, lengthT+2.
    return torch.tensor([[(211+173*b+37*j+19*j*j) % 50257 for j in range(time+2)]
                         for b in range(batch)], dtype=torch.int64, device=device)


def clone_tree(value, device='cpu'):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {k: clone_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_tree(v, device) for v in value)
    return copy.deepcopy(value)


def tree_rows(prefix, actual, expected, atol=0., rtol=0.):
    if isinstance(expected, torch.Tensor):
        return [compare(prefix, actual, expected, atol, rtol)]
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or list(actual) != list(expected):
            return [fact(prefix+'.keys', False)]
        return [row for k in expected for row in tree_rows(prefix+'.'+str(k), actual[k], expected[k], atol, rtol)]
    if isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            return [fact(prefix+'.length', False)]
        return [row for i, (a, e) in enumerate(zip(actual, expected)) for row in tree_rows(prefix+'.'+str(i), a, e, atol, rtol)]
    return [fact(prefix, type(actual) is type(expected) and actual == expected)]


def tensors_digest(named):
    # Used only for exact new initialization/paired copies, never old weights.
    result = {}
    for name, value in named:
        a = value.detach().cpu().contiguous()
        result[name] = dict(shape=list(a.shape), dtype=str(a.dtype),
                            sha256=hashlib.sha256(a.numpy().tobytes()).hexdigest())
    return result


def release():
    gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()


def own(arm, seed, device):
    return GPT(model_config(arm), initializer_seed=seed).to(device)


def inventory(model, common, draw, paired):
    arm = model.signature['arm']; named = list(model.named_parameters(remove_duplicate=False))
    rows = [fact('parameter_elements', sum(p.numel() for _, p in named) == (28492416 if arm == 'native_next' else 29183008)),
            fact('parameter_tensors', len(named) == (88 if arm == 'native_next' else 111)),
            fact('common_native_initial_values_order', tensors_digest(model.native.named_parameters()) == common),
            fact('native_CPU_draw_digest', model.native_cpu_draw_digest == draw),
            fact('single_shared_unembedding', [n for n, p in named if p is model.native.lm_head.weight] == ['native.lm_head.weight']),
            fact('unique_parameter_storage', len({p.untyped_storage().data_ptr() for _, p in named}) == len(named))]
    if arm != 'native_next':
        copied = list(model.auxiliary.block.named_parameters())+[("norm."+n,p) for n,p in model.auxiliary.norm.named_parameters()]
        wanted = dict(model.native.transformer.h[3].named_parameters())
        wanted.update({'norm.'+n:p for n,p in model.native.transformer.ln_f.named_parameters()})
        rows += [compare('copy.'+n, p, wanted[n], 0., 0.) for n, p in copied]
        rows += [fact('copy_independent_storage.'+n, p.untyped_storage().data_ptr() != wanted[n].untyped_storage().data_ptr()) for n,p in copied]
        value = tensors_digest(copied)
        rows.append(fact('paired_aux_initialization', paired is None or paired == value))
        paired = value
    checked(arm+'-inventory', rows)
    return paired


def grouping(model, optimizer):
    groups = optimizer.param_groups
    muon = len(optimizer.optimizers[0].param_groups); adam = len(optimizer.optimizers[1].param_groups)
    checked(model.signature['arm']+'-groups', [fact('actual_muon', muon == model.signature['muon_groups'], count=muon),
        fact('actual_adam', adam == model.signature['adam_groups'], count=adam),
        fact('all_parameters_once', len(groups) == len(list(model.parameters())) and len({id(g['params'][0]) for g in groups}) == len(groups)),
        fact('shared_W_one_group', sum(g['params'][0] is model.native.lm_head.weight for g in groups) == 1)])


def complete_step(model, optimizer):
    norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(norm.detach().cpu())


def native_pair(seed):
    with CpuPrerequisiteGuard(R/'native-init-cpu-guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            ref = Native(model_config('native_next')['native_template'])
            draw = hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest()
        common = tensors_digest(ref.named_parameters())
        m = own('native_next', seed, 'cpu')
        inventory(m, common, draw, None)
        checked('native-exact-initial-state', tree_rows('state', m.native.state_dict(), ref.state_dict()))
    ref, m = ref.to('mps'), m.to('mps')
    ro = native_optimizer(ref, SimpleNamespace(training=TRAINING)); mo = build_optimizer(m, TRAINING)
    grouping(m, mo)
    seq = ids(2, 11, 'mps'); x, y = seq[:, :-2], seq[:, 1:-1]
    rlogits, rloss = ref(x, y)
    checked('native-exact-forward', [compare('logits', m(x).values, rlogits, 0., 0.),
        compare('loss', m.loss(x,y), rloss, 0., 0.)])
    rloss.backward(); actual = sequential_backward(m, seq)
    checked('native-exact-backward', [compare(n, p.grad, dict(ref.named_parameters())[n.removeprefix('native.')].grad, 0., 0.)
        for n,p in m.named_parameters()]+[fact('reported_primary_CE_exact',actual['primary_ce']==float(rloss.detach().cpu()),actual=actual['primary_ce'],expected=float(rloss.detach().cpu()),atol=0.,rtol=0.)])
    del rlogits, rloss
    step_cache=m.caches(1)
    rn = complete_step(ref, ro); mn = complete_step(m, mo)
    stale_step(m,step_cache,'native_next');del step_cache
    rows = [compare(n, p, dict(ref.named_parameters())[n.removeprefix('native.')], 0., 0.) for n,p in m.named_parameters()]
    for child_a, child_e in zip(mo.optimizers, ro.optimizers):
        sa, se = clone_tree(child_a.state_dict()), clone_tree(child_e.state_dict())
        for g in sa['param_groups']:
            g['parameter_names'] = [n.removeprefix('native.') for n in g['parameter_names']]
        rows += tree_rows('child', sa, se)
        del sa,se
    rows += [fact('one_clip_norm_exact', mn == rn, own=mn, reference=rn),
             fact('one_actual_update_all_exposures', mo.completed_steps == 1 and set(mo.gradient_steps.values()) == {1})]
    checked('native-exact-one-update', rows)
    del ref,ro,seq,rows
    release()
    m.zero_grad(set_to_none=True)
    roundtrip_and_invalids(m,mo)
    del m,mo
    release()
    return common, draw


def auxiliary_pair(arm, seed, common, draw, paired):
    with CpuPrerequisiteGuard(R/(arm+'-init-cpu-guard.json'), ROOT, timeout_seconds=120, poll_seconds=.25):
        joint, seq = own(arm,seed,'cpu'), own(arm,seed,'cpu')
        paired = inventory(seq,common,draw,paired)
        checked(arm+'-paired-own-start', tree_rows('state',joint.state_dict(),seq.state_dict()))
    joint, seq = joint.to('mps'), seq.to('mps')
    jo, so = build_optimizer(joint,TRAINING), build_optimizer(seq,TRAINING)
    grouping(seq,so)
    data = ids(2,11,'mps'); before = data.clone()
    objective = joint_reference(joint,data)
    expected = {k: None if v is None else float(v.detach().cpu()) for k,v in objective.items()}
    objective['training_loss'].backward(); del objective
    actual = sequential_backward(seq,data)
    checked(arm+'-joint-sequential-all-gradients',
        [compare(n,p.grad,dict(joint.named_parameters())[n].grad,4e-6,4e-5) for n,p in seq.named_parameters()]+
        [scalar(k,actual[k],expected[k]) for k in actual]+[compare('caller_sequence',data,before,0.,0.)])
    aux_initial={n:p.detach().cpu().clone() for n,p in seq.named_parameters() if n.startswith('auxiliary.block.mlp.')}
    step_cache=seq.caches(1)
    checked(arm+'-auxiliary-FFN-gradient-exposure',[fact(n,p.grad is not None and bool((p.grad!=0).any())) for n,p in seq.named_parameters() if n in aux_initial])
    jn,sn = complete_step(joint,jo),complete_step(seq,so)
    stale_step(seq,step_cache,arm);del step_cache
    checked(arm+'-auxiliary-FFN-changed',[fact(n,not torch.equal(p.detach().cpu(),aux_initial[n])) for n,p in seq.named_parameters() if n in aux_initial])
    del aux_initial
    checked(arm+'-joint-sequential-one-update', [compare(n,p,dict(joint.named_parameters())[n],4e-6,4e-5)
        for n,p in seq.named_parameters()]+tree_rows('moments',so.state_dict(),jo.state_dict(),4e-6,4e-5)+
        [fact('one_actual_update_all_exposures',so.completed_steps == jo.completed_steps == 1 and set(so.gradient_steps.values()) == {1}),
         fact('matched_finite_clip_norms',0 <= sn < float('inf') and 0 <= jn < float('inf') and abs(sn-jn)<=4e-6+4e-5*abs(jn),sequential=sn,joint=jn,atol=4e-6,rtol=4e-5)])
    del joint,jo,data,before
    release()
    seq.zero_grad(set_to_none=True)
    roundtrip_and_invalids(seq,so)
    del seq,so
    release()
    return paired


def stale_step(model,cache,arm):
    refused=False
    try:model.validate_cache(cache,1,0)
    except ValueError:refused=True
    checked(arm+'-actual-step-stale-cache',[fact('actual_step_epoch_stale',refused)])
    cache.reset()
    checked(arm+'-actual-step-reset-cache',[fact('actual_step_reset_epoch',cache.epoch==model.weight_epoch and all(c.position==0 for c in cache.layers))])


def snapshot(model, optimizer, cache):
    # Do not call validating model/optimizer APIs while testing deliberately
    # malformed configuration/cache inputs. Read the owned state directly.
    return dict(weights=clone_tree(model.state_dict()),buffers=clone_tree(dict(model.named_buffers())),
        grads={n:None if p.grad is None else p.grad.detach().cpu().clone() for n,p in model.named_parameters()},
        optimizer=clone_tree([c.state_dict() for c in optimizer.optimizers]),
        exposure=copy.deepcopy(optimizer.gradient_steps),steps=optimizer.completed_steps,decay=optimizer.decay,
        epoch=model.weight_epoch,modes={n:v.training for n,v in model.named_modules()},cfg=copy.deepcopy(model.cfg),signature=copy.deepcopy(model.signature),
        cache_signature=copy.deepcopy(cache.signature),cache_epoch=cache.epoch,cache_batch=cache.batch,
        cache_owner_id=id(cache.owner()),
        cache_values=[(c.position,c.c_cache.detach().cpu().clone(),c.r_cache.detach().cpu().clone()) for c in cache.layers])


def exact_tree(actual, expected):
    if isinstance(expected,torch.Tensor):
        # Bitwise comparison also establishes preservation of deliberate NaN
        # preflight fixtures; numerical acceptance elsewhere still requires finite.
        return isinstance(actual,torch.Tensor) and actual.shape==expected.shape and actual.dtype==expected.dtype and torch.equal(
            actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),expected.contiguous().reshape(-1).view(torch.uint8))
    if isinstance(expected,dict):
        return isinstance(actual,dict) and list(actual)==list(expected) and all(exact_tree(actual[k],v) for k,v in expected.items())
    if isinstance(expected,(list,tuple)):
        return type(actual) is type(expected) and len(actual)==len(expected) and all(exact_tree(a,e) for a,e in zip(actual,expected))
    return type(actual) is type(expected) and actual==expected


def roundtrip_and_invalids(m,o):
    arm = m.signature['arm']
    cache = m.caches(1)
    with torch.no_grad(): m(ids(1,3,'mps')[:,:-2],cache=cache)
    payload = dict(kind='future_token_native_weights_v1',architecture=copy.deepcopy(m.signature),model_state=clone_tree(m.state_dict()))
    opt_state = clone_tree(o.state_dict(),'mps')
    # Adam's scalar step must remain CPU even while nonstep moments remain MPS.
    for child in opt_state['optimizers']:
        for moments in child['state'].values():
            if 'step' in moments: moments['step'] = moments['step'].cpu()
    old_epoch = m.weight_epoch
    load_weights(m,payload); o.load_state_dict(opt_state)
    checked(arm+'-own-load-roundtrip', tree_rows('weights',m.state_dict(),payload['model_state'])+
        tree_rows('optimizer',o.state_dict(),opt_state)+[fact('two_epoch_increments',m.weight_epoch == old_epoch+2)])
    stale = False
    try:
        with torch.no_grad(): m(ids(1,1,'mps')[:,:-2],cache=cache,start_pos=3)
    except ValueError: stale = True
    checked(arm+'-stale-after-load', [fact('stale_cache_rejected',stale)])
    cache.reset()
    del payload,opt_state
    baseline = snapshot(m,o,cache); reports=[]
    def reject(label,action):
        refused=False
        try: action()
        except (ValueError,TypeError): refused=True
        unchanged=exact_tree(snapshot(m,o,cache),baseline) and torch.equal(sequence,sequence_before)
        row=fact(label,refused and unchanged,rejected=refused,all_state_values_gradslots_optimizer_cache_unchanged=unchanged)
        checked(arm+'-reject-'+label,[row]);reports.append(row)
    sequence=ids(2,11,'mps');sequence_before=sequence.clone();x=sequence[:,:-2]
    reject('float_IDs',lambda:m(x.float()))
    reject('out_of_range_ID',lambda:m(torch.full_like(x,50257)))
    reject('negative_ID',lambda:m(torch.full_like(x,-1)))
    reject('offset_without_cache',lambda:m(x,start_pos=1))
    reject('target_dtype',lambda:m.loss(x,sequence[:,1:-1].float()))
    reject('target_shape',lambda:m.loss(x,sequence[:,:1]))
    reject('target_range',lambda:m.loss(x,torch.full_like(x,50257)))
    reject('foreign_cache_owner',lambda:m.validate_cache(SimpleNamespace(),1,0))
    reject('cache_grad_mode',lambda:m(x[:1,:1],cache=cache))
    reject('joint_full_batch_bound',lambda:joint_reference(m,ids(8,128,'mps')))
    reject('foreign_weight_kind',lambda:load_weights(m,dict(kind='foreign',architecture=m.signature,model_state={})))
    reject('foreign_weight_signature',lambda:load_weights(m,dict(kind='future_token_native_weights_v1',architecture={},model_state={})))
    weights=m.state_dict()
    reject('missing_weight',lambda:load_weights(m,dict(kind='future_token_native_weights_v1',architecture=m.signature,model_state=dict(list(weights.items())[1:]))))
    # Replace only a small RMSNorm tensor; never mutate caller/model-owned state.
    name='native.transformer.h.0.ln1.weight'
    for label,value in [('weight_dtype',weights[name].double()),('weight_shape',weights[name][:-1]),('weight_nonfinite',torch.full_like(weights[name],float('nan')))]:
        altered=dict(weights);altered[name]=value
        reject(label,lambda altered=altered:load_weights(m,dict(kind='future_token_native_weights_v1',architecture=m.signature,model_state=altered)))
    state=o.state_dict()
    bad=dict(state);bad['future_token_signature']={}
    reject('optimizer_signature',lambda:o.load_state_dict(bad))
    bad_steps=dict(state);bad_steps['completed_steps']=2
    reject('optimizer_exposure',lambda:o.load_state_dict(bad_steps))
    bad_decay=dict(state);bad_decay['independent_decay']=0.
    reject('optimizer_decay',lambda:o.load_state_dict(bad_decay))
    bad_groups=copy.copy(state);bad_groups['optimizers']=[dict(c) for c in state['optimizers']]
    bad_groups['optimizers'][1]['param_groups']=[dict(g) for g in state['optimizers'][1]['param_groups']]
    bad_groups['optimizers'][1]['param_groups'][0]['eps']=1e-8
    reject('optimizer_epsilon',lambda:o.load_state_dict(bad_groups))
    reject('step_missing_gradients',lambda:o.step())
    reject('incompatible_distribution',lambda:Distribution('foreign',m.signature['model_fingerprint'],'future_token_primary_native_logits',torch.zeros(1)).log_probs())
    # Structural mutation fixtures snapshot the deliberately modified preflight
    # state, then restore it exactly after rejection; no numeric source rescue.
    def modified(label,install,restore,action):
        nonlocal baseline
        install();del_baseline=baseline;baseline=snapshot(m,o,cache)
        reject(label,action);restore();baseline=del_baseline
    original=m.cfg['readout']['divisor']
    modified('mutable_config',lambda:m.cfg['readout'].__setitem__('divisor',2.),lambda:m.cfg['readout'].__setitem__('divisor',original),m.validate)
    parameter=dict(m.named_parameters())[name]
    modified('populated_gradient_slot',lambda:setattr(parameter,'grad',torch.full_like(parameter,.125)),
        lambda:setattr(parameter,'grad',None),lambda:sequential_backward(m,sequence))
    saved=parameter.detach().clone()
    def set_parameter(value):
        with torch.no_grad():parameter.copy_(value)
    modified('nonfinite_learned_parameter',lambda:set_parameter(torch.full_like(parameter,float('nan'))),
        lambda:set_parameter(saved),lambda:sequential_backward(m,sequence))
    del saved
    modified('evaluation_backward',lambda:m.eval(),lambda:m.train(),lambda:sequential_backward(m,sequence))
    cache_sig=copy.deepcopy(cache.signature)
    modified('cache_signature',lambda:setattr(cache,'signature',{}),lambda:setattr(cache,'signature',cache_sig),lambda:m.validate_cache(cache,1,0))
    modified('cache_epoch',lambda:setattr(cache,'epoch',m.weight_epoch-1),lambda:setattr(cache,'epoch',m.weight_epoch),lambda:m.validate_cache(cache,1,0))
    old=cache.layers[1].c_cache
    modified('cache_alias',lambda:setattr(cache.layers[1],'c_cache',cache.layers[0].c_cache),lambda:setattr(cache.layers[1],'c_cache',old),lambda:m.validate_cache(cache,1,0))
    modified('cache_position',lambda:setattr(cache.layers[0],'position',1),lambda:setattr(cache.layers[0],'position',0),lambda:m.validate_cache(cache,1,0))
    rope_name='native.transformer.h.0.freq_cos';altered=dict(weights);altered[rope_name]=weights[rope_name].clone()+.125
    reject('weight_RoPE_value',lambda:load_weights(m,dict(kind='future_token_native_weights_v1',architecture=m.signature,model_state=altered)))
    reject('sequence_dtype',lambda:sequential_backward(m,sequence.float()))
    reject('sequence_too_long',lambda:sequential_backward(m,ids(1,129,'mps')))
    reject('distribution_kind',lambda:Distribution(m.signature['policy'],m.signature['model_fingerprint'],'probability_mix_log_probs',torch.zeros(1)).log_probs())
    reject('distribution_dtype',lambda:Distribution(m.signature['policy'],m.signature['model_fingerprint'],'future_token_primary_native_logits',torch.zeros(1,1,50257,dtype=torch.float64)).log_probs())
    reject('distribution_shape',lambda:Distribution(m.signature['policy'],m.signature['model_fingerprint'],'future_token_primary_native_logits',torch.zeros(1)).log_probs())
    def autocast_rejection():
        with torch.autocast('cpu',dtype=torch.bfloat16):m(x)
    reject('autocast',autocast_rejection)
    owner=cache.owner
    modified('cache_foreign_owner',lambda:setattr(cache,'owner',lambda:m.native),lambda:setattr(cache,'owner',owner),lambda:m.validate_cache(cache,1,0))
    old_dtype=cache.layers[0].c_cache
    modified('cache_dtype',lambda:setattr(cache.layers[0],'c_cache',old_dtype.half()),lambda:setattr(cache.layers[0],'c_cache',old_dtype),lambda:m.validate_cache(cache,1,0))
    assert len(reports)==39
    checked(arm+'-invalid-summary',reports)
    del baseline,cache,sequence,sequence_before,weights,state,bad,bad_steps,bad_decay,bad_groups


def primary_gradients(m,sequence,gather):
    x,y=sequence[:,:-2],sequence[:,1:-1]
    logits=m(x).values
    loss=(-F.log_softmax(logits,dim=-1).gather(-1,y.unsqueeze(-1)).squeeze(-1).mean()
          if gather else F.cross_entropy(logits.reshape(-1,50257),y.reshape(-1)))
    params=list(m.named_parameters())
    gradients=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
    result={n: None if g is None else g.detach().cpu().clone() for (n,_),g in zip(params,gradients)}
    return float(loss.detach().cpu()),result


def cache_and_causality(m,device):
    arm=m.signature['arm'];rows=[]
    with torch.no_grad():
        sequence=ids(2,16,device);x=sequence[:,:-2];changed=x.clone();changed[:,9:]=(changed[:,9:]+701)%50257
        h=trunk(m,x);p=head(m,h,auxiliary=False);hc=trunk(m,changed);pc=head(m,hc,auxiliary=False)
        rows.append(compare('primary_future_causality',pc[:,:9],p[:,:9],0.,0.))
        if arm!='native_next': rows.append(compare('aux_future_causality',head(m,hc,auxiliary=True)[:,:9],head(m,h,auxiliary=True)[:,:9],0.,0.))
        target_changed=sequence.clone();target_changed[:,-2:]=(target_changed[:,-2:]+809)%50257
        ht=trunk(m,target_changed[:,:-2]);rows.append(compare('primary_target_only',head(m,ht,auxiliary=False),p,0.,0.))
        if arm!='native_next': rows.append(compare('aux_target_only',head(m,ht,auxiliary=True),head(m,h,auxiliary=True),0.,0.))
        del sequence,x,changed,h,p,hc,pc,target_changed,ht
        for batch in (1,8):
            x=ids(batch,128,device)[:,:-2];cache=m.caches(batch)
            size=sum(t.numel()*t.element_size() for c in cache.layers for t in (c.c_cache,c.r_cache))
            rows.append(fact('cache_bytes_'+str(batch),size==196608*batch,bytes=size))
            full=m(x).values
            for reset in (0,1):
                cache.reset()
                first=m(x[:,:61],cache=cache,start_pos=0).values
                second=m(x[:,61:],cache=cache,start_pos=61).values
                cached=torch.cat((first,second),dim=1);del first,second
                rows.append(compare(f'raw_B{batch}_reset{reset}',cached,full,3e-5,3e-5))
                # Chunk log-probability comparison avoids two further fullB8
                # vocabulary buffers; normalization remains the genuine kernel.
                for start,end in ((0,61),(61,128)):
                    rows.append(compare(f'LP_B{batch}_reset{reset}_{start}',F.log_softmax(cached[:,start:end],dim=-1),F.log_softmax(full[:,start:end],dim=-1),3e-5,3e-5))
                rows.append(fact(f'position_B{batch}_reset{reset}',all(c.position==128 for c in cache.layers)))
                del cached
            del full,cache,x
    checked(arm+'-'+device+'-causality-cache',rows)


def backend_and_full_batch(arm,seed):
    with CpuPrerequisiteGuard(R/(arm+'-backend-init-cpu-guard.json'),ROOT,timeout_seconds=120,poll_seconds=.25):
        cpu=own(arm,seed,'cpu');mps=own(arm,seed,'mps')
    with CpuPrerequisiteGuard(R/(arm+'-backend-cpu-guard.json'),ROOT,timeout_seconds=120,poll_seconds=.25):
        sequence=ids(2,11,'cpu');before=sequence.clone()
        cr=sequential_backward(cpu,sequence)
        cg={n:p.grad.detach().clone() for n,p in cpu.named_parameters()};cpu.zero_grad(set_to_none=True)
        cp=cpu(sequence[:,:-2]).values.detach();cl=float(cpu.loss(sequence[:,:-2],sequence[:,1:-1]).detach())
        checked(arm+'-CPU-input-preserved',[compare('sequence',sequence,before,0.,0.)])
    mr=sequential_backward(mps,sequence.to('mps'))
    checked(arm+'-CPU-MPS-primary-objective-gradients',[compare('primary_logits',mps(sequence[:,:-2].to('mps')).values,cp,4e-5,4e-4),
        scalar('primary_CE',float(mps.loss(sequence[:,:-2].to('mps'),sequence[:,1:-1].to('mps')).detach().cpu()),cl)]+
        [scalar(k,mr[k],cr[k]) for k in cr if cr[k] is not None]+
        [compare(n,p.grad,cg[n],4e-5,4e-4) for n,p in mps.named_parameters()])
    mps.zero_grad(set_to_none=True);del cg,cp,sequence,before
    for device,m in [('cpu',cpu),('mps',mps)]:
        with CpuPrerequisiteGuard(R/(arm+'-'+device+'-gather-guard.json'),ROOT,timeout_seconds=120,poll_seconds=.25):
            # NEW B1T13=13 positions, independent of the objective B2T11 case.
            sequence=ids(1,13,device)
            loss,grads=primary_gradients(m,sequence,False)
            gather,gg=primary_gradients(m,sequence,True)
            rows=[scalar('target_CE_vs_full_gather',gather,loss)]
            for name in grads:
                rows.append(fact('unused_aux.'+name,gg[name] is None) if grads[name] is None else compare(name,gg[name],grads[name],0.,0.))
            checked(arm+'-'+device+'-full-gather-gradients',rows)
            del sequence,grads,gg,rows
    # Two small backend models are needed only for equality. Release CPU now.
    del cpu,m
    release()
    # Reuse the sole MPS model, then migrate it to CPU and back for primary cache
    # proofs; there is never a second resident model in a fullB8 phase.
    for device in ('mps','cpu'):
        mps=mps.to(device)
        with CpuPrerequisiteGuard(R/(arm+'-'+device+'-cache-guard.json'),ROOT,timeout_seconds=120,poll_seconds=.25):
            cache_and_causality(mps,device)
        release()
    mps=mps.to('mps')
    sequence=ids(8,128,'mps');before=sequence.clone()
    report=sequential_backward(mps,sequence)
    checked(arm+'-full-B8T128-sequential-backward',
        [fact(n,p.grad is not None and p.grad.shape==p.shape and p.grad.device==p.device and p.grad.dtype==p.dtype and bool(torch.isfinite(p.grad).all()))
         for n,p in mps.named_parameters()]+[compare('caller_sequence',sequence,before,0.,0.),fact('no_optimizer_or_step_in_phase',True,report=report)])
    del mps,sequence,before
    release()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    assert sha(R/'plan.json')==arg.plan_sha and set(p.name for p in R.iterdir())=={'plan.json','source-review.json','admission.json'}
    plan=json.loads((R/'plan.json').read_text())
    assert plan['initializer_census']['status']=='required_before_execution'
    census_root=ROOT/plan['initializer_census']['directory']
    assert sha(census_root/'plan.json')==plan['initializer_census']['plan_sha256']
    census=json.loads((census_root/'result.json').read_text())
    assert census['initializer']==json.loads((census_root/'plan.json').read_text())['initializer']
    assert census['status']=='new_future_token_initializer_unused_before_construction' and census['prior_matches']==[]
    admission=json.loads((R/'admission.json').read_text())
    assert admission['status']=='accepted_new_census_and_source_resource_gate_only_generated_proof'
    assert admission['proof_source_review_sha256']==sha(R/'source-review.json')
    assert admission['census_result_sha256']==sha(census_root/'result.json') and admission['census_STOP_sha256']==sha(census_root/'STOP')
    assert json.loads((census_root/'STOP').read_text())['status']=='complete_original_future_token_initializer_census'
    for name,h in plan['execution_sources'].items():assert sha(ROOT/name)==h,name
    assert shutil.disk_usage(ROOT).free>=plan['complete_target_bytes']
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)
        assert str(torch.__version__)==plan['torch_version'] and torch.version.git_version==plan['torch_git']
        assert all(os.environ.get(k)==v for k,v in plan['environment'].items())
        publish('attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha,initializer=census['initializer']))
        cpu_rng=mps_rng=None
        try:
            with StrictMpsGuard(R/'mps-guard.json',ROOT):
                cpu_rng,mps_rng=torch.get_rng_state().clone(),torch.mps.get_rng_state().clone()
                common,draw=native_pair(census['initializer']);paired=None
                for arm in ('duplicate_next','future_two'):
                    paired=auxiliary_pair(arm,census['initializer'],common,draw,paired)
                for arm in ARMS:backend_and_full_batch(arm,census['initializer'])
                checked('caller-global-RNG',[compare('CPU',torch.get_rng_state(),cpu_rng,0.,0.),compare('MPS',torch.mps.get_rng_state(),mps_rng,0.,0.)])
                for name,h in plan['execution_sources'].items():assert sha(ROOT/name)==h,name
                publish('result.json',dict(status='complete_original_generated_future_token_model_proof',reports=ROWS,
                    actual_optimizer_steps=6,full_B8T128_sequential_backwards=3,tensor_files=0,
                    corpus_checkpoint_scoring_quality_or_resume_calls=0,source_global_cpu_mps_rng_unchanged=True))
            publish('STOP',dict(status='complete_original_generated_future_token_model_proof',no_retry=True,no_replay=True))
        except BaseException as error:
            if not (R/'STOP').exists():publish('STOP',dict(status='closed_failed',error=repr(error),no_retry=True,no_replay=True))
            raise
        finally:
            if cpu_rng is not None:torch.set_rng_state(cpu_rng)
            if mps_rng is not None:torch.mps.set_rng_state(mps_rng)


if __name__=='__main__':main()
