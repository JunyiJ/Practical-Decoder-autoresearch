"""New state proof: no historical numerical entry is called."""
import gc
import hashlib
import json
from pathlib import Path
import torch
from research.byte_input_model_v3 import Spec, ByteInputModel, build_codebook, build_optimizer
from research import byte_input_state_v2 as adapter
from research.byte_input_proof_v1.cases import inputs
from research.byte_input_proof_v1.codebook_reference import raw_rows
from research.byte_input_proof_v1.transactions import tensor_record, snapshot
from .cases import BACKENDS, VARIANTS, SHAPE, REJECTIONS, IO_REJECTIONS


def cleanup():
    gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while x := f.read(1048576): h.update(x)
    return h.hexdigest()


def tree(value):
    if isinstance(value, torch.Tensor): return tensor_record(value, False)
    if isinstance(value, dict): return {k: tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [tree(v) for v in value]
    return value


def copy_tree(value):
    # Metadata copies only; each injected tensor replacement is separately owned.
    if isinstance(value, dict): return {k: copy_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [copy_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(copy_tree(v) for v in value)
    return value


def semantic(model, opt):
    parameters = dict(model.ordered_parameters())
    by_id = {id(p): n for n,p in parameters.items()}
    return dict(parameters=tree(parameters), buffers=tree(dict(model.named_buffers())),
        children=[dict(kind=type(child).__name__, groups=[dict(name=by_id[id(g['params'][0])],
            options={k:v for k,v in g.items() if k!='params'}) for g in child.param_groups],
            state={by_id[id(p)]:tree(v) for p,v in child.state.items()}) for child in opt._optimizers],
        steps=opt._steps, training=model.training, cpu_rng=tree(torch.random.get_rng_state()),
        mps_rng=tree(torch.mps.get_rng_state()))


def cpu_record(value):
    result = tensor_record(value, False)
    result['device'] = 'cpu'
    result['requires_grad'] = False
    return result


def verify_capture(model, opt, payload):
    # Independent walk of actual live fields, not capture->capture comparison.
    parameters = dict(model.ordered_parameters())
    assert list(payload['parameters']) == list(parameters)
    for name,p in parameters.items(): assert tree(payload['parameters'][name]) == cpu_record(p)
    for saved, child in zip(payload['children'], opt._optimizers):
        for group in child.param_groups:
            p=group['params'][0]; name=next(n for n,v in parameters.items() if v is p)
            if p not in child.state: assert name not in saved['state']; continue
            for key,value in child.state[p].items(): assert tree(saved['state'][name][key]) == cpu_record(value)
    assert torch.equal(payload['cpu_rng'], torch.random.get_rng_state())
    assert torch.equal(payload['mps_rng'], torch.mps.get_rng_state())


def construct(plan, spec, book, backend):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(plan['roles']['native_initializer'])
        model=ByteInputModel(spec,book)
    return model.to(backend)


def mutate(name, p):
    result=copy_tree(p)
    params=result['parameters']; small=min(params,key=lambda n:params[n].numel())
    muon,adam=result['children']; mn=next(iter(muon['state'])); an=next(iter(adam['state']))
    ms=muon['state'][mn]; ast=adam['state'][an]
    if name=='extra_field': result['foreign']=True
    elif name=='missing_field': del result['training']
    elif name=='kind': result['kind']='foreign'
    elif name=='identity': result['identity']=('foreign',)
    elif name=='binding': result['binding']['source_sha256']='0'*64
    elif name=='backend': result['backend']='cuda:0'
    elif name=='training_type': result['training']=1
    elif name=='negative_step': result['completed_steps']=-1
    elif name=='bool_step': result['completed_steps']=True
    elif name=='missing_parameter': del params[small]
    elif name=='reordered_parameters': result['parameters']=dict(reversed(list(params.items())))
    elif name=='nonfinite_parameter': params[small]=torch.full_like(params[small],float('nan'))
    elif name=='parameter_shape': params[small]=params[small].reshape(*params[small].shape,1).clone()
    elif name=='parameter_dtype': params[small]=params[small].double()
    elif name=='parameter_requires_grad': params[small]=params[small].clone().requires_grad_()
    elif name=='parameter_alias':
        names=list(params)
        pairs=[(a,b) for i,a in enumerate(names) for b in names[i+1:] if params[a].shape==params[b].shape]
        assert pairs, 'Actual equal-shape pair required for the alias fixture'
        target,source=min(pairs,key=lambda pair:(params[pair[0]].numel(),pair))
        params[target]=params[source]
    elif name=='buffer_identity': result['immutable_buffer_hashes']={}
    elif name=='child_kind': muon['kind']='AdamW'
    elif name=='missing_child': result['children']=result['children'][:1]
    elif name=='group_name': muon['groups'][0]['name']='foreign'
    elif name=='group_option': muon['groups'][0]['options']['weight_decay']=1.
    elif name=='lr_mismatch': muon['groups'][0]['options']['lr']=.9
    elif name=='missing_state': del muon['state'][mn]
    elif name=='extra_state_field': ms['foreign']=torch.zeros(())
    elif name=='state_shape': ms['momentum_buffer']=torch.zeros(1)
    elif name=='state_dtype': ms['momentum_buffer']=ms['momentum_buffer'].double()
    elif name=='state_nonfinite': ms['momentum_buffer']=torch.full_like(ms['momentum_buffer'],float('inf'))
    elif name=='negative_variance': ast['exp_avg_sq']=-torch.ones_like(ast['exp_avg_sq'])
    elif name=='adam_step': ast['step']=torch.tensor(2.)
    elif name=='state_alias': ast['exp_avg_sq']=ast['exp_avg']
    elif name=='cpu_rng_shape': result['cpu_rng']=torch.zeros(1,dtype=torch.uint8)
    elif name=='cpu_rng_encoding': result['cpu_rng']=torch.zeros_like(result['cpu_rng'])
    elif name=='mps_rng_dtype': result['mps_rng']=result['mps_rng'].float()
    else: raise AssertionError(name)
    return result


def invalid(name, model, opt, payload, binding, x, y, publish, prefix):
    bad=mutate(name,payload)
    before=snapshot(model,opt,x,y); payload_before=tree(bad)
    error=None
    try: adapter.restore_pristine(model,opt,bad,binding)
    except (ValueError,TypeError,RuntimeError) as exc: error=repr(exc)
    after=snapshot(model,opt,x,y); payload_after=tree(bad)
    publish(prefix+'-reject-'+name+'.json',dict(name=name,error=error,before=before,after=after,
                payload_before=payload_before,payload_after=payload_after))
    assert error and before==after and payload_before==payload_after, name


def step(model,opt,x,y,publish,label):
    model.train()
    result=model.forward_train(x,y)
    assert tuple(result.primary.values.shape)==SHAPE
    assert bool(torch.isfinite(result.primary.values).all()) and bool(torch.isfinite(result.loss))
    output=dict(logits=tree(result.primary.values),loss=tree(result.loss))
    publish(label+'-forward.json',output)
    result.loss.backward(); del result; cleanup()
    gradients={n:tree(p.grad) for n,p in model.ordered_parameters()}
    assert all(v['shape']==list(p.shape) for (_,p),v in zip(model.ordered_parameters(),gradients.values()))
    publish(label+'-gradients.json',gradients)
    if 'composer.byte_weight' in gradients:
        byte_gradient=dict(model.ordered_parameters())['composer.byte_weight'].grad.detach().cpu()
        publish(label+'-byte-gradient.json',dict(shape=list(byte_gradient.shape),values=byte_gradient.tolist()))
        del byte_gradient
    opt.step();opt.zero_grad(set_to_none=True);cleanup()
    return dict(output=output,gradients=gradients)


def draw():
    return dict(cpu=tree(torch.rand(257)), mps=tree(torch.rand(257,device='mps')))


def one(plan, archives, publish, rows, books, backend,variant):
    prefix=backend+'-'+variant;binding=plan['binding'];book=books.get(variant)
    spec=Spec(variant) if book is None else Spec(variant,plan['roles']['composer_initializer'],book.identity)
    model=construct(plan,spec,book,backend);opt=build_optimizer(model,.00125)
    x,y=inputs(torch,rows,backend)
    # Empty state is a real supported boundary: capture and restore it before training.
    before=snapshot(model,opt,x,y);zero=adapter.capture(model,opt,binding);verify_capture(model,opt,zero)
    assert snapshot(model,opt,x,y)==before
    before_semantic=semantic(model,opt);assert adapter.restore_pristine(model,opt,zero,binding)==0
    assert semantic(model,opt)==before_semantic
    publish(prefix+'-step0.json',dict(before=before_semantic,after=semantic(model,opt),exact=True))
    del zero,before,before_semantic;cleanup()
    step(model,opt,x,y,publish,prefix+'-step1')
    for index,child in enumerate(opt._optimizers):
        for g in child.param_groups:g['lr']=.000625*(.5 if index==0 else 1.)
    model.eval()  # Explicit mode and changed LR must survive restore.
    expected_start=semantic(model,opt);before=snapshot(model,opt,x,y)
    payload=adapter.capture(model,opt,binding);verify_capture(model,opt,payload)
    assert snapshot(model,opt,x,y)==before
    receipt=adapter.write_new((archives/(prefix+'-step1.pt')).resolve(),payload)
    publish(prefix+'-archive-step1.json',receipt)
    loaded=adapter.read_bound(receipt)
    assert tree(loaded)==tree(payload)
    verify_capture(model,opt,loaded);del payload,before;cleanup()
    # Receipt rejection and exclusive publication preserve original bytes and RNG.
    for name in IO_REJECTIONS:
        r=dict(receipt);before=snapshot(model,opt,x,y);old_hash=sha(receipt['path']);error=None
        if name=='receipt_hash':r['sha256']='0'*64
        elif name=='receipt_size':r['size']+=1
        elif name=='receipt_kind':r['kind']='foreign'
        elif name=='receipt_path':r['path']=str(Path(receipt['path']).parent/'..'/'foreign.pt')
        try:
            if name=='overwrite':adapter.write_new(Path(receipt['path']),loaded)
            else:adapter.read_bound(r)
        except (ValueError,TypeError,RuntimeError) as exc:error=repr(exc)
        after=snapshot(model,opt,x,y)
        publish(prefix+'-io-'+name+'.json',dict(name=name,error=error,before=before,after=after,
                original_sha256=old_hash,after_sha256=sha(receipt['path'])))
        assert error and before==after and old_hash==sha(receipt['path']),name
    expected_draw=draw()
    expected_step=step(model,opt,x,y,publish,prefix+'-control-next')
    expected_end=semantic(model,opt)
    final=adapter.capture(model,opt,binding);verify_capture(model,opt,final)
    end_receipt=adapter.write_new((archives/(prefix+'-control-step2.pt')).resolve(),final)
    publish(prefix+'-archive-step2.json',end_receipt)
    del final,model,opt;cleanup()
    restored=construct(plan,spec,book,backend);restored_opt=build_optimizer(restored,.00125)
    for name in REJECTIONS:invalid(name,restored,restored_opt,loaded,binding,x,y,publish,prefix)
    # Advance both caller streams to demonstrate restoration, not accidental equality.
    draw()
    assert adapter.restore_pristine(restored,restored_opt,loaded,binding)==1
    actual_start=semantic(restored,restored_opt)
    publish(prefix+'-restore.json',dict(expected=expected_start,actual=actual_start))
    assert actual_start==expected_start
    actual_draw=draw();publish(prefix+'-rng.json',dict(expected=expected_draw,actual=actual_draw))
    assert actual_draw==expected_draw
    actual_step=step(restored,restored_opt,x,y,publish,prefix+'-restored-next')
    assert actual_step==expected_step,'Complete next forward/loss/gradient bitwise equality'
    actual_end=semantic(restored,restored_opt)
    publish(prefix+'-next-state.json',dict(expected=expected_end,actual=actual_end))
    assert actual_end==expected_end,'Complete next parameters/state/RNG bitwise equality'
    # A populated destination is expressly forbidden, even for a valid payload.
    before=snapshot(restored,restored_opt,x,y);error=None
    try:adapter.restore_pristine(restored,restored_opt,loaded,binding)
    except ValueError as exc:error=repr(exc)
    after=snapshot(restored,restored_opt,x,y)
    publish(prefix+'-nonpristine.json',dict(error=error,before=before,after=after))
    assert error and before==after
    publish(prefix+'-complete.json',dict(backend=backend,variant=variant,shape=list(SHAPE),
        parameters=len(actual_end['parameters']),steps=2,payload_rejections=len(REJECTIONS),
        io_rejections=len(IO_REJECTIONS),nonpristine_rejections=1,exact=True,
        scope='Numerical state only; no sampler/schedule/clipping/H200/training-worker qualification'))
    del restored,restored_opt,loaded,x,y;cleanup()


def run(plan,archives,publish):
    assert torch.__version__==plan['runtime']['torch_version'] and torch.version.git_version==plan['runtime']['torch_git']
    assert torch.backends.mps.is_available();torch.set_num_threads(4)
    old_cpu=torch.random.get_rng_state().clone();old_mps=torch.mps.get_rng_state().clone()
    try:
        assets=[Path(pin['path']).read_bytes() for pin in plan['assets']]
        rows=raw_rows(*assets)
        books={v:build_codebook(*assets,v,None if v=='raw_bytes' else plan['roles']['nonlexical_key']) for v in VARIANTS if v!='native'}
        for backend in BACKENDS:
            for variant in VARIANTS:one(plan,archives,publish,rows,books,backend,variant)
    finally:
        torch.random.set_rng_state(old_cpu);torch.mps.set_rng_state(old_mps)
    assert torch.equal(old_cpu,torch.random.get_rng_state()) and torch.equal(old_mps,torch.mps.get_rng_state())
    publish('result.json',dict(accepted=True,cases=6,caller_rng_restored=True,exact=True,
        no_real_data_quality=True,numerical_checkpoint_only=True,roles=plan['roles']))
