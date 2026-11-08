"""Own complete training-context SOURCE; numerical admission still required.

Reuses the frozen paired sampler, clipping, cooldown and H200 arithmetic.
No worker/quality/scorer admission follows from this source. Outer entry supplies
fresh audited roles, source/runtime binding, guards and complete output forecast.
The frozen outer constructor binds native initialization to its declared role;
this context records that identity and does not rerun an initializer to infer it.
"""
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
import tiktoken
from omegaconf import OmegaConf
from research import byte_input_state_v2 as numerical
from research.byte_input_model_v3 import ByteInputModel, ByteInputOptimizer
from research.byte_input_model_v3.contract import require, same_typed
from research.block_norm_v1.runtime.data.loaders import _get_batch
from research.block_norm_v1.runtime.models.output_bias import digest
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule, schedule_contract, multiplier
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages, averaging_contract
from research.gradient_clip_model_v1.clipping import clip_gradients

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001'
KIND='byte_input_training_context_v2'


def recipe(steps, learning_rate):
    require(type(steps) is int and steps in (4,20,1200), 'Declared fixture or scientific horizon')
    require(type(learning_rate) is float and learning_rate in (.000625,.00125,.0025), 'Declared base LR')
    original=json.loads((BASE/'quality002-slot03-evidence001/config.json').read_text())
    train=copy.deepcopy(original['training'])
    for key in ('seed','eval_seed','device','checkpoint_dir','metrics_file','summary_file','resume_from'):
        train.pop(key)
    start={4:2,20:13,1200:800}[steps]
    train.update(max_iters=steps,learning_rate=learning_rate)
    train['lr_schedule'].update(total_steps=steps,start_step=start)
    train['weight_averaging'].update(total_steps=steps,start_step=start,interval=10 if steps==1200 else 1)
    return train


def roles(value):
    wanted={'native_initializer','composer_initializer','nonlexical_key','train_sampler','validation_sampler'}
    require(type(value) is dict and set(value)==wanted and all(type(x) is int and 0<=x<2**63 for x in value.values())
            and len(set(value.values()))==len(wanted), 'Five explicit distinct roles required')


class View(SimpleNamespace):
    def get_batch(self, split='train'):
        require(split in ('train','val'), 'No holdout sampler')
        return _get_batch(self,split)


def load_view(audited_roles, backend):
    roles(audited_roles);require(backend in ('cpu','mps'), 'Own CPU/MPS view')
    cfg=json.loads((BASE/'quality002-slot03-evidence001/config.json').read_text())['data']
    expected=json.loads((BASE/'quality002-plan.json').read_text())['data']
    path=Path(cfg['path']);raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected['corpus_sha256'], 'Original corpus hash')
    text=raw.decode('utf8');a=int(.8*len(text));b=int((.8+.1)*len(text))
    require((len(text),a,b)==(expected['raw_characters'],expected['raw_train_end'],expected['raw_val_end']), 'Original raw split boundaries')
    enc=tiktoken.get_encoding('gpt2')
    tokenizer=dict(name='gpt2',implementation='tiktoken',version=importlib.metadata.version('tiktoken'),
        vocabulary_sha256=hashlib.sha256(json.dumps(sorted((t.hex(),r)for t,r in enc._mergeable_ranks.items())).encode()).hexdigest(),
        pattern=enc._pat_str,special_tokens=dict(enc._special_tokens),special_token_policy='ordinary_text',split_unit='raw_text',vocab_size=enc.n_vocab)
    require(tokenizer==expected['tokenizer'], 'Original tokenizer identity')
    values={n:enc.encode(s,disallowed_special=())for n,s in [('train',text[:a]),('val',text[a:b])]}
    require(len(values['train'])==267688 and len(values['val'])==34279
            and digest(values['train'])==expected['train_ids_sha256']
            and digest(values['val'])==expected['validation_ids_sha256']
            and digest(values['val'][1:])==expected['validation_target_ids_sha256'], 'Original native train/validation IDs')
    require(hashlib.sha256(path.read_bytes()).hexdigest()==expected['corpus_sha256'], 'Corpus unchanged during load')
    tensors={n:torch.tensor(v,dtype=torch.int64)for n,v in values.items()}
    return View(train_data=tensors['train'],val_data=tensors['val'],splits=tensors,
        generators={n:torch.Generator(device='cpu').manual_seed(audited_roles[k])for n,k in [('train','train_sampler'),('val','validation_sampler')]},
        batch_size=8,block_size=128,batch_layout='paired_blocks512',vocab_size=50257,device=backend,
        data_identity=expected,tokenizer_contract=tokenizer)


def check_view(view, audited_roles):
    roles(audited_roles)
    require(type(view) is View and set(view.splits)==set(view.generators)=={'train','val'}
            and view.batch_size==8 and view.block_size==128 and view.batch_layout=='paired_blocks512'
            and view.vocab_size==50257 and view.device in ('cpu','mps'), 'Own paired native view')
    expected=json.loads((BASE/'quality002-plan.json').read_text())['data']
    require(same_typed(view.data_identity,expected) and same_typed(view.tokenizer_contract,expected['tokenizer']), 'Bound original data identity')
    for name,size,key,seed_key in [('train',267688,'train_ids_sha256','train_sampler'),('val',34279,'validation_ids_sha256','validation_sampler')]:
        value=view.splits[name];gen=view.generators[name]
        require(value is getattr(view,name+'_data') and type(value) is torch.Tensor and value.device.type=='cpu'
                and value.dtype==torch.int64 and value.layout==torch.strided and value.is_contiguous()
                and tuple(value.shape)==(size,) and digest(value.tolist())==expected[key], 'Complete immutable native data split')
        require(type(gen) is torch.Generator and gen.device.type=='cpu' and gen.initial_seed()==audited_roles[seed_key], 'Bound private sampler seed')
    return copy.deepcopy(expected)


class CanonicalParameters:
    def __init__(self, model): self.model=model
    def named_parameters(self): return iter(self.model.ordered_parameters())


class Context:
    def __init__(self, model, optimizer, view, steps, learning_rate, audited_roles, admitted_binding):
        require(type(model) is ByteInputModel and type(optimizer) is ByteInputOptimizer and optimizer.model is model
                and optimizer._steps==0, 'Fresh own model/optimizer context')
        numerical.binding(admitted_binding);check_view(view,audited_roles)
        require(view.device==model.validate().type, 'Sampler destination equals model device')
        for index,child in enumerate(optimizer._optimizers):
            require(all(g['lr']==learning_rate*(.5 if index==0 else 1.)for g in child.param_groups), 'Declared unscheduled initial LR')
        if model.spec.variant!='native':
            require(model.spec.composer_seed==audited_roles['composer_initializer'], 'Bound composer initializer')
            if model.spec.variant=='nonlexical_codes':
                require(model.spec.codebook_identity[2]==audited_roles['nonlexical_key'], 'Bound nonlexical code key')
        self.model=model;self.optimizer=optimizer;self.view=view
        self.roles=copy.deepcopy(audited_roles);self.binding=copy.deepcopy(admitted_binding)
        self.recipe=recipe(steps,learning_rate);self.training=OmegaConf.create(self.recipe)
        self.schedule=LearningRateSchedule(optimizer,self.training)
        self.averages=TailWeightAverages(CanonicalParameters(model),self.training)
        self.history=[]
        self._identity=dict(kind=KIND,recipe=copy.deepcopy(self.recipe),roles=copy.deepcopy(self.roles),
                            binding=copy.deepcopy(self.binding),data=copy.deepcopy(view.data_identity),model=model.identity)

    def check(self):
        parameters=numerical.live(self.model,self.optimizer)
        check_view(self.view,self.roles)
        require(self.view.device==next(iter(parameters.values())).device.type, 'Sampler device must match before any draw or restore')
        require(same_typed(self.recipe,recipe(self.recipe['max_iters'],self.recipe['learning_rate']))
                and same_typed(OmegaConf.to_container(self.training,resolve=True),self.recipe)
                and same_typed(self._identity,dict(kind=KIND,recipe=self.recipe,roles=self.roles,
                      binding=self.binding,data=self.view.data_identity,model=self.model.identity)), 'Unchanged training identity')
        require(type(self.averages) is TailWeightAverages and type(self.averages.model) is CanonicalParameters
                and self.averages.model.model is self.model and self.averages.completed_steps==self.optimizer._steps
                and same_typed(self.averages.spec,averaging_contract(self.training)), 'Aligned own H200 state')
        self.averages._check_parameters()
        require(len(self.averages.parameters)==len(parameters) and all(n==old and p is original
                for (n,p),(old,original) in zip(self.averages.parameters,parameters.items())), 'Canonical averaging parameter ownership')
        spec=self.averages.spec;step=self.optimizer._steps
        snapshots=0 if step<spec['start_step'] else 1+(step-spec['start_step'])//spec['interval']
        require(type(self.averages.snapshots) is int and self.averages.snapshots==snapshots
                and same_typed(self.averages.betas,{'200':2.**(-spec['interval']/200)})
                and type(self.averages.averages) is dict and set(self.averages.averages)==({'200'} if snapshots else set()), 'Exact live H200 counters and coefficients')
        live=list(parameters.values())+list(self.model.buffers())
        live += [v for child in self.optimizer._optimizers for fields in child.state.values()for v in fields.values()]
        seen={v.untyped_storage().data_ptr()for v in live}
        for panel in self.averages.averages.values():
            require(type(panel) is dict and list(panel)==list(parameters), 'Complete live H200 panel')
            for name,value in panel.items():
                parameter=parameters[name]
                require(type(value) is torch.Tensor and value.layout==torch.strided and value.is_contiguous()
                        and value.dtype==torch.float32 and value.device==parameter.device and value.shape==parameter.shape
                        and not value.requires_grad and bool(torch.isfinite(value).all())
                        and value.untyped_storage().data_ptr()not in seen, 'Finite independent live H200 tensor')
                seen.add(value.untyped_storage().data_ptr())
        history(self.history,self.optimizer._steps,list(parameters),self.recipe['max_iters'])
        require(type(self.schedule) is LearningRateSchedule and self.schedule.optimizer is self.optimizer
                and same_typed(self.schedule.spec,schedule_contract(self.training))
                and self.schedule.base_lrs==tuple(self.recipe['learning_rate']*(.5 if i==0 else 1.)
                    for i,child in enumerate(self.optimizer._optimizers)for _ in child.param_groups), 'Own fixed schedule and original base rates')
        return parameters

    def step(self):
        parameters=self.check();step=self.optimizer._steps
        require(step<self.recipe['max_iters'], 'Fixed horizon reached')
        self.schedule.apply(step);self.model.train()
        x,y=self.view.get_batch('train');result=self.model.forward_train(x,y)
        loss=float(result.loss.detach());require(math.isfinite(loss) and loss>=0., 'Finite ordinary CE')
        result.loss.backward();del result
        self.optimizer.validate(require_gradients=True)
        observed=clip_gradients(parameters.values(),1.)
        require(observed['gradient_tensors']==len(parameters), 'Every canonical parameter receives a gradient')
        self.optimizer.step();self.averages.update(step+1)
        self.history.append(dict(step=step+1,**observed,active_parameter_names=list(parameters),ce_loss=loss))
        self.optimizer.zero_grad(set_to_none=True)
        self.check()
        return copy.deepcopy(self.history[-1])


def history(rows, step, names, horizon):
    require(type(step) is int and 0<=step<=horizon and type(rows) is list and len(rows)==step, 'Complete bounded clipping history')
    for i,row in enumerate(rows,1):
        require(type(row) is dict and set(row)=={'step','threshold','pre_norm','post_norm','coefficient','gradient_tensors','active_parameter_names','ce_loss'}
                and type(row['step']) is int and row['step']==i and type(row['threshold']) is float and row['threshold']==1.
                and type(row['gradient_tensors']) is int and row['gradient_tensors']==len(names) and row['active_parameter_names']==names, 'Exact per-parameter clipping history')
        require(all(type(row[k]) is float and math.isfinite(row[k])for k in ('pre_norm','post_norm','coefficient','ce_loss'))
                and min(row['pre_norm'],row['post_norm'],row['ce_loss'])>=0., 'Finite nonnegative history')
        c=min(1.,1./(row['pre_norm']+1e-6))
        require(0<=row['coefficient']<=1 and row['post_norm']<=1.+1e-5 and abs(row['coefficient']-c)<2e-5
                and abs(row['post_norm']-row['pre_norm']*row['coefficient'])<=max(1e-5,row['post_norm']*2e-5), 'Original clipping arithmetic')


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return numerical.clone(value)
    if isinstance(value,dict):return {k:cpu_tree(v)for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v)for v in value]
    return copy.deepcopy(value)


def capture(context):
    require(type(context) is Context,'Own training context')
    context.check()
    value=dict(kind=KIND,identity=copy.deepcopy(context._identity),numerical=numerical.capture(context.model,context.optimizer,context.binding),
        averages=cpu_tree(context.averages.state_dict()),history=copy.deepcopy(context.history),
        samplers={n:g.get_state().clone()for n,g in context.view.generators.items()})
    validate(context,value)
    return value


def validate(context,value):
    require(type(context) is Context,'Own training context')
    parameters=context.check()
    require(type(value) is dict and set(value)=={'kind','identity','numerical','averages','history','samplers'}
            and value['kind']==KIND and same_typed(value['identity'],context._identity), 'Exact own training envelope')
    step=numerical.validate(context.model,context.optimizer,value['numerical'],context.binding)
    history(value['history'],step,list(parameters),context.recipe['max_iters'])
    ratio=multiplier(context.schedule.spec,step-1) if step else 1.
    for index,child in enumerate(value['numerical']['children']):
        require(all(g['options']['lr']==context.recipe['learning_rate']*(.5 if index==0 else 1.)*ratio for g in child['groups']), 'Exact absolute scheduled LR')
    saved=value['averages'];spec=averaging_contract(context.training)
    count=0 if step<spec['start_step'] else 1+(step-spec['start_step'])//spec['interval']
    require(type(saved) is dict and set(saved)=={'version','spec','completed_steps','snapshots','parameter_names','averages'}
            and type(saved['version']) is int and saved['version']==1 and same_typed(saved['spec'],spec)
            and type(saved['completed_steps']) is int and saved['completed_steps']==step
            and type(saved['snapshots']) is int and saved['snapshots']==count and saved['parameter_names']==list(parameters)
            and type(saved['averages']) is dict and set(saved['averages'])==({'200'} if count else set()), 'Complete canonical H200 state')
    seen=set()
    def collect(x):
        if isinstance(x,torch.Tensor):seen.add(x.untyped_storage().data_ptr())
        elif isinstance(x,dict):
            for y in x.values():collect(y)
        elif isinstance(x,(tuple,list)):
            for y in x:collect(y)
    collect(value['numerical']);collect(parameters);collect(dict(context.model.named_buffers()))
    collect([fields for child in context.optimizer._optimizers for fields in child.state.values()])
    collect(context.averages.averages)
    for panel in saved['averages'].values():
        require(type(panel) is dict and list(panel)==list(parameters), 'Ordered complete EMA inventory')
        for name,tensor in panel.items():numerical._tensor(tensor,parameters[name].shape,seen)
    require(type(value['samplers']) is dict and set(value['samplers'])=={'train','val'}, 'Exact private sampler inventory')
    for name,key in [('train','train_sampler'),('val','validation_sampler')]:
        state=value['samplers'][name];numerical._tensor(state,context.view.generators[name].get_state().shape,seen,dtype=torch.uint8)
        private=torch.Generator(device='cpu').set_state(state)
        require(private.initial_seed()==context.roles[key], 'Saved private sampler identity')
    return step


def restore_pristine(context,value):
    require(type(context) is Context,'Own training context')
    require(context.optimizer._steps==0 and context.averages.completed_steps==0 and not context.history, 'Fresh unexposed training destination only')
    step=validate(context,value)
    # All decoding and allocations precede the numerical commit. A resource/driver
    # failure after a valid commit discards this unexposed context; never retry it.
    staged=TailWeightAverages(CanonicalParameters(context.model),context.training)
    staged.load_state_dict(value['averages'],step)
    generators={n:torch.Generator(device='cpu').set_state(v)for n,v in value['samplers'].items()}
    rows=copy.deepcopy(value['history'])
    numerical.restore_pristine(context.model,context.optimizer,value['numerical'],context.binding)
    context.averages=staged;context.view.generators=generators;context.history=rows
    context.check()
    return step


def write_new(context,path):
    """Capture and publish one complete RAW context, preserving failed temporaries."""
    import os
    require(type(context) is Context,'Own training context')
    target=numerical._path(path);temporary=target.with_suffix(target.suffix+'.tmp')
    require(not target.exists() and not temporary.exists(),'Exclusive own RAW output')
    payload=capture(context)
    class Bounded:
        def __init__(self,stream):self.stream=stream
        def write(self,value):
            require(self.stream.tell()+len(value)<=numerical.FILE_CAP,'Complete RAW serialization cap')
            return self.stream.write(value)
        def flush(self):return self.stream.flush()
        def tell(self):return self.stream.tell()
    with temporary.open('xb') as stream:
        torch.save(payload,Bounded(stream));stream.flush();os.fsync(stream.fileno())
    os.link(temporary,target);temporary.unlink()
    descriptor=os.open(target.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
    return dict(kind=KIND,path=str(target),size=target.stat().st_size,sha256=numerical._sha(target))


def read_bound(receipt):
    require(type(receipt) is dict and set(receipt)=={'kind','path','size','sha256'} and receipt['kind']==KIND
            and type(receipt['path']) is str and type(receipt['size']) is int and 0<receipt['size']<=numerical.FILE_CAP
            and type(receipt['sha256']) is str and len(receipt['sha256'])==64
            and all(c in '0123456789abcdef'for c in receipt['sha256']),'Exact own RAW receipt')
    path=numerical._path(receipt['path'])
    require(path.is_file() and path.stat().st_nlink==1 and path.stat().st_size==receipt['size']
            and numerical._sha(path)==receipt['sha256'],'Exact retained RAW before read')
    value=torch.load(path,map_location='cpu',weights_only=True)
    require(path.stat().st_size==receipt['size'] and numerical._sha(path)==receipt['sha256'],'RAW unchanged during read')
    return value  # validate(context, value) remains mandatory before restoration.
