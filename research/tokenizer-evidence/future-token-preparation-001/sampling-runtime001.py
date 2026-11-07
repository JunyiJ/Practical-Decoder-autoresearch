"""NEW CPU sampler draw/gather/private-state integration proof; no model/corpus."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[3]
R=Path(__file__).with_suffix('')
SERIAL=0


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw)<=2*1024**2
    with (R/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def check(label,flags,**details):
    global SERIAL
    SERIAL+=1
    report=dict(label=label,predicates=flags,passed=all(flags.values()),**details)
    publish(f'{SERIAL:03d}-{label}.json',report)
    assert report['passed'],label


def clone(value):
    if isinstance(value,torch.Tensor):return value.detach().clone()
    if isinstance(value,dict):return {k:clone(v) for k,v in value.items()}
    if isinstance(value,list):return [clone(v) for v in value]
    if isinstance(value,tuple):return tuple(clone(v) for v in value)
    return copy.deepcopy(value)


def equal(a,b):
    if isinstance(b,torch.Tensor):return isinstance(a,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and a.device==b.device and torch.equal(a,b)
    if isinstance(b,dict):return isinstance(a,dict) and list(a)==list(b) and all(equal(a[k],b[k]) for k in b)
    if isinstance(b,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def raw_sampler(s):
    # This deliberately avoids validating APIs while a structural negative is installed.
    return dict(data=clone(s.data),roles=copy.deepcopy(s.roles),signature=copy.deepcopy(s.signature),
        identity=copy.deepcopy(s.identity),signature_identity=s._signature_identity,calls=copy.deepcopy(s.completed_calls),
        generator_ids={k:id(v) for k,v in s.generators.items()},states={k:g.get_state().clone() for k,g in s.generators.items()})


def expected(split,data,generator):
    # Independent sampler reference: draw eligible anchors on a separate private
    # stream, then take514-token physical blocks, slice four130-token rows.
    length=len(data);count=2 if split=='train' else 8;span=514 if split=='train' else 130
    anchors=torch.randint(length-span+1,(count,),generator=generator,dtype=torch.int64,device='cpu')
    values=[];locations=[]
    if split=='train':
        for start in anchors.tolist():
            physical=[data[i] for i in range(start,start+514)]
            absolute=list(range(start,start+514))
            for left,right in ((0,130),(128,258),(256,386),(384,514)):
                values.append(physical[left:right]);locations.append(absolute[left:right])
    else:
        for start in anchors.tolist():
            values.append(data[start:start+130]);locations.append(list(range(start,start+130)))
    return dict(sequence=torch.tensor(values,dtype=torch.int64),positions=torch.tensor(locations,dtype=torch.int64),anchors=anchors)


def verify_sample(s,split,ref_generator,data,ordinal):
    before_inputs=clone(data);cpu_rng=torch.get_rng_state().clone()
    wanted=expected(split,data[split],ref_generator[split])
    other='val' if split=='train' else 'train';other_before=s.generators[other].get_state().clone()
    actual=s.next(split)
    flags={k:equal(actual[k],wanted[k]) for k in ('sequence','positions','anchors')}
    flags.update(private_reference_state=torch.equal(s.generators[split].get_state(),ref_generator[split].get_state()),
        other_private_stream_unchanged=torch.equal(s.generators[other].get_state(),other_before),
        global_CPU_RNG_unchanged=torch.equal(torch.get_rng_state(),cpu_rng),source_lists_unchanged=equal(data,before_inputs),
        shape=tuple(actual['sequence'].shape)==(8,130),dtype=actual['sequence'].dtype==torch.int64,
        cpu=actual['sequence'].device.type=='cpu',targets=actual['primary_targets']==actual['second_next_targets']==1024,
        split_tag=actual['policy']=='future_token_split_local_paired512_v1' and actual['split']==split,
        source_ownership=actual['sequence'].untyped_storage().data_ptr() not in {v.untyped_storage().data_ptr() for v in s.data.values()})
    # Every actual token and its offset1/2 follows its declared physical index.
    for offset in (0,1,2):
        flags['all1024_offset_'+str(offset)]=actual['sequence'][:,offset:128+offset].tolist()==[
            [data[split][i] for i in row[offset:128+offset]] for row in wanted['positions'].tolist()]
    check('draw-'+str(ordinal)+'-'+split,flags,anchors=actual['anchors'].tolist(),
        actual_sequence=actual['sequence'].tolist(),expected_sequence=wanted['sequence'].tolist(),
        actual_positions=actual['positions'].tolist(),expected_positions=wanted['positions'].tolist())
    return actual


def run(roles):
    cpu_rng=torch.get_rng_state().clone()
    data={'train':[(67*i+31)%50257 for i in range(1793)],'val':[(71*i+43)%50257 for i in range(419)]}
    input_tensors={name:torch.tensor(values,dtype=torch.int64) for name,values in data.items()}
    inputs=clone(input_tensors)
    a=Sampler(input_tensors['train'],input_tensors['val'],train_seed=roles['train'],eval_seed=roles['val'])
    ref={name:torch.Generator(device='cpu').manual_seed(seed) for name,seed in roles.items()}
    check('construction',dict(source_tensors_preserved=equal(input_tensors,inputs),global_CPU_RNG_unchanged=torch.equal(torch.get_rng_state(),cpu_rng),
        owned_clones=all(a.data[n].untyped_storage().data_ptr()!=input_tensors[n].untyped_storage().data_ptr() for n in inputs),
        zero_calls=a.completed_calls=={'train':0,'val':0}))
    for i,split in enumerate(('train','val','train','val','train')):
        out=verify_sample(a,split,ref,data,i)
    del out
    prefix=clone(a.state_dict());prefix_saved=clone(prefix)
    b=Sampler(input_tensors['train'],input_tensors['val'],train_seed=roles['train'],eval_seed=roles['val'])
    b.load_state_dict(prefix)
    check('prefix-restore',dict(exact_private_states=equal(a.state_dict(),b.state_dict()),caller_prefix_preserved=equal(prefix,a.state_dict())))
    for i,split in enumerate(('val','train','train','val'),start=5):
        wanted=verify_sample(a,split,ref,data,i);got=b.next(split)
        check('resume-'+str(i),dict(complete_output=equal(got,wanted),complete_state=equal(a.state_dict(),b.state_dict()),
            independent_generator_objects=all(a.generators[n] is not b.generators[n] for n in ref)))
    check('final-private-counters',dict(exact_calls=a.completed_calls==b.completed_calls=={'train':5,'val':4}))
    owned=raw_sampler(b)
    # Outputs are detached, owned data; mutation cannot alter either stream/state.
    for key in ('sequence','positions','anchors'):got[key].zero_()
    check('returned-buffer-ownership',dict(sampler_state_unchanged=equal(raw_sampler(b),owned),caller_tensors_preserved=equal(input_tensors,inputs)))
    state=clone(a.state_dict());negative_rows=[]
    def reject(label,action,payload=None):
        previous=raw_sampler(a);caller=clone(payload);global_before=torch.get_rng_state().clone();ref_before={k:g.get_state().clone() for k,g in ref.items()}
        refused=False;error=None
        try:action()
        except (ValueError,TypeError,RuntimeError) as exc:refused=True;error=type(exc).__name__+': '+str(exc)
        flags=dict(rejected=refused,destination_all_state_unchanged=equal(raw_sampler(a),previous),caller_payload_preserved=equal(payload,caller),
            global_CPU_RNG_unchanged=torch.equal(torch.get_rng_state(),global_before),reference_private_RNG_unchanged=all(torch.equal(g.get_state(),ref_before[k]) for k,g in ref.items()))
        check('reject-'+label,flags,error=error);negative_rows.append(label)
    for label,split in [('holdout','holdout'),('test','test'),('none',None),('bool',True),('foreign','validation')]:
        reject('split-'+label,lambda split=split:a.next(split))
    bad=dict(state);bad['kind']='foreign'
    reject('state-kind',lambda:a.load_state_dict(bad),bad)
    bad_signature=dict(state);bad_signature['signature']={}
    reject('state-signature',lambda:a.load_state_dict(bad_signature),bad_signature)
    missing=dict(state);missing.pop('generators')
    reject('missing-generators',lambda:a.load_state_dict(missing),missing)
    for label,calls in [('negative',{'train':-1,'val':4}),('bool',{'train':True,'val':4}),('missing',{'train':5})]:
        payload=dict(state);payload['completed_calls']=calls
        reject('counter-'+label,lambda payload=payload:a.load_state_dict(payload),payload)
    for label,value in [('dtype',state['generators']['train'].int()),('shape',state['generators']['train'][:-1]),
                        ('noncontiguous',state['generators']['train'].repeat_interleave(2)[::2])]:
        payload=dict(state);payload['generators']=dict(state['generators'],train=value)
        reject('generator-'+label,lambda payload=payload:a.load_state_dict(payload),payload)
    foreign=dict(state);foreign['generators']=dict(state['generators'],train=state['generators']['val'])
    reject('generator-role',lambda:a.load_state_dict(foreign),foreign)
    second=dict(prefix);second['generators']=dict(prefix['generators'],val=prefix['generators']['val'].int())
    reject('second-stream-preflight',lambda:a.load_state_dict(second),second)
    extra=dict(state);extra['generators']=dict(state['generators'],test=state['generators']['train'])
    reject('extra-stream',lambda:a.load_state_dict(extra),extra)
    malformed=dict(state);malformed['generators']=dict(state['generators'],val=torch.zeros_like(state['generators']['val']))
    reject('invalid-generator-format',lambda:a.load_state_dict(malformed),malformed)
    def altered(label,install,restore):
        install();reject(label,lambda:a.next('train'));restore()
    original=a.signature['context']
    altered('mutable-signature',lambda:a.signature.__setitem__('context',127),lambda:a.signature.__setitem__('context',original))
    value=int(a.data['train'][0])
    altered('owned-token-identity',lambda:a.data['train'].__setitem__(0,(value+1)%50257),lambda:a.data['train'].__setitem__(0,value))
    altered('owned-token-range',lambda:a.data['train'].__setitem__(0,50257),lambda:a.data['train'].__setitem__(0,value))
    old=a.roles['train']
    altered('mutable-role',lambda:a.roles.__setitem__('train',old+1),lambda:a.roles.__setitem__('train',old))
    old_count=a.completed_calls['train']
    altered('mutable-counter',lambda:a.completed_calls.__setitem__('train',True),lambda:a.completed_calls.__setitem__('train',old_count))
    # Constructor rejections cannot mutate caller tensors or global streams.
    def constructor_reject(label,train_tensor,val_tensor,train_seed,eval_seed):
        reject(label,lambda:Sampler(train_tensor,val_tensor,train_seed=train_seed,eval_seed=eval_seed),dict(train=train_tensor,val=val_tensor))
    constructor_reject('constructor-dtype',input_tensors['train'].int(),input_tensors['val'],roles['train'],roles['val'])
    constructor_reject('constructor-rank',input_tensors['train'].unsqueeze(0),input_tensors['val'],roles['train'],roles['val'])
    constructor_reject('constructor-short-train',input_tensors['train'][:511],input_tensors['val'],roles['train'],roles['val'])
    constructor_reject('constructor-short-val',input_tensors['train'],input_tensors['val'][:127],roles['train'],roles['val'])
    constructor_reject('constructor-role-type',input_tensors['train'],input_tensors['val'],True,roles['val'])
    constructor_reject('constructor-role-alias',input_tensors['train'],input_tensors['val'],roles['train'],roles['train'])
    check('negative-count',dict(count=len(negative_rows)==29,unique=len(set(negative_rows))==29),labels=negative_rows)
    check('final-caller-and-owned-state',dict(global_CPU_RNG_unchanged=torch.equal(torch.get_rng_state(),cpu_rng),
        caller_inputs_preserved=equal(input_tensors,inputs),prefix_preserved=equal(prefix,prefix_saved),final_states_equal=equal(a.state_dict(),b.state_dict())))
    return dict(negative_cases=negative_rows,draw_calls={'uninterrupted':9,'restored_continuation':4},
        actual_sampler_anchor_draws=62,reference_anchor_draws=42,roles=roles)


def main():
    global torch,Sampler
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    assert sha(R/'plan.json')==arg.plan_sha and set(p.name for p in R.iterdir())=={'plan.json','source-review.json','admission.json'}
    plan=json.loads((R/'plan.json').read_text())
    for name,h in plan['execution_sources'].items():assert sha(ROOT/name)==h,name
    assert shutil.disk_usage(ROOT).free>=plan['complete_target_bytes']
    c=ROOT/plan['census_directory'];admission=json.loads((R/'admission.json').read_text())
    assert sha(c/'plan.json')==plan['census_plan_sha256'] and admission['census_result_sha256']==sha(c/'result.json')
    census=json.loads((c/'result.json').read_text())
    assert census['status']=='new_future_token_sampler_roles_unused_before_construction' and census['prior_matches']==[]
    assert json.loads((c/'guard.json').read_text())['status']=='finished' and json.loads((c/'STOP').read_text())['status']=='complete_original_sampler_role_census'
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);publish('attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
        try:
            with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
                import torch
                from research.future_token_sampling_v1 import Sampler
                torch.set_num_threads(4)
                assert str(torch.__version__)==plan['torch_version'] and torch.version.git_version==plan['torch_git']
                assert torch.get_num_threads()==4 and torch.get_default_device()==torch.device('cpu')
                assert all(os.environ.get(k)==v for k,v in plan['environment'].items())
                rows=run(census['roles'])
                for name,h in plan['execution_sources'].items():assert sha(ROOT/name)==h,name
                publish('result.json',dict(status='complete_original_CPU_sampler_draw_gather_state_proof',**rows,
                    no_models_MPS_corpus_tokenizer_holdout_checkpoints_or_quality_calls=True,
                    scope='CPU_private_sampler_only;closed_index_and_model_proofs_remain_unqualified'))
            publish('STOP',dict(status='complete_original_CPU_sampler_proof',no_retry=True,no_replay=True))
        except BaseException as error:
            if not (R/'STOP').exists():publish('STOP',dict(status='closed_failed',error=repr(error),no_retry=True,no_replay=True))
            raise


if __name__=='__main__':main()
