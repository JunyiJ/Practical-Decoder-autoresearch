"""NEW finite physical T+2 data-view proof; no model, scorer, MPS or tensor I/O."""
import argparse
import copy
from dataclasses import replace
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[3]
R=Path(__file__).with_suffix('')
SERIAL=0


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw)<=4*1024**2
    with (R/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def check(label,flags,**values):
    global SERIAL
    SERIAL+=1
    report=dict(label=label,predicates=flags,passed=all(flags.values()),**values)
    publish(f'{SERIAL:03d}-{label}.json',report)
    assert report['passed'],label


def clone(v):
    if isinstance(v,torch.Tensor):return v.detach().clone()
    if isinstance(v,dict):return {k:clone(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return type(v)(clone(x) for x in v)
    return copy.deepcopy(v)


def equal(a,b):
    if isinstance(b,torch.Tensor):return isinstance(a,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and a.device==b.device and torch.equal(a,b)
    if isinstance(b,dict):return isinstance(a,dict) and list(a)==list(b) and all(equal(a[k],b[k]) for k in b)
    if isinstance(b,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def input_snapshot(value):
    if type(value) is D.Decoded:return dict(train_ids=value.train_ids,val_ids=value.val_ids,contract=copy.deepcopy(value.contract))
    return clone(value)


def raw_view(v):
    # Read directly while testing malformed owned metadata, without validate().
    s=v.sampler
    return dict(signature=copy.deepcopy(v.signature),signature_json=v._signature_json,
        decoded=input_snapshot(v.decoded),sampler_signature=copy.deepcopy(s.signature),
        sampler_signature_json=s._signature_identity,roles=copy.deepcopy(s.roles),identity=copy.deepcopy(s.identity),
        data={n:t.clone() for n,t in s.data.items()},counts=copy.deepcopy(s.completed_calls),
        generator_ids={n:id(g) for n,g in s.generators.items()},states={n:g.get_state().clone() for n,g in s.generators.items()})


def decode_witness(plan):
    # These reads occur only inside the admitted original CPU proof. No raw
    # corpus or tokenizer call occurs in source review or the role census.
    expected=json.loads((ROOT/plan['physical_manifest']['path']).read_text())['data']
    p=Path(expected['path'])
    with p.open('rb') as f:raw=f.read(8*1024**2+1)
    assert len(raw)<=8*1024**2
    raw_hash=hashlib.sha256(raw).hexdigest();text=raw.decode('utf8')
    a,b=expected['raw_train_end'],expected['raw_val_end']
    wanted=[dict(characters=len(s),utf8_sha256=hashlib.sha256(s.encode('utf8')).hexdigest()) for s in (text[:a],text[a:b])]
    del text,raw
    calls=[];seen={};genuine=tiktoken.Encoding.encode
    def observed(encoder,value,*args,**kwargs):
        calls.append(dict(characters=len(value),utf8_sha256=hashlib.sha256(value.encode('utf8')).hexdigest(),
            no_positional_options=len(args)==0,ordinary_special_policy=kwargs=={'disallowed_special':()}))
        seen['encoding']=encoder
        return genuine(encoder,value,*args,**kwargs)
    fetches=[]
    def no_fetch(path):
        fetches.append(path);raise RuntimeError('Offline fixture: tokenizer cache miss/download forbidden')
    before=torch.get_rng_state().clone()
    with patch.object(tiktoken.Encoding,'encode',observed),patch.object(tiktoken.load,'read_file',no_fetch),patch.object(torch,'Generator',side_effect=AssertionError('Decode must not construct any generator')) as generators:
        spec=D.reference_spec();decoded=D.decode()
        generator_calls=generators.call_count
    # Genuine decoding of validation target IDs establishes the physical byte
    # denominator, without computing or reconstructing a single model loss.
    target_bytes=len(seen['encoding'].decode(list(decoded.val_ids[1:])).encode('utf8'))
    flags=dict(exactly_two_encoder_calls=len(calls)==2,
        exact_train_val_text_inputs=[{k:v for k,v in x.items() if k in ('characters','utf8_sha256')} for x in calls]==wanted,
        ordinary_special_policy=all(x['no_positional_options'] and x['ordinary_special_policy'] for x in calls),
        no_other_or_holdout_encoder_input=len(calls)==2 and [x['characters'] for x in calls]==[892315,111539],
        no_decode_generator=generator_calls==0,offline_cache_only=len(fetches)==0,
        global_CPU_RNG_preserved=torch.equal(before,torch.get_rng_state()),
        corpus_prepost_SHA=raw_hash==expected['corpus_sha256']==sha(p),
        metadata_contract=decoded.contract==spec,
        train_IDs=len(decoded.train_ids)==267688 and D.ids_digest(decoded.train_ids)==expected['train_ids_sha256'],
        val_IDs=len(decoded.val_ids)==34279 and D.ids_digest(decoded.val_ids)==expected['validation_ids_sha256'],
        full_primary_target_IDs=len(decoded.val_ids[1:])==34278 and D.ids_digest(decoded.val_ids[1:])==expected['validation_target_ids_sha256'],
        physical_target_bytes=target_bytes==111536,
        no_holdout_retained=set(decoded.__dict__)=={'train_ids','val_ids','contract'},
        original_codec=decoded.contract['tokenizer']==expected['tokenizer'])
    check('genuine-physical-decode',flags,encoder_inputs=calls,expected_inputs=wanted,decode_generator_calls=generator_calls,
        target_bytes=target_bytes,corpus_sha256=raw_hash,train_IDs_sha256=D.ids_digest(decoded.train_ids),
        validation_IDs_sha256=D.ids_digest(decoded.val_ids),target_IDs_sha256=D.ids_digest(decoded.val_ids[1:]),
        holdout_BPE_calls=0,model_or_score_calls=0)
    return decoded


def verify_output(view,split,output,decoded,label):
    physical=decoded.train_ids if split=='train' else decoded.val_ids
    anchors=output['anchors'].tolist();positions=output['positions'].tolist()
    # Independent physical-block slicing oracle, never D/Sampler.positions().
    wanted=[]
    for anchor in anchors:
        if split=='train':
            block=list(range(anchor,anchor+514))
            wanted.extend([block[0:130],block[128:258],block[256:386],block[384:514]])
        else:wanted.append(list(range(anchor,anchor+130)))
    expected=[[physical[i] for i in row] for row in wanted]
    actual=output['sequence'].tolist();layout=decoded.contract['sequence_layout']
    flags=dict(view_policy=output['policy']==D.POLICY,sampler_policy=output['sampler_policy']=='future_token_split_local_paired512_v1',
        exact_physical_identity=output['physical_identity']==view.signature['physical_identity'],
        exact_all1040_positions=positions==wanted,exact_all1040_tokens=actual==expected,
        no_cross_split=all(0<=i<len(physical) for row in positions for i in row),
        exact_shape=output['sequence'].shape==output['positions'].shape==(8,130),
        native_cpu_int64=output['sequence'].dtype==output['positions'].dtype==torch.int64 and output['sequence'].device.type==output['positions'].device.type=='cpu',
        declared1024_targets=output['primary_targets']==output['second_next_targets']==1024,
        anchor_inventory=len(anchors)==(2 if split=='train' else 8),
        eligible_anchors=all(0<=x<layout['train_eligible_starts' if split=='train' else 'val_eligible_starts'] for x in anchors))
    for offset in (0,1,2):flags['all1024_offset_'+str(offset)]=[row[offset:128+offset] for row in actual]==[row[offset:128+offset] for row in expected]
    check(label,flags,split=split,anchors=anchors,actual_positions=positions,expected_positions=wanted,actual_sequence=actual,expected_sequence=expected)


def runtime(plan,roles):
    caller=torch.get_rng_state().clone();decoded=decode_witness(plan);decoded_before=input_snapshot(decoded)
    a=D.View(decoded,train_seed=roles['train'],eval_seed=roles['val'])
    check('physical-view-construction',dict(global_CPU_RNG_preserved=torch.equal(caller,torch.get_rng_state()),
        physical_input_preserved=equal(input_snapshot(decoded),decoded_before),
        sampler_counts_zero=a.sampler.completed_calls=={'train':0,'val':0},
        own_explicit_interface=not hasattr(a,'get_batch') and hasattr(a,'next_sequence')))
    for i,split in enumerate(('train','val')):
        out=a.next_sequence(split);verify_output(a,split,out,decoded,'prefix-'+str(i)+'-'+split)
    prefix=clone(a.state_dict());prefix_copy=clone(prefix)
    b=D.View(decoded,train_seed=roles['train'],eval_seed=roles['val']);b.load_state_dict(prefix)
    check('physical-view-prefix-restore',dict(exact_private_view_state=equal(a.state_dict(),b.state_dict()),caller_prefix_preserved=equal(prefix,prefix_copy),
        independent_private_generators=all(a.sampler.generators[n] is not b.sampler.generators[n] for n in ('train','val'))))
    for i,split in enumerate(('train','val')):
        original=a.next_sequence(split);restored=b.next_sequence(split)
        verify_output(a,split,original,decoded,'continuation-'+str(i)+'-'+split)
        verify_output(b,split,restored,decoded,'restored-'+str(i)+'-'+split)
        check('exact-resume-'+str(i),dict(output_exact=equal(original,restored),complete_state_exact=equal(a.state_dict(),b.state_dict())))
    before=raw_view(b)
    for key in ('sequence','positions','anchors'):restored[key].zero_()
    check('physical-output-ownership',dict(private_data_and_state_preserved=equal(raw_view(b),before),decoded_preserved=equal(input_snapshot(decoded),decoded_before)))
    state=clone(a.state_dict());negatives=[]
    def reject(label,action,payload=None):
        previous=raw_view(a);saved=input_snapshot(payload);rng=torch.get_rng_state().clone();refused=False;error=None
        try:action()
        except (ValueError,TypeError,RuntimeError) as exc:refused=True;error=type(exc).__name__+': '+str(exc)
        flags=dict(rejected=refused,complete_destination_unchanged=equal(raw_view(a),previous),
            caller_payload_preserved=equal(input_snapshot(payload),saved),global_CPU_RNG_unchanged=torch.equal(rng,torch.get_rng_state()))
        check('reject-'+label,flags,error=error);negatives.append(label)
    def constructor(label,value,train_seed,eval_seed):reject(label,lambda:D.View(value,train_seed=train_seed,eval_seed=eval_seed),value)
    constructor('foreign-decoded',SimpleNamespace(),roles['train'],roles['val'])
    constructor('role-bool',decoded,True,roles['val'])
    constructor('role-alias',decoded,roles['train'],roles['train'])
    constructor('role-negative',decoded,-1,roles['val'])
    constructor('role-overflow',decoded,roles['train'],2**63)
    bad_contract=copy.deepcopy(decoded.contract);bad_contract['corpus']['raw_train_end']+=1
    bad=replace(decoded,contract=bad_contract);constructor('decoded-physical-contract',bad,roles['train'],roles['val'])
    bad=replace(decoded,train_ids=decoded.train_ids[:-1]);constructor('decoded-train-length',bad,roles['train'],roles['val'])
    bad=replace(decoded,train_ids=(True,)+decoded.train_ids[1:]);constructor('decoded-train-ID-type',bad,roles['train'],roles['val'])
    bad=replace(decoded,val_ids=decoded.val_ids[:-1]+((decoded.val_ids[-1]+19)%50257,));constructor('decoded-val-ID-identity',bad,roles['train'],roles['val'])
    altered=dict(state);altered['kind']='foreign'
    reject('state-kind',lambda:a.load_state_dict(altered),altered)
    wrong=dict(state);wrong['signature']={}
    reject('state-signature',lambda:a.load_state_dict(wrong),wrong)
    wrong_data=copy.deepcopy(state);wrong_data['signature']['physical_identity']='0'*64
    reject('state-dataset-identity',lambda:a.load_state_dict(wrong_data),wrong_data)
    wrong_sampler=copy.deepcopy(state);wrong_sampler['sampler_state']['kind']='foreign'
    reject('sampler-state-kind',lambda:a.load_state_dict(wrong_sampler),wrong_sampler)
    missing=copy.deepcopy(state);missing['sampler_state']['generators'].pop('val')
    reject('missing-private-stream',lambda:a.load_state_dict(missing),missing)
    wrong_role=copy.deepcopy(state);wrong_role['sampler_state']['generators']['train']=wrong_role['sampler_state']['generators']['val'].clone()
    reject('wrong-private-role',lambda:a.load_state_dict(wrong_role),wrong_role)
    second=copy.deepcopy(prefix);second['sampler_state']['generators']['val']=second['sampler_state']['generators']['val'].int()
    reject('second-private-stream-preflight',lambda:a.load_state_dict(second),second)
    for label,value in [('holdout','holdout'),('test','test'),('none',None),('bool',False)]:reject('split-'+label,lambda value=value:a.next_sequence(value))
    old=a.signature['physical_identity'];a.signature['physical_identity']='0'*64
    reject('mutable-view-identity',lambda:a.next_sequence('train'));a.signature['physical_identity']=old
    value=int(a.sampler.data['val'][0]);a.sampler.data['val'][0]=(value+1)%50257
    reject('owned-physical-ID-mutation',lambda:a.next_sequence('val'));a.sampler.data['val'][0]=value
    value=a.sampler.roles['train'];a.sampler.roles['train']=value+1
    reject('mutable-owned-role',lambda:a.next_sequence('train'));a.sampler.roles['train']=value
    missing_key=dict(state);missing_key.pop('sampler_state')
    reject('missing-view-state',lambda:a.load_state_dict(missing_key),missing_key)
    check('final-case-counters-and-RNG',dict(negative_count=len(negatives)==24,unique_case_labels=len(set(negatives))==24,
        actual_sampler_counters=a.sampler.completed_calls==b.sampler.completed_calls=={'train':2,'val':2},
        caller_CPU_RNG_preserved=torch.equal(caller,torch.get_rng_state()),decoded_preserved=equal(input_snapshot(decoded),decoded_before),
        caller_prefix_preserved=equal(prefix,prefix_copy),views_final_state_exact=equal(a.state_dict(),b.state_dict())),negative_labels=negatives)
    return dict(negative_cases=negatives,actual_view_calls=6,actual_anchor_draws=30,private_roles=roles,physical_train_IDs=267688,
        physical_val_IDs=34279,physical_next_targets=34278,target_bytes=111536,encoder_calls=2,holdout_BPE_calls=0)


def main():
    global torch,tiktoken,D
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    assert sha(R/'plan.json')==arg.plan_sha and set(p.name for p in R.iterdir())=={'plan.json','source-review.json','admission.json'}
    plan=json.loads((R/'plan.json').read_text());c=ROOT/plan['census_directory']
    for name,h in plan['execution_sources'].items():assert sha(ROOT/name)==h,name
    assert sha(c/'plan.json')==plan['census_plan_sha256']
    admission=json.loads((R/'admission.json').read_text());assert admission['census_result_sha256']==sha(c/'result.json')
    census=json.loads((c/'result.json').read_text())
    assert census['status']=='new_physical_data_private_roles_unused_before_construction' and census['prior_matches']==[]
    assert json.loads((c/'guard.json').read_text())['status']=='finished' and json.loads((c/'STOP').read_text())['status']=='complete_original_physical_data_role_census'
    assert shutil.disk_usage(ROOT).free>=plan['complete_target_bytes']
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);publish('attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
        try:
            with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
                for ref in plan['codec_cache']:
                    assert Path(ref['path']).stat().st_size==ref['size'] and sha(ref['path'])==ref['sha256']
                import torch
                import tiktoken
                import tiktoken.load
                from research import future_token_data_v1 as D
                torch.set_num_threads(4)
                assert str(torch.__version__)==plan['torch_version'] and torch.version.git_version==plan['torch_git']
                assert importlib.metadata.version('tiktoken')==plan['tiktoken_version']
                assert torch.get_default_device()==torch.device('cpu') and all(os.environ.get(k)==v for k,v in plan['environment'].items())
                result=runtime(plan,census['roles'])
                for name,h in plan['execution_sources'].items():assert sha(ROOT/name)==h,name
                for ref in plan['codec_cache']:assert sha(ref['path'])==ref['sha256']
                publish('result.json',dict(status='complete_original_physical_train_val_Tplus2_view_proof',**result,
                    source_and_codec_cache_unchanged=True,no_model_optimizer_MPS_score_holdout_or_tensor_file_calls=True,
                    scope='physical_CPU_data_view_and_inmemory_private_state_only;not_closed_index_model_or_full_training_qualification'))
            publish('STOP',dict(status='complete_original_physical_data_view_proof',no_retry=True,no_replay=True))
        except BaseException as error:
            if not (R/'STOP').exists():publish('STOP',dict(status='closed_failed',error=repr(error),no_retry=True,no_replay=True))
            raise


if __name__=='__main__':main()
