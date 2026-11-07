"""New generated CPU/MPS transport proof; no closed proof execution or real model."""
import copy,json,math,os,shutil,struct
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_resume_v1.contract import runtime as environment
from research.tokenizer_moe_worker_v1.execution import idle
from research.tokenizer_optimizer_response_mps_v1.component import KIND,response
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
E=Path(__file__).resolve().parent;RUN=ROOT/'research_runs/tokenizer-optimizer-response-mps-component-v1-001';META=2*1024**2
FLOOR=8*1024**3+META+1267904512+1458176;TARGET=FLOOR+2*1024**3

def checked(r):
    p=Path(r['path']);require(reference(p)=={k:r[k] for k in ('path','sha256')},'Immutable reference changed');return json.loads(p.read_text())
def serialized(value,used):
    require(type(used) is int and used>=0,'Exact nonnegative output usage required')
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(used+len(raw)<=META,'Complete2MiB metadata cap required');return raw
def publish(p,value):
    used=sum(v.stat().st_size for v in RUN.glob('*') if v.is_file());raw=serialized(value,used)
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def same(x,y):
    if type(x) is torch.Tensor:return type(y) is torch.Tensor and x.dtype==y.dtype and x.shape==y.shape and torch.equal(x.cpu(),y.cpu())
    if type(x) is dict:return type(y) is dict and x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
    return x==y
def equal_bad(x,y):
    if type(x) is torch.Tensor:
        a=x.detach().cpu();b=y.detach().cpu();return a.shape==b.shape and a.dtype==b.dtype and bool(((a==b)|(torch.isnan(a)&torch.isnan(b))).all())
    if type(x) is dict:return x.keys()==y.keys() and all(equal_bad(x[k],y[k]) for k in x)
    return x==y

def f32(x):return struct.unpack('<f',struct.pack('<f',float(x)))[0]

def bf(x):
    # Independent IEEE round-to-nearest-even bfloat16, not a PyTorch cast/kernel.
    bits=struct.unpack('<I',struct.pack('<f',float(x)))[0]
    rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    return struct.unpack('<f',struct.pack('<I',rounded))[0]

def transpose(x):return [list(v) for v in zip(*x)]

def dot(x,y):return math.fsum(a*b for a,b in zip(x,y))

def scalar_muon(g):
    # No torch matmul/addmm/private Muon helper; scalar loops and explicit bf16 boundaries.
    tall=len(g)>len(g[0]);x=[[bf(v) for v in row] for row in g]
    if tall:x=transpose(x)
    denom=max(bf(math.sqrt(math.fsum(v*v for row in x for v in row))),bf(1e-7))
    x=[[bf(v/denom) for v in row] for row in x]
    for _ in range(5):
        xt=transpose(x);a=[[bf(dot(row,col)) for col in transpose(xt)] for row in x]
        at=transpose(a)
        b=[[bf(-4.775*a[i][j]+2.0315*dot(row,col)) for j,col in enumerate(at)] for i,row in enumerate(a)]
        x=[[bf(3.4445*x[i][j]+dot(row,col)) for j,col in enumerate(transpose(x))] for i,row in enumerate(b)]
    return transpose(x) if tall else x

def oracle(state,grads):
    out=copy.deepcopy(state);norm=math.sqrt(math.fsum(float(v)**2 for g in grads.values() if g is not None for v in g.flatten()));coef=min(1.,1./(norm+1e-6));directions={}
    for name,s in out['parameters'].items():
        if grads[name] is None:continue
        old=state['parameters'][name];p=old['value'].tolist();g=[f32(float(v)*coef) for v in grads[name].flatten()];m=[float(v) for v in old['first'].flatten()]
        s['updates']+=1
        if s['algorithm']=='muon':
            first=[f32(v+.05*(w-v)) for v,w in zip(m,g)]
            nesterov=[f32(v+.95*(w-v)) for v,w in zip(g,first)]
            cols=old['value'].shape[1];matrix=[nesterov[i:i+cols] for i in range(0,len(nesterov),cols)];q=scalar_muon(matrix);u=[v for row in q for v in row];lr=old['lr']*.2*math.sqrt(max(old['value'].shape))
            directions[name]=u
        else:
            first=[.9*v+.1*w for v,w in zip(m,g)];v=[float(w) for w in old['second'].flatten()];second=[.999*z+.001*w*w for z,w in zip(v,g)];t=s['updates'];u=[(a/(1-.9**t))/(math.sqrt(b/(1-.999**t))+1e-6) for a,b in zip(first,second)];lr=old['lr']
            s['second']=torch.tensor(second,dtype=torch.float32).reshape(old['value'].shape)
        s['first']=torch.tensor(first,dtype=torch.float32).reshape(old['value'].shape)
        s['value']=torch.tensor([float(v)*(1-.00006)-lr*w for v,w in zip(old['value'].flatten(),u)],dtype=torch.float32).reshape(old['value'].shape)
    return out,norm,coef,directions

def fixture(shape,updates):
    out={}
    for name,algorithm,sz in [('matrix','muon',shape),('vector','adamw',(3,))]:
        n=math.prod(sz);value=torch.tensor([(i+1)/32 for i in range(n)],dtype=torch.float32).reshape(sz)
        first=torch.tensor([((i%3)-1)/32 if updates else 0. for i in range(n)],dtype=torch.float32).reshape(sz)
        second=torch.tensor([(i+1)/128 if updates else 0. for i in range(n)],dtype=torch.float32).reshape(sz) if algorithm=='adamw' else None
        out[name]=dict(algorithm=algorithm,value=value,first=first,second=second,updates=updates,lr=.000625 if algorithm=='muon' else .00125)
    return dict(kind=KIND,parameters=out)

def gradients(state,kind,scale):
    return {n:torch.tensor([0. if kind=='zero' else .25*scale if kind=='constant' else ((i%5)-2)*scale for i in range(s['value'].numel())],dtype=torch.float32).reshape(s['value'].shape) for n,s in state['parameters'].items()}


with session_lock(GLOBAL_LOCK.parent):
    idle();torch.set_num_threads(4);env=environment();require(not RUN.exists(),'Fresh CPU/MPS proof only;never retry closed root')
    closure_ref=reference(E/'continuation-completed004.json');closure=checked(closure_ref);prior=checked(closure['source_plan']);prior_result=checked(closure['proof']);review=checked(closure['review']);stop=checked(closure['terminal']);design=checked(closure['next_design']);prior_guard=checked(review['original_guard']);review_guard=checked(review['review_guard'])
    require(closure['current_bindings']==1050 and prior_result['status']=='generated_fixed_state_clipping_muon_adam_response_independent_oracle_passed' and review['status']=='generated_optimizer_response_proof_independently_reviewed' and stop['status']=='complete_generated_optimizer_response_proof','Complete closed CPU proof/review required')
    require(all(g['status']=='finished' and g['reason'] is None for g in (prior_guard,review_guard)),'Actual completed CPU resource guards required')
    pins=dict(prior['sources']);require(len(pins)==1050,'All1050 immutable predecessor pins required')
    for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Immutable predecessor changed '+p)
    for p in [*sorted((ROOT/'research/tokenizer_optimizer_response_mps_v1').glob('*.py')),Path(__file__).resolve()]:pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    require(len(pins)==1053,'Exactly two new device files plus new proof required')
    for r in [*prior['libraries'],prior['actual_inherited_decay_source']]:require(reference(r['path'])==r,'Exact runtime optimizer source changed')
    TOL=dict(prior['tolerances']);require(TOL==design['tolerances'],'No tolerance changes allowed')
    free=shutil.disk_usage(ROOT).free;require(free>=TARGET and torch.backends.mps.is_available(),'Complete LIVE budget and native MPS availability required')
    cost=design['cost'];require(cost['conservative_rss_bytes']<cost['rss_limit_bytes'] and cost['conservative_mps_bytes']<cost['mps_limit_bytes'] and cost['conservative_seconds']<cost['timeout_seconds'],'Complete declared resource forecast must fit')
    RUN.mkdir();publish(RUN/'plan.json',dict(kind='generated_cpu_mps_optimizer_response_transport_v1',sources=pins,predecessor_closure=closure_ref,predecessor_review=closure['review'],design=closure['next_design'],environment=env,libraries=prior['libraries'],actual_inherited_decay_source=prior['actual_inherited_decay_source'],pure_equation_source=reference(E/'optimizer-response001.py'),pure_equation_functions=['f32','bf','transpose','dot','scalar_muon','oracle','fixture','gradients'],closed_proof_executed=False,case_shapes=prior['case_shapes'],case_updates=prior['case_updates'],case_kinds=prior['case_kinds'],case_scales=prior['case_scales'],devices=['cpu','mps'],tolerances=TOL,generated_only=True,no_rng_draws=True,no_model_raw_corpus=True,full_floor=FLOOR,target=TARGET,free_bytes=free,cost=cost,metadata_cap=META,adversarial_expected=51))
    try:
        with StrictMpsGuard(RUN/'guard.json',ROOT):
            mps_rng=torch.mps.get_rng_state().clone();cpu_rng=torch.get_rng_state().clone();cases=[];maxima={dev:dict(parameter=0.,moment=0.,direction=0.,clip=0.) for dev in ('cpu','mps')};transport=dict(parameter=0.,moment=0.,clip=0.);rejections=0
            def verify(state,grads,got,want,norm,coef,dirs,device):
                errors={};maximum=maxima[device]
                err=abs(got['clipping']['coefficient']-coef);maximum['clip']=max(maximum['clip'],err)
                require(err<TOL['clip_atol'] and abs(got['clipping']['pre_norm']-norm)<max(2e-7,norm*1e-7),'Independent clip response differs on '+device)
                for name,s in got['state']['parameters'].items():
                    target=want['parameters'][name];err=float((s['value']-target['value']).abs().max());errors[name]=err;maximum['parameter']=max(maximum['parameter'],err)
                    require(err<=TOL['muon_parameter_atol' if s['algorithm']=='muon' else 'adam_parameter_atol'] and s['updates']==target['updates'],'Independent parameter/exposure response differs on '+device)
                    for k in ('first','second'):
                        if s[k] is not None:
                            err=float((s[k]-target[k]).abs().max());maximum['moment']=max(maximum['moment'],err);require(err<TOL['moment_atol'],'Independent moment response differs on '+device)
                    if grads[name] is None:require(same(s,state['parameters'][name]),'Missing-gradient entry changed on '+device)
                    elif s['algorithm']=='muon':
                        old=state['parameters'][name];factor=old['lr']*.2*math.sqrt(max(old['value'].shape));actual=(old['value']*(1-.00006)-s['value'])/factor;expected=torch.tensor(dirs[name]).reshape(actual.shape);err=float((actual-expected).abs().max());maximum['direction']=max(maximum['direction'],err)
                        require(torch.allclose(actual,expected,atol=TOL['muon_direction_atol'],rtol=TOL['muon_direction_rtol']),'Independent scalar bf16 direction differs on '+device)
                return errors
            def compare(cpu,mps):
                for name,s in cpu['state']['parameters'].items():
                    t=mps['state']['parameters'][name];err=float((s['value']-t['value']).abs().max());transport['parameter']=max(transport['parameter'],err)
                    require(err<=TOL['muon_parameter_atol' if s['algorithm']=='muon' else 'adam_parameter_atol'] and s['updates']==t['updates'],'CPU/MPS parameter/exposure transport differs')
                    for k in ('first','second'):
                        if s[k] is not None:
                            err=float((s[k]-t[k]).abs().max());transport['moment']=max(transport['moment'],err);require(err<TOL['moment_atol'],'CPU/MPS moment transport differs')
                err=abs(cpu['clipping']['coefficient']-mps['clipping']['coefficient']);transport['clip']=max(transport['clip'],err);require(err<TOL['clip_atol'],'CPU/MPS clip transport differs')
            for shape in ((3,2),(2,3),(2,2)):
                for updates in (0,7,1200):
                    for kind in ('ordinary','constant','zero'):
                        for scale in (.05,4.):
                            state=fixture(shape,updates);g=gradients(state,kind,scale);before=copy.deepcopy((state,g));want,norm,coef,dirs=oracle(state,g);results={};row=dict(shape=shape,updates=updates,kind=kind,scale=scale,devices={})
                            for device in ('cpu','mps'):
                                results[device]=response(state,g,device);errors=verify(state,g,results[device],want,norm,coef,dirs,device)
                                require(same(state,before[0]) and same(g,before[1]),'Caller state/gradients changed')
                                row['devices'][device]=dict(parameter_maxerror=errors,clip_coefficient=results[device]['clipping']['coefficient'])
                            compare(results['cpu'],results['mps']);cases.append(row)
            for inactive in ('matrix','vector'):
                state=fixture((3,2),7);g=gradients(state,'ordinary',.25);g[inactive]=None;before=copy.deepcopy((state,g));want,norm,coef,dirs=oracle(state,g);results={}
                for device in ('cpu','mps'):results[device]=response(state,g,device);verify(state,g,results[device],want,norm,coef,dirs,device)
                compare(results['cpu'],results['mps']);require(same(state,before[0]) and same(g,before[1]),'Missing-gradient caller mutated')
            gaps={};sequential={}
            state=fixture((2,2),7);a=gradients(state,'ordinary',.5);b=gradients(state,'constant',2.);mean={n:(a[n]+b[n])/2 for n in a}
            linear=max(float((mean[n]-(a[n]+b[n])/2).abs().max()) for n in a);require(linear==0,'Linear identity reference differs')
            for device in ('cpu','mps'):
                ra=response(state,a,device);rb=response(state,b,device);rm=response(state,mean,device);seq=response(ra['state'],b,device)
                gaps[device]={n:float((rm['delta'][n]-(ra['delta'][n]+rb['delta'][n])/2).abs().max()) for n in a};sequential[device]={n:float((seq['state']['parameters'][n]['value']-rm['state']['parameters'][n]['value']).abs().max()) for n in a}
                require(max(gaps[device].values())>1e-7 and max(sequential[device].values())>1e-7,'Nonlinear/state-evolution controls failed to discriminate')
            base=fixture((3,2),7);good=gradients(base,'ordinary',.25);invalid=[]
            def altered(fn):
                s=copy.deepcopy(base);g=copy.deepcopy(good);fn(s,g);invalid.append((s,g))
            for fn in [lambda s,g:s.update(kind='foreign'),lambda s,g:s.update(extra=1),lambda s,g:s['parameters']['matrix'].update(algorithm='adam'),lambda s,g:s['parameters']['matrix'].update(updates=True),lambda s,g:s['parameters']['matrix'].update(updates=-1),lambda s,g:s['parameters']['matrix'].update(lr=.00125),lambda s,g:s['parameters']['vector'].update(second=torch.full((3,),-1.)),lambda s,g:s['parameters']['vector'].update(first=torch.zeros(4)),lambda s,g:s['parameters']['vector'].update(value=torch.ones(3,dtype=torch.float64)),lambda s,g:s['parameters']['vector'].update(updates=0),lambda s,g:g.pop('vector'),lambda s,g:g.update(extra=torch.ones(1)),lambda s,g:g.update(vector=torch.ones(4)),lambda s,g:g.update(vector=torch.ones(3,dtype=torch.float64)),lambda s,g:g.update(vector=torch.tensor([float('nan'),0.,0.])),lambda s,g:g.update(vector=torch.tensor([float('inf'),0.,0.])),lambda s,g:g.update(matrix=None,vector=None),lambda s,g:s['parameters']['matrix'].update(second=torch.zeros(3,2)),lambda s,g:s['parameters']['vector'].update(first=torch.tensor([float('nan'),0.,0.])),lambda s,g:g.update(vector=torch.ones(3,requires_grad=True)),lambda s,g:s.update(kind='generated_fixed_state_optimizer_response_v1'),lambda s,g:g.update(vector=g['vector'].to('mps'))]:altered(fn)
            for s,g in invalid:
                before=copy.deepcopy((s,g))
                for device in ('cpu','mps'):
                    try:response(s,g,device)
                    except (ValueError,RuntimeError,TypeError):pass
                    else:raise AssertionError('Foreign state/gradient accepted')
                    require(equal_bad(s,before[0]) and equal_bad(g,before[1]),'Rejected caller state mutated');rejections+=1
            for device in ('cuda','meta','mps:0',None):
                before=copy.deepcopy((base,good))
                try:response(base,good,device)
                except ValueError:pass
                else:raise AssertionError('Foreign response device accepted')
                require(same(base,before[0]) and same(good,before[1]),'Rejected device mutated caller');rejections+=1
            for value,used in [({},META),({'x':float('nan')},0),({},-1)]:
                try:serialized(value,used)
                except ValueError:rejections+=1
                else:raise AssertionError('Invalid output cap accepted')
            require(rejections==51,'Exact complete adversarial panel required')
            torch.mps.synchronize();require(torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(mps_rng,torch.mps.get_rng_state()),'Global CPU/MPS RNG changed')
            for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Frozen source changed during proof')
            for r in [*prior['libraries'],prior['actual_inherited_decay_source']]:require(reference(r['path'])==r,'Optimizer runtime changed during proof')
        result=dict(status='generated_cpu_mps_optimizer_response_independent_oracle_and_transport_passed',source_plan=reference(RUN/'plan.json'),source_bindings=1053,cases=cases,none_gradient_cases=2,none_gradient_device_checks=4,rejections_before_mutation=rejections,maxima=maxima,transport_maxima=transport,noncommutation_max_abs=gaps,sequential_vs_fixed_state_max_abs=sequential,linear_reference_error=linear,global_cpu_mps_rng_unchanged=True,caller_inputs_unchanged=True,generated_optimizer_steps_only=True,actual_model_optimizer_steps=0,models_or_raw_loaded=0,corpus_loaded=False,validation_encoded=False,holdout_encoded=False,weight_files_written=0,closed_proof_executed=False,quality_result=False,adopted=False,scope='Tiny generated CPU/MPS responses only;not full model/checkpoint-state transport or batch/quality evidence.')
        publish(RUN/'result.json',result);publish(RUN/'STOP',dict(status='complete_generated_cpu_mps_optimizer_response_proof',result=reference(RUN/'result.json'),never_rerun=True));print(json.dumps({k:v for k,v in result.items() if k not in ('cases','source_plan')}),flush=True)
    except BaseException as exc:
        if not(RUN/'STOP').exists():publish(RUN/'STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
        raise
