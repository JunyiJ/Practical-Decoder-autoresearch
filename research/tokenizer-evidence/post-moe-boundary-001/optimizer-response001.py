"""Once-only generated CPU optimizer response versus independent scalar arithmetic."""
import copy,hashlib,inspect,json,math,shutil,struct
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_resume_v1.contract import runtime as environment
from research.tokenizer_moe_worker_v1.execution import idle
from research.tokenizer_optimizer_response_v1.component import KIND,response,MAX_ELEMENTS,EpsilonOptimizer
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
E=Path(__file__).resolve().parent;RUN=E/'optimizer-response001';META=2*1024**2
FLOOR=8*1024**3+META+1267904512+1458176;TARGET=FLOOR+2*1024**3
TOL=dict(muon_parameter_atol=3e-6,adam_parameter_atol=2e-7,moment_atol=2e-7,clip_atol=2e-7,muon_direction_atol=.015625,muon_direction_rtol=.01)

def checked(r):
    p=Path(r['path']);require(reference(p)=={k:r[k] for k in ('path','sha256')},'Immutable evidence changed');return json.loads(p.read_text())
def serialized(value,used):
    require(type(used) is int and used>=0,'Exact nonnegative used metadata bytes required')
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    require(used+len(raw)<=META,'Complete2MiB metadata cap required')
    return raw
def publish(p,value):
    used=sum(p.stat().st_size for p in RUN.glob('*') if p.is_file());raw=serialized(value,used)
    with p.open('xb') as f:f.write(raw);f.flush()
def same(x,y):
    if type(x) is torch.Tensor:return type(y) is torch.Tensor and x.dtype==y.dtype and x.shape==y.shape and torch.equal(x,y)
    if type(x) is dict:return type(y) is dict and x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
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
    idle();torch.set_num_threads(4);env=environment();require(not RUN.exists(),'Fresh generated proof only;closed failures cannot retry')
    closure_ref=reference(E/'continuation-completed003.json');closure=checked(closure_ref);oldplan=checked(closure['source_plan']);review=checked(closure['cohort_review']);stop=checked(closure['cohort_terminal']);analysis=checked(closure['existing_analysis']);design=checked(closure['next_design'])
    require(closure['current_bindings']==1047 and review['status']=='complete_training_free_gradient_cohorts_independently_reviewed' and stop['status']=='complete_training_free_gradient_cohorts' and analysis['status']=='closed_complete_existing_cohort_scalar_analysis','Complete actual closed cohort prerequisites required')
    pins=dict(oldplan['sources']);require(len(pins)==1047,'All1047 predecessor bindings required')
    for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Frozen predecessor changed '+p)
    for p in [*sorted((ROOT/'research/tokenizer_optimizer_response_v1').glob('*.py')),Path(__file__).resolve()]:pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    require(len(pins)==1050,'Exactly two new component files plus this independent proof required')
    for r in design['source_audit']:checked_source=reference(r['path']);require(checked_source==r,'Original optimizer source audit changed')
    actual_hybrid=EpsilonOptimizer.__mro__[2];hybrid_source=reference(inspect.getsourcefile(actual_hybrid))
    import importlib
    libraries=[reference(inspect.getsourcefile(importlib.import_module(n))) for n in ('torch.optim._muon','torch.optim.adam','torch.optim.adamw','torch.nn.utils.clip_grad')]
    free=shutil.disk_usage(ROOT).free;require(free>=TARGET,'Complete generated metadata/reserve/external/headroom budget required');RUN.mkdir()
    publish(RUN/'plan.json',dict(kind='generated_fixed_state_optimizer_response_proof_v1',sources=pins,predecessor_closure=closure_ref,prospective_design=closure['next_design'],environment=env,libraries=libraries,actual_inherited_decay_source=hybrid_source,case_shapes=[[3,2],[2,3],[2,2]],case_updates=[0,7,1200],case_kinds=['ordinary','constant','zero'],case_scales=[.05,4.],tolerances=TOL,generated_only=True,no_rng=True,no_corpus=True,no_actual_model_or_raw=True,max_elements=MAX_ELEMENTS,full_floor=FLOOR,target=TARGET,free_bytes=free,metadata_cap=META,cpu_seconds=120))
    try:
        with CpuPrerequisiteGuard(RUN/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
            rng=torch.get_rng_state().clone();cases=[];max_parameter=0.;max_moment=0.;max_direction=0.;max_clip=0.;rejections=0;clipped=0
            for shape in ((3,2),(2,3),(2,2)):
                for updates in (0,7,1200):
                    for kind in ('ordinary','constant','zero'):
                        for scale in (.05,4.):
                            state=fixture(shape,updates);grads=gradients(state,kind,scale);before=copy.deepcopy((state,grads));got=response(state,grads);want,norm,coef,dirs=oracle(state,grads)
                            require(same(state,before[0]) and same(grads,before[1]),'Fixed-state response mutated caller inputs')
                            ce=abs(got['clipping']['coefficient']-coef);max_clip=max(max_clip,ce);require(ce<TOL['clip_atol'] and abs(got['clipping']['pre_norm']-norm)<max(2e-7,norm*1e-7),'Independent global clip arithmetic differs')
                            clipped+=got['clipping']['coefficient']<1
                            errors={}
                            for name,s in got['state']['parameters'].items():
                                target=want['parameters'][name];error=float((s['value']-target['value']).abs().max());errors[name]=error;max_parameter=max(max_parameter,error)
                                require(error<=TOL['muon_parameter_atol' if s['algorithm']=='muon' else 'adam_parameter_atol'],'Independent parameter response differs')
                                require(s['updates']==target['updates'],'Active perparameter update count differs')
                                for k in ('first','second'):
                                    if s[k] is not None:
                                        error=float((s[k]-target[k]).abs().max());max_moment=max(max_moment,error);require(error<TOL['moment_atol'],'Independent moment update differs')
                                if s['algorithm']=='muon':
                                    old=state['parameters'][name];factor=old['lr']*.2*math.sqrt(max(old['value'].shape));actual=(old['value']*(1-.00006)-s['value'])/factor;expected=torch.tensor(dirs[name]).reshape(actual.shape);error=float((actual-expected).abs().max());max_direction=max(max_direction,error)
                                    require(torch.allclose(actual,expected,atol=TOL['muon_direction_atol'],rtol=TOL['muon_direction_rtol']),'Independent bf16 scalar Newton-Schulz direction differs')
                            cases.append(dict(shape=shape,updates=updates,kind=kind,scale=scale,clip_coefficient=got['clipping']['coefficient'],parameter_maxerror=errors))
            # Missing gradients skip BOTH decay and optimizer exposure; zero gradients do neither.
            for inactive in ('matrix','vector'):
                state=fixture((3,2),7);grads=gradients(state,'ordinary',.25);grads[inactive]=None;got=response(state,grads);want,_,_,_=oracle(state,grads)
                require(same(got['state']['parameters'][inactive],state['parameters'][inactive]),'None-gradient slot decayed/advanced')
                for name,s in got['state']['parameters'].items():require(torch.allclose(s['value'],want['parameters'][name]['value'],atol=3e-6,rtol=0),'Active/None gradient oracle differs')
            # Linear response commutes; clipping plus stateful optimizer need not.
            state=fixture((2,2),7);a=gradients(state,'ordinary',.5);b=gradients(state,'constant',2.);mean={n:(a[n]+b[n])/2 for n in a};ra=response(state,a);rb=response(state,b);rm=response(state,mean);seq=response(ra['state'],b)
            gaps={n:float((rm['delta'][n]-(ra['delta'][n]+rb['delta'][n])/2).abs().max()) for n in a}
            sequential={n:float((seq['state']['parameters'][n]['value']-rm['state']['parameters'][n]['value']).abs().max()) for n in a}
            require(max(gaps.values())>1e-7 and max(sequential.values())>1e-7,'Generated nonlinear/state-evolution controls did not discriminate')
            linear=max(float((mean[n]-(a[n]+b[n])/2).abs().max()) for n in a);require(linear==0,'Linear identity reference differs')
            # Foreign/malformed policy/state/gradient inputs must leave every caller unchanged.
            base=fixture((3,2),7);good=gradients(base,'ordinary',.25);invalid=[]
            def altered(fn):
                s=copy.deepcopy(base);g=copy.deepcopy(good);fn(s,g);invalid.append((s,g))
            for fn in [lambda s,g:s.update(kind='foreign'),lambda s,g:s.update(extra=1),lambda s,g:s['parameters']['matrix'].update(algorithm='adam'),lambda s,g:s['parameters']['matrix'].update(updates=True),lambda s,g:s['parameters']['matrix'].update(updates=-1),lambda s,g:s['parameters']['matrix'].update(lr=.00125),lambda s,g:s['parameters']['vector'].update(second=torch.full((3,),-1.)),lambda s,g:s['parameters']['vector'].update(first=torch.zeros(4)),lambda s,g:s['parameters']['vector'].update(value=torch.ones(3,dtype=torch.float64)),lambda s,g:s['parameters']['vector'].update(updates=0),lambda s,g:g.pop('vector'),lambda s,g:g.update(extra=torch.ones(1)),lambda s,g:g.update(vector=torch.ones(4)),lambda s,g:g.update(vector=torch.ones(3,dtype=torch.float64)),lambda s,g:g.update(vector=torch.tensor([float('nan'),0.,0.])),lambda s,g:g.update(vector=torch.tensor([float('inf'),0.,0.])),lambda s,g:g.update(matrix=None,vector=None),lambda s,g:s['parameters']['matrix'].update(second=torch.zeros(3,2)),lambda s,g:s['parameters']['vector'].update(first=torch.tensor([float('nan'),0.,0.])),lambda s,g:g.update(vector=torch.ones(3,requires_grad=True))]:altered(fn)
            for s,g in invalid:
                old=copy.deepcopy((s,g))
                try:response(s,g)
                except (ValueError,RuntimeError,TypeError):pass
                else:raise AssertionError('Invalid synthetic response accepted')
                # equal_nan needed solely to compare rejected malformed NaN fixtures.
                def equal_bad(x,y):
                    if type(x) is torch.Tensor:return x.shape==y.shape and x.dtype==y.dtype and bool(((x==y)|(torch.isnan(x)&torch.isnan(y))).all())
                    if type(x) is dict:return x.keys()==y.keys() and all(equal_bad(x[k],y[k]) for k in x)
                    return x==y
                require(equal_bad(s,old[0]) and equal_bad(g,old[1]),'Invalid input mutated caller');rejections+=1
            for value,used in [({},META),({'x':float('nan')},0),({},-1)]:
                try:serialized(value,used)
                except ValueError:rejections+=1
                else:raise AssertionError('Invalid metadata publication cap accepted')
            require(torch.equal(rng,torch.get_rng_state()),'Global CPU RNG changed')
            for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Frozen source changed during generated proof')
            for r in libraries:require(reference(r['path'])==r,'Installed optimizer source changed')
        result=dict(status='generated_fixed_state_clipping_muon_adam_response_independent_oracle_passed',source_plan=reference(RUN/'plan.json'),source_bindings=1050,cases=cases,none_gradient_cases=2,rejections_before_mutation=rejections,max_parameter_error=max_parameter,max_moment_error=max_moment,max_muon_direction_error=max_direction,max_clip_coefficient_error=max_clip,linear_reference_error=linear,noncommutation_max_abs=gaps,sequential_vs_fixed_state_max_abs=sequential,clipped_cases=clipped,unclipped_cases=len(cases)-clipped,global_cpu_rng_unchanged=True,mps_api_called=False,generated_optimizer_steps_only=True,actual_model_optimizer_steps=0,models_or_raw_loaded=0,corpus_loaded=False,validation_encoded=False,holdout_encoded=False,weight_files_written=0,quality_result=False,adopted=False,scope='Tiny CPU deterministic synthetic states only. No actual-model/MPS response or batch/quality gain established.')
        publish(RUN/'result.json',result);publish(RUN/'STOP',dict(status='complete_generated_optimizer_response_proof',result=reference(RUN/'result.json'),never_rerun=True))
        print(json.dumps({k:v for k,v in result.items() if k not in ('cases','source_plan')}),flush=True)
    except BaseException as exc:
        if not(RUN/'STOP').exists():publish(RUN/'STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
        raise
