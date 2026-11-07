"""New generated cohort/combined-stream oracle; no model,corpus or gradient replay."""
import copy,json,math
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_gradient_cohorts_v1 import contract as K
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.tokenizer_moe_worker_v1.execution import idle
E=K.C.E

def publish(p,value):
    with p.open('xb') as f:f.write(K.serialized(value,0))

def snapshot(m):
    def one(a):
        return (a.count,copy.deepcopy(a.schema),dict(a.m2),{k:v.clone() for k,v in a.means.items()})
    return (one(m.combined),one(m.current) if m.current is not None else None,copy.deepcopy(m.reports))

def same(a,b):
    if type(a) is torch.Tensor:return torch.equal(a,b)
    if type(a) in (list,tuple):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if type(a) is dict:return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    return a==b

def rejected(m,call):
    before=snapshot(m)
    try:call()
    except (ValueError,RuntimeError,TypeError,AssertionError):pass
    else:raise AssertionError('Malformed/partial contract accepted')
    require(same(before,snapshot(m)),'Rejected input mutated either accumulator/cohort state')

with session_lock(GLOBAL_LOCK.parent):
    idle();torch.set_num_threads(4)
    require(not(E/'cohort-contracts001-plan.json').exists() and not(E/'cohort-contracts001.json').exists(),'One new proof only;no retry')
    pins=K.sources();prerequisites=K.prerequisites()
    publish(E/'cohort-contracts001-plan.json',dict(sources=pins,prerequisites=prerequisites,generated_seed=K.GENERATED,models_loaded=0,corpus_loaded=False,old_proofs_replayed=False,math_kernel_unchanged=True))
    with CpuPrerequisiteGuard(E/'cohort-contracts001-guard.json',ROOT,timeout_seconds=600,poll_seconds=.25):
        before=torch.get_rng_state().clone();rng=torch.Generator().manual_seed(K.GENERATED);cases=[];maxerror=0.;rejections=0
        for dtype in (torch.float32,torch.float64):
            for kind in ('ordinary','constant','zero_mean'):
                if kind=='ordinary':values=torch.randn(128,7,generator=rng,dtype=dtype)+.3
                elif kind=='constant':values=torch.ones(128,7,dtype=dtype)*.5
                else:values=torch.tensor([1.,-1.],dtype=dtype).repeat(64).view(128,1).repeat(1,7)
                m=K.Cohorts()
                for j,x in enumerate(values):
                    r=m.update(j//32,K.SEEDS[j//32],j%32+1,dict(a=x[:4].reshape(2,2),b=x[4:]))
                    require((r is not None)==(j%32==31),'Only complete32draw cohort may report')
                    if j+1 in (1,31,32,33,64,96,127):rejected(m,lambda:m.report());rejections+=1
                report=m.report()
                for r,v in [(report['combined'],values),*[(q['statistics'],values[i*32:(i+1)*32]) for i,q in enumerate(report['cohorts'])]]:
                    v=v.double();mu=v.mean(0);norm=float(mu.square().sum());cov=float((v-mu).square().sum()/(len(v)-1));signal=norm-cov/len(v)
                    error=max(abs(norm-r['mean_gradient_squared_norm']),abs(cov-r['trace_sample_covariance']),abs(signal-r['signal_squared_unbiased']));maxerror=max(maxerror,error)
                    require(error<1e-12,'Independent stacked-vector oracle differs')
                    if kind=='zero_mean':require(r['signal_squared_unbiased']<0 and r['ratio_in_independent_B8_draw_units'] is None,'Nonpositive signal must remain undefined')
                    if kind=='constant':require(r['trace_sample_covariance']==0 and r['ratio_in_independent_B8_draw_units']==0,'Constant positive signal ratio must be zero')
                # Independent within+between-cohort decomposition of combined M2.
                combined=report['combined'];norm=combined['mean_gradient_squared_norm']
                total_norms=math.fsum(float(x.double().square().sum()) for x in values)
                require(abs((total_norms-128*norm)/127-combined['trace_sample_covariance'])<1e-12,'Combined norm identity differs')
                rejected(m,lambda:m.update(4,K.SEEDS[0],1,dict(a=torch.ones(2,2,dtype=dtype),b=torch.ones(3,dtype=dtype))));rejections+=1
                cases.append(dict(dtype=str(dtype),kind=kind,cohorts=4,draws=128,maxerror=error))
        m=K.Cohorts();good=dict(a=torch.ones(2,2),b=torch.ones(3));m.update(0,K.SEEDS[0],1,good)
        for args in [(0,600492,2),(1,K.SEEDS[1],1),(0,K.SEEDS[0],1),(0,K.SEEDS[0],3),(False,K.SEEDS[0],2),(0,True,2),(0,K.SEEDS[0],2.0),(-1,K.SEEDS[-1],2)]:
            rejected(m,lambda args=args:m.update(*args,good));rejections+=1
        for bad in [None,{},dict(a=good['a']),dict(a=good['a'],b=good['b'],c=torch.ones(1)),dict(a=torch.ones(4),b=good['b']),dict(a=good['a'].double(),b=good['b']),dict(a=good['a'],b=torch.tensor([1.,float('nan'),2.])),dict(a=good['a'],b=torch.tensor([1.,float('inf'),2.])),dict(a=good['a'],b=torch.ones(3,dtype=torch.int64)),dict(a=good['a'],b=torch.ones(0)),{1:torch.ones(2)},[]]:
            rejected(m,lambda bad=bad:m.update(0,K.SEEDS[0],2,bad));rejections+=1
        for args in [({},-1,K.META),({},0,0),({},K.META,K.META),({'x':float('nan')},0,K.META)]:
            try:K.serialized(*args)
            except (RuntimeError,ValueError):rejections+=1
            else:raise AssertionError('Invalid output cap accepted')
        for seed in (600493,600494,600492,True):
            try:K.load_dense({},'cpu',seed)
            except ValueError:rejections+=1
            else:raise AssertionError('Foreign initializer passed policy before load')
        require(torch.equal(before,torch.get_rng_state()),'Global CPU RNG mutated')
        require(K.sources()==pins,'Sources changed during generated proof')
    publish(E/'cohort-contracts001.json',dict(status='generated_cohort_combined_stream_and_transactional_contracts_passed',sources=pins,cases=cases,maxerror=maxerror,rejections_before_mutation=rejections,global_cpu_rng_unchanged=True,models_loaded=0,corpus_loaded=False,weight_files_written=0,optimizer_steps=0,old_proofs_replayed=False))
    print(json.dumps(dict(status='passed',cases=len(cases),rejections=rejections,maxerror=maxerror)),flush=True)
