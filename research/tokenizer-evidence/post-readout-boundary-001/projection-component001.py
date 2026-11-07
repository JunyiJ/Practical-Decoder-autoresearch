"""New literal single-projection forward/analyticbackward proof; no model or RNG draws."""
import argparse,fcntl,hashlib,json,math,os,shutil,time
from pathlib import Path
import torch
from research.readout_projection_v1.component import KINDS,logits,log_probs,nll
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[3];E=Path(__file__).resolve().parent/'projection-component001'
def ref(p):
    p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def require(value,message):
    if not value:raise ValueError(message)
def publish(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    require(sum(p.stat().st_size for p in E.iterdir() if p.is_file())+len(raw)<2*1024**2,'Complete two MiB metadata cap required')
    with (E/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def sources(pins):
    for name,digest in pins.items():require(ref(ROOT/name)['sha256']==digest,'Pinned source changed: '+name)
def dot(x,y):return math.fsum(a*b for a,b in zip(x,y))
def lsoft(x):
    top=max(x);normalizer=top+math.log(math.fsum(math.exp(v-top) for v in x))
    return [v-normalizer for v in x]
def oracle(kind,z,w,a,targets):
    # Literal scalar calculus only: no torch, matrix API, autograd or component calls.
    n,d,v=len(z),len(z[0]),len(w)
    require(n<=16 and d<=8 and v<=31,'Independent tiny oracle scope')
    gw=[[[] for _ in range(d)] for _ in range(v)];ga=[[[] for _ in range(d)] for _ in range(d)];gz=[];lp=[];ls=[];out=[]
    for i,x in enumerate(z):
        u=x if kind=='native' else [dot(row,x) for row in a]
        h=[math.tanh(t) for t in u] if kind=='tanh_projection1' else u
        scores=[dot(row,h)/4 for row in w];p=lsoft(scores);out.append(scores);lp.append(p);ls.append(-p[targets[i]])
        err=[(math.exp(p[j])-(j==targets[i]))/n for j in range(v)]
        gh=[math.fsum(err[j]*w[j][q]/4 for j in range(v)) for q in range(d)]
        for j in range(v):
            for q in range(d):gw[j][q].append(err[j]*h[q]/4)
        if kind=='native':gz.append(gh)
        else:
            gu=[gh[q]*(1-h[q]**2) if kind=='tanh_projection1' else gh[q] for q in range(d)]
            gz.append([math.fsum(gu[p]*a[p][q] for p in range(d)) for q in range(d)])
            for p in range(d):
                for q in range(d):ga[p][q].append(gu[p]*x[q])
    grads=[gz,[[math.fsum(t) for t in row] for row in gw]]
    if kind!='native':grads.append([[math.fsum(t) for t in row] for row in ga])
    return dict(logits=out,logp=lp,nll=ls,loss=math.fsum(ls)/n,grads=grads)

def fixture(name):
    z=[[.35,-.65],[1.15,.25],[-.45,.85]];w=[[.15,-.55],[-.75,.45],[.95,.2],[-.3,-.9]];a=[[.55,-.35],[.25,.8]];y=[2,0,3]
    if name=='zero_A':a=[[0.,0.],[0.,0.]]
    elif name=='identity_A':a=[[1.,0.],[0.,1.]]
    elif name=='signed_A':a=[[-.7,.45],[.35,-1.05]]
    elif name=='saturated_A':a=[[t*18 for t in row] for row in a]
    elif name=='sharp_vocabulary':w=[[t*55 for t in row] for row in w]
    elif name=='one_dimensional':z,w,a,y=[[.45]],[[-.7],[1.1],[.2]],[[-.65]],[1]
    elif name=='boundary_copies':
        # Deterministic copies/permutations of literal rational vectors, no random fixture.
        zx=[.125,-.375,.625,-.875,.25,-.5,.75,-1.];wx=[-.3,.1,-.7,.5,.9,-.2,.4,-.6]
        z=[zx[i%8:]+zx[:i%8] for i in range(16)];w=[[(1 if i%2 else -1)*t for t in wx[i%8:]+wx[:i%8]] for i in range(31)]
        a=[[.4 if p==q else .05*(-1 if (p+q)%2 else 1) for q in range(8)] for p in range(8)];y=[0,3,6,9,12,15,18,21,24,27,30,2,5,8,11,14]
    else:require(name=='ordinary','Known new literal fixture required')
    return z,w,a,y
def compare(actual,expected,atol,rtol):
    x=actual.detach().cpu().double();target=torch.tensor(expected,dtype=torch.float64)
    require(x.shape==target.shape and torch.allclose(x,target,atol=atol,rtol=rtol),'Independent elementwise forward/backward predicate failed')
    return float((x-target).abs().max())
def backend(device,dtype,tol):
    rows=[];distinguished=False
    atol=tol['cpu64_forward_atol'] if dtype==torch.float64 else tol['fp32_atol'];gatol=tol['cpu64_grad_atol'] if dtype==torch.float64 else tol['fp32_atol'];rtol=0. if dtype==torch.float64 else tol['fp32_rtol']
    for name in ['ordinary','zero_A','identity_A','signed_A','saturated_A','sharp_vocabulary','one_dimensional','boundary_copies']:
        zz,ww,aa,yy=fixture(name);args=[torch.tensor(v,dtype=dtype,device=device,requires_grad=True) for v in [zz,ww,aa]];z,w,a=args;y=torch.tensor(yy,dtype=torch.int64,device=device)
        before=[t.detach().cpu().clone() for t in args+[y]];rounded=[t.detach().cpu().double().tolist() for t in args];outputs={}
        for kind in KINDS:
            projection=None if kind=='native' else a;active=args[:2] if kind=='native' else args;expected=oracle(kind,*rounded,yy)
            value=logits(kind,z,w,projection);lp=log_probs(kind,z,w,projection);losses=nll(kind,z,w,projection,y,reduction='none');loss=nll(kind,z,w,projection,y)
            differences=[compare(value,expected['logits'],atol,rtol),compare(lp,expected['logp'],atol,rtol),compare(losses,expected['nll'],atol,rtol),compare(loss,expected['loss'],atol,rtol)]
            grads=torch.autograd.grad(loss,active);grad_error=max(compare(g,expected['grads'][i],gatol,rtol) for i,g in enumerate(grads))
            gathered=-lp.gather(1,y[:,None]).squeeze(1).mean();compare(gathered,expected['loss'],atol,rtol);other=torch.autograd.grad(gathered,active)
            for g,h in zip(grads,other):compare(g,h.detach().cpu().double().tolist(),gatol,rtol)
            compare(lp.exp().sum(-1),[1.]*len(yy),atol,rtol);outputs[kind]=value.detach().cpu()
            rows.append(dict(fixture=name,kind=kind,device=str(device),dtype=str(dtype),forward_max=max(differences),gradient_max=grad_error,all_elementwise_predicates_passed=True,target_only_fullgather_gradients_agree=True))
        if name=='ordinary':
            distinguished=float((outputs['linear_projection1']-outputs['tanh_projection1']).abs().max())>1e-3
            # Represented-map identity only, never an optimizer-dynamics assertion.
            compare(logits('native',z,w@a,None),outputs['linear_projection1'].double().tolist(),atol,rtol)
        if name=='identity_A':compare(outputs['native'],outputs['linear_projection1'].double().tolist(),atol,rtol)
        require(all(torch.equal(t.detach().cpu(),old) for t,old in zip(args+[y],before)) and all(t.grad is None for t in args),'Literal inputs or caller gradient slots changed')
    require(distinguished,'At least one nonlinear discriminator required');return rows

def invalids():
    vals=fixture('ordinary');z,w,a=[torch.tensor(t,dtype=torch.float32,requires_grad=True) for t in vals[:3]];y=torch.tensor(vals[3],dtype=torch.int64);rows=[]
    def reject(name,fn,inputs=None):
        tensors=inputs or [z,w,a,y];before=[t.detach().cpu().clone() for t in tensors]
        try:fn()
        except (ValueError,TypeError,RuntimeError):pass
        else:raise AssertionError('Invalid component input accepted: '+name)
        require(all(torch.allclose(t.detach().cpu(),v,rtol=0,atol=0,equal_nan=True) for t,v in zip(tensors,before)) and all(t.grad is None for t in tensors if t.is_leaf),'Invalid input mutated caller values/gradients')
        rows.append(name)
    reject('unknown_kind',lambda:logits('context_mix2',z,w,a));reject('nonstring_kind',lambda:logits(1,z,w,a));reject('native_projection',lambda:logits('native',z,w,a));reject('missing_A',lambda:logits('linear_projection1',z,w,None))
    reject('z_type',lambda:logits('linear_projection1',vals[0],w,a));reject('W_type',lambda:logits('linear_projection1',z,vals[1],a));reject('A_type',lambda:logits('linear_projection1',z,w,vals[2]));reject('target_type',lambda:nll('linear_projection1',z,w,a,vals[3]))
    for name,zz,ww,aa in [('z_rank',z[0],w,a),('W_rank',z,w[0],a),('A_rank',z,w,a[0]),('W_width',z,w[:,:1],a),('A_shape',z,w,a[:1]),('empty_N',z[:0],w,a),('N17',z[:1].expand(17,-1),w,a),('D9',torch.zeros(1,9),torch.zeros(3,9),torch.zeros(9,9)),('V32',z,torch.zeros(32,2),a),('V1',z,w[:1],a),('z_int',z.to(torch.int64),w,a),('W_dtype',z,w.double(),a),('A_dtype',z,w,a.half()),('unsupported_dtype',z.half(),w.half(),a.half())]:
        reject(name,lambda zz=zz,ww=ww,aa=aa:logits('tanh_projection1',zz,ww,aa),[zz,ww,aa])
    for k in range(3):
        v=[t.detach().clone() for t in [z,w,a]];v[k].reshape(-1)[0]=float('nan') if k%2==0 else float('inf');reject('nonfinite_'+str(k),lambda v=v:logits('tanh_projection1',*v),v)
    for name,yy in [('target_rank',y[:,None]),('target_count',y[:1]),('target_dtype',y.int()),('target_negative',torch.tensor([-1,0,1])),('target_V',torch.tensor([4,0,1]))]:reject(name,lambda yy=yy:nll('linear_projection1',z,w,a,yy),[z,w,a,yy])
    reject('reduction',lambda:nll('linear_projection1',z,w,a,y,reduction='sum'))
    # Every data role is checked on mixed devices, plus both ambient autocast flags.
    for anchor_device,other_device in [('cpu','mps'),('mps','cpu')]:
        base=[t.detach().to(anchor_device) for t in [z,w,a,y]]
        for i in range(4):
            v=list(base);v[i]=v[i].to(other_device)
            reject('mixed_'+anchor_device+'_'+str(i),lambda v=v:nll('tanh_projection1',v[0],v[1],v[2],v[3]),v)
        for ambient,dt in [('cpu',torch.bfloat16),('mps',torch.float16)]:
            with torch.autocast(device_type=ambient,dtype=dt):reject('autocast_'+anchor_device+'_'+ambient,lambda:nll('tanh_projection1',base[0],base[1],base[2],base[3]),base)
    return rows

def main(arg):
    require(ref(E/'plan.json')['sha256']==arg.plan_sha,'Observed exact plan SHA required');plan=json.loads((E/'plan.json').read_text());require(plan['source']==ref(__file__) and not(E/'attempt.json').exists() and not(E/'STOP').exists(),'New pinned command only')
    sources(plan['sources']);require(len(plan['sources'])==1130,'All1127 predecessors plus new3sources required');require(shutil.disk_usage(ROOT).free>=8*1024**3+2*1024**2,'Reserve and complete tiny metadata output required')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4)
        environment=dict(torch_version=str(torch.__version__),torch_git=torch.version.git_version,threads=torch.get_num_threads(),flags={k:os.environ.get(k) for k in plan['environment']['flags']})
        require(environment==plan['environment'],'Exact admitted runtime and four-thread flags required')
        publish('attempt.json',dict(pid=os.getpid(),plan=ref(E/'plan.json'),environment=environment))
        try:
            with StrictMpsGuard(E/'mps-guard.json',ROOT),CpuPrerequisiteGuard(E/'cpu-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
                cpu=torch.get_rng_state().clone();mps=torch.mps.get_rng_state().clone();began=time.perf_counter()
                rows=[]
                for device,dtype in [('cpu',torch.float64),('cpu',torch.float32),('mps',torch.float32)]:rows.extend(backend(device,dtype,plan['tolerances']))
                rejected=invalids();require(torch.equal(cpu,torch.get_rng_state()) and torch.equal(mps,torch.mps.get_rng_state()),'Global CPU/MPS RNG changed');sources(plan['sources'])
                publish('result.json',dict(status='passed_generated_single_projection_component_forward_and_analytic_backward',plan=ref(E/'plan.json'),sources=plan['sources'],rows=rows,rejected_cases=rejected,positive_cases=len(rows),cpu_mps_rng_unchanged=True,literal_inputs_unchanged=True,all_elementwise_predicates_passed=True,models_optimizersteps_corpus_checkpoints_qualityseeds_rngdraws_tensorfiles=0,holdout_scored=False,fullmodel_admitted=False,elapsed_seconds=time.perf_counter()-began))
            publish('STOP',dict(status='complete_generated_single_projection_component',no_retry=True,no_replay=True,result=ref(E/'result.json')));print(json.dumps(dict(status='complete',cases=len(rows),rejected=len(rejected),forward_max=max(r['forward_max'] for r in rows),grad_max=max(r['gradient_max'] for r in rows))),flush=True)
        except BaseException as exc:
            if not(E/'STOP').exists():publish('STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
            raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan-sha',required=True);main(p.parse_args())
