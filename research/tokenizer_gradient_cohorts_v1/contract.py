"""New cohort ordering/streaming contracts around the unchanged gradient kernel."""
import gc,json,math,os
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_gradient_noise_v1.component import Moments
from research.tokenizer_gradient_probe_v1 import runtime as C
from research.tokenizer_moe_model_v1.model import GPT,load_weights

SEEDS=(600501,600502,600503,600504)
INITIALIZER=600505
GENERATED=600506
META=16*1024**2
FLOOR=8*1024**3+META+1267904512+1458176
TARGET=FLOOR+2*1024**3
RUN=ROOT/'research_runs/tokenizer-gradient-noise-cohorts-v1-001'

def sources():
    pins=C.sources()
    require(len(pins)==1043,'All1043 proved predecessor bindings required')
    for p in [*sorted(Path(__file__).parent.glob('*.py')),C.E/'cohort-contracts001.py']:
        pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    require(len(pins)==1047,'Exact new three-file runtime and generated proof required')
    return pins

def prerequisites():
    done=C.checked(reference(C.E/'continuation-completed002.json'))
    review=C.checked(done['probe_review']);stop=C.checked(review['terminal'])
    require(review['status']=='complete_training_free_gradient_diagnostic_independently_reviewed' and stop['status']=='complete_training_free_gradient_diagnostic' and stop['never_rerun'],'First probe must remain closed and reviewed')
    prior=C.checked(review['plan']);prior_result=C.checked(review['result']);prior_guard=C.checked(review['guard'])
    integration=C.checked(done['integration']);integration_plan=C.checked(done['source_plan'])
    integration_guard=C.checked(integration['guards'][0])
    require(prior['sources']==prior_result['sources']==integration['sources']==integration_plan['sources']==C.sources(),'Exact unchanged1043 sources and original integration required')
    require(prior_guard['status']=='finished' and prior_guard['reason'] is None and integration['status']=='own_fixed_weight_gradient_integration_cpu_mps_and_actual_payloads_passed','Original complete successful proofs required')
    require(integration_guard['status']=='finished' and integration_guard['reason'] is None,'Original CPU/MPS integration guard required')
    roles=C.checked(reference(C.E/'fixture-role-census002.json'))
    require(roles['no_prior_matches'] and roles['cohorts']==list(SEEDS) and roles['initializer']==INITIALIZER and roles['generated']==GENERATED and roles['scientific_seed'] is False,'New previously unused private fixture roles required')
    require(len(set((*SEEDS,INITIALIZER,GENERATED)))==6 and not set(SEEDS)&{600490,600491,600492,600493,600494},'No old private role may enter cohorts')
    return dict(first_review=done['probe_review'],first_terminal=review['terminal'],integration=done['integration'],roles=reference(C.E/'fixture-role-census002.json'),design=reference(C.E/'cohort-design001.json'))

def load_dense(worker,device,initializer_seed):
    # Same proven construction/load equations, with this namespace's OWN role policy.
    require(type(initializer_seed) is int and initializer_seed==INITIALIZER,'Only declared new CPU initializer allowed')
    C.verify_artifact(worker['parent'])
    raw=torch.load(worker['parent']['path'],map_location='cpu',weights_only=True)
    weights=C.raw_weights(raw,worker)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(initializer_seed)
        model=GPT(plain(worker['config']['moe_model']));load_weights(model,weights)
    del weights,raw
    gc.collect()
    return model.to(device).train()

class Cohorts:
    def __init__(self):
        self.combined=Moments();self.current=None;self.reports=[]

    def update(self,index,seed,draw,gradients):
        n=self.combined.count
        require(all(type(x) is int for x in (index,seed,draw)) and n<128 and index==n//32 and seed==SEEDS[index] and draw==n%32+1,'Exact ordered cohort/seed/draw;no substitution,duplicate or extra draw')
        require(len(self.reports)==index and ((draw==1 and self.current is None) or (draw>1 and self.current is not None and self.current.count==draw-1)),'Consistent complete cohort state required')
        # Entire gradient inventory/finite values checked before EITHER accumulator changes.
        require(type(gradients) is dict and bool(gradients) and all(type(k) is str and bool(k) for k in gradients),'Complete named gradient dictionary required')
        schema={}
        for name,g in gradients.items():
            require(type(g) is torch.Tensor and g.device.type=='cpu' and g.dtype in (torch.float32,torch.float64) and g.layout==torch.strided and g.numel()>0 and bool(torch.isfinite(g).all()),'Finite nonempty CPU gradients required')
            schema[name]=(tuple(g.shape),g.dtype)
        require(self.combined.schema is None or schema==self.combined.schema,'Complete stable combined gradient inventory required')
        require(self.current is None or self.current.schema==schema,'Complete stable cohort gradient inventory required')
        if self.current is None:self.current=Moments()
        self.combined.update(gradients);self.current.update(gradients)
        if draw==32:
            result=self.current.report();self.reports.append(dict(cohort=index,seed=seed,statistics=result))
            self.current=None
            return result
        return None

    def report(self):
        require(self.combined.count==128 and self.current is None and len(self.reports)==4,'All four complete32draw cohorts required;no partial result')
        return dict(cohorts=self.reports,combined=self.combined.report())

def serialized(value,used,cap=META):
    require(type(used) is int and used>=0 and type(cap) is int and 0<cap<=META,'Exact bounded metadata budget required')
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    require(used+len(raw)<=cap,'Complete16MiB metadata cap required before publication')
    return raw

def publish(path,value):
    require(path.parent==RUN and path.name not in ('.','..'),'Exact new diagnostic root only')
    used=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
    raw=serialized(value,used)
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def check_norm_identity(report,draws):
    n=report['draws'];require(len(draws)==n,'Exact complete draw count required')
    for name,stat in report['parameter_statistics'].items():
        cov=(math.fsum(d['gradient_squared_norms'][name] for d in draws)-n*stat['mean_gradient_squared_norm'])/(n-1)
        require(math.isclose(cov,stat['trace_sample_covariance'],rel_tol=1e-8,abs_tol=1e-10),'Independent perparameter norm/covariance identity differs')
