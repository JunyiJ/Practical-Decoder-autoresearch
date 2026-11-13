"""Bounded CPU parity and shape checks needed by the new depth wrapper."""
import argparse,datetime,fcntl,gc,hashlib,json,platform,shutil,traceback
from pathlib import Path
import torch
from research.byte_input_model_v6 import ByteInputModel,Spec as OldSpec,build_codebook,build_optimizer as old_optimizer
from research.general_rsi_cross_size_v1.model import DepthModel
from research.general_rsi_cross_size_v1.contract import Spec,counts
from research.general_rsi_cross_size_v1.optimizer import build_optimizer
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-fixture-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def put(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def prepare():
    prior=read(ROOT/'research/tokenizer-evidence/general-rsi-panel-001/plan.json')
    sources=dict(prior['sources'])
    for path in [Path(__file__),* (ROOT/'research/general_rsi_cross_size_v1').glob('*.py'),ROOT/'src/research/cpu_prerequisite_guard.py']:
        sources[str(path.relative_to(ROOT))]=sha(path)
    p=dict(kind='cross_size_cpu_parity_v1',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,codebook_assets=prior['codebook_assets'],torch_version=torch.__version__,python_version=platform.python_version(),roles=dict(initializer=62111312020001,composer=62111312020002,extra_groups=[62111312020003,62111312020004,62111312020005]),prior_role_audit='rg research/research_state JSON and JSONL before assignment returned no matches',criteria='Exact old/new depth4 initialization/logits/loss/gradients/one clipped optimizer update and optimizer child state. Exact common body/output/prefix parameters and expected counts at depths4/8/16. Distinct extra block values/storage. Invalid depth/codebook rejected. No data or validation scoring.',timeout_seconds=600,rss_limit_bytes=4*1024**3,reserve_bytes=12*1024**3,max_output_bytes=8*1024**2,scientific_training=False,test_opened=False)
    OUT.mkdir();put(OUT/'plan.json',p);print(sha(OUT/'plan.json'))
def make(arm,depth,p,book,old=False):
    r=p['roles'];before=torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r['initializer'])
        if old:
            spec=OldSpec(arm) if arm=='native' else OldSpec(arm,r['composer'],book.identity)
            m=ByteInputModel(spec,None if arm=='native' else book)
        else:
            spec=Spec(arm,depth,tuple(r['extra_groups']),None if arm=='native' else r['composer'],None if arm=='native' else book.identity)
            m=DepthModel(spec,None if arm=='native' else book)
    assert torch.equal(before,torch.random.get_rng_state())
    return m

def run(p):
    torch.set_num_threads(4)
    assert torch.__version__==p['torch_version'] and platform.python_version()==p['python_version']
    for path,h in p['sources'].items():assert sha(ROOT/path)==h,path
    raw=[]
    for a in p['codebook_assets']:
        assert sha(a['path'])==a['sha256'];raw.append(Path(a['path']).read_bytes())
    book=build_codebook(*raw,'raw_bytes')
    x=torch.tensor([[0,1,42,256,999,50256,1000,17]],dtype=torch.int64);y=x.roll(-1,1)
    parity=[]
    for arm in ['native','raw_bytes']:
        old=make(arm,4,p,book,True);new=make(arm,4,p,book)
        oi,ni=dict(old.ordered_parameters()),dict(new.ordered_parameters());assert list(oi)==list(ni)
        assert all(torch.equal(oi[k],ni[k]) for k in oi)
        old.eval();new.eval()
        with torch.no_grad(),FP32RMSNormInference():
            a=old.primary_logits(x).values;b=new.primary_logits(x).values
        assert torch.equal(a,b)
        old.train();new.train();a=old.forward_train(x,y);b=new.forward_train(x,y)
        assert torch.equal(a.primary.values,b.primary.values) and torch.equal(a.loss,b.loss)
        a.loss.backward();b.loss.backward();assert all(torch.equal(oi[k].grad,ni[k].grad) for k in oi)
        op,np=old_optimizer(old,.0025),build_optimizer(new,.0025)
        ca=clip_gradients(oi.values(),1.0);cb=clip_gradients(ni.values(),1.0);assert ca==cb
        op.step();np.step();assert all(torch.equal(oi[k],ni[k]) for k in oi)
        for oc,nc in zip(op._optimizers,np._optimizers):
            for og,ng in zip(oc.param_groups,nc.param_groups):
                assert {k:v for k,v in og.items() if k!='params'}=={k:v for k,v in ng.items() if k!='params'}
                os,ns=oc.state[og['params'][0]],nc.state[ng['params'][0]];assert os.keys()==ns.keys()
                assert all(torch.equal(os[k],ns[k]) for k in os)
        parity.append(dict(arm=arm,parameters=len(oi),loss=float(a.loss.detach()),initialization_logits_gradients_update_optimizer_exact=True))
        del old,new,op,np,oi,ni,a,b,oc,nc,og,ng,os,ns;gc.collect()
    inventories=[];prefix=None
    for depth in [4,8,16]:
        common=None
        for arm in ['native','raw_bytes']:
            m=make(arm,depth,p,book);items=m.ordered_parameters();actual={n:digest(v) for n,v in items}
            shared={n:h for n,h in actual.items() if not n.startswith('composer.') and n!='transformer.wte.weight'}
            if common is None:common=shared
            else:assert shared==common
            if prefix is not None:assert all(shared[k]==h for k,h in prefix.items())
            assert len(items)==counts(m.spec)[0] and sum(v.numel() for _,v in items)==counts(m.spec)[1]
            addresses=[v.untyped_storage().data_ptr() for _,v in items];assert len(set(addresses))==len(addresses)
            # Nonconstant attention matrices must differ between every separately drawn block.
            draws=[actual[f'transformer.h.{i}.attn.dkv.weight'] for i in range(depth)]
            assert len(set(draws))==depth
            opt=build_optimizer(m,.0025)
            inventories.append(dict(arm=arm,depth=depth,parameter_tensors=len(items),parameters=sum(v.numel() for _,v in items),muon_groups=len(opt._optimizers[0].param_groups),adam_groups=len(opt._optimizers[1].param_groups),parameter_sha256=actual,private_extra_blocks_distinct=True))
            del opt,m,items;gc.collect()
        prefix=common
    rejects=[]
    for d in [0,5,12,True]:
        try:Spec('native',d,tuple(p['roles']['extra_groups']))
        except ValueError:rejects.append(str(d))
        else:raise AssertionError('invalid depth accepted')
    try:DepthModel(Spec('native',4,tuple(p['roles']['extra_groups'])),book)
    except ValueError:rejects.append('native_with_codebook')
    else:raise AssertionError('native codebook accepted')
    return dict(accepted=True,plan_sha256=sha(OUT/'plan.json'),parity=parity,inventories=inventories,negative_controls=rejects,caller_rng_unchanged=True,common_arm_and_depth_prefix_hashes_equal=True,test_opened=False,scope='CPU wrapper parity and initialization/inventory only. New-depth MPS numerical/restart/scoring/timing/memory remain unqualified.')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);a=ap.parse_args()
    if a.action=='prepare':prepare();return
    p=read(OUT/'plan.json');attempt=OUT/'attempt-01';attempt.mkdir()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
        with CpuPrerequisiteGuard(attempt/'guard.json',ROOT,timeout_seconds=p['timeout_seconds'],rss_limit_bytes=p['rss_limit_bytes'],reserve_bytes=p['reserve_bytes']):
            try:result=run(p);put(attempt/'result.json',result);print(json.dumps(result['parity']))
            except BaseException as e:put(attempt/'failure.json',dict(exception=repr(e),traceback=traceback.format_exc(),plan_sha256=sha(OUT/'plan.json')));raise
if __name__=='__main__':main()
