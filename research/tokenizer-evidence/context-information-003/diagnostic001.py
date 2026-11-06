"""Single predeclared local128-at256 fixed-weight diagnostic; no model selection."""
import gc,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_worker_v1.execution import checked,sources
from research.body_allocation_worker_v1.validation import verify_artifact
from research.body_allocation_worker_v1.data import load as data_view
from research.final_ln_scale_worker_v1.validation import target_contract
from research.mla_context_extension_v1.model import GPT,config,import_original_h200
from research.mla_context_locality_v1.session import Session
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;B=ROOT/'research/tokenizer-evidence/body-allocation-preparation-001';P=ROOT/'research/tokenizer-evidence/context-locality-preparation-001'
torch.set_num_threads(4)
def inventory():
    result=sources()
    for ns in ('mla_context_extension_v1','mla_context_locality_v1'):
        for p in sorted((ROOT/'research'/ns).glob('*.py')):result[str(p.relative_to(ROOT))]=reference(p)['sha256']
    return result
def publish(path,value):
    data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(data)<=512*1024,'Bounded diagnostic output')
    with path.open('xb') as f:f.write(data)
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();require(not(E/'plan.json').exists(),'Closed diagnostic cannot retry')
    cpu=checked(reference(P/'correctness001-cpu.json'));mps=checked(reference(P/'correctness001-mps.json'))
    require(cpu['status']==mps['status']=='passed' and cpu['sources']==mps['sources']==inventory(),'Exact CPU/MPS prerequisites required')
    global_ref=reference(ROOT/'research/tokenizer-evidence/context-information-002/result.json');global_result=checked(global_ref)
    require(global_result['status']=='completed_fixed_weight_context256_diagnostic' and len(global_result['reports'])==3,'Recorded global256 control required')
    global_rows={r['seed']:r for r in global_result['reports']}
    closure=checked(reference(B/'quality001-closure001.json'));require(closure['status']=='CLOSED_NOT_ADOPTED','Completed original study required')
    rows=[r for r in closure['trajectories'] if r['slot']>=9 and r['arm']=='incumbent'];require(len(rows)==3,'All three original fixed models required')
    plan=dict(kind='fixed_weight_local128_at256_diagnostic_v1',source=reference(__file__),sources=inventory(),
        closure=reference(B/'quality001-closure001.json'),cpu_proof=reference(P/'correctness001-cpu.json'),mps_proof=reference(P/'correctness001-mps.json'),
        original_models=[r['accepted'] for r in rows],conditioning_window=256,attention_window=128,reference_window=128,global256_control=global_ref,
        hypothesis='Trained-range keys may avoid harmful extra-key competition while preserving cross-boundary hidden context. Compare all three fixed models; a failure lowers priority. No tuning or adoption.',
        reference='Original frozen accepted full-validation128 scores; never rerun original closed inference/review or select a new model.',
        scope='One fixed attention window128 derived from training. Distinct session policy; immutable model unchanged. Same original targets and all three weights. No training, seed or adoption.',
        resources='Three serial existing H200 models, no new weights,512KiB metadata cap,600s/RSS4GiB/MPS6GiB/reserve8GiB.',
        new_scientific_seed=False,adoption_permitted=False,holdout_scored=False)
    publish(E/'plan.json',plan);reports=[];started=time.monotonic()
    try:
        with StrictMpsGuard(E/'guard.json',ROOT),torch.random.fork_rng(devices=[]):
            for row in rows:
                a=checked(row['accepted']);w=checked(a['worker']);ref=w['h200'];verify_artifact(ref)
                model=GPT(config()).to('mps');saved=torch.load(ref['path'],map_location='cpu',weights_only=True);import_original_h200(model,saved,ref['sha256']);del saved;session=Session(model)
                data=data_view(OmegaConf.create(w['config']),'mps');identity=target_contract(data);baseline=a['acceptance']['full_validation'];require(identity==baseline['target'],'Exact validation target identity required')
                chunks=[];counts=[]
                with torch.no_grad():
                    for start in range(0,len(data.val_data)-1,256):
                        n=min(256,len(data.val_data)-1-start);x=data.val_data[start:start+n].unsqueeze(0).to('mps');y=data.val_data[start+1:start+n+1].to('mps')
                        logits=session(x);require(logits.shape==(1,n,50257),'Exact original vocabulary/logits')
                        values=F.cross_entropy(logits[0],y,reduction='none').cpu().to(torch.float64);require(bool(torch.isfinite(values).all()),'Finite losses required');chunks.append(values);counts.append(n)
                        require(time.monotonic()-started<600,'Bounded diagnostic timed out')
                losses=torch.cat(chunks);require(len(losses)==sum(counts)==34278,'Full target coverage required');total=float(losses.sum())
                control=global_rows[w['config']['training']['seed']];require(control['h200']==ref and control['target']==identity,'Same fixed model and target controls required')
                reports.append(dict(seed=w['config']['training']['seed'],original_accepted=row['accepted'],h200=ref,target=identity,
                    conditioning_window=256,attention_window=128,window_counts=counts,tail_retained=True,block_loss_sums=[float(losses[i:i+2048].sum()) for i in range(0,34278,2048)],
                    total_loss=total,ce=total/34278,original_reference128_ce=baseline['ce'],conditioning_gain128_to_local256=baseline['ce']-total/34278,global256_ce=control['ce'],gain_global256_to_local256=control['ce']-total/34278,
                    parameters=28492416,cache_bytes_per_item=393216,training=False))
                del session,model,data,losses,chunks,logits,x,y;gc.collect();torch.mps.empty_cache();verify_artifact(ref)
            require(inventory()==plan['sources'],'Source changed during diagnostic')
        result=dict(status='completed_fixed_weight_local128_at256_diagnostic',plan=reference(E/'plan.json'),reports=reports,
            mean_local128_at256_ce=sum(r['ce'] for r in reports)/3,mean_original_reference128_ce=sum(r['original_reference128_ce'] for r in reports)/3,
            mean_conditioning_gain128_to_local256=sum(r['conditioning_gain128_to_local256'] for r in reports)/3,
            mean_global256_ce=sum(r['global256_ce'] for r in reports)/3,mean_gain_global256_to_local256=sum(r['gain_global256_to_local256'] for r in reports)/3,
            all_models_gain=all(r['conditioning_gain128_to_local256']>0 for r in reports),adopted=False,new_scientific_seed=False,training=False,holdout_scored=False,
            interpretation='One predeclared locality diagnostic on the same fixed models and targets; no adoption or fresh-model claim. Multilayer receptive field can exceed one attention window. Distinguishes extra-key competition from simple reset fragmentation imperfectly; native128 and global256 controls are immutable recorded values.',guard=reference(E/'guard.json'))
        publish(E/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('reports','plan','guard')}))
    except Exception as exc:publish(E/'STOP',dict(error=repr(exc),no_retry=True));raise
