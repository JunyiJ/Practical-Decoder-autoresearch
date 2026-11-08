"""Export/load and full-validation qualification using existing six RAW parents."""
import copy
import gc
import hashlib
from pathlib import Path
import numpy as np
import torch
from research import byte_input_training_v2 as training
from research import byte_input_inference_v1 as inference
from research.byte_input_model_v3 import Spec, ByteInputModel, build_codebook, build_optimizer

def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def parameters(model):
    return {n: digest(p) for n,p in model.ordered_parameters()}

def clean():
    gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()

def construct(plan, variant, backend, book):
    old_mps = torch.mps.get_rng_state().clone()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(plan['roles']['native_initializer'])
            spec = Spec(variant) if book is None else Spec(variant,plan['roles']['composer_initializer'],book.identity)
            model = ByteInputModel(spec,book).to(backend)
    finally: torch.mps.set_rng_state(old_mps)
    return training.Context(model,build_optimizer(model,.00125),training.load_view(plan['roles'],backend),
                            4,.00125,plan['roles'],plan['binding'])

def invalid_payload(name,value):
    p=dict(value); p['parameters']=dict(value['parameters'])
    small=min(p['parameters'],key=lambda n:p['parameters'][n].numel())
    if name=='kind':p['kind']='foreign'
    elif name=='identity':p['identity']=copy.deepcopy(value['identity']);p['identity']['roles']['train_sampler']+=1
    elif name=='parent':p['parent']=dict(value['parent'],sha256='0'*64)
    elif name=='step':p['completed_steps']=2
    elif name=='averaging':p['averaging']=dict(value['averaging'],half_life=100)
    elif name=='buffers':p['immutable_buffer_hashes']={}
    elif name=='missing':del p['parameters'][small]
    elif name=='shape':p['parameters'][small]=p['parameters'][small].reshape(-1,1)
    elif name=='dtype':p['parameters'][small]=p['parameters'][small].double()
    elif name=='nonfinite':p['parameters'][small]=p['parameters'][small].clone();p['parameters'][small].reshape(-1)[0]=float('nan')
    elif name=='alias':
        names=list(p['parameters']);pair=next((a,b) for a in names for b in names if a!=b and p['parameters'][a].shape==p['parameters'][b].shape)
        p['parameters'][pair[1]]=p['parameters'][pair[0]]
    else:raise AssertionError(name)
    return p

INVALID=('kind','identity','parent','step','averaging','buffers','missing','shape','dtype','nonfinite','alias')

def run(plan, archives, publish):
    torch.set_num_threads(4)
    cpu=torch.random.get_rng_state().clone();mps=torch.mps.get_rng_state().clone()
    assert torch.get_num_threads()==4 and torch.backends.mps.is_available()
    old=plan['training_plan'];exports=[];oracles=[];completed=[]
    assets=[Path(r['path']).read_bytes() for r in old['codebook_assets']]
    try:
        for backend in ('cpu','mps'):
            for variant in ('native','raw_bytes','nonlexical_codes'):
                prefix=backend+'-'+variant
                parent=next(r for r in plan['parents'] if Path(r['path']).name==prefix+'-step4.pt')
                book=None if variant=='native' else build_codebook(*assets,variant,
                    old['roles']['nonlexical_key'] if variant=='nonlexical_codes' else None)
                c=construct(old,variant,backend,book)
                expected=copy.deepcopy(c._identity);view=c.view
                before=parameters(c.model);rng=(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
                value=inference.export(c,parent)
                assert parameters(c.model)==before and rng==(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
                # Verify export bytes against the independently accepted RAW EMA, not online weights.
                raw=training.read_bound(parent)
                expected_weights={n:digest(t) for n,t in raw['averages']['averages']['200'].items()}
                assert {n:digest(t) for n,t in value['parameters'].items()}==expected_weights
                del raw
                rejected=[]
                for name in INVALID:
                    try: inference.validate(c.model,invalid_payload(name,value),expected,parent)
                    except ValueError:rejected.append(name)
                    else:raise AssertionError('Accepted invalid '+name)
                assert parameters(c.model)==before and rng==(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
                record=inference.write_new(archives/(prefix+'-h200.pt'),value);exports.append(record)
                for name,call in [('wrong_hash',lambda:inference.read_bound(dict(record,sha256='0'*64))),
                                  ('overwrite',lambda:inference.write_new(record['path'],value))]:
                    try:call()
                    except ValueError:rejected.append(name)
                    else:raise AssertionError('Accepted invalid '+name)
                assert inference.numerical._sha(Path(record['path']))==record['sha256']
                # Direct accepted model-v3 reference with the exact parent EMA.
                with torch.no_grad():
                    for n,p in c.model.ordered_parameters():p.copy_(value['parameters'][n])
                c.model.eval()
                starts=(0,2048,34176);reference={};labels={}
                with torch.no_grad():
                    for start in starts:
                        count=min(128,len(view.val_data)-1-start)
                        x=view.val_data[start:start+count].unsqueeze(0).to(backend)
                        reference[start]=c.model.primary_logits(x).values.detach().cpu().numpy().copy()
                        labels[start]=view.val_data[start+1:start+count+1].numpy().copy()
                del c,value;clean()
                rng=(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
                model=inference.load(record,expected,parent,book,backend)
                assert parameters(model)==expected_weights and not model.training
                assert rng==(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
                with torch.no_grad():
                    for start in starts:
                        x=view.val_data[start:start+len(labels[start])].unsqueeze(0).to(backend)
                        actual=inference._logits(model,x).cpu().numpy()
                        assert np.array_equal(actual,reference[start]), 'Exact direct/exported primary logits'
                del actual,x
                publish(prefix+'-loaded.json',dict(backend=backend,variant=variant,export=record,parent=parent,
                    weights=expected_weights,invalid_rejections=rejected,exact_parent_ema=True,
                    exact_loaded_logits=True,caller_rng_preserved=True))
                oracle=archives/(prefix+'-oracle.npz')
                np.savez(oracle,**{str(k):v for k,v in reference.items()},
                    **{'labels'+str(k):v for k,v in labels.items()})
                oracle_record=dict(path=str(oracle),size=oracle.stat().st_size,sha256=inference.numerical._sha(oracle))
                assert oracle_record['size']<80*1024**2
                oracles.append(oracle_record)
                result=inference.score(model,view,expected,record)
                scores=np.array(result['per_target_losses'],dtype='float64')
                errors=[]
                for start in starts:
                    values=reference[start][0].astype('float64')
                    high=values.max(axis=-1);wanted=high+np.log(np.exp(values-high[:,None]).sum(axis=-1))-values[np.arange(len(labels[start])),labels[start]]
                    error=np.abs(scores[start:start+len(wanted)]-wanted)
                    assert np.all(error<=4e-6+4e-5*np.abs(wanted)), 'Independent FP64 CE'
                    errors.append(dict(start=start,count=len(wanted),max_abs=float(error.max())))
                assert parameters(model)==expected_weights
                assert rng==(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
                publish(prefix+'-scores.json',result)
                row=dict(backend=backend,variant=variant,oracle=oracle_record,windows=268,targets=34278,
                    independent_ce=errors,atol=4e-6,rtol=4e-5,invalid_rejections=len(rejected),
                    weights_unchanged=True,caller_rng_preserved=True,no_training_updates=True)
                publish(prefix+'-complete.json',row);completed.append(row)
                del model,view,book,reference,labels,scores,result,values,wanted,error;clean()
        assert len(completed)==6 and len(exports)==6
        publish('result.json',dict(accepted=True,cases=completed,exports=exports,oracles=oracles,
            updates_executed=0,full_validation_targets=6*34278,invalid_transactions=6*(len(INVALID)+2),
            scope='Existing four-step fixture weights only; export/load and uncached full validation. No quality selection or cache/worker qualification.'))
    finally:
        torch.random.set_rng_state(cpu);torch.mps.set_rng_state(mps);clean()
