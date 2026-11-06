"""Train/validation composition controls, no model loading or scoring."""
import json,math
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_worker_v1.execution import checked,sources
from research.body_allocation_worker_v1.data import load as data_view
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;B=ROOT/'research/tokenizer-evidence/body-allocation-preparation-001';torch.set_num_threads(4)
def publish(p,d):
    raw=(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<=256*1024,'Bounded result')
    with p.open('xb') as f:f.write(raw)
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();require(not(E/'frequency001-plan.json').exists(),'Closed analysis cannot retry')
    fitref=reference(ROOT/'research/tokenizer-evidence/train-fit-diagnostic-001/result.json');fit=checked(fitref)
    closure=checked(reference(B/'quality001-closure001.json'));row=next(r for r in closure['trajectories'] if r['slot']==9);worker=checked(row['worker']);pinned=sources()
    plan=dict(kind='train_validation_frequency_composition_controls_v1',source=reference(__file__),sources=pinned,train_fit=fitref,
        controls='Fixed Jeffreys(.5) and Laplace(1) smoothed unigram estimates from train counts only,all50257IDs. No tuning/model scoring/checkpoint loads/holdout.',
        strata='Original train-fit counts0/1-5/6-50/51-500/501+. Coarse reweighting assumes unchanged within-bin model losses;unknownIDs not imputed.',
        resources='CPU120s/RSS4GiB/reserve8GiB,256KiB per metadata file,zero weight outputs.',
        interpretation='Unigram cross-entropy gap is a descriptive frequency-shift comparator,not a causal or additive decomposition of the model gap.')
    publish(E/'frequency001-plan.json',plan)
    try:
        with CpuPrerequisiteGuard(E/'frequency001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
            data=data_view(OmegaConf.create(worker['config']));train=data.train_data;val=data.val_data
            require((len(train),len(val))==(267688,34279),'Exact split counts required')
            estimation=torch.bincount(train,minlength=50257).double();tc=torch.bincount(train[1:],minlength=50257).double();vc=torch.bincount(val[1:],minlength=50257).double()
            pt=tc/tc.sum();pv=vc/vc.sum();mix=(pt+pv)/2
            def kl(p,q):
                keep=p>0;return float((p[keep]*(p[keep]/q[keep]).log()).sum())
            js=(kl(pt,mix)+kl(pv,mix))/2;baselines=[]
            for alpha in (.5,1.):
                q=(estimation+alpha)/(estimation.sum()+alpha*50257);nll=-q.log()
                tr=float((pt*nll).sum());vl=float((pv*nll).sum())
                baselines.append(dict(dirichlet_alpha=alpha,train_target_ce=tr,validation_target_ce=vl,gap=vl-tr))
            bins=[]
            for lo,hi in ((0,0),(1,5),(6,50),(51,500),(501,267688)):
                mask=(estimation>=lo)&(estimation<=hi);nt=int(tc[mask].sum());nv=int(vc[mask].sum())
                bins.append(dict(min_occurrences=lo,max_occurrences=hi,train_targets=nt,validation_targets=nv,train_mass=nt/267687,validation_mass=nv/34278,observed_token_ids=int(mask.sum())))
            require(sum(r['train_targets'] for r in bins)==267687 and sum(r['validation_targets'] for r in bins)==34278,'Complete bin coverage')
            known=34278-bins[0]['validation_targets'];reweighted=[]
            for report in fit['reports']:
                require([r['targets'] for r in report['frequency_strata']]==[r['train_targets'] for r in bins[1:]],'Original train-fit bin target counts differ')
                weighted=sum(b['validation_targets']*r['ce'] for b,r in zip(bins[1:],report['frequency_strata']))/known
                reweighted.append(dict(seed=report['seed'],original_train_ce=report['ce'],validation_known_mass=known/34278,
                    train_loss_reweighted_to_validation_known_bin_mix=weighted,coarse_composition_change=weighted-report['ce'],
                    assumption='Within-bin train losses held fixed;unseen-token val mass excluded,not a model validation CE or causal decomposition.'))
            require(sources()==pinned,'Sources changed')
        result=dict(status='completed_frequency_composition_analysis',plan=reference(E/'frequency001-plan.json'),data_identity=data.data_identity,
            training_unigram_support=int((estimation>0).sum()),validation_target_unseen_token_ids=int(((vc>0)&(estimation==0)).sum()),
            validation_target_unseen_mass=bins[0]['validation_mass'],jensen_shannon_nats=js,unigram_controls=baselines,frequency_strata=bins,coarse_reweighting=reweighted,
            original_model_mean_gap=fit['mean_validation_minus_train_ce'],model_loaded=False,model_rescored=False,training=False,holdout_scored=False,
            guard=reference(E/'frequency001-guard.json'))
        publish(E/'frequency001.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('data_identity','plan','guard','frequency_strata','coarse_reweighting')}))
    except Exception as exc:publish(E/'STOP-frequency001',dict(error=repr(exc),closed=True));raise
