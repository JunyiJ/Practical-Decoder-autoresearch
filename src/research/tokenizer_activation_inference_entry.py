"""Serial finite activation inference after the entire accepted quality panel.

Separate explicit command; never changes training ledger, freezes a plan, retries
an attempt, deletes a checkpoint or adopts a model. One seed per invocation.
"""
import argparse
import os
from pathlib import Path
import sys
import time

from .protocol import ROOT,digest
from .tokenizer_activation_session import Session,reference,require
from .tokenizer_activation_entry import (bind,idle,PhaseGuard,guard_ref,verify_guard,GLOBAL_LOCK,
    session_lock,MetadataWriter,run_process,_group_exists,observation,violation)
from .tokenizer_activation_timing import CONTRACT,FRESH_SEEDS,slot,summarize
from .tokenizer_activation_inference import quality_binding,budget,produce,review_outputs

KIND='activation_inference_dispatch_v1'


def argv_for(ledger,seed,request,fds=None):
    fds=ledger.writer.lock_fds if fds is None else fds
    return [sys.executable,'-m','src.research.tokenizer_activation_inference_entry','worker',
        '--root',str(ledger.root),'--plan-sha256',ledger.plan['sha256'],'--seed',str(seed),
        '--request-sha256',request['sha256'],'--global-fd',str(fds[0]),
        '--campaign-fd',str(fds[1])]


def process_review(ledger,seed,request):
    physical=slot(seed);rq=ledger.document(request);ref=reference(ledger.root,f'evidence/inference/{physical}/process/process.json')
    p=ledger.document(ref)
    require(p.get('status')=='completed' and p.get('returncode')==0 and p.get('error') is None
        and type(p.get('pid')) is int and p['pid']>1 and p['pid']!=rq['parent_pid']
        and p.get('parent_pid')==rq['parent_pid'] and p.get('argv')==argv_for(ledger,seed,request,rq['lock_fds'])
        and p.get('capture_kind')=='bounded_pipe_v2' and p.get('log_truncated') is False,
        'Complete canonical inference transport required')
    expected=dict(required_free_bytes=rq['required_free_bytes'],reserve_bytes=CONTRACT['reserve_bytes'],
        output_limit_bytes=CONTRACT['output_limit_bytes'],log_limit_bytes=CONTRACT['log_limit_bytes'],timeout_seconds=CONTRACT['job_seconds'])
    require(all(p.get(k)==v for k,v in expected.items()) and
        not any(k in p for k in ('cleanup_error','log_close_error','failure_observation_error')),'Inference process limits/cleanup differ')
    sample=p['observation'];elapsed=p['elapsed_seconds']
    require(set(sample)=={'free_bytes','output_bytes','log_bytes'} and all(type(v) is int and v>=0 for v in sample.values())
        and sample['free_bytes']>=CONTRACT['reserve_bytes'] and sample['output_bytes']<=CONTRACT['output_limit_bytes']
        and sample['log_bytes']<=CONTRACT['log_limit_bytes'] and type(elapsed) in (float,int) and 0<=elapsed<600,
        'Inference process observation differs')
    return ref,p


def request_check(ledger,seed,request,prior):
    rq=ledger.document(request)
    require(set(rq)=={'kind','plan','seed','binding','prior','parent_pid','lock_fds','prepare_guard','required_free_bytes'}
        and rq['kind']==KIND and rq['plan']==ledger.plan and type(rq['seed']) is int and rq['seed']==seed
        and request['path']==f'evidence/inference/{slot(seed)}/request.json' and rq['prior']==prior
        and type(rq['parent_pid']) is int and rq['parent_pid']>1
        and type(rq['required_free_bytes']) is int and rq['required_free_bytes']>=CONTRACT['reserve_bytes'],
        'Inference request binding differs')
    require(rq['binding']==quality_binding(ledger,seed),'Inference quality binding differs')
    require(rq['prepare_guard']['path']==f'control/activation-prepare-inference-{slot(seed):02d}.json','Preparation path differs')
    verify_guard(ledger,rq['prepare_guard'],rq['parent_pid'])
    return rq


def accepted_prefix(ledger):
    refs=[]
    for seed in FRESH_SEEDS:
        physical=slot(seed);stem=f'evidence/inference/{physical}';path=ledger.root/stem/'accepted.json'
        if not path.exists():break
        ref=reference(ledger.root,stem+'/accepted.json');done=ledger.document(ref)
        rq=request_check(ledger,seed,done['request'],refs)
        process,p=process_review(ledger,seed,done['request'])
        expected=dict(kind=KIND,status='accepted',seed=seed,request=done['request'],process=process,
            review=done['review'],entry_guard=done['entry_guard'],worker_guard=done['worker_guard'],
            review_guard=done['review_guard'],files=done['files'])
        require(done==expected,'Inference completion schema differs')
        for phase,pid in [('entry',p['pid']),('worker',p['pid']),('review',rq['parent_pid'])]:
            g=done[phase+'_guard'];require(g['path']==f'control/activation-{phase}-inference-{physical:02d}.json','Guard path differs')
            verify_guard(ledger,g,pid)
        require(done['review']['path']==stem+'/review.json','Review path differs')
        independent=review_outputs(ledger,done['request'],rq['binding'],p['pid'],rq['parent_pid'])
        require(ledger.document(done['review'])==independent,'Independent inference review differs')
        actual=[reference(ledger.root,str(q.relative_to(ledger.root))) for q in sorted((ledger.root/stem).rglob('*'))
                if q.is_file() and q.name!='accepted.json']
        require(done['files']==actual,'Inference producer inventory changed')
        refs.append(ref)
    return refs


def worker(ledger,seed,request):
    physical=slot(seed)
    with PhaseGuard(ledger.writer,'inference',physical,'entry'):
        bind(ledger);prior=accepted_prefix(ledger);rq=request_check(ledger,seed,request,prior)
        require(len(prior)<3 and FRESH_SEEDS[len(prior)]==seed,'Finite inference order differs')
        require(os.getppid()==rq['parent_pid'] and rq['lock_fds']==list(ledger.writer.lock_fds)
                and os.getpgid(os.getpid())==os.getpid(),'Inherited process identity differs')
        idle((rq['parent_pid'],))
    def recheck():
        bind(ledger);budget(ledger)
        require(ledger.document(request)==rq and quality_binding(ledger,seed)==rq['binding'],'Inference inputs changed')
    return produce(ledger,request,rq['binding'],recheck)


def run_once(ledger):
    bind(ledger);idle();quality_binding(ledger,FRESH_SEEDS[0]);prior=accepted_prefix(ledger)
    if len(prior)==3:
        name='evidence/meta/activation-inference-summary.json'
        require(not (ledger.root/name).exists(),'Completed inference summary cannot be repeated')
        results=[ledger.document(reference(ledger.root,f'evidence/inference/{slot(s)}/result.json')) for s in FRESH_SEEDS]
        b=quality_binding(ledger,FRESH_SEEDS[0]);report=summarize(results,b['selection'],b['inputs']['training96_sha256'])
        ledger.writer.publish_json(name,dict(report=report,accepted=prior,adopted=False))
        return reference(ledger.root,name)
    seed=FRESH_SEEDS[len(prior)];physical=slot(seed);stem=f'evidence/inference/{physical}';request=None
    seen=budget(ledger)['files']
    for future in FRESH_SEEDS[len(prior):]:
        bucket=slot(future)
        require(not any(n.startswith(f'evidence/inference/{bucket}/') or
            n.startswith('control/') and (n.endswith(f'inference-{bucket}.json') or n==f'control/process-inference-{bucket}.json')
            for n in seen),'Existing incomplete or future inference attempt cannot retry')
    try:
        with PhaseGuard(ledger.writer,'inference',physical,'prepare'):
            b=quality_binding(ledger,seed);idle()
        # Future output is entirely metadata already covered by the full panel
        # reserve. Also reserve this bounded seed output at the process boundary.
        free=CONTRACT['reserve_bytes']+CONTRACT['output_limit_bytes']
        ledger.writer.publish_json(stem+'/request.json',dict(kind=KIND,plan=ledger.plan,seed=seed,binding=b,prior=prior,
            parent_pid=os.getpid(),lock_fds=list(ledger.writer.lock_fds),required_free_bytes=free,
            prepare_guard=guard_ref(ledger,'inference',physical,'prepare')))
        request=reference(ledger.root,stem+'/request.json');started=time.monotonic()
        def monitor():
            why=violation(observation(ledger.root),time.monotonic()-started,
                rss_limit_bytes=CONTRACT['rss_limit_bytes'],reserve_bytes=CONTRACT['reserve_bytes'],timeout_seconds=600)
            require(why is None,'Parent inference guard: '+str(why))
        run_process(ledger.writer,'inference',physical,argv_for(ledger,seed,request),ROOT,
            required_free_bytes=free,output_limit_bytes=CONTRACT['output_limit_bytes'],
            log_limit_bytes=CONTRACT['log_limit_bytes'],timeout_seconds=600,parent_check=monitor)
        with PhaseGuard(ledger.writer,'inference',physical,'review'):
            bind(ledger);idle();process,p=process_review(ledger,seed,request)
            require(not _group_exists(p['pid']),'Inference child left descendants')
            verify_guard(ledger,guard_ref(ledger,'inference',physical,'entry'),p['pid'])
            result=review_outputs(ledger,request,b,p['pid'],os.getpid())
            ledger.writer.publish_json(stem+'/review.json',result)
        ledger.writer.publish_json(stem+'/accepted.json',dict(kind=KIND,status='accepted',seed=seed,request=request,process=process,
            review=reference(ledger.root,stem+'/review.json'),entry_guard=guard_ref(ledger,'inference',physical,'entry'),
            worker_guard=guard_ref(ledger,'inference',physical,'worker'),review_guard=guard_ref(ledger,'inference',physical,'review'),
            files=[reference(ledger.root,str(q.relative_to(ledger.root))) for q in sorted((ledger.root/stem).rglob('*')) if q.is_file()]))
        return reference(ledger.root,stem+'/accepted.json')
    except BaseException as exc:
        try:ledger.writer.publish_json(stem+'/dispatch-failure.json',dict(error=repr(exc)[:1000],request=request,retry_allowed=False))
        except BaseException as error:exc.add_note('Failure publication: '+repr(error)[:1000])
        raise


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('step','worker'))
    parser.add_argument('--root',required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--seed',type=int);parser.add_argument('--request-sha256')
    parser.add_argument('--global-fd',type=int);parser.add_argument('--campaign-fd',type=int)
    args=parser.parse_args(argv);root=Path(args.root).absolute()
    require(root==root.resolve() and root.is_dir(),'Canonical existing root required')
    def run(fds):
        writer=MetadataWriter(root,fds);plan=reference(root,'evidence/meta/activation-plan.json')
        require(plan['sha256']==args.plan_sha256,'Frozen CLI plan hash differs');ledger=Session(writer,plan)
        if args.mode=='step':return run_once(ledger)
        request=reference(root,f'evidence/inference/{slot(args.seed)}/request.json')
        require(request['sha256']==args.request_sha256,'Inference request CLI hash differs')
        return worker(ledger,args.seed,request)
    values=(args.seed,args.request_sha256,args.global_fd,args.campaign_fd)
    if args.mode=='step':
        require(all(v is None for v in values),'Step mode cannot supply child arguments')
        with session_lock(GLOBAL_LOCK.parent) as g,session_lock(root) as c:return run((g,c))
    require(all(v is not None for v in values),'Complete inherited worker CLI required')
    return run((args.global_fd,args.campaign_fd))


if __name__=='__main__':main()
