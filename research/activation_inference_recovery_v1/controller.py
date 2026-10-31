"""Finite separate-root inference controller. Requires a complete external freeze."""
import argparse
import os
from pathlib import Path
import sys
import time

from src.research.protocol import ROOT,digest
from src.research.tokenizer_activation_session import reference,require
from src.research.tokenizer_activation_entry import idle,GLOBAL_LOCK,session_lock,MetadataWriter
from src.research.tokenizer_wide_hybrid_process_v2 import run_process
from src.research.tokenizer_serial_process import _group_exists
from src.research.cpu_prerequisite_guard import observation,violation
from src.research.tokenizer_activation_timing import CONTRACT,FRESH_SEEDS,slot,summarize
from research.activation_inference_recovery_v1.context import (
    KIND,PLAN_NAME,Recovery,bind,budget,original_session,verify_original,PhaseGuard,guard_ref,verify_guard)
from research.activation_inference_recovery_v1.worker import produce,review_outputs

MODULE='research.activation_inference_recovery_v1.controller'
SUMMARY='evidence/meta/recovery-inference-summary.json'


def argv_for(ctx,seed,request,fds=None):
    fds=ctx.writer.lock_fds if fds is None else fds
    return [sys.executable,'-m',MODULE,'worker','--root',str(ctx.root),'--plan-sha256',ctx.plan['sha256'],
        '--seed',str(seed),'--request-sha256',request['sha256'],'--global-fd',str(fds[0]),'--campaign-fd',str(fds[1])]


def request_check(ctx,seed,request,prior,imported):
    rq=ctx.document(request)
    require(set(rq)=={'kind','plan','seed','binding','original_import_sha256','prior','parent_pid','lock_fds','prepare_guard','required_free_bytes'}
        and rq['kind']==KIND and rq['plan']==ctx.plan and type(rq['seed']) is int and rq['seed']==seed
        and request['path']==f'evidence/inference/{slot(seed)}/request.json' and rq['prior']==prior
        and type(rq['parent_pid']) is int and rq['parent_pid']>1
        and type(rq['lock_fds']) is list and len(rq['lock_fds'])==2 and len(set(rq['lock_fds']))==2
        and all(type(fd) is int and fd>2 for fd in rq['lock_fds'])
        and type(rq['required_free_bytes']) is int and rq['required_free_bytes']==CONTRACT['reserve_bytes']+CONTRACT['output_limit_bytes'],
        'Complete recovery request identity required')
    require(rq['binding']==imported['pair_bindings'][str(seed)] and rq['original_import_sha256']==digest(imported),
            'Original imported pair binding differs')
    require(rq['prepare_guard']['path']==f'control/recovery-prepare-inference-{slot(seed)}.json','Preparation guard path differs')
    verify_guard(ctx,rq['prepare_guard'],rq['parent_pid'])
    return rq


def process_review(ctx,seed,request):
    rq=ctx.document(request);ref=reference(ctx.root,f'evidence/inference/{slot(seed)}/process/process.json');p=ctx.document(ref)
    require(p.get('status')=='completed' and p.get('returncode')==0 and p.get('error') is None
        and type(p.get('pid')) is int and p['pid']>1 and p['pid']!=rq['parent_pid']
        and p.get('parent_pid')==rq['parent_pid'] and p.get('argv')==argv_for(ctx,seed,request,rq['lock_fds'])
        and p.get('capture_kind')=='bounded_pipe_v2' and p.get('log_truncated') is False,'Complete canonical recovery transport required')
    expected=dict(required_free_bytes=rq['required_free_bytes'],reserve_bytes=CONTRACT['reserve_bytes'],
        output_limit_bytes=CONTRACT['output_limit_bytes'],log_limit_bytes=CONTRACT['log_limit_bytes'],timeout_seconds=600)
    require(all(p.get(k)==v for k,v in expected.items()) and
        not any(k in p for k in ('cleanup_error','log_close_error','failure_observation_error')),'Recovery process limits/cleanup differ')
    sample=p['observation'];elapsed=p['elapsed_seconds']
    require(set(sample)=={'free_bytes','output_bytes','log_bytes'} and all(type(v) is int and v>=0 for v in sample.values())
        and sample['free_bytes']>=CONTRACT['reserve_bytes'] and sample['output_bytes']<=CONTRACT['output_limit_bytes']
        and sample['log_bytes']<=CONTRACT['log_limit_bytes'] and type(elapsed) in (float,int) and 0<=elapsed<600,
        'Recovery process resource observation differs')
    return ref,p


def inventory(ctx,seed,*,exclude_acceptance=False):
    stem=ctx.root/f'evidence/inference/{slot(seed)}'
    return [reference(ctx.root,str(path.relative_to(ctx.root))) for path in sorted(stem.rglob('*'))
            if path.is_file() and (not exclude_acceptance or path.name!='accepted.json')]


def accepted_prefix(ctx,source,imported):
    refs=[]
    for seed in FRESH_SEEDS:
        stem=f'evidence/inference/{slot(seed)}'
        if not (ctx.root/stem/'accepted.json').exists():break
        ref=reference(ctx.root,stem+'/accepted.json');done=ctx.document(ref)
        rq=request_check(ctx,seed,done['request'],refs,imported)
        process,p=process_review(ctx,seed,done['request'])
        require(done==dict(kind=KIND,status='accepted',seed=seed,request=done['request'],process=process,
            review=done['review'],entry_guard=done['entry_guard'],worker_guard=done['worker_guard'],
            review_guard=done['review_guard'],files=done['files']),'Recovery acceptance schema differs')
        for phase,pid in [('entry',p['pid']),('worker',p['pid']),('review',rq['parent_pid'])]:
            g=done[phase+'_guard']
            require(g['path']==f'control/recovery-{phase}-inference-{slot(seed)}.json','Recovery guard path differs')
            verify_guard(ctx,g,pid)
        require(done['review']['path']==stem+'/review.json','Recovery review path differs')
        independent=review_outputs(ctx,source,done['request'],rq['binding'],p['pid'],rq['parent_pid'])
        require(ctx.document(done['review'])==independent and done['files']==inventory(ctx,seed,exclude_acceptance=True),
                'Independent recovery review or inventory changed')
        refs.append(ref)
    return refs


def worker(ctx,seed,request):
    with original_session(ctx) as source:
        with PhaseGuard(ctx.writer,seed,'entry'):
            imported=bind(ctx);verify_original(ctx,source,imported)
            prior=accepted_prefix(ctx,source,imported);rq=request_check(ctx,seed,request,prior,imported)
            require(len(prior)<3 and FRESH_SEEDS[len(prior)]==seed,'Finite inference order differs')
            require(os.getppid()==rq['parent_pid'] and rq['lock_fds']==list(ctx.writer.lock_fds)
                and os.getpgid(os.getpid())==os.getpid(),'Inherited recovery process identity differs')
            idle((rq['parent_pid'],))
        return produce(ctx,source,request,rq['binding'])


def run_once(ctx):
    imported=bind(ctx,capacity=True);idle()
    with original_session(ctx) as source:
        verify_original(ctx,source,imported);prior=accepted_prefix(ctx,source,imported)
    if len(prior)==3:
        require(not (ctx.root/SUMMARY).exists(),'Completed recovery summary cannot repeat')
        results=[ctx.document(reference(ctx.root,f'evidence/inference/{slot(seed)}/result.json')) for seed in FRESH_SEEDS]
        report=summarize(results,imported['selection'],imported['inputs']['training96_sha256'])
        ctx.writer.publish_json(SUMMARY,dict(report=report,accepted=prior,original_import_sha256=digest(imported),adopted=False))
        return reference(ctx.root,SUMMARY)
    seed=FRESH_SEEDS[len(prior)];stem=f'evidence/inference/{slot(seed)}';request=None
    seen=budget(ctx)['files']
    for future in FRESH_SEEDS[len(prior):]:
        bucket=slot(future)
        require(not any(n.startswith(f'evidence/inference/{bucket}/') or n.startswith('control/') and
            (n.endswith(f'inference-{bucket}.json') or n.endswith(f'inference-{bucket}.json.tmp')) for n in seen),
            'Existing incomplete or future recovery attempt cannot retry')
    try:
        with PhaseGuard(ctx.writer,seed,'prepare'):
            with original_session(ctx) as source:
                require(bind(ctx,capacity=True)==imported,'Recovery plan changed');verify_original(ctx,source,imported);idle()
        # The original root is unlocked here. The child takes its own original
        # lock; only the global/new-root descriptors are inherited by supervisor.
        free=CONTRACT['reserve_bytes']+CONTRACT['output_limit_bytes']
        ctx.writer.publish_json(stem+'/request.json',dict(kind=KIND,plan=ctx.plan,seed=seed,binding=imported['pair_bindings'][str(seed)],
            original_import_sha256=digest(imported),prior=prior,parent_pid=os.getpid(),lock_fds=list(ctx.writer.lock_fds),
            required_free_bytes=free,prepare_guard=guard_ref(ctx,seed,'prepare')))
        request=reference(ctx.root,stem+'/request.json');started=time.monotonic()
        def monitor():
            why=violation(observation(ctx.root),time.monotonic()-started,
                rss_limit_bytes=CONTRACT['rss_limit_bytes'],reserve_bytes=CONTRACT['reserve_bytes'],timeout_seconds=600)
            require(why is None,'Parent recovery guard: '+str(why));budget(ctx)
        run_process(ctx.writer,'inference',slot(seed),argv_for(ctx,seed,request),ROOT,required_free_bytes=free,
            output_limit_bytes=CONTRACT['output_limit_bytes'],log_limit_bytes=CONTRACT['log_limit_bytes'],
            timeout_seconds=600,parent_check=monitor)
        with PhaseGuard(ctx.writer,seed,'review'):
            require(bind(ctx)==imported,'Recovery context changed');idle();process,p=process_review(ctx,seed,request)
            require(not _group_exists(p['pid']),'Recovery child left descendants')
            verify_guard(ctx,guard_ref(ctx,seed,'entry'),p['pid'])
            with original_session(ctx) as source:
                result=review_outputs(ctx,source,request,imported['pair_bindings'][str(seed)],p['pid'],os.getpid())
            ctx.writer.publish_json(stem+'/review.json',result)
        ctx.writer.publish_json(stem+'/accepted.json',dict(kind=KIND,status='accepted',seed=seed,request=request,process=process,
            review=reference(ctx.root,stem+'/review.json'),entry_guard=guard_ref(ctx,seed,'entry'),worker_guard=guard_ref(ctx,seed,'worker'),
            review_guard=guard_ref(ctx,seed,'review'),files=inventory(ctx,seed)))
        return reference(ctx.root,stem+'/accepted.json')
    except BaseException as exc:
        try:ctx.writer.publish_json(stem+'/dispatch-failure.json',dict(error=repr(exc)[:1000],request=request,retry_allowed=False))
        except BaseException as error:exc.add_note('Recovery dispatch failure publication: '+repr(error)[:1000])
        raise


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('step','worker'))
    parser.add_argument('--root',required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--seed',type=int);parser.add_argument('--request-sha256')
    parser.add_argument('--global-fd',type=int);parser.add_argument('--campaign-fd',type=int)
    args=parser.parse_args(argv);root=Path(args.root).absolute()
    require(root==root.resolve() and root.is_dir(),'Canonical existing recovery root required')
    def run(fds):
        writer=MetadataWriter(root,fds);plan=reference(root,PLAN_NAME)
        require(plan['sha256']==args.plan_sha256,'Recovery CLI plan hash differs');ctx=Recovery(writer,plan)
        if args.mode=='step':return run_once(ctx)
        request=reference(root,f'evidence/inference/{slot(args.seed)}/request.json')
        require(request['sha256']==args.request_sha256,'Recovery CLI request hash differs')
        return worker(ctx,args.seed,request)
    values=(args.seed,args.request_sha256,args.global_fd,args.campaign_fd)
    if args.mode=='step':
        require(all(v is None for v in values),'Step cannot supply child arguments')
        with session_lock(GLOBAL_LOCK.parent) as g,session_lock(root) as c:return run((g,c))
    require(all(v is not None for v in values),'Complete inherited recovery CLI required')
    return run((args.global_fd,args.campaign_fd))


if __name__=='__main__':main()
