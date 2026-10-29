"""Compact live inference handoff for the finite three-seed serial panel.

The parent replays full retained history before publishing readiness. The child
reconstructs the same immutable binding, checks frozen source/runtime/input and
numerical evidence, and verifies live transport at every worker boundary.
The child CLI requires exact inherited descriptors and parent readiness.
The separate serial controller owns idle replay, resource checks and reaping.
"""
import math
import os
from pathlib import Path
import sys
import time

from .protocol import digest,file_digest
from .cpu_prerequisite_guard import violation
from .tokenizer_wide_hybrid_cleanup import _check,TIMEOUT
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_retention_receipts import _document,_sha
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_plan_v2 import bind_campaign
from .tokenizer_wide_hybrid_quality_handoff_v2 import _log_prefix
from . import tokenizer_wide_hybrid_inference_preparation_v2 as prep
from . import tokenizer_wide_hybrid_inference_outputs_v2 as outputs

MODULE='src.research.tokenizer_wide_hybrid_inference_entry_v2'
PHASES={'entry':(), 'loaded':('worker-attempt.json',), 'measured':('worker-attempt.json',),
        'saved':('worker-attempt.json','result.json','models.json')}


def _seed(seed):
    _require(type(seed) is int and seed in outputs.SEEDS,'Exact finite inference seed required')


def worker_argv(writer,seed,ready_ref):
    _require(type(seed) is int and seed in outputs.SEEDS,'Exact inference seed required')
    _require(isinstance(writer,MetadataWriter) and set(ready_ref)=={'path','sha256'}
        and ready_ref['path']==f'inference/{seed}/ready.json','Exact readiness reference required')
    return [sys.executable,'-m',MODULE,'worker','--campaign',str(writer.root),'--seed',str(seed),
        '--readiness-sha256',_sha(ready_ref['sha256']),'--global-fd',str(writer.lock_fds[0]),
        '--campaign-fd',str(writer.lock_fds[1])]


def prepare_first_readiness(writer,request_ref):
    """Full idle replay then exclusive small seal; caller owns an outer CPU guard.

    No child is created here. A partial readiness publication is not resumable.
    The saved request's historical parent may differ from this live parent.
    """
    outputs._budget(writer)
    bound=prep.verify_prepared_inference(writer,request_ref)
    value=dict(kind=outputs.READY_KIND,seed=3511,parent_pid=os.getpid(),request_ref=request_ref,
        bindings_sha256=digest(bound['triples'][3511]),protected_files_sha256=digest(bound['files']),
        contract=bound['contract'])
    outputs.recheck(bound['files']);outputs._budget(writer)
    path=writer.publish_json('evidence/inference/3511/ready.json',value)
    outputs.recheck(bound['files']);outputs._budget(writer)
    return dict(path='inference/3511/ready.json',sha256=file_digest(path))


def _immutable(writer,ready):
    """Reconstruct original prepared bytes, excluding exact live process files."""
    seed=ready['seed']
    _require(type(seed) is int and seed in outputs.SEEDS,'Exact inference readiness seed required')
    root=writer.root/'evidence';ref=ready['request_ref'];request=_document(root,ref)
    args={k:request[k] for k in ('plan_sha256','completion_ref','projection_ref','rank_completion_ref')}
    with writer._serialized():prep._request(writer,ref,**args)
    prep._guard(writer,request['parent_pid'],finished=True)
    # Only the mutable namespace census is locked, never frozen-file hashing.
    frozen=bind_campaign(root/'meta/plan.json',request['plan_sha256'],live_writer=writer)
    with writer._serialized():census=snapshot(writer.root)
    inventory={};controls={};deadline=time.monotonic()+TIMEOUT
    for name,item in census['files'].items():
        if name.startswith('evidence/') and not name.startswith('evidence/inference/'):
            target=inventory;key=name.removeprefix('evidence/')
        elif name.startswith('control/') and name not in {
                'control/.metadata.lock',prep.GUARD} and not name.startswith(
                    ('control/inference-worker-','control/process-inference-')):
            target=controls;key=name
        else:continue
        binding=dict(size=item['size'],sha256=file_digest(writer.root/name))
        _check(writer.root,name,binding,deadline);target[key]=binding
    _require(digest(inventory)==request['quality_inventory_sha256']
        and digest(controls)==request['quality_controls_sha256'],
        'Original quality evidence or control identity changed')
    plan=_document(root,dict(path='meta/plan.json',sha256=request['plan_sha256']))
    specs=outputs.confirmation_specs(request['selection']);pairs={}
    for slot,spec in enumerate(specs,15):
        stem=f'quality/{slot:02d}'
        result=_document(root,dict(path=stem+'/result.json',sha256=inventory[stem+'/result.json']['sha256']))
        pair={}
        for role in ('raw','h200'):
            path=Path(result[role]['checkpoint'])
            _require(path.is_absolute() and path.parent==root/stem/'checkpoints','Exact retained fresh pair required')
            name=str(path.relative_to(root));pair[role]=dict(path=name,**inventory[name])
        pairs[stem]=pair
    report=dict(selection=request['selection'],records=[None]*15+specs,checkpoint_pairs=pairs)
    triples=prep.gate._triples(root,report,frozen,inventory,plan['tokenizer'])
    _require(digest(triples)==request['triples_sha256'],'Reconstructed triples differ from full idle audit')
    files=dict(frozen['files'])
    for prefix,items in ((root,inventory),(writer.root,controls)):
        for name,item in items.items():
            path=str(prefix/name)
            _require(path not in files or files[path]==item['sha256'],'Conflicting frozen file identity')
            files[path]=item['sha256']
    files[str(root/ref['path'])]=ref['sha256']
    files[str(writer.root/prep.GUARD)]=file_digest(writer.root/prep.GUARD)
    # Earlier completed inference files are immutable inputs to this seed.
    # Exact finite scope and their semantics are checked by the idle parent;
    # readiness's protected-files digest binds them again in the child.
    for prior in outputs.SEEDS[:outputs.SEEDS.index(seed)]:
        for name in census['files']:
            if name.startswith(f'evidence/inference/{prior}/') or name in {
                    f'control/inference-worker-{prior}.json',f'control/process-inference-{prior}.json'}:
                files[str(writer.root/name)]=file_digest(writer.root/name)
    return dict(kind=outputs.ENTRY_KIND,seed=seed,parent_pid=ready['parent_pid'],request_ref=ref,
        ready_ref=dict(path=f'inference/{seed}/ready.json',sha256=file_digest(root/f'inference/{seed}/ready.json')),
        binding=dict(bindings=triples[seed],selection=request['selection'],tokenizer=plan['tokenizer'],
                     token_sha256=plan['training_token_sha256'],files=files))


def _launch(argv,parent):
    return dict(status='preflight',argv=argv,parent_pid=parent,pid=None,returncode=None,error=None,
        required_free_bytes=outputs.CONTRACT['reserve_bytes'],reserve_bytes=outputs.CONTRACT['reserve_bytes'],
        output_limit_bytes=outputs.LIMIT,log_limit_bytes=outputs.LIMIT,timeout_seconds=600,
        observation=None,capture_kind='bounded_pipe_v2',log_truncated=False)


def _live(writer,argv,parent,phase):
    """Read only under the shared metadata lock; never hash weights while held."""
    seed=int(argv[7]);_seed(seed)
    stem=f'evidence/inference/{seed}';control=f'control/process-inference-{seed}.json'
    with writer._serialized():
        seen=snapshot(writer.root)['files']
        _require(not any(n.startswith('emergency/') for n in seen),'Emergency evidence blocks live inference')
        inference={n for n in seen if n.startswith('evidence/inference/')}
        expected={ 'evidence/'+prep.REQUEST,stem+'/ready.json',stem+'/process/launch.json',stem+'/process/worker.log'}
        expected|={stem+'/'+n for n in PHASES[phase]}
        prior_controls=set()
        from .tokenizer_wide_hybrid_inference_acceptance_v2 import COMPLETED
        for prior in outputs.SEEDS[:outputs.SEEDS.index(seed)]:
            expected|={f'evidence/inference/{prior}/'+n for n in COMPLETED|{'review.json'}}
            controller=f'evidence/inference/{prior}/controller.json'
            if controller in seen:expected.add(controller)
            prior_controls|={f'control/inference-worker-{prior}.json',f'control/process-inference-{prior}.json'}
        _require(inference==expected,'Exact live inference phase files required')
        current={n for n in seen if n.startswith(('control/inference-','control/process-inference-'))}
        _require(current=={prep.GUARD,control,f'control/inference-worker-{seed}.json'}|prior_controls,'Exact live inference controls required')
        launch=_document(writer.root,dict(path=stem+'/process/launch.json',sha256=file_digest(writer.root/stem/'process/launch.json')))
        expected_launch=_launch(argv,parent)
        _require(digest(launch)==digest(expected_launch),'Live launch identity or limits differ')
        state=_document(writer.root,dict(path=control,sha256=file_digest(writer.root/control)))
        fixed={k:v for k,v in expected_launch.items() if k not in ('status','pid','observation')}
        _require(set(state)==set(expected_launch)|{'elapsed_seconds'}
            and digest({k:state[k] for k in fixed})==digest(fixed),'Live supervisor contract differs')
        elapsed=state['elapsed_seconds'];sample=state['observation']
        _require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<600
            and type(sample) is dict and set(sample)=={'free_bytes','output_bytes','log_bytes'}
            and all(type(v) is int and v>=0 for v in sample.values())
            and sample['free_bytes']>=outputs.CONTRACT['reserve_bytes']
            and sample['output_bytes']+outputs.TERMINAL_RESERVE<=outputs.LIMIT,
            'Live supervisor observation violates limits')
        # Popen may have started us just before the parent's first running write.
        starting=state['status']=='preflight' and state['pid'] is None
        _require(starting or state['status']=='running' and type(state['pid']) is int and state['pid']==os.getpid(),
                 'Live supervisor child identity differs')
        path=writer.root/stem/'process/worker.log';st=path.stat()
        _require(sample['log_bytes']<=st.st_size<=outputs.LIMIT and st.st_nlink==1,'Live log observation differs')
        guard_name=f'control/inference-worker-{seed}.json'
        guard=_document(writer.root,dict(path=guard_name,sha256=file_digest(writer.root/guard_name)))
        limits=dict(timeout_seconds=600,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)
        _require(set(guard)=={'kind','status','reason','pid','limits','elapsed_seconds','observation'}
            and guard['kind']==outputs.KIND and guard['status']=='starting' and guard['reason'] is None
            and type(guard['pid']) is int and guard['pid']==os.getpid()
            and digest(guard['limits'])==digest(limits),'Live worker guard differs')
        sample=guard['observation'];elapsed=guard['elapsed_seconds']
        _require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<600
            and set(sample)=={'peak_rss_bytes','free_bytes'} and all(type(v) is int and v>0 for v in sample.values())
            and violation(sample,elapsed,**limits) is None,'Live worker resource violation')
        return starting,(st.st_dev,st.st_ino,st.st_size),_log_prefix(path,st.st_size)


def verify_live_entry(writer,seed,ready_ref,actual_argv,phase):
    _seed(seed);outputs._budget(writer)
    argv=worker_argv(writer,seed,ready_ref)
    _require(actual_argv==argv and phase in PHASES,'Exact child command and boundary required')
    root=writer.root/'evidence';ready=_document(root,ready_ref);parent=os.getppid()
    _require(type(ready.get('parent_pid')) is int and ready['parent_pid']==parent>1
        and os.getpgid(os.getpid())==os.getpid(),'Live parent and owned child process group required')
    deadline=time.monotonic()+2
    while True:
        starting,stamp,prefix=_live(writer,argv,parent,phase)
        if not starting:break
        _require(phase=='entry' and time.monotonic()<deadline,'Supervisor did not establish running child')
        time.sleep(.01)
    entry=_immutable(writer,ready)
    _require(entry['ready_ref']==ready_ref,'Readiness changed while reconstructing entry')
    outputs._binding(writer,seed,entry)
    starting,after,_=_live(writer,argv,parent,phase)
    path=root/f'inference/{seed}/process/worker.log'
    _require(not starting and after[:2]==stamp[:2] and after[2]>=stamp[2]
        and _log_prefix(path,stamp[2])==prefix,'Live log replaced or previous bytes changed')
    _document(root,ready_ref);outputs._budget(writer)
    _require(os.getppid()==parent and os.getpgid(os.getpid())==os.getpid(),'Live process identity changed')
    return entry


def run_worker(writer,seed,ready_ref,actual_argv):
    """Bounded child composition with independent live verification."""
    _seed(seed)
    return outputs.execute_bound_triple(writer,seed,
        entry_check=lambda phase:verify_live_entry(writer,seed,ready_ref,actual_argv,phase))



def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['worker'])
    parser.add_argument('--campaign',required=True)
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--readiness-sha256',required=True)
    parser.add_argument('--global-fd',required=True,type=int)
    parser.add_argument('--campaign-fd',required=True,type=int)
    args=parser.parse_args(argv)
    writer=MetadataWriter(args.campaign,(args.global_fd,args.campaign_fd))
    ref=dict(path=f'inference/{args.seed}/ready.json',sha256=_sha(args.readiness_sha256))
    actual=[sys.executable,'-m',MODULE,*(sys.argv[1:] if argv is None else argv)]
    _require(actual==worker_argv(writer,args.seed,ref),'Noncanonical inference child command')
    return run_worker(writer,args.seed,ref,actual)


if __name__=='__main__':main()
