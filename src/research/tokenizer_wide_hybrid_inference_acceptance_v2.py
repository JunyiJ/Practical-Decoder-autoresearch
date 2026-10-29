"""Exact finite idle inference boundaries and independent transport acceptance.

Readiness, completed outputs and accepted receipts are the only admitted states.
Every public boundary review replays the complete retained quality history.
No live process, retries, training, cleanup or adoption are initiated here.
"""
import math
import os

from .protocol import digest,file_digest
from .tokenizer_serial_process import _group_exists
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_retention_receipts import _document
from .tokenizer_wide_hybrid_scores import _require
from . import tokenizer_wide_hybrid_inference_preparation_v2 as prep
from . import tokenizer_wide_hybrid_inference_outputs_v2 as outputs
from . import tokenizer_wide_hybrid_inference_entry_v2 as entry

KIND='wide_hybrid_inference_acceptance_v2'
COMPLETED={'ready.json','worker-attempt.json','result.json','models.json','worker-outputs.json',
           'process/launch.json','process/worker.log','process/process.json'}
STATES={'ready.json':'ready','worker-outputs.json':'completed','review.json':'accepted'}
QUALITY_KEYS=('plan_sha256','records_sha256','checkpoint_pairs_sha256','retention_sha256',
              'quality_inventory_sha256','quality_controls_sha256','triples_sha256')


def _ref(root,name):
    return dict(path=name,sha256=file_digest(root/name))


def _identity(ref):
    _require(type(ref) is dict and set(ref)=={'path','sha256'},'Exact inference boundary reference required')
    parts=ref['path'].split('/')
    _require(len(parts)==3 and parts[0]=='inference' and parts[1] in {str(s) for s in outputs.SEEDS}
        and parts[2] in STATES,'Only exact finite idle inference boundaries are supported')
    return int(parts[1]),STATES[parts[2]]


def _transport(writer,seed,proof):
    """Reconstruct historical argv/FDs and the self-inclusive terminal census."""
    root=writer.root/'evidence';stem=f'inference/{seed}'
    refs={k:_ref(root,stem+'/'+name) for k,name in (
        ('launch','process/launch.json'),('process','process/process.json'),('log','process/worker.log'))}
    terminal=_document(root,refs['process']);launch=_document(root,refs['launch'])
    argv=terminal.get('argv');expected=entry.worker_argv(writer,seed,proof['ready_ref'])
    _require(type(argv) is list and len(argv)==len(expected),'Exact historical inference command required')
    fds=[]
    for index in (11,13):
        value=argv[index]
        _require(type(value) is str and value.isascii() and value.isdecimal()
            and str(int(value))==value and int(value)>2,'Exact historical lock descriptors required')
        fds.append(int(value));expected[index]=value
    _require(fds[0]!=fds[1] and argv==expected,'Canonical historical command/lock identity differs')
    expected_launch=entry._launch(expected,proof['parent_pid'])
    _require(digest(launch)==digest(expected_launch),'Original inference launch differs')
    pid=terminal.get('pid')
    _require(type(pid) is int and pid>1 and pid!=proof['parent_pid'],'Distinct completed inference child required')
    fixed=dict(expected_launch,status='completed',pid=pid,returncode=0)
    dynamic={'observation'}
    _require(set(terminal)==set(fixed)|{'elapsed_seconds'}
        and digest({k:terminal[k] for k in fixed if k not in dynamic})==
            digest({k:v for k,v in fixed.items() if k not in dynamic}), 'Completed inference process contract differs')
    elapsed=terminal['elapsed_seconds'];sample=terminal['observation']
    census=snapshot(writer.root)
    # Acceptance is appended after process completion; it is not retroactive
    # child output and must not alter the saved self-inclusive byte census.
    total=sum(v['size'] for n,v in census['files'].items()
        if n.startswith('evidence/'+stem+'/') and n!='evidence/'+stem+'/review.json')
    log=(root/refs['log']['path']).stat().st_size
    _require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<600
        and type(sample) is dict and set(sample)=={'free_bytes','output_bytes','log_bytes'}
        and all(type(v) is int and v>=0 for v in sample.values())
        and sample['free_bytes']>=outputs.CONTRACT['reserve_bytes']
        and sample['output_bytes']==total<=outputs.LIMIT and sample['log_bytes']==log<=outputs.LIMIT,
        'Completed inference resource census differs')
    supervisor_ref=_ref(writer.root,f'control/process-inference-{seed}.json')
    supervisor=_document(writer.root,supervisor_ref)
    dynamic={'elapsed_seconds','observation'}
    _require(set(supervisor)==set(terminal) and digest({k:v for k,v in supervisor.items() if k not in dynamic})==
        digest({k:v for k,v in terminal.items() if k not in dynamic}), 'Completed inference supervisor provenance differs')
    elapsed=supervisor['elapsed_seconds'];sample=supervisor['observation']
    _require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<=terminal['elapsed_seconds']
        and type(sample) is dict and set(sample)=={'free_bytes','output_bytes','log_bytes'}
        and all(type(v) is int and v>=0 for v in sample.values())
        and sample['free_bytes']>=outputs.CONTRACT['reserve_bytes']
        and sample['output_bytes']==total-(root/refs['process']['path']).stat().st_size
        and sample['log_bytes']==log,'Completed inference supervisor observation differs')
    return dict(seed=seed,parent_pid=proof['parent_pid'],worker_pid=pid,lock_fds=fds,
        ready_ref=proof['ready_ref'],refs=refs,supervisor_ref=supervisor_ref)


def _receipt(request,proof,artifact,transport,previous):
    return dict(kind=KIND,status='inference_seed_reviewed',seed=proof['seed'],request_ref=proof['request_ref'],
        ready_ref=proof['ready_ref'],worker_ref=artifact['worker_ref'],process_ref=transport['refs']['process'],
        supervisor_ref=transport['supervisor_ref'],previous_refs=previous,entry_sha256=digest(proof),
        artifact_review_sha256=digest(artifact),transport_sha256=digest(transport),
        quality_binding={k:request[k] for k in QUALITY_KEYS},quality_history_reaudited=True,
        adopted=False,dispatch_authorized=False,next_seed_authorized=False)


def _scope(writer,reference,**expected):
    """Independently validate a specific idle state, never an arbitrary allowlist.

    This scopes subsequent quality replay; it does not itself confer science
    credit. Lower history consumers revalidate this exact scope independently.
    """
    seed,state=_identity(reference);outputs._budget(writer);root=writer.root/'evidence'
    _document(root,reference);before=snapshot(writer.root)
    _require(not any(n.endswith('.tmp') or 'failure' in n or n.startswith('emergency/')
        for n in before['files']), 'Failed or partial campaign cannot enter idle inference review')
    request_ref=_ref(root,prep.REQUEST);request=_document(root,request_ref)
    args={k:request[k] for k in ('plan_sha256','completion_ref','projection_ref','rank_completion_ref')}
    _require(set(expected)<=set(args) and all(digest(args[k])==digest(v) for k,v in expected.items()),
             'Inference scope differs from requested quality history')
    prep._request(writer,request_ref,**args);guard=prep._guard(writer,request['parent_pid'],finished=True)
    evidence_names={prep.REQUEST};control_names={prep.GUARD};prefix=outputs.SEEDS[:outputs.SEEDS.index(seed)+1]
    for current in prefix:
        mode=state if current==seed else 'accepted'
        names={'ready.json'} if mode=='ready' else COMPLETED|({'review.json'} if mode=='accepted' else set())
        evidence_names|={f'inference/{current}/{n}' for n in names}
        if mode!='ready':control_names|={f'control/inference-worker-{current}.json',f'control/process-inference-{current}.json'}
    inventory=prep.gate.history.acceptance._inventory(root)
    controls=prep.gate.history.acceptance.readiness._controls(writer,'not-a-control')
    _require({n for n in inventory if n.startswith('inference/')}==evidence_names
        and {n for n in controls if n.startswith(('control/inference-','control/process-inference-'))}==control_names,
        'Exact finite idle inference evidence/control prefix required')
    original={n:v for n,v in inventory.items() if n not in evidence_names}
    original_controls={n:v for n,v in controls.items() if n not in control_names}
    saved=_document(root,request['completion_ref'])['review']
    _require(digest(original)==request['quality_inventory_sha256']
        and digest(original_controls)==request['quality_controls_sha256']
        and digest({n:v for n,v in original.items() if n!=request['completion_ref']['path']})==saved['evidence_inventory_sha256'],
        'Original quality evidence or controls changed at inference boundary')
    previous=[];current_review=None
    for current in prefix:
        mode=state if current==seed else 'accepted';stem=f'inference/{current}'
        ready_ref=_ref(root,stem+'/ready.json');ready=_document(root,ready_ref)
        proof=entry._immutable(writer,ready);outputs._binding(writer,current,proof)
        _require(proof['request_ref']==request_ref and proof['ready_ref']==ready_ref,'Shared request/readiness lineage differs')
        artifact=transport=None
        if mode!='ready':
            artifact=outputs.review_bound_triple(writer,current,proof,_ref(root,stem+'/worker-outputs.json'))
            transport=_transport(writer,current,proof)
            _require(artifact['worker_pid']==transport['worker_pid'] and artifact['parent_pid']==transport['parent_pid'],
                     'Output and process child identities differ')
            receipt=_receipt(request,proof,artifact,transport,list(previous))
            if mode=='accepted':
                review_ref=_ref(root,stem+'/review.json')
                _require(digest(_document(root,review_ref))==digest(receipt),'Independent accepted inference receipt differs')
                previous.append(review_ref)
            current_review=receipt
        else:current_review=None
    outputs._budget(writer)
    _require(snapshot(writer.root)==before,'Idle inference census changed during review')
    return dict(request=request,request_ref=request_ref,guard=guard,seed=seed,state=state,
        evidence={n:inventory[n] for n in evidence_names},controls={n:controls[n] for n in control_names},
        proof=proof,artifact=artifact,transport=transport,receipt=current_review,previous_refs=previous)


def verify_boundary(writer,reference):
    """Read-only full scientific and transport replay at a stable idle boundary."""
    scope=_scope(writer,reference);request=scope['request']
    args={k:request[k] for k in ('plan_sha256','completion_ref','projection_ref','rank_completion_ref')}
    bound=prep.gate._bind(writer,**args,inference_request_ref=reference)
    _require(digest(request)==digest(prep._payload(bound,args['projection_ref'],args['rank_completion_ref'],request['parent_pid'])),
             'Full quality replay differs from original inference request')
    again=_scope(writer,reference)
    _require(digest(again)==digest(scope),'Idle inference binding changed during full quality replay')
    return dict(bound=bound,scope=scope,quality_history_reaudited=True,dispatch_authorized=False,adopted=False)


def accept_completed(writer,worker_ref,process):
    """Accept one reaped same-parent child only after full independent replay."""
    seed,state=_identity(worker_ref)
    _require(state=='completed','Exact unaccepted worker outputs required')
    root=writer.root/'evidence';ready=_document(root,_ref(root,f'inference/{seed}/ready.json'))
    _require(type(ready['parent_pid']) is int and ready['parent_pid']==os.getpid(),'Current parent must accept its own child')
    checked=verify_boundary(writer,worker_ref);scope=checked['scope'];transport=scope['transport']
    _require(digest(_document(root,transport['refs']['process']))==digest(process)
        and not _group_exists(transport['worker_pid']), 'Returned process differs or owned process group still exists')
    expected=entry.worker_argv(writer,seed,scope['proof']['ready_ref'])
    _require(process['argv']==expected,'Current parent descriptor identity differs')
    outputs.recheck(checked['bound']['files']);outputs._budget(writer)
    path=writer.publish_json(f'evidence/inference/{seed}/review.json',scope['receipt'])
    ref=dict(path=f'inference/{seed}/review.json',sha256=file_digest(path))
    outputs.recheck(checked['bound']['files']);_scope(writer,ref)
    return ref


def prepare_next_readiness(writer,previous_ref):
    """Full prior acceptance replay, then one exclusive next-seed readiness."""
    seed,state=_identity(previous_ref)
    _require(state=='accepted' and seed in outputs.SEEDS[:-1],'Only a nonfinal accepted seed has a successor')
    checked=verify_boundary(writer,previous_ref);bound=checked['bound'];seed=outputs.SEEDS[outputs.SEEDS.index(seed)+1]
    request_ref=checked['scope']['request_ref']
    ready=dict(kind=outputs.READY_KIND,seed=seed,parent_pid=os.getpid(),request_ref=request_ref,
        bindings_sha256=digest(bound['triples'][seed]),protected_files_sha256=digest(bound['files']),contract=bound['contract'])
    outputs.recheck(bound['files']);outputs._budget(writer)
    path=writer.publish_json(f'evidence/inference/{seed}/ready.json',ready)
    ref=dict(path=f'inference/{seed}/ready.json',sha256=file_digest(path))
    outputs.recheck(bound['files']);_scope(writer,ref)
    return ref
