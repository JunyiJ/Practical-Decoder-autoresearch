"""Bind v2 quality producer lineage before full LR/rank stage acceptance.

Read-only; the caller still supplies frozen selected requests, complete immutable
inventory and context. This neither executes tensor audits nor grants cleanup,
launch, storage credit or fresh-confirmation acceptance.
"""
from pathlib import Path

from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_quality_tensors_v2 import verify_producer
from .tokenizer_wide_hybrid_retention_audit import (bind_quality_boundary, _require,
    numerical_review,_review_quality_members,_check_quality_inventory)
from .tokenizer_wide_hybrid_pilot_audit_v2 import _worker_review
from .tokenizer_wide_hybrid_retention import allocations
from .tokenizer_wide_hybrid_retention_receipts import _document, _sha, _path
from . import tokenizer_wide_hybrid_rules as rules

QUALITY_ARTIFACTS=frozenset(('request.json','result.json','worker-attempt.json','worker-outputs.json',
    'training-attempt.json','training-outputs.json','training-summary.json','metrics.jsonl',
    'evaluation-attempt.json','tensor-attempt.json','tensor-audit.json','resource-audit.json',
    'resource-events.jsonl','process/launch.json','process/process.json','process/worker.log'))


def _bind_producers(root, *, training_token_sha256, **kwargs):
    """Bind producer lineage without implying complete stage selection."""
    root=Path(root).resolve();_sha(training_token_sha256)
    evidence=kwargs['evidence'];context=kwargs['context']
    budgets=allocations(_document(root,kwargs['projection_ref']))
    inodes=set();corpora=set()
    for index,stem in kwargs['trials'].items():
        _require(type(index) is int and stem==f'quality/{index:02d}',
                 'Canonical v2 quality slot required')
        def load(name):
            path=stem+'/'+name
            _require(path in evidence,'Quality producer artifact is not bound')
            return _document(root,dict(path=path,sha256=evidence[path]))
        request=load('request.json');result=load('result.json')
        # Worker completion binds this ref before the full resource audit below.
        load('resource-audit.json')
        verify_producer(root,stem,index,load('tensor-audit.json'),evidence=evidence,context=context,
                        projection_ref=kwargs['projection_ref'],training_token_sha256=training_token_sha256)
        pair={}
        for role in ('raw','h200'):
            score=result[role];path=Path(score['checkpoint'])
            _require(path.is_absolute() and path==path.resolve()
                     and path.parent==root/stem/'checkpoints' and path.suffix=='.pt',
                     'Quality checkpoint outside declared slot')
            stat=path.stat();inode=(stat.st_dev,stat.st_ino)
            _require(inode not in inodes and stat.st_nlink==1 and 0<stat.st_size<=2*1024**3,
                     'Quality checkpoint alias or invalid size')
            inodes.add(inode)
            pair[role]=dict(path=str(path.relative_to(root)),size=stat.st_size,
                            sha256=score['checkpoint_sha256'])
        allowed={stem+'/'+name for name in QUALITY_ARTIFACTS}|{entry['path'] for entry in pair.values()}
        for path in (root/stem).rglob('*'):
            _require(not path.is_symlink(),'Quality output symlink refused')
            if path.is_dir():
                _require(path in (root/stem/'checkpoints',root/stem/'process'),
                         'Undeclared quality output directory')
            else:
                _require(path.is_file() and str(path.relative_to(root)) in allowed,
                         'Undeclared quality output artifact')
        _require(result['training'].get('final_checkpoint')==str(root/pair['raw']['path']),
                 'Quality raw training endpoint differs')
        corpora.add(Path(request['config']['data']['path']).resolve())
        _worker_review(root,stem,index,context,request,result,pair,evidence,
                       kwargs['projection_ref'],training_token_sha256,
                       budgets[request['spec']['family'],request['spec']['divisor']],stage='quality')
    return corpora


def bind_quality_stage(root, *, training_token_sha256, **kwargs):
    """Require producer lineage in addition to every existing stage audit."""
    corpora=_bind_producers(root,training_token_sha256=training_token_sha256,**kwargs)
    report=bind_quality_boundary(root,**kwargs)
    context=kwargs['context']
    for corpus in corpora:
        _require(file_digest(corpus)==context['corpus_sha256'],'Quality corpus changed')
    return dict(report,kind='wide_hybrid_quality_stage_v2',producer_provenance_bound=True)



def bind_lr_prefix(root,*,context,requests,evidence,numerical_ref,projection_ref,training_token_sha256,
                   pending_request=None):
    """Review every member of a nonempty LR prefix before its next trial.

    The caller supplies independent frozen requests and complete immutable
    evidence, holds worker locks and an outer resource guard. Pilot history,
    frozen-plan readiness, storage and process handoff remain separate gates.
    No rate selection or cleanup manifest is produced, even after all nine screens.
    Optional pending_request permits only the exact next frozen request file;
    it grants that slot no completed-record, checkpoint or process credit.
    """
    root=Path(root).resolve();_sha(training_token_sha256)
    _require(isinstance(requests,list) and 1<=len(requests)<=9,'Nonempty ordered LR prefix required')
    _require(set(context)=={'plan_sha256','source_sha256','runtime_sha256','corpus_sha256'},
             'Independent frozen provenance required')
    for sha in context.values():_sha(sha)
    quality=_path(root,'quality');expected={f'{i:02d}' for i in range(len(requests))}
    if pending_request is not None:
        i=len(requests)
        _require(i<9 and isinstance(pending_request,dict),'Pending request must be another LR screen')
        stem=f'quality/{i:02d}';name=stem+'/request.json';trial=_path(root,stem)
        _require(digest(pending_request.get('spec'))==digest(rules.lr_specs()[i])
                 and pending_request.get('result_path')==str(root/stem/'result.json'),
                 'Pending frozen LR request/order differs')
        _require(trial.is_dir() and {p.name for p in trial.iterdir()}=={'request.json'}
                 and name in evidence and digest(_document(root,dict(path=name,sha256=evidence[name])))==digest(pending_request),
                 'Only the exact pending LR request may follow the completed prefix')
        expected.add(f'{i:02d}')
    _require(quality.is_dir() and {p.name for p in quality.iterdir()}==expected
             and all(p.is_dir() and not p.is_symlink() for p in quality.iterdir()),
             'Quality directory must equal the completed LR prefix')
    bound=dict(evidence)
    for name,sha in {**numerical_review(root,numerical_ref,context),projection_ref['path']:projection_ref['sha256']}.items():
        _require(name not in bound or bound[name]==sha,'Conflicting prefix evidence hash')
        bound[name]=sha
    projection=_document(root,projection_ref);budgets=allocations(projection)
    records=[];trials={};expected_requests={}
    for i,(request,spec) in enumerate(zip(requests,rules.lr_specs())):
        stem=f'quality/{i:02d}';trials[i]=stem;expected_requests[i]=request
        _require(digest(request.get('spec'))==digest(spec) and request.get('result_path')==str(root/stem/'result.json'),
                 'LR prefix request/order differs')
        name=stem+'/result.json';_require(name in bound,'Completed prefix result not bound')
        result=_document(root,dict(path=name,sha256=bound[name]))
        row=dict(spec,status=result.get('status'),ce=result.get('h200',{}).get('mean_ce'))
        rules._checked([row],[spec]);records.append(row)
    corpora=_bind_producers(root,training_token_sha256=training_token_sha256,context=context,
        requests=expected_requests,trials=trials,evidence=bound,projection_ref=projection_ref)
    pairs=_review_quality_members(root,records,expected_requests,trials,bound,context,projection,budgets)
    _check_quality_inventory(root,[root/name for name in trials.values()],pairs,bound)
    for corpus in corpora:_require(file_digest(corpus)==context['corpus_sha256'],'LR prefix corpus changed')
    return dict(kind='wide_hybrid_lr_prefix_v2',records=records,
        checkpoint_pairs={f'quality/{i:02d}':pair for i,pair in pairs.items()},evidence=bound,
        producer_provenance_bound=True,prefix_audits_bound=True,stage_inventory_complete=True,
        stage_audits_bound=False,inventory_completeness_verified=False,frozen_plan_verified=False,
        storage_verified=False,dispatch_authorized=False,deletion_authorized=False,selection=None,manifest=None)


def bind_rank_prefix(root,*,context,lr_records,rates,lr_requests,rank_requests,evidence,
                     numerical_ref,rates_ref,projection_ref,training_token_sha256,pending_request=None):
    """Audit live LR anchors and zero through six completed rank members.

    Caller supplies independently frozen requests, historical LR records and
    complete immutable metadata under worker locks and an outer resource guard.
    This re-audits every surviving anchor but cannot establish the deleted LR
    losers' history. The controller must also call verify_lr_history with the
    complete current inventory, bind records/rates/pairs to that history, and
    verify full remaining storage before any preparation or dispatch. No final
    rank choice or cleanup credit is granted here, even for all six members.
    """
    root=Path(root).resolve();_sha(training_token_sha256)
    _require(set(context)=={'plan_sha256','source_sha256','runtime_sha256','corpus_sha256'},
             'Independent frozen provenance required')
    for sha in context.values():_sha(sha)
    _require(isinstance(lr_records,list) and len(lr_records)==9
             and isinstance(lr_requests,list) and len(lr_requests)==9
             and isinstance(rank_requests,list) and 0<=len(rank_requests)<=6,
             'Complete LR history and bounded rank prefix required')
    rules._rates(rates)
    for i,(record,request,spec) in enumerate(zip(lr_records,lr_requests,rules.lr_specs())):
        rules._checked([record],[spec])
        _require(digest(request.get('spec'))==digest(spec)
                 and request.get('result_path')==str(root/f'quality/{i:02d}/result.json'),
                 'Independently frozen ordered LR requests required')
    _require(digest(rules.choose_rates(lr_records))==digest(rates)
             and digest(_document(root,rates_ref))==digest(rates),'Persisted LR selection differs')
    count=len(rank_requests);specs=rules.rank_specs(rates)
    expected={f'{i:02d}' for i in range(9+count)}
    if pending_request is not None:
        i=9+count;stem=f'quality/{i:02d}';name=stem+'/request.json';trial=_path(root,stem)
        _require(count<6 and isinstance(pending_request,dict),'Pending request must be another rank screen')
        _require(digest(pending_request.get('spec'))==digest(specs[count])
                 and pending_request.get('result_path')==str(root/stem/'result.json'),
                 'Pending frozen rank request/order differs')
        _require(trial.is_dir() and {p.name for p in trial.iterdir()}=={'request.json'}
                 and name in evidence and digest(_document(root,dict(path=name,sha256=evidence[name])))==digest(pending_request),
                 'Only the exact pending rank request may follow the completed prefix')
        expected.add(f'{i:02d}')
    quality=_path(root,'quality')
    _require(quality.is_dir() and {p.name for p in quality.iterdir()}==expected
             and all(p.is_dir() and not p.is_symlink() for p in quality.iterdir()),
             'Quality directory must equal historical LR plus completed rank prefix')
    bound=dict(evidence)
    for ref in (numerical_ref,rates_ref,projection_ref):
        _require(ref['path'] not in bound or bound[ref['path']]==ref['sha256'],'Conflicting rank prefix evidence hash')
        bound[ref['path']]=ref['sha256']
    for name,sha in numerical_review(root,numerical_ref,context).items():
        _require(name not in bound or bound[name]==sha,'Conflicting numerical evidence hash')
        bound[name]=sha
    projection=_document(root,projection_ref);budgets=allocations(projection)
    anchors=[i for i,r in enumerate(lr_records) if r['lr']==rates[r['family']]]
    _require(len(anchors)==3,'Exactly three selected LR anchors required')
    trials={i:f'quality/{i:02d}' for i in anchors}
    requests={i:lr_requests[i] for i in anchors};records=list(lr_records)
    for offset,(request,spec) in enumerate(zip(rank_requests,specs)):
        i=9+offset;stem=f'quality/{i:02d}';trials[i]=stem;requests[i]=request
        _require(digest(request.get('spec'))==digest(spec)
                 and request.get('result_path')==str(root/stem/'result.json'),
                 'Independently frozen rank prefix request/order differs')
        name=stem+'/result.json';_require(name in bound,'Completed rank result not bound')
        result=_document(root,dict(path=name,sha256=bound[name]))
        row=dict(spec,status=result.get('status'),ce=result.get('h200',{}).get('mean_ce'))
        rules._checked([row],[spec]);records.append(row)
    corpora=_bind_producers(root,training_token_sha256=training_token_sha256,context=context,
        requests=requests,trials=trials,evidence=bound,projection_ref=projection_ref)
    pairs=_review_quality_members(root,records,requests,trials,bound,context,projection,budgets)
    _check_quality_inventory(root,[root/name for name in trials.values()],pairs,bound)
    for corpus in corpora:_require(file_digest(corpus)==context['corpus_sha256'],'Rank prefix corpus changed')
    return dict(kind='wide_hybrid_rank_prefix_v2',records=records,rates=rates,
        checkpoint_pairs={f'quality/{i:02d}':pair for i,pair in pairs.items()},evidence=bound,
        audited_indices=sorted(trials),historical_lr_indices=sorted(set(range(9))-set(anchors)),
        producer_provenance_bound=True,prefix_audits_bound=True,stage_inventory_complete=True,
        lr_history_verified=False,stage_audits_bound=False,inventory_completeness_verified=False,
        frozen_plan_verified=False,storage_verified=False,dispatch_authorized=False,
        deletion_authorized=False,selection=None,manifest=None)
