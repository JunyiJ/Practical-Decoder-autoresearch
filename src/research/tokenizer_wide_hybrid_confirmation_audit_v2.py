"""Audit selected screen anchors and a fresh confirmation prefix, without adoption.

Independently frozen requests and original rank history are caller inputs. This
component checks live producer/tensor/resource/validation evidence; it cannot
validate deletion history, parent transport, whole-campaign storage or dispatch.
"""
from pathlib import Path

from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_retention_receipts import _document,_path,_sha
from .tokenizer_wide_hybrid_retention_audit import (
    _require,numerical_review,_review_quality_members,_check_quality_inventory)
from .tokenizer_wide_hybrid_quality_audit_v2 import _bind_producers
from .tokenizer_wide_hybrid_retention import allocations
from . import tokenizer_wide_hybrid_rules as rules


def bind_confirmation_prefix(root,*,context,screen_records,rates,selection,screen_requests,confirmation_requests,
        evidence,numerical_ref,rates_ref,selection_ref,projection_ref,training_token_sha256,pending_request=None):
    root=Path(root).resolve();_sha(training_token_sha256)
    _require(set(context)=={'plan_sha256','source_sha256','runtime_sha256','corpus_sha256'},'Independent frozen provenance required')
    for sha in context.values():_sha(sha)
    _require(isinstance(screen_records,list) and len(screen_records)==15
        and isinstance(screen_requests,list) and len(screen_requests)==15
        and isinstance(confirmation_requests,list) and 0<=len(confirmation_requests)<=9,'Complete screening and bounded fresh prefix required')
    specs=rules.lr_specs()+rules.rank_specs(rates)
    for i,(record,request,spec) in enumerate(zip(screen_records,screen_requests,specs)):
        rules._checked([record],[spec])
        _require(digest(request.get('spec'))==digest(spec) and request.get('result_path')==str(root/f'quality/{i:02d}/result.json'),
                 'Independently frozen screening request/order differs')
    _require(digest(rules.choose_rates(screen_records[:9]))==digest(rates)==digest(_document(root,rates_ref))
        and digest(rules.choose_ranks(screen_records[:9],screen_records[9:]))==digest(selection)==digest(_document(root,selection_ref)),
        'Persisted screening choices differ')
    count=len(confirmation_requests);specs=rules.confirmation_specs(selection)
    expected={f'{i:02d}' for i in range(15+count)}
    if pending_request is not None:
        i=15+count;stem=f'quality/{i:02d}';name=stem+'/request.json';trial=_path(root,stem)
        _require(count<9 and isinstance(pending_request,dict),'Pending request must be another fresh confirmation')
        _require(digest(pending_request.get('spec'))==digest(specs[count])
            and pending_request.get('result_path')==str(root/stem/'result.json'),'Pending frozen fresh request/order differs')
        _require(trial.is_dir() and {p.name for p in trial.iterdir()}=={'request.json'} and name in evidence
            and digest(_document(root,dict(path=name,sha256=evidence[name])))==digest(pending_request),
            'Only exact pending fresh request may follow completed prefix')
        expected.add(f'{i:02d}')
    quality=_path(root,'quality')
    _require(quality.is_dir() and {p.name for p in quality.iterdir()}==expected
        and all(p.is_dir() and not p.is_symlink() for p in quality.iterdir()),'Quality directory must equal screening history plus fresh prefix')
    bound=dict(evidence)
    for ref in (numerical_ref,rates_ref,selection_ref,projection_ref):
        _require(ref['path'] not in bound or bound[ref['path']]==ref['sha256'],'Conflicting confirmation evidence hash')
        bound[ref['path']]=ref['sha256']
    for n,sha in numerical_review(root,numerical_ref,context).items():
        _require(n not in bound or bound[n]==sha,'Conflicting numerical evidence hash');bound[n]=sha
    projection=_document(root,projection_ref);budgets=allocations(projection)
    anchors=[i for i,r in enumerate(screen_records) if r['lr']==selection[r['family']]['lr'] and r['divisor']==selection[r['family']]['divisor']]
    _require(len(anchors)==3 and {screen_records[i]['family'] for i in anchors}==set(rules.FAMILIES),'Exactly three final selected screens required')
    trials={i:f'quality/{i:02d}' for i in anchors};requests={i:screen_requests[i] for i in anchors};records=list(screen_records)
    for offset,(request,spec) in enumerate(zip(confirmation_requests,specs)):
        i=15+offset;stem=f'quality/{i:02d}';trials[i]=stem;requests[i]=request
        _require(digest(request.get('spec'))==digest(spec) and request.get('result_path')==str(root/stem/'result.json'),
                 'Independently frozen fresh request/order differs')
        name=stem+'/result.json';_require(name in bound,'Fresh prefix result is not bound')
        result=_document(root,dict(path=name,sha256=bound[name]))
        row=dict(spec,status=result.get('status'),ce=result.get('h200',{}).get('mean_ce'))
        rules._checked([row],[spec]);records.append(row)
    corpora=_bind_producers(root,training_token_sha256=training_token_sha256,context=context,requests=requests,
        trials=trials,evidence=bound,projection_ref=projection_ref)
    pairs=_review_quality_members(root,records,requests,trials,bound,context,projection,budgets)
    _check_quality_inventory(root,[root/n for n in trials.values()],pairs,bound)
    for corpus in corpora:_require(file_digest(corpus)==context['corpus_sha256'],'Confirmation corpus changed')
    return dict(kind='wide_hybrid_confirmation_prefix_v2',records=records,rates=rates,selection=selection,
        checkpoint_pairs={f'quality/{i:02d}':pair for i,pair in pairs.items()},evidence=bound,audited_indices=sorted(trials),
        historical_screen_indices=sorted(set(range(15))-set(anchors)),producer_provenance_bound=True,
        prefix_audits_bound=True,stage_inventory_complete=True,rank_history_verified=False,stage_audits_bound=False,
        inventory_completeness_verified=False,frozen_plan_verified=False,storage_verified=False,
        adopted=False,dispatch_authorized=False,deletion_authorized=False,manifest=None)
