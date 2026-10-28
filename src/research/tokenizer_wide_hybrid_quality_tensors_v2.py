"""One bounded quality tensor/export audit within the active trial watchdog.

No CLI, training, recurrence replay, selection, cleanup or dispatch. Frozen plan,
selected request, input identity and full storage remain caller prerequisites.
"""
import time
from pathlib import Path

from .protocol import digest,file_digest
from .tokenizer_wide_hybrid_campaign_resources import CampaignQualityResourceRecorder
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_plan_v2 import request_options
from .tokenizer_wide_hybrid_retention_audit import RESOURCE_LIMITS, reject_output_failures
from .tokenizer_wide_hybrid_retention_receipts import _path,_sha,_document
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_tensors import review_values

KIND='wide_hybrid_quality_tensor_producer_v2'


def attempt_contract(slot,plan_sha,request_sha,result_sha,projection_ref,contract_sha,token_sha):
    return dict(kind=KIND,status='started',slot=slot,plan_sha256=plan_sha,
        request_sha256=request_sha,result_sha256=result_sha,projection_ref=projection_ref,
        training_contract_sha256=contract_sha,training_token_sha256=token_sha,
        resource_kind='wide_hybrid_trial_resources_v2',resource_limits=RESOURCE_LIMITS)


def review_and_publish(writer,slot,binding,*,projection_ref,resources,training_token_sha256):
    """Map one locally produced raw/H200 pair under the same7200s watchdog."""
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and 0<=slot<24,
             'Exact quality writer/slot required')
    root=writer.root/'evidence';stem=f'quality/{slot:02d}';trial=_path(root,stem)
    _require(set(binding)=={'request','request_sha256','result_sha256','training_contract'},
             'Exact quality tensor binding required')
    rqsha=_sha(binding['request_sha256']);rsha=_sha(binding['result_sha256']);_sha(training_token_sha256)
    _require(sum(digest(binding['request'])==digest(r) for r in request_options(writer.root)['quality'][str(slot)])==1,
             'Tensor audit request differs from frozen slot options')
    _require(isinstance(projection_ref,dict) and set(projection_ref)=={'path','sha256'}
             and projection_ref['path'].startswith('meta/'),'Bound campaign projection required')
    def active():
        _require(isinstance(resources,CampaignQualityResourceRecorder) and resources.writer is writer
                 and resources.trial==trial and resources.request_sha==rqsha
                 and resources.started is not None and resources.thread is not None
                 and resources.thread.is_alive() and not resources.stop.is_set()
                 and resources.result_sha is None and resources.LIMITS==RESOURCE_LIMITS
                 and 0<=time.monotonic()-resources.started<7200,
                 'Active matching quality resource interval required')
    active();_sha(resources.plan_sha)
    paths=('request.json','result.json','training-outputs.json','evaluation-attempt.json')
    refs={name:dict(path=stem+'/'+name,sha256=file_digest(trial/name)) for name in paths}
    _require(refs['request.json']['sha256']==rqsha and refs['result.json']['sha256']==rsha,
             'Tensor audit request/result changed')
    docs={name:_document(root,ref) for name,ref in refs.items()}
    _require(digest(docs['request.json'])==digest(binding['request']),'Tensor audit request changed')
    training=docs['training-outputs.json']
    _require(training.get('kind')=='wide_hybrid_training_outputs_v2' and training.get('status')=='completed'
             and training.get('config_sha256')==digest(binding['request']['config'])
             and training.get('numerical_audits_passed') is False and training.get('dispatch_authorized') is False,
             'Completed bounded training outputs required')
    evaluation=docs['evaluation-attempt.json'];contract_sha=digest(binding['training_contract'])
    _require(evaluation.get('kind')=='wide_hybrid_evaluation_v2' and evaluation.get('status')=='started'
             and evaluation.get('request_sha256')==rqsha
             and evaluation.get('training_contract_sha256')==contract_sha
             and evaluation.get('training_token_sha256')==training_token_sha256
             and evaluation.get('training_outputs_sha256')==refs['training-outputs.json']['sha256'],
             'Tensor audit evaluation lineage differs')
    files={str(root/ref['path']):ref['sha256'] for ref in refs.values()}
    files[str(_path(root,projection_ref['path']))]=projection_ref['sha256']
    _document(root,projection_ref)
    pairs={};inodes=set()
    for role in ('raw','h200'):
        score=docs['result.json'][role];p=Path(score['checkpoint'])
        _require(p.is_absolute() and p==p.resolve() and p.parent==trial/'checkpoints' and p.suffix=='.pt',
                 'Tensor audit checkpoint outside slot')
        stat=p.stat();inode=(stat.st_dev,stat.st_ino)
        _require(0<stat.st_size<=2*1024**3 and stat.st_nlink==1 and inode not in inodes,
                 'Distinct regular raw/H200 files required')
        inodes.add(inode);pairs[role]=dict(path=str(p),size=stat.st_size,sha256=_sha(score['checkpoint_sha256']))
        _require(digest(docs['training-outputs.json'].get(role))==digest(pairs[role]),
                 'Tensor audit checkpoint receipt differs')
        files[str(p)]=score['checkpoint_sha256']
    names=[stem+'/'+n for n in ('tensor-attempt.json','tensor-audit.json','tensor-failure.json')]
    def recheck():
        active()
        with writer._serialized():
            _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),
                     'Campaign STOP prevents tensor audit')
            snapshot(writer.root);reject_output_failures(trial)
        # Large checkpoint hashing must not hold the metadata namespace lock:
        # the independent resource thread still needs to publish observations.
        _require(all(file_digest(Path(p))==sha for p,sha in files.items()),'Tensor audit inputs changed')
    recheck()
    with writer._serialized():
        _require(not any(_path(root,name+s).exists() for name in names for s in ('','.tmp')),
                 'Existing tensor audit attempt cannot be retried')
    attempt=attempt_contract(slot,resources.plan_sha,rqsha,rsha,projection_ref,contract_sha,training_token_sha256)
    marker=writer.publish_json('evidence/'+names[0],attempt)
    marker_ref=dict(path=names[0],sha256=file_digest(marker))
    try:
        report=review_values(trial,binding['request'],rqsha,rsha,binding['training_contract'],kind='quality')
        recheck()
        _require(digest(_document(root,marker_ref))==digest(attempt),'Tensor attempt changed')
        report['producer']=dict(kind=KIND,attempt=marker_ref,files=files,
                               numerical_audits_passed=False,dispatch_authorized=False)
        path=writer.publish_json('evidence/'+names[1],report)
        return dict(path=names[1],sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+names[2],dict(kind=KIND,status='failed',attempt=marker_ref,
            error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Tensor failure publication failed: '+repr(error)[:1000])
        raise


def verify_producer(root,stem,slot,report,*,evidence,context,projection_ref,training_token_sha256):
    """Read-only provenance check; full independent stage audit remains required."""
    root=Path(root);producer=report.get('producer',{})
    _require(isinstance(report.get('files'),dict),'Tensor producer file bindings required')
    required=('request.json','result.json','training-outputs.json','evaluation-attempt.json')
    _require(all(stem+'/'+n in evidence for n in required),'Tensor producer input not bound')
    name=stem+'/tensor-attempt.json';_require(name in evidence,'Tensor attempt not bound')
    ref=dict(path=name,sha256=evidence[name]);attempt=_document(root,ref)
    ep=stem+'/evaluation-attempt.json';_require(ep in evidence,'Evaluation attempt not bound')
    evaluation=_document(root,dict(path=ep,sha256=evidence[ep]))
    expected=attempt_contract(slot,context['plan_sha256'],evidence[stem+'/request.json'],
        evidence[stem+'/result.json'],projection_ref,evaluation['training_contract_sha256'],training_token_sha256)
    files={str(root/(stem+'/'+n)):evidence[stem+'/'+n] for n in required}
    files[str(_path(root,projection_ref['path']))]=projection_ref['sha256']
    files.update({p:sha for p,sha in report['files'].items() if Path(p).suffix=='.pt'})
    _require(digest(attempt)==digest(expected) and digest(producer)==digest(dict(kind=KIND,attempt=ref,
        files=files,numerical_audits_passed=False,dispatch_authorized=False)),
        'Tensor producer provenance differs')
    for path,sha in files.items():_require(file_digest(Path(path))==sha,'Tensor producer input changed')
