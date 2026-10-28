"""Bounded v2 cache-audit producer using the frozen campaign projection copy.

No launcher: caller must bind the full frozen plan, numerical prerequisites,
input identity, storage and inherited locks, inside the600s pilot resource guard.
"""
from pathlib import Path

from .protocol import digest, file_digest
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_retention import allocations
from .tokenizer_wide_hybrid_retention_receipts import _path, _sha, _document
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_pilot_config import specs
from . import tokenizer_wide_hybrid_pilot_cache as cache

KIND='wide_hybrid_pilot_cache_v2'


def attempt_contract(slot,request_sha256,result_sha256,projection_ref,tokenizer,token_sha256):
    return dict(kind=KIND,status='started',slot=slot,request_sha256=request_sha256,
        result_sha256=result_sha256,projection_ref=projection_ref,
        tokenizer_sha256=digest(tokenizer),training_token_sha256=token_sha256,
        cache_contract_sha256=digest(cache.CACHE_CONTRACT))


def review_and_publish(writer,slot,binding,expected_tokenizer,expected_token_sha256,*,projection_ref):
    """Exactly one raw/H200 audit attempt; preserve failure/partial evidence.

    Returns an audit reference for resource completion. It does not authorize
    cleanup/quality dispatch or claim the caller's prerequisite gates were run.
    """
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and 0<=slot<3,
             'Bounded campaign writer and exact pilot slot required')
    stem=f'pilots/{slot:02d}';evidence=writer.root/'evidence';trial=_path(evidence,stem)
    _require(set(binding)=={'trial','request','request_sha256','result_sha256','training_contract'}
             and binding['trial']==str(trial),'Pilot binding must name exact campaign slot')
    _require(digest(binding['request']['spec'])==digest(specs()[slot]),'Pilot family/order differs')
    _sha(binding['request_sha256']);_sha(binding['result_sha256']);_sha(expected_token_sha256)
    _require(isinstance(expected_tokenizer,dict) and bool(expected_tokenizer),'Frozen tokenizer required')
    _require(isinstance(projection_ref,dict) and set(projection_ref)=={'path','sha256'}
             and type(projection_ref['path']) is str and projection_ref['path'].startswith('meta/'),
             'Frozen projection reference must be in immutable campaign metadata')
    names=[stem+'/'+name for name in ('audit-attempt.json','audit.json','audit-failure.json')]
    def recheck():
        with writer._serialized():
            _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),
                     'Campaign STOP prevents cache audit')
            snapshot(writer.root)
            return _document(evidence,projection_ref)
    projection=recheck();allocations(projection)
    row=next(r for r in projection['records'] if r['width']==512 and
             r['family']==specs()[slot]['family'] and r['bottleneck_divisor']==2)
    with writer._serialized():
        _require(not any(_path(evidence,name+suffix).exists() for name in names for suffix in ('','.tmp')),
                 'Existing cache audit attempt cannot be retried')
    attempt=attempt_contract(slot,binding['request_sha256'],binding['result_sha256'],
                             projection_ref,expected_tokenizer,expected_token_sha256)
    marker=writer.publish_json('evidence/'+names[0],attempt)
    marker_ref=dict(path=names[0],sha256=file_digest(marker))
    try:
        def check(*args):
            recheck()
            result=cache._check_model(*args,row)
            recheck()
            return result
        report=cache._review_caches(binding,expected_tokenizer,expected_token_sha256,
            projection_path=_path(evidence,projection_ref['path']),
            projection_sha256=projection_ref['sha256'],model_check=check)
        recheck()
        _require(_document(evidence,marker_ref)==attempt,'Cache audit attempt changed')
        report.update(kind=KIND,attempt=marker_ref)
        path=writer.publish_json('evidence/'+names[1],report)
        return dict(path=names[1],sha256=file_digest(path))
    except BaseException as exc:
        try:
            writer.publish_json('evidence/'+names[2],dict(kind=KIND,status='failed',
                attempt=marker_ref,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as report_error:
            exc.add_note('Cache failure evidence publication failed: '+repr(report_error)[:1000])
        raise
