"""Bind v2 quality producer lineage before full LR/rank stage acceptance.

Read-only; the caller still supplies frozen selected requests, complete immutable
inventory and context. This neither executes tensor audits nor grants cleanup,
launch, storage credit or fresh-confirmation acceptance.
"""
from pathlib import Path

from .protocol import file_digest
from .tokenizer_wide_hybrid_retention_audit import bind_quality_boundary, _require
from .tokenizer_wide_hybrid_pilot_audit_v2 import _worker_review
from .tokenizer_wide_hybrid_retention import allocations
from .tokenizer_wide_hybrid_retention_receipts import _document, _sha


def bind_quality_stage(root, *, training_token_sha256, **kwargs):
    """Require producer lineage in addition to every existing stage audit."""
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
        _require(result['training'].get('final_checkpoint')==str(root/pair['raw']['path']),
                 'Quality raw training endpoint differs')
        corpora.add(Path(request['config']['data']['path']).resolve())
        _worker_review(root,stem,index,context,request,result,pair,evidence,
                       kwargs['projection_ref'],training_token_sha256,
                       budgets[request['spec']['family'],request['spec']['divisor']],stage='quality')
    report=bind_quality_boundary(root,**kwargs)
    for corpus in corpora:
        _require(file_digest(corpus)==context['corpus_sha256'],'Quality corpus changed')
    return dict(report,kind='wide_hybrid_quality_stage_v2',producer_provenance_bound=True)
