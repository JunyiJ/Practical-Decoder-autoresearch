"""Reopen a hash-pinned disposable completed prefix; never skip history audits.

Only tiny synthetic /tmp fixtures are eligible. This reinstalls the original
fixture's frozen-config/corpus mocks under new real lock descriptors. It neither
recreates trajectories nor mocks transport, retention, scientific receipt or
capacity verification. Caller must hold an independent bounded resource guard
whose disk observer was captured before this fixture patches shutil.
"""
from contextlib import contextmanager
import copy
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from src.research.protocol import digest,file_digest
from src.research import tokenizer_wide_hybrid_inference_preparation_v2 as prep
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_plan_v2 import request_options


def inventory(root):
    return {str(p.relative_to(root)):file_digest(p) for p in root.rglob('*') if p.is_file()}


@contextmanager
def reopen_prefix(binding,*,allow_preparation=False,allow_first_readiness=False,allow_inference=False):
    root=Path(binding['root'])
    assert binding['kind']=='wide_hybrid_reusable_synthetic_prefix_v1'
    assert root==root.resolve() and root.is_relative_to(Path('/tmp').resolve())
    assert not any(p.is_symlink() for p in root.rglob('*'))
    assert inventory(root)==binding['files']
    evidence=root/'evidence'
    assert (evidence/'meta/source.txt').read_bytes()==b'synthetic source identity'
    assert (evidence/'meta/corpus.txt').read_bytes()==b'synthetic corpus, no actual training data'
    assert sum(p.stat().st_size for p in evidence.rglob('*.pt'))<1024**2
    plan=json.loads((evidence/'meta/plan.json').read_text())
    numeric=json.loads((evidence/plan['numerical_ref']['path']).read_text())
    pilot=json.loads((evidence/'pilots/00/worker-attempt.json').read_text())
    context={k:numeric[k] for k in ('source_sha256','runtime_sha256','corpus_sha256')}
    context['plan_sha256']=pilot['plan_sha256']
    options=request_options(root)
    for request in options['pilots']+[r for values in options['quality'].values() for r in values]:
        request['config']=dict(fixture_only=True,data=dict(path=str(evidence/'meta/corpus.txt')))
    frozen={str(evidence/name):binding['files']['evidence/'+name] for name in
            ('meta/plan.json','meta/projection.json','meta/numerical.json','meta/source.txt')}
    bound=dict(context=context,requests=options,files=frozen)
    def campaign(path,sha):
        assert path==evidence/'meta/plan.json' and sha==frozen[str(path)]
        if any(file_digest(Path(name))!=value for name,value in frozen.items()):
            raise ValueError('Synthetic frozen source changed')
        return copy.deepcopy(bound)
    global_path=root.parent/'global.lock'
    assert global_path.is_file() and not global_path.is_symlink()
    with global_path.open('rb') as g,(root/'.lock').open('rb') as c,pytest.MonkeyPatch.context() as patch:
        fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(c,fcntl.LOCK_EX|fcntl.LOCK_NB)
        patch.setattr(cleanup,'GLOBAL_LOCK',global_path)
        # Reinstall only the original fixture environment, not audit functions.
        for name,module in list(sys.modules.items()):
            if not name.startswith('src.research.tokenizer_wide_hybrid_'):continue
            if hasattr(module,'bind_campaign'):patch.setattr(module,'bind_campaign',campaign)
            if hasattr(module,'PROJECTION'):patch.setattr(module,'PROJECTION',evidence/'meta/projection.json')
            if hasattr(module,'shutil'):
                patch.setattr(module,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
        patch.setattr(prep.gate,'_contract',lambda config:dict(tokenizer=plan['tokenizer']))
        writer=MetadataWriter(root,(g.fileno(),c.fileno()))
        try:
            yield writer
        finally:
            after=inventory(root)
            assert all(after.get(n)==v for n,v in binding['files'].items()), 'Original prefix changed'
            extra=set(after)-set(binding['files'])
            allowed={prep.GUARD,'evidence/'+prep.REQUEST,'evidence/'+prep.FAILURE}
            if allow_preparation:allowed|={n+'.tmp' for n in allowed}
            else:allowed=set()
            if allow_first_readiness:allowed|={'evidence/inference/3511/ready.json','evidence/inference/3511/ready.json.tmp'}
            if allow_inference:
                allowed|={prep.GUARD,'evidence/'+prep.REQUEST,'evidence/'+prep.FAILURE}
                from src.research.tokenizer_wide_hybrid_inference_acceptance_v2 import COMPLETED
                for seed in (3511,3512,3513):
                    allowed|={f'evidence/inference/{seed}/'+n for n in COMPLETED|{'review.json','worker-failure.json'}}
                    allowed|={f'control/inference-worker-{seed}.json',f'control/process-inference-{seed}.json'}
                allowed|={n+'.tmp' for n in allowed}
            assert extra<=allowed, 'Undeclared reusable-fixture append'
