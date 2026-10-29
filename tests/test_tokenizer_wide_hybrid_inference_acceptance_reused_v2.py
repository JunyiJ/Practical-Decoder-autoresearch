"""Three synthetic output triples; real complete retained-history audits each time.

No OS child or MPS model is launched. Transport documents and child identities
are synthetic. Only this already-successful hash-pinned disposable prefix may
append; all original files are preserved and a failure closes the new attempt.
"""
import copy
import json
import os
from pathlib import Path

import pytest

from src.research import tokenizer_wide_hybrid_inference_acceptance_v2 as acceptance
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from tokenizer_wide_hybrid_reusable_fixture import reopen_prefix
from test_tokenizer_wide_hybrid_inference_acceptance_v2 import terminal

entry=acceptance.entry;outputs=acceptance.outputs;prep=acceptance.prep


def synthetic_token_adapter(patch):
    """Adapt the historical fixture's placeholder in all independent consumers."""
    from src.research import tokenizer_wide_hybrid_timing as timing
    from src.research import tokenizer_wide_hybrid_timing_review as timing_review
    def synthetic_digest(value):
        assert list(value)==list(range(96))
        return outputs.digest('96 synthetic training IDs')
    for module in (outputs,timing,timing_review):patch.setattr(module,'token_digest',synthetic_digest)


def test_synthetic_adapter_covers_producer_and_independent_reviewer(monkeypatch):
    from src.research.tokenizer_wide_hybrid_timing_review import review_result
    with monkeypatch.context() as patch:
        synthetic_token_adapter(patch)
        selection={f:dict(lr=.00125,divisor=4) for f in outputs.FAMILIES}
        sha=outputs.digest('96 synthetic training IDs')
        result=outputs.measure_triple(3511,selection,list(range(96)),sha,
            check=lambda f,b:dict(passed=True,max_absolute_error=0.),
            measure=lambda f,b:dict(prefill_seconds=.1,decode_seconds=.2,state_bytes=100*b,
                observed_mps_current_bytes=1000,observed_mps_driver_bytes=2000))
        assert len(review_result(result,3511,selection,sha)['cells'])==6
        with pytest.raises(AssertionError):outputs.token_digest([1]*96)
    assert outputs.token_digest(list(range(96)))!=sha


def synthetic_triple(writer,ready_ref,monkeypatch):
    root=writer.root/'evidence';ready=outputs._document(root,ready_ref);seed=ready['seed']
    proof=entry._immutable(writer,ready);bound=proof['binding'];metadata={}
    rows=json.loads((root/'meta/projection.json').read_text())['records']
    for family,binding in bound['bindings'].items():
        spec=binding['request']['spec'];result=json.loads((Path(binding['trial'])/'result.json').read_text())
        row=next(r for r in rows if r['width']==512 and r['family']==family and r['bottleneck_divisor']==spec['divisor'])
        metadata[family]=dict(checkpoint=result['h200']['checkpoint'],sha256=result['h200']['checkpoint_sha256'],
            seed=seed,family=family,divisor=spec['divisor'],lr=spec['lr'],config_sha256=outputs.digest(binding['request']['config']),
            tokenizer_sha256=outputs.digest(bound['tokenizer']),parameters=row['parameters'],parameter_storage_bytes=row['parameter_bytes'])
    ids=list(range(96))
    assert bound['token_sha256']==outputs.digest('96 synthetic training IDs')
    parent=ready['parent_pid'];child=900000+seed;argv=entry.worker_argv(writer,seed,ready_ref);launch=entry._launch(argv,parent)
    writer.publish_json(f'evidence/inference/{seed}/process/launch.json',launch)
    with writer.open_append(f'evidence/inference/{seed}/process/worker.log') as log:log.write(b'synthetic inference transport; no OS child\n')
    state=dict(launch,status='running',pid=child,elapsed_seconds=.1,
        observation=dict(free_bytes=64*1024**3,output_bytes=1000,log_bytes=0))
    writer.publish_json(f'control/process-inference-{seed}.json',state)
    with monkeypatch.context() as patch:
        patch.setattr(outputs.os,'getpid',lambda:child);patch.setattr(outputs.os,'getppid',lambda:parent)
        patch.setattr(entry.os,'getpgid',lambda pid:pid)
        patch.setattr(outputs,'start_memory_guard',lambda:None);patch.setattr(outputs.torch,'set_num_threads',lambda n:None)
        patch.setattr(outputs,'load_selected_exports',lambda *a:dict(models={},configs={},metadata=copy.deepcopy(metadata),token_ids=ids))
        patch.setattr(outputs,'mps_callbacks',lambda *a:(lambda f,b:dict(passed=True,max_absolute_error=0.),
            lambda f,b:dict(prefill_seconds=.1,decode_seconds=.2,state_bytes=100*b,
                observed_mps_current_bytes=1000,observed_mps_driver_bytes=2000)))
        if seed==3511:ref=entry.run_worker(writer,seed,ready_ref,argv)
        else:
            ref=outputs.execute_bound_triple(writer,seed,
                entry_check=lambda phase:entry._immutable(writer,outputs._document(root,ready_ref)))
    return ref,terminal(writer,seed)


def test_three_finite_boundaries_replay_complete_quality_history(monkeypatch):
    name=os.environ.get('TOKENIZER_READY_PREFIX_BINDING')
    if not name:pytest.skip('Explicit successful ready prefix binding required')
    binding=json.loads(Path(name).read_text());calls=[];original=prep.gate.history._verify
    def full(*a,**kw):
        value=original(*a,**kw);calls.append(kw.get('inference_request_ref'));return value
    with reopen_prefix(binding,allow_inference=True) as writer,monkeypatch.context() as patch:
        patch.setattr(prep.gate.history,'_verify',full)
        patch.setattr(acceptance,'_group_exists',lambda pid:False)
        synthetic_token_adapter(patch)
        root=writer.root/'evidence'
        assert not (root/prep.REQUEST).exists(), 'Only untouched successful idle prefix'
        complete=acceptance._ref(root,'meta/quality-dispatch-23.json')
        saved=outputs._document(root,complete)
        prep.prepare_inference(writer,plan_sha256=saved['plan_sha256'],completion_ref=complete,
            projection_ref=saved['projection_ref'],rank_completion_ref=saved['rank_completion_ref'])
        assert not (root/'inference/3511/ready.json').exists(), 'Only fresh readiness on this alternate prefix'
        ready=entry.prepare_first_readiness(writer,acceptance._ref(root,prep.REQUEST))
        # Full replay must admit only the explicitly scoped readiness append.
        report=acceptance.verify_boundary(writer,ready)
        assert report['quality_history_reaudited'] and len(report['bound']['history']['records'])==24
        accepted=[]
        for seed in outputs.SEEDS:
            ref,process=synthetic_triple(writer,ready,patch)
            with patch.context() as parent:
                parent.setattr(acceptance.os,'getpid',lambda:process['parent_pid'])
                reviewed=acceptance.accept_completed(writer,ref,process)
            accepted.append(reviewed)
            if seed!=outputs.SEEDS[-1]:ready=acceptance.prepare_next_readiness(writer,reviewed)
        final=acceptance.verify_boundary(writer,accepted[-1])
        assert final['scope']['previous_refs']==accepted
        assert len(final['bound']['history']['checkpoint_pairs'])==12
        assert len(list(root.rglob('*.pt')))==24
        assert [None if ref is None else ref['path'] for ref in calls]==[
            None,prep.REQUEST,prep.REQUEST,'inference/3511/ready.json','inference/3511/worker-outputs.json','inference/3511/review.json',
            'inference/3512/worker-outputs.json','inference/3512/review.json',
            'inference/3513/worker-outputs.json','inference/3513/review.json']
        with pytest.raises(ValueError):prep.verify_prepared_inference(writer,acceptance._ref(root,prep.REQUEST))
        request=outputs._document(root,acceptance._ref(root,prep.REQUEST))
        with pytest.raises(ValueError):
            prep.gate.history.verify_confirmation_dispatch(writer,23,request['plan_sha256'],request['completion_ref'],
                projection_ref=request['projection_ref'],rank_completion_ref=request['rank_completion_ref'])
        with pytest.raises(ValueError):acceptance.prepare_next_readiness(writer,accepted[-1])
