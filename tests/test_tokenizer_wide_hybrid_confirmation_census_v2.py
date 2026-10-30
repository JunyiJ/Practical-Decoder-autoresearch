"""Live metadata growth during fresh startup on disposable history; no training."""
import importlib
import pytest

from src.research import tokenizer_wide_hybrid_first_confirmation_startup_v2 as first
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from src.research.tokenizer_wide_hybrid_quality_readiness_v2 import _same_frozen_identity
from test_tokenizer_wide_hybrid_metadata_writer import writer


@pytest.mark.parametrize('module',[
    'confirmation_startup', 'later_confirmation_startup', 'remaining_confirmation_startup',
])
def test_startup_accepts_owned_metadata_growth_and_replays_receipt(writer,monkeypatch,module):
    fixture=importlib.import_module('test_tokenizer_wide_hybrid_'+module+'_v2')
    start=fixture.start
    args=fixture.setup(writer,monkeypatch)
    original=first.bind_campaign
    observed=[]
    def live(*a,**kw):
        bound=dict(original(*a,**kw),metadata_bytes=snapshot(writer.root)['metadata_bytes'])
        observed.append(bound['metadata_bytes'])
        return bound
    monkeypatch.setattr(first,'bind_campaign',live)
    try:
        ref=start.prepare_confirmation_request(**args)
    finally:
        # Exercise a real census delta, not an arbitrary ignored field.
        assert len(observed)>=2 and max(observed)>min(observed)
    checked=start.verify_confirmation_request(writer,args['slot'],ref)
    assert checked['confirmation_request_verified'] and not checked['dispatch_authorized']
    root=writer.root/'evidence'
    assert sorted(p.name for p in (root/f"quality/{args['slot']:02d}").iterdir())==['request.json']
    assert not (root/f"quality/{args['slot']+1:02d}").exists()
    before=snapshot(writer.root)
    with pytest.raises(ValueError,match='cannot be retried'):
        start.prepare_confirmation_request(**args)
    assert snapshot(writer.root)==before


@pytest.mark.parametrize('count',[True,-1,1.5,'123',None,256*1024**2+1])
def test_invalid_census_is_not_treated_as_identity_only(count):
    assert not _same_frozen_identity(dict(metadata_bytes=100),dict(metadata_bytes=count))
    assert not _same_frozen_identity(dict(metadata_bytes=count),dict(metadata_bytes=100))


@pytest.mark.parametrize('field',[
    'kind','context','files','requests','storage_contract','numerical_review_bound',
    'training_input_bound','current_storage_fit_verified','measured_pilots_verified',
    'producer_enforcement_verified','dispatch_authorized','unexpected',
])
def test_census_exception_does_not_hide_any_other_binding_change(field):
    before=dict(metadata_bytes=100,**{field:'original'})
    after=dict(metadata_bytes=200,**{field:'changed'})
    assert not _same_frozen_identity(before,after)
    assert not _same_frozen_identity(before,dict(metadata_bytes=200))


def test_census_presence_must_match():
    assert not _same_frozen_identity({},dict(metadata_bytes=100))
    assert not _same_frozen_identity(dict(metadata_bytes=100),{})
