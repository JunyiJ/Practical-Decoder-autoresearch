"""Live metadata growth during fresh startup on disposable history; no training."""
import importlib
import pytest

from src.research import tokenizer_wide_hybrid_first_confirmation_startup_v2 as first
from src.research.tokenizer_wide_hybrid_metadata import snapshot
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
