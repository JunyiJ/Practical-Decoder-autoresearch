"""Disposable actual child-entry regression with explicit source/model doubles.

Not a production worker: only temp roots with a synthetic plan are permitted.
Calls the production entry/prefix/review code; never calls model production.
"""
from contextlib import contextmanager
import json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research.tokenizer_activation_session import reference,Session
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from research.activation_inference_recovery_v2 import context as c,controller as e,worker as w

root=Path(sys.argv[1]).resolve();cleanup.GLOBAL_LOCK=Path(sys.argv[2]);fds=tuple(map(int,sys.argv[3:5]));case=sys.argv[5]
assert 'pytest-' in str(root) and root.name=='recovery'
writer=MetadataWriter(root,fds);ctx=c.Recovery(writer,reference(root,c.PLAN_NAME))
assert ctx.document(ctx.plan)=={'synthetic_plan':True}
imported=json.loads((root/c.IMPORT_NAME).read_text());carried=json.loads((root/c.CARRIED_NAME).read_text())
class Source:
    verify=Session.verify;document=Session.document
    root=Path(imported['original_root']);plan=imported['plan']
source=Source()
@contextmanager
def original(ctx_):yield source
for mod in (c,e,w):mod.carried_import=lambda ctx_:carried
for mod in (e,w):mod.bind=lambda ctx_,**kw:(c.budget(ctx_,**kw),imported)[1]
e.original_session=original;e.idle=lambda *a:None
seen=[]
def verify(ctx_,source_,pins):
    assert pins==imported
    for binding in imported['pair_bindings'].values():
        for model in binding['bindings'].values():
            source.verify(model['request'])
            for ref in model['weights'].values():source.verify(ref)
e.verify_original=verify
# This hook is exactly the long source-audit boundary inside production
# review_outputs, AFTER it captured the completed-slot hash proof.
def concurrent_verify(ctx_,source_,pins):
    verify(ctx_,source_,pins)
    control=root/'control/process-inference-3513.json'
    with writer._serialized():before=control.read_bytes()
    print('x'*8192,flush=True);started=time.monotonic()
    while True:
        with writer._serialized():
            monitor=json.loads(control.read_text())
            ack=json.loads((root/'control/recovery-worker-inference-3513.json').read_text()) if (root/'control/recovery-worker-inference-3513.json').exists() else {}
        if ack.get('synthetic_ack') and monitor['observation']['log_bytes']>=8193:break
        if time.monotonic()-started>5:raise TimeoutError('Actual entry handshake timeout')
        time.sleep(.01)
    with writer._serialized():assert before!=control.read_bytes()
    seen.append(True)
w.verify_original=concurrent_verify
produced=[]
e.produce=lambda *a:produced.append(True)
request=reference(root,'evidence/inference/3513/request.json')
try:e.worker(ctx,3613,request)
except (ValueError,KeyError) as exc:
    assert case!='allowed' and seen==[True] and not produced
    report=dict(status='rejected_expected',case=case,error=str(exc))
else:
    assert case=='allowed' and seen==[True] and produced==[True]
    report=dict(status='passed',case=case,entry_reached_production_boundary=True)
print('ENTRY_RESULT '+json.dumps(report),flush=True)
