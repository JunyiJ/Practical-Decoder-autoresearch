"""Invoke exactly frozen quality slot0 after accepted measured pilot boundary.

All readiness, process, resource and independent acceptance gates remain in the
production dispatcher. No next slot, cleanup, retry or adoption is performed.
"""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_quality_dispatch_v2 import run_lr_once
HERE=Path(__file__).resolve().parent
root=ROOT/'research_runs/tokenizer-wide-hybrid-001'
freeze=json.loads((HERE/'finite-campaign-freeze-v2-review.json').read_text())
boundary=json.loads((HERE/'resource-pilot-boundary-v2-review.json').read_text())
assert boundary['review']['pilot_boundary_verified'] is True
cache=ROOT/'research_state/tiktoken-cache'
for name,sha in [('vocab.bpe','1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5'),
                 ('encoder.json','196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783')]:
    url='https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/'+name
    assert file_digest(cache/hashlib.sha1(url.encode()).hexdigest())==sha
os.environ['TIKTOKEN_CACHE_DIR']=str(cache)
assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='1'
assert file_digest(root/'evidence/meta/plan.json')==freeze['plan_sha256']
import torch
assert torch.backends.mps.is_available(), 'MPS required; no CPU fallback'
start=time.monotonic()
def save(name,value):
    with (HERE/name).open('x') as out:json.dump(value,out,indent=2);out.write('\n')
save('first-quality-launch-v2.json',dict(pid=os.getpid(),started_unix=time.time(),slot=0,
    plan_sha256=freeze['plan_sha256'],pilot_completion_ref=boundary['reference'],
    script_sha256=file_digest(Path(__file__)),cache=str(cache),automatic_next_slot=False))
try:
    with session_lock(ROOT/'research_state/tokenizer-worker-lock') as global_fd, session_lock(root) as campaign_fd:
        writer=MetadataWriter(root,(global_fd,campaign_fd))
        projection=dict(path='meta/projection.json',sha256=file_digest(root/'evidence/meta/projection.json'))
        ref=run_lr_once(writer,0,freeze['plan_sha256'],projection_ref=projection,pilot_completion_ref=boundary['reference'])
    save('first-quality-terminal-v2.json',dict(status='first_quality_trial_accepted',reference=ref,
        elapsed_seconds=time.monotonic()-start,automatic_next_slot=False))
    print(json.dumps(ref),flush=True)
except BaseException as error:
    save('first-quality-terminal-v2.json',dict(status='failed_closed',elapsed_seconds=time.monotonic()-start,
        error=repr(error),retry_allowed=False,automatic_next_slot=False))
    raise
