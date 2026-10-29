"""Invoke only the three frozen production pilot transitions, serially, once.

Each transition owns its original parent/child guards and process supervisor.
No quality dispatch, cleanup, retry, source mutation or plan modification.
"""
import hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_pilot_dispatch_v2 import run_pilot_once
HERE=Path(__file__).resolve().parent
root=ROOT/'research_runs/tokenizer-wide-hybrid-001'
freeze=json.loads((HERE/'finite-campaign-freeze-v2-review.json').read_text())
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
start=time.monotonic();refs=[]
def save(name,value):
    with (HERE/name).open('x') as out:json.dump(value,out,indent=2);out.write('\n')
save('resource-pilot-launch-v2.json',dict(pid=os.getpid(),started_unix=time.time(),
    plan_sha256=freeze['plan_sha256'],script_sha256=file_digest(Path(__file__)),
    cache=str(cache),slots=[0,1,2],quality_launch=False))
try:
    with session_lock(ROOT/'research_state/tokenizer-worker-lock') as global_fd, session_lock(root) as campaign_fd:
        writer=MetadataWriter(root,(global_fd,campaign_fd))
        projection=dict(path='meta/projection.json',sha256=file_digest(root/'evidence/meta/projection.json'))
        for slot in range(3):
            refs.append(run_pilot_once(writer,slot,freeze['plan_sha256'],projection_ref=projection))
            print(json.dumps(dict(slot=slot,status='pilot_reviewed',reference=refs[-1])),flush=True)
    save('resource-pilot-terminal-v2.json',dict(status='three_pilots_reviewed',references=refs,
        elapsed_seconds=time.monotonic()-start,quality_launch=False))
except BaseException as error:
    save('resource-pilot-terminal-v2.json',dict(status='failed_closed',references=refs,
        elapsed_seconds=time.monotonic()-start,error=repr(error),retry_allowed=False,quality_launch=False))
    raise
