"""Serial frozen002 resource pilots, audited pilot boundary, then quality slot0.

Each original production dispatcher owns its bounded supervisor and guards.
No other quality slot, retry, selection or adoption is initiated here.
"""
import base64,hashlib,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_pilot_dispatch_v2 import run_pilot_once
from src.research.tokenizer_wide_hybrid_pilot_boundary_v2 import cleanup_pilot_boundary,verify_pilot_boundary
from src.research.tokenizer_wide_hybrid_quality_dispatch_v2 import run_lr_once
HERE=Path(__file__).resolve().parent
ROOT_RUN=ROOT/'research_runs/tokenizer-wide-hybrid-002'
freeze=json.loads((HERE/'campaign002-freeze-v2-review.json').read_text())
cache=ROOT/'research_state/tiktoken-cache'
for name,sha in [('vocab.bpe','1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5'),
                 ('encoder.json','196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783')]:
    url='https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/'+name
    assert file_digest(cache/hashlib.sha1(url.encode()).hexdigest())==sha
os.environ['TIKTOKEN_CACHE_DIR']=str(cache)
assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='1'
assert file_digest(ROOT_RUN/'evidence/meta/plan.json')==freeze['plan_sha256']
import torch
assert torch.backends.mps.is_available(), 'MPS required; no CPU fallback'
start=time.monotonic();refs=[];phase='pilots'
def save(name,value):
    with (HERE/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def stopped(signum,frame):raise KeyboardInterrupt('Controller signal '+str(signum))
for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stopped)
save('campaign002-launch-v2.json',dict(pid=os.getpid(),started_unix=time.time(),
    plan_sha256=freeze['plan_sha256'],script_sha256=file_digest(Path(__file__)),
    cache=str(cache),pilot_slots=[0,1,2],quality_slots=[0],automatic_later_slots=False))
try:
    with session_lock(ROOT/'research_state/tokenizer-worker-lock') as global_fd,session_lock(ROOT_RUN) as campaign_fd:
        writer=MetadataWriter(ROOT_RUN,(global_fd,campaign_fd))
        projection=dict(path='meta/projection.json',sha256=file_digest(ROOT_RUN/'evidence/meta/projection.json'))
        for slot in range(3):
            refs.append(run_pilot_once(writer,slot,freeze['plan_sha256'],projection_ref=projection))
            print(json.dumps(dict(phase='pilot_accepted',slot=slot,reference=refs[-1])),flush=True)
        # Archive exact prior bytes before the authorized six-weight cleanup.
        files={str(p.relative_to(ROOT_RUN)):file_digest(p) for p in ROOT_RUN.rglob('*') if p.is_file()}
        metadata={n:base64.b64encode((ROOT_RUN/n).read_bytes()).decode() for n in files if not n.endswith('.pt')}
        save('campaign002-pilots-v2-archive.json',dict(root=str(ROOT_RUN),files=files,noncheckpoint_bytes_base64=metadata,references=refs))
        phase='pilot_boundary'
        boundary=cleanup_pilot_boundary(writer,freeze['plan_sha256'],projection_ref=projection)
        reviewed=verify_pilot_boundary(writer,freeze['plan_sha256'],boundary)
        save('campaign002-pilot-boundary-v2.json',dict(reference=boundary,review=reviewed))
        phase='quality0'
        save('campaign002-quality0-start-v2.json',dict(pid=os.getpid(),started_unix=time.time(),
            plan_sha256=freeze['plan_sha256'],pilot_completion_ref=boundary,quality_slot=0))
        print(json.dumps(dict(phase=phase,status='entering_production_readiness')),flush=True)
        final=run_lr_once(writer,0,freeze['plan_sha256'],projection_ref=projection,pilot_completion_ref=boundary)
    save('campaign002-terminal-v2.json',dict(status='first_quality_trial_accepted',reference=final,
        elapsed_seconds=time.monotonic()-start,automatic_later_slots=False))
    print(json.dumps(final),flush=True)
except BaseException as error:
    save('campaign002-terminal-v2.json',dict(status='failed_closed',phase=phase,pilot_references=refs,
        elapsed_seconds=time.monotonic()-start,error=repr(error),retry_allowed=False,automatic_later_slots=False))
    raise
