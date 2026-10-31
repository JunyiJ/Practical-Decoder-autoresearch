"""Run the frozen three-seed inference panel once; no training, retry or adoption."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import digest,file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_inference_preparation_v2 import prepare_inference
from src.research.tokenizer_wide_hybrid_inference_dispatch_v2 import run_panel
HERE=Path(__file__).resolve().parent
RUN=ROOT/'research_runs/tokenizer-wide-hybrid-005'
STEM='campaign005-inference-v2'
def save(suffix,value):
    with (HERE/(STEM+suffix+'.json')).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def stopped(signum,frame):raise KeyboardInterrupt('Inference controller signal '+str(signum))
def main():
    freeze=json.loads((HERE/'campaign005-freeze-v2-review.json').read_text());sha=freeze['plan_sha256']
    review_path=HERE/'campaign005-fresh-prefix-24-v2-review.json'
    review=json.loads(review_path.read_text())
    guard=json.loads((HERE/'campaign005-fresh-prefix-24-v2-guard.json').read_text())
    assert review['status']=='independently_reviewed_slot23_fresh_prefix' and review['completed']==24
    assert review['plan_sha256']==sha and guard['status']=='finished' and guard['reason'] is None
    assert digest(source_fingerprint())==freeze['source_sha256']
    assert file_digest(RUN/'evidence/meta/plan.json')==sha
    mapping=json.loads((HERE/'campaign005-inference-resumption-map-v2.json').read_text())
    assert all(file_digest(ROOT/p)==pin for p,pin in mapping['source_files'].items())
    for suffix in ('-launch','-terminal','-prepared','-startup-guard'):
        assert not (HERE/(STEM+suffix+'.json')).exists()
    assert not (RUN/'evidence/quality/24').exists()
    assert not (RUN/'evidence/inference').exists()
    cache=ROOT/'research_state/tiktoken-cache'
    for name,pin in [('vocab.bpe','1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5'),('encoder.json','196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783')]:
        url='https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/'+name
        assert file_digest(cache/hashlib.sha1(url.encode()).hexdigest())==pin
    os.environ['TIKTOKEN_CACHE_DIR']=str(cache)
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='1'
    import torch
    assert torch.backends.mps.is_available(),'MPS required; no CPU fallback'
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stopped)
    projection=dict(path='meta/projection.json',sha256=file_digest(RUN/'evidence/meta/projection.json'))
    start=time.monotonic()
    save('-launch',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=sha,
        script_sha256=file_digest(Path(__file__)),independent_review_sha256=file_digest(review_path),
        seed_order=[3511,3512,3513],automatic_later_quality=False,adopted=False))
    try:
        with session_lock(ROOT/'research_state/tokenizer-worker-lock') as g,session_lock(RUN) as c:
            writer=MetadataWriter(RUN,(g,c))
            with CpuPrerequisiteGuard(HERE/(STEM+'-startup-guard.json'),ROOT,timeout_seconds=1800):
                request=prepare_inference(writer,plan_sha256=sha,completion_ref=review['dispatch_references'][-1],
                    projection_ref=projection,rank_completion_ref=review['rank_completion_ref'])
            save('-prepared',dict(status='prepared_pending_serial_panel',reference=request,adopted=False))
            # The outer idle guard ends before any child; run_panel owns all phase/process guards.
            result=run_panel(writer,request)
        save('-terminal',dict(result,elapsed_seconds=time.monotonic()-start))
        print(json.dumps(result),flush=True)
    except BaseException as error:
        save('-terminal',dict(status='failed_closed',error=repr(error),elapsed_seconds=time.monotonic()-start,
            retry_allowed=False,adopted=False))
        raise
if __name__=='__main__':main()
