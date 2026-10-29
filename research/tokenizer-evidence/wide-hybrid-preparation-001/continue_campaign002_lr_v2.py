"""One later frozen LR slot through original production gates; no retry or cleanup."""
import argparse,hashlib,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_quality_dispatch_v2 import run_lr_once
HERE=Path(__file__).resolve().parent
RUN=ROOT/'research_runs/tokenizer-wide-hybrid-002'
def save(name,value):
    with (HERE/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def stopped(signum,frame):raise KeyboardInterrupt('Controller signal '+str(signum))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--slot',type=int,required=True);a=ap.parse_args();assert 1<=a.slot<=8
    stem=f'campaign002-lr-{a.slot:02d}-v2'
    freeze=json.loads((HERE/'campaign002-freeze-v2-review.json').read_text());sha=freeze['plan_sha256']
    review_path=HERE/f'campaign002-lr-prefix-{a.slot:02d}-v2-review.json'
    review=json.loads(review_path.read_text())
    guard=json.loads((HERE/f'campaign002-lr-prefix-{a.slot:02d}-v2-guard.json').read_text())
    assert review['status']=='independently_reviewed_lr_prefix' and review['completed']==a.slot and review['plan_sha256']==sha
    assert guard['status']=='finished' and guard['reason'] is None
    assert file_digest(RUN/'evidence/meta/plan.json')==sha
    assert not (HERE/(stem+'-launch.json')).exists() and not (HERE/(stem+'-terminal.json')).exists()
    cache=ROOT/'research_state/tiktoken-cache'
    for name,pin in [('vocab.bpe','1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5'),('encoder.json','196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783')]:
        url='https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/'+name
        assert file_digest(cache/hashlib.sha1(url.encode()).hexdigest())==pin
    os.environ['TIKTOKEN_CACHE_DIR']=str(cache)
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='1'
    import torch
    assert torch.backends.mps.is_available(),'MPS required; no CPU fallback'
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stopped)
    boundary=json.loads((HERE/'campaign002-pilot-boundary-v2.json').read_text())['reference']
    projection=dict(path='meta/projection.json',sha256=file_digest(RUN/'evidence/meta/projection.json'))
    start=time.monotonic()
    save(stem+'-launch.json',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=sha,slot=a.slot,
        script_sha256=file_digest(Path(__file__)),independent_review_sha256=file_digest(review_path),automatic_later_slots=False))
    try:
        with session_lock(ROOT/'research_state/tokenizer-worker-lock') as g,session_lock(RUN) as c:
            writer=MetadataWriter(RUN,(g,c))
            ref=run_lr_once(writer,a.slot,sha,projection_ref=projection,pilot_completion_ref=boundary)
        save(stem+'-terminal.json',dict(status='quality_trial_accepted',slot=a.slot,reference=ref,elapsed_seconds=time.monotonic()-start,automatic_later_slots=False))
        print(json.dumps(ref),flush=True)
    except BaseException as error:
        save(stem+'-terminal.json',dict(status='failed_closed',slot=a.slot,error=repr(error),elapsed_seconds=time.monotonic()-start,retry_allowed=False,automatic_later_slots=False))
        raise
if __name__=='__main__':main()
