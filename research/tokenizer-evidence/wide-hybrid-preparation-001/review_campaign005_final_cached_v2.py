"""One guarded independent final005 review; no training, inference, cleanup or adoption."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest,digest
from src.research.runner import source_fingerprint
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter,json_encoder
from review_final_campaign_v2 import review_completed_campaign,LIMIT
HERE=Path(__file__).resolve().parent
RUN=ROOT/'research_runs/tokenizer-wide-hybrid-005'
STEM='campaign005-final-cached-v2'
def save(suffix,value):
    data=(json_encoder().encode(value)+'\n').encode('ascii')
    assert len(data)<=LIMIT
    with (HERE/(STEM+suffix+'.json')).open('xb') as f:f.write(data)
def stopped(signum,frame):raise KeyboardInterrupt('Final review signal '+str(signum))
def main():
    for suffix in ('-launch','-guard','-review','-failure'):
        assert not (HERE/(STEM+suffix+'.json')).exists()
    terminal_path=HERE/'campaign005-inference-v2-terminal.json'
    terminal=json.loads(terminal_path.read_text())
    assert terminal['status']=='inference_panel_reviewed_pending_quality_cost_decision'
    assert [r['path'] for r in terminal['reviews']]==[f'inference/{s}/review.json' for s in (3511,3512,3513)]
    freeze=json.loads((HERE/'campaign005-freeze-v2-review.json').read_text())
    assert digest(source_fingerprint())==freeze['source_sha256']
    assert file_digest(RUN/'evidence/meta/plan.json')==freeze['plan_sha256']
    plan=json.loads((RUN/'evidence/meta/plan.json').read_text())
    procedure=HERE/'review_final_campaign_v2.py'
    assert dict(path=str(procedure),sha256=file_digest(procedure)) in plan['files']
    for ref in terminal['reviews']:
        assert file_digest(RUN/'evidence'/ref['path'])==ref['sha256']
    for seed in (3511,3512,3513):
        value=json.loads((RUN/f'evidence/inference/{seed}/controller.json').read_text())
        assert value['status']=='completed' and value['seed']==seed
        assert all(g['status']=='finished' and g['reason'] is None for g in value['guards'].values())
    cache=ROOT/'research_state/tiktoken-cache'
    for name,pin in [('vocab.bpe','1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5'),('encoder.json','196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783')]:
        url='https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/'+name
        assert file_digest(cache/hashlib.sha1(url.encode()).hexdigest())==pin
    os.environ['TIKTOKEN_CACHE_DIR']=str(cache)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stopped)
    save('-launch',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=freeze['plan_sha256'],
        script_sha256=file_digest(Path(__file__)),procedure_sha256=file_digest(procedure),
        inference_terminal_sha256=file_digest(terminal_path),adopted=False))
    try:
        with session_lock(ROOT/'research_state/tokenizer-worker-lock') as g,session_lock(RUN) as c, CpuPrerequisiteGuard(HERE/(STEM+'-guard.json'),ROOT,timeout_seconds=1800):
            report=review_completed_campaign(MetadataWriter(RUN,(g,c)),terminal['reviews'][-1])
            assert report['status']=='complete_independent_quality_cost_review'
            save('-review',report)
        print(json.dumps({k:report[k] for k in ('status','adoption_eligible','passing_routes','adopted')}),flush=True)
    except BaseException as error:
        save('-failure',dict(status='failed_closed',error=repr(error),retry_allowed=False))
        raise
if __name__=='__main__':main()
