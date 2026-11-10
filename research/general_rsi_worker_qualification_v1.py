"""One bounded assembled-worker check, excluding scientific selection."""
import argparse,fcntl,json,os,shutil,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-worker-qualification-001'
RUNS=ROOT/'research_runs/general-rsi-worker-qualification-001'


def main():
    from research.general_rsi_training_v2 import sha
    from research.general_rsi_equal_time_worker_v1 import put
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('binding',nargs='?');a=p.parse_args()
    if a.action=='prepare':
        previous=json.loads((ROOT/'research/tokenizer-evidence/general-rsi-io-qualification-001/plan.json').read_text())
        timeplan=json.loads((ROOT/previous['context_plan']).read_text())
        sources=dict(previous['sources'])
        for name in ('research/general_rsi_equal_time_worker_v1.py','research/general_rsi_worker_qualification_v1.py'):
            sources[name]=sha(ROOT/name)
        plan=dict(kind='assembled_equal_time_worker_engineering_check_v1',sources=sources,torch_version=timeplan['torch_version'],
            codebook_assets=timeplan['codebook_assets'],ema_half_life_ns=1_000_000_000,checkpoint_interval_ns=4_000_000_000,
            probe_documents=[0,1,2,3,85,86,87,88],slots=[dict(phase='engineering',arm='native',lr=.0025,budget_ns=12_000_000_000,roles=timeplan['roles'])],
            timeout_seconds=600,rss_limit_bytes=4*1024**3,mps_limit_bytes=6*1024**3,reserve_bytes=10*1024**3,
            incremental_peak_bytes=1536*1024**2,scientific_selection=False,test_opened=False,
            scope='Only assembled checkpoint publication/read/retirement and final endpoint saving/scoring; no new scoring oracle, seed selection, or old campaign replay.')
        assert not OUT.exists() and not RUNS.exists();OUT.mkdir();put(OUT/'plan.json',plan);print(sha(OUT/'plan.json'),flush=True);return
    import research.general_rsi_equal_time_worker_v1 as worker
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    assert sha(OUT/'plan.json')==a.binding;plan=json.loads((OUT/'plan.json').read_text())
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not worker.stopped()
        assert shutil.disk_usage(ROOT).free>=plan['reserve_bytes']+plan['incremental_peak_bytes']
        for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
        worker.PANEL=OUT;worker.RUNS=RUNS
        out=OUT/'slot-00';weights=RUNS/'slot-00';out.mkdir();weights.mkdir(parents=True)
        put(OUT/'attempt.json',dict(pid=os.getpid(),started=time.time(),binding=a.binding))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=plan['timeout_seconds'],rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):
                result=worker.execute(plan,0,a.binding,1,out,weights)
                assert result['status']=='completed' and result['snapshots']>0
                assert set(p.name for p in weights.iterdir())=={'raw.pt','time_ema.pt'}
                journal=[json.loads(l) for l in (out/'deletion-journal.jsonl').read_text().splitlines()]
                assert len(journal)>=4 and len(journal)%2==0
                for i in range(0,len(journal),2):
                    assert journal[i]['event']=='before_unlink' and journal[i+1]['event']=='unlinked'
                    assert journal[i]['receipt']==journal[i+1]['receipt']
                    assert not Path(journal[i]['receipt']['path']).exists()
                for e in result['endpoints']:
                    assert sha(e['weights']['path'])==e['weights']['sha256'] and e['count']==100416
                assert json.loads((out/'latest.json').read_text())['status']=='retired_after_completion'
            for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
            put(OUT/'result.json',dict(qualified=True,transient_retirements=len(journal)//2,endpoints_retained=2,
                endpoint_result_sha256=sha(out/'result.json'),scientific_selection=False,test_opened=False))
            print('QUALIFIED',flush=True)
        except BaseException:put(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
