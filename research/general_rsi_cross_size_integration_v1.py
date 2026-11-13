"""Only new worker publication/loading/generation paths; training-only fixture."""
import argparse,datetime,fcntl,gc,json,os,shutil,signal,traceback
from pathlib import Path
import torch
from research.general_rsi_cross_size_worker_v1 import execute,put,stop
from research.general_rsi_training_v2 import sha,read_checkpoint,clone_tree
from research.general_rsi_cross_size_endpoint_v1 import KIND,apply,load
from research.general_rsi_cross_size_v1.model import DepthModel
from research.general_rsi_cross_size_v1.contract import Spec
from research.general_rsi_cross_size_pilot_v1 import TrainScoreCase
from research.general_rsi_pseudo_update_diagnostic_v1 import check_stop,digest
from research.byte_input_model_v6 import build_codebook
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-integration-001'
RUN=ROOT/'research_runs/general-rsi-cross-size-integration-001'

def read(p):return json.loads(p.read_text())
def prepare():
    prior=read(ROOT/'research/tokenizer-evidence/general-rsi-cross-size-pilot-001/plan.json')
    sources=dict(prior['sources'])
    for name in ['general_rsi_cross_size_endpoint_v1.py','general_rsi_cross_size_generation_v1.py','general_rsi_cross_size_worker_v1.py','general_rsi_cross_size_integration_v1.py','general_rsi_context_length_worker_v1.py','general_rsi_time_journal_v1.py','general_rsi_transfer_data_v1.py','general-rsi-generation-eval-v1.json']:
        p=ROOT/'research'/name;sources[str(p.relative_to(ROOT))]=sha(p)
    roles=dict(zip(['native_initializer','composer_initializer','train_sampler','validation_sampler'],range(62111312033001,62111312033005)))
    p=dict(kind='cross_size_worker_integration_v1',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,codebook_assets=prior['codebook_assets'],torch_version=torch.__version__,scientific_training=False,
        slots=[dict(arm='raw_bytes',depth=16,roles=roles,extra_group_seeds=list(range(62111312033005,62111312033008)),horizon=40)],
        fresh_seed_audit='rg62111312033[0-9]+ over59536 JSON/JSONL files returned0matches before assignment',
        generation_protocol=read(ROOT/'research/general-rsi-generation-eval-v1.json'),
        checkpoint_interval=20,probe_steps=[20],probe_documents=[0],validation_targets=258,broader_targets=258,
        mps_limit_bytes=6*2**30,rss_limit_bytes=4*2**30,reserve_bytes=12*2**30,timeout_seconds=600,
        retained_weights_cap_bytes=224*2**20,checkpoint_cap_bytes=640*2**20,
        required_free_bytes=read(ROOT/'research/tokenizer-evidence/general-rsi-cross-size-pilot-001/completion-review.json')['storage_remaining_planning_bytes'],
        criteria='Six retained engineering-shape payloads load exactly; wrong identity and last-tensor NaN reject before mutation. Actual largest-byte worker pauses30/restores40 with activeEMA; standalone RAW/H200 loading and training-only evaluation complete;24first generation outputs retained. No quality selection, no new depth-numerics replay.',test_opened=False)
    assert shutil.disk_usage(ROOT).free>=p['required_free_bytes'];OUT.mkdir();put(OUT/'plan.json',p);print(sha(OUT/'plan.json'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);args=ap.parse_args()
    if args.action=='prepare':prepare();return
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop();p=read(OUT/'plan.json');binding=sha(OUT/'plan.json')
        assert all(sha(ROOT/n)==h for n,h in p['sources'].items()) and shutil.disk_usage(ROOT).free>=p['required_free_bytes']
        attempt=OUT/'attempt-01';attempt.mkdir();RUN.mkdir();put(attempt/'launch.json',dict(pid=os.getpid(),binding=binding))
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);torch.set_num_threads(4)
        try:
            with CpuPrerequisiteGuard(attempt/'guard.json',ROOT,timeout_seconds=p['timeout_seconds'],rss_limit_bytes=p['rss_limit_bytes'],reserve_bytes=p['reserve_bytes']):
                book=build_codebook(*[Path(a['path']).read_bytes() for a in p['codebook_assets']],'raw_bytes')
                pilot=read(ROOT/'research/tokenizer-evidence/general-rsi-cross-size-pilot-001/attempt-01/result.json');checks=[]
                for s in pilot['summaries']:
                    payload=read_checkpoint(s['endpoint']);cfg=payload['config'];arm=cfg['arm'];depth=cfg['depth']
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(cfg['roles']['native_initializer'])
                        m=DepthModel(Spec(arm,depth,tuple(cfg['extra_group_seeds']),None if arm=='native' else cfg['roles']['composer_initializer'],None if arm=='native' else book.identity),None if arm=='native' else book)
                    # In-memory schema fixture, not a new training endpoint claim.
                    fixture=dict(kind=KIND,identity=m.identity,config=cfg,endpoint='raw',step=cfg['horizon'],immutable_buffers=m._own_expected_buffer_hashes,parameters=payload['parameters'])
                    apply(m,fixture,cfg,'raw');assert digest(clone_tree(dict(m.ordered_parameters())))==digest(payload['parameters'])
                    before=digest(clone_tree(dict(m.ordered_parameters())))
                    bad=dict(fixture,identity=('wrong-depth',))
                    try:apply(m,bad,cfg,'raw')
                    except AssertionError:pass
                    else:raise AssertionError('wrong identity accepted')
                    assert digest(clone_tree(dict(m.ordered_parameters())))==before
                    last=next(reversed(fixture['parameters']));fixture['parameters'][last].flatten()[-1]=float('nan')
                    try:apply(m,fixture,cfg,'raw')
                    except AssertionError:pass
                    else:raise AssertionError('late nonfinite parameter accepted')
                    assert digest(clone_tree(dict(m.ordered_parameters())))==before
                    checks.append(dict(depth=depth,arm=arm,exact_parameters=True,invalid_rejection_before_mutation=True));del m,payload,fixture,bad;gc.collect()
                put(attempt/'loader-checks.json',checks)
                out=attempt/'slot-00';out.mkdir();weights=RUN/'slot-00';weights.mkdir()
                first=execute(p,0,binding,1,out,weights,engineering_data=TrainScoreCase,pause_at=30)
                assert first['status']=='stopped' and first['step']==30
                gc.collect();torch.mps.empty_cache()
                result=execute(p,0,binding,2,out,weights,engineering_data=TrainScoreCase)
                assert result['status']=='completed' and result['steps']==40 and result['snapshots']==2
                assert read(out/'recovery-02.json')['exact_complete_state']
                assert len(read(out/'generation/result.json')['rows'])==24
                retained=sum(w.stat().st_size for w in RUN.rglob('*.pt'));assert retained<=p['retained_weights_cap_bytes']
                assert not list(RUN.rglob('restart-*.pt'))
                review=dict(accepted_integration=True,binding=binding,loader_checks=checks,new_updates=40,paused_at=30,restored_complete_state_exact=True,raw_h200_scored_through_standalone_loader=True,training_only_targets_per_endpoint_per_score=258,first_outputs=24,retained_bytes=retained,scientific_training=False,validation_scored=False,test_opened=False,result_sha256=sha(out/'result.json'))
            assert all(sha(ROOT/n)==h for n,h in p['sources'].items());put(attempt/'result.json',review);print(json.dumps(review),flush=True)
        except BaseException:put(attempt/'failure.json',dict(traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
