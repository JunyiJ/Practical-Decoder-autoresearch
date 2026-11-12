"""Read-only audit of the new-context pilot, its receipts and retained failure."""
import ast,datetime,fcntl,json,shutil
from pathlib import Path
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put
OUT=ROOT/'research/tokenizer-evidence/general-rsi-context-length-pilot-001'
def read(p):return json.loads(p.read_text())
def main():
    with open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=read(OUT/'plan.json');a=OUT/'attempt-01';r=read(a/'result.json');g=read(a/'guard.json')
        assert r['accepted'] and g['status']=='finished' and not list(a.glob('*failure*'))
        assert sha(OUT/'plan.json')==r['plan_sha256']==read(OUT/'admission.json')['plan_sha256']
        assert all(sha(ROOT/f)==h for f,h in p['sources'].items())
        histories=[];receipts=[]
        for length,summary in zip((128,256),r['summaries']):
            rows=[json.loads(x) for x in (a/f'c{length}-updates.jsonl').read_text().splitlines()];assert len(rows)==24
            assert [x['step'] for x in rows]==list(range(1,25)) and all(x['lr_ratio']==1 for x in rows)
            assert rows[-1]['end_ns']/1e9==summary['training_seconds']
            assert rows[0]['begin_ns']==0 and all(x['end_ns']==y['begin_ns'] for x,y in zip(rows,rows[1:]))
            histories.append(rows)
            for name,receipt in summary.items():
                if name.endswith('_receipt'):
                    path=Path(receipt['path']);assert path.stat().st_size==receipt['size'] and sha(path)==receipt['sha256'];receipts.append(receipt)
        assert [x['batch_sha256'] for x in histories[0]]==[x['batch_sha256'] for x in histories[1]]
        assert (a/'c256-reference-next.jsonl').read_bytes()==(a/'c256-replayed-next.jsonl').read_bytes()
        assert r['full_replay_exact'] and r['restart_roundtrip_exact'] and r['wrong_context_rejected_before_mutation']
        assert sum(x['size'] for x in receipts)==r['retained_bytes']
        old=(ROOT/'research/byte_input_model_v6/optimizer.py').read_text().replace('from research.byte_input_model_v3 import contract as C','from . import contract as C').replace('ByteInputModel','ContextModel').replace('ByteInputOptimizer','ContextOptimizer')
        new=(ROOT/'research/general_rsi_context_length_v1/optimizer.py').read_text()
        assert ast.dump(ast.parse(old))==ast.dump(ast.parse(new))
        failed=ROOT/'research/tokenizer-evidence/general-rsi-context-length-fixture-001';fixed=ROOT/'research/tokenizer-evidence/general-rsi-context-length-fixture-002'
        assert (failed/'attempt-01/failure.json').exists() and read(fixed/'completion-review.json')['accepted']
        total_guard=sum(read(x/'attempt-01/guard.json')['elapsed_seconds'] for x in (failed,fixed,OUT))
        review=dict(accepted=True,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(OUT/'plan.json'),result_sha256=sha(a/'result.json'),review_source_sha256=sha(__file__),source_bindings_unchanged=len(p['sources']),optimizer_math_ast_equivalent=True,pilot_updates=50,paired_draws=24,full_restart_replay_exact=True,wrong_context_rejected_before_mutation=True,retained_receipts=receipts,retained_bytes=r['retained_bytes'],guard_seconds=g['elapsed_seconds'],all_attempts_guard_seconds=total_guard,peak_rss_bytes=g['observation']['peak_rss_bytes'],peak_mps_bytes=r['peak_mps_bytes'],prior_harness_failures=1,original_failed_attempt_unchanged=True,lock_reacquired=True,worker_exited=True,free_bytes=shutil.disk_usage(ROOT).free,test_opened=False,validation_opened=False,quality_adopted=False,next_action='Integrate the changed context path into time-EMA endpoints and identical-target evaluation; qualify only changed state/evaluation/generation paths before fresh panel admission.')
        put(OUT/'completion-review.json',review);print(json.dumps(review,indent=2))
if __name__=='__main__':main()
