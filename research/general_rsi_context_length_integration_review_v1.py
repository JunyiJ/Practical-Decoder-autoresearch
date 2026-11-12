"""Audit new context integration, all first generations and worker binding."""
import ast,copy,datetime,fcntl,json,math,shutil
from pathlib import Path
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put
OUT=ROOT/'research/tokenizer-evidence/general-rsi-context-length-integration-001'
def read(p):return json.loads(p.read_text())
def main():
    from research.general_rsi_context_length_score_v1 import aggregate
    with open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=read(OUT/'plan.json');a=OUT/'attempt-01';r=read(a/'result.json');g=read(a/'guard.json')
        assert r['accepted'] and g['status']=='finished' and not list(a.glob('*failure*'))
        assert r['plan_sha256']==sha(OUT/'plan.json')==read(OUT/'admission.json')['plan_sha256']
        assert all(sha(ROOT/f)==h for f,h in p['sources'].items())
        assert (a/'reference-next.jsonl').read_bytes()==(a/'replayed-next.jsonl').read_bytes()
        records=[read(a/'train-document-c128.json'),read(a/'train-document-c256.json')]
        for record in records:
            rows=record['rows'];assert sum(x['targets'] for x in rows)==653
            assert math.fsum(x['loss_sum'] for x in rows)==record['loss_sum']
            for x in rows:assert math.fsum(x['token_losses'])==x['loss_sum'] and len(x['target_ids'])==x['targets']
            result=aggregate(copy.deepcopy(record),[1],[dict(source='unused'),p['training_document']])
            assert result['count']==653 and result['documents'][0]['document']==1 and all(x['document']==1 for x in result['chunks'])
            assert result['sources']['fineweb_edu']['loss_sum']==record['loss_sum']
        synthetic=dict(context=256,rows=[dict(document=0,targets=2,loss_sum=4.),dict(document=1,targets=1,loss_sum=3.),dict(document=2,targets=3,loss_sum=12.)])
        result=aggregate(synthetic,[2,0,1],[dict(source='wikipedia'),dict(source='fineweb_edu'),dict(source='fineweb_edu')])
        assert result['count']==6 and result['loss_sum']==19. and result['sources']['fineweb_edu']['count']==5 and result['sources']['wikipedia']['ce']==3.
        generations=read(a/'generations/result.json');assert len(generations['rows'])==24
        for row in generations['rows']:
            events=[json.loads(x) for x in (a/'generations'/f"{row['id']}.jsonl").read_text().splitlines()]
            assert events[0]['event']=='start' and events[-1]['event']=='complete' and events[-1]['row']==row
            assert [e['token'] for e in events if e['event']=='token']==row['generated_token_ids']
        receipts=r['restart_receipts']+list(r['endpoint_receipts'].values())
        for receipt in receipts:assert Path(receipt['path']).stat().st_size==receipt['size'] and sha(receipt['path'])==receipt['sha256']
        assert sum(x['size'] for x in receipts)==r['retained_bytes']
        old=ast.parse((ROOT/'research/general_rsi_native_horizon_worker_v1.py').read_text());new=ast.parse((ROOT/'research/general_rsi_context_length_worker_v1.py').read_text())
        unchanged=[]
        for name in ['put','replace','stopped','boot','retire','remaining_guard_seconds','main']:
            x=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name);y=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name);assert ast.dump(x)==ast.dump(y);unchanged.append(name)
        review=dict(accepted=True,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(OUT/'plan.json'),result_sha256=sha(a/'result.json'),source_bindings_unchanged=len(p['sources']),review_source_sha256=sha(__file__),aggregation_source_sha256=sha(ROOT/'research/general_rsi_context_length_score_v1.py'),worker_source_sha256=sha(ROOT/'research/general_rsi_context_length_worker_v1.py'),unchanged_worker_functions=unchanged,aggregation_checks=3,full_active_ema_replay_exact=True,first_outputs_verified=24,per_token_oracle_max_abs=r['per_token_ce_max_abs_vs_fp64'],guard_seconds=g['elapsed_seconds'],peak_rss_bytes=g['observation']['peak_rss_bytes'],retained_bytes=r['retained_bytes'],worker_runtime_admitted=False,next_action='Freeze fresh paired roles, exact quality protocol and full measured resource admission, then use the qualified context/evaluator/generator in the inherited serial worker. No completed prerequisite reruns.',free_bytes=shutil.disk_usage(ROOT).free,test_opened=False,validation_opened=False,quality_adopted=False)
        put(OUT/'completion-review.json',review);print(json.dumps(review,indent=2))
if __name__=='__main__':main()
