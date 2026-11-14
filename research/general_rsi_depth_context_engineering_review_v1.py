"""Read-only audit of the completed new combined-shape qualification."""
import json,math,datetime
from pathlib import Path
from research.general_rsi_training_v2 import sha,read_checkpoint
from research.general_rsi_pseudo_update_diagnostic_v1 import put,digest
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-depth-context-integration-003'
def read(p):return json.loads(p.read_text())
def review():
    p=read(OUT/'plan.json');binding=sha(OUT/'plan.json');a=OUT/'attempt-01';r=read(a/'result.json')
    assert r['accepted'] and r['plan_sha256']==binding and all(sha(ROOT/f)==h for f,h in p['sources'].items())
    assert [s['shape'] for s in r['summaries']]==p['slots']
    checks=[]
    for s in r['summaries']:
        d=s['shape']['depth'];c=s['shape']['context'];out=a/f'd{d}-c{c}';guard=read(out/'guard.json');rr=read(out/'review.json')
        assert rr==s and guard['status']=='finished' and guard['reason'] is None
        assert guard['observation']['peak_rss_bytes']<p['rss_limit_bytes'] and s['peak_mps_bytes']<p['mps_limit_bytes']
        assert s['full_active_time_ema_restart_replay_exact'] and s['wrong_shape_restore_and_endpoint_rejected'] and s['endpoint_parameters_exact']
        assert s['scoring_targets_exact'] and s['ce_max_absolute_error']<=p['ce_atol']
        assert s['state_unchanged_after_scoring_generation'] and s['first_outputs']==24 and not s['test_opened'] and not s['validation_opened']
        for e,receipt in s['endpoint_receipts'].items():
            payload=read_checkpoint(receipt)
            assert payload['kind']=='general_rsi_depth_context_inference_v1' and payload['endpoint']==e
            assert payload['config']['binding']==binding and payload['config']['depth']==d and payload['config']['context']==c
            assert len(payload['parameters'])==21*d+4 and sum(t.numel() for t in payload['parameters'].values())==25732096+690080*d
            assert payload['step']==s['checkpoint_step']+1;del payload
        gen=read(out/'generations/result.json');assert len(gen['rows'])==24 and gen['binding']==binding
        for row in gen['rows']:
            log=[json.loads(x) for x in (out/'generations'/f"{row['id']}.jsonl").read_text().splitlines()]
            assert log[-1]==dict(event='complete',row=row) and [x['token'] for x in log[1:-1]]==row['generated_token_ids']
        original=read(out/'reference-next.jsonl');replayed=read(out/'replayed-next.jsonl');assert original==replayed
        memory=[json.loads(x) for x in (out/'memory.jsonl').read_text().splitlines()]
        assert max(x['mps_driver_bytes'] for x in memory)==s['peak_mps_bytes']
        checks.append(dict(depth=d,context=c,peak_rss_bytes=guard['observation']['peak_rss_bytes'],peak_mps_bytes=s['peak_mps_bytes'],guard_seconds=guard['elapsed_seconds'],phases=[x['phase'] for x in memory if not x['phase'].startswith('training')],pressure_values=sorted(set(x['host']['pressure'].strip() for x in memory)),endpoint_bytes={k:v['size'] for k,v in s['endpoint_receipts'].items()},max_restart_bytes=max(x['size'] for x in s['restart_receipts']),steady_median_seconds=s['steady_median_seconds']))
    logs=[json.loads(x) for x in (a/'deletion-journal.jsonl').read_text().splitlines()];assert len(logs)==16
    for before,after in zip(logs[::2],logs[1::2]):
        assert before['event']=='intent' and after['event']=='deleted'
        assert {k:v for k,v in before.items() if k!='event'}=={k:v for k,v in after.items() if k!='event'}
        assert not Path(before['path']).exists()
    assert r['paired_prefix_batch_streams_exact'] and r['retained_bytes']<=p['retained_inference_cap_bytes']
    result=dict(accepted=True,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=binding,result_sha256=sha(a/'result.json'),review_source_sha256=sha(__file__),shape_checks=checks,retained_endpoint_bytes=r['retained_bytes'],first_outputs=96,exact_transient_retirements=8,original_failure='research/tokenizer-evidence/general-rsi-depth-context-integration-002/attempt-01/failure.json',diagnosis='research/tokenizer-evidence/general-rsi-depth-context-integration-002/attempt-01/rng-diagnosis.json',scope='New combined wrapper qualified. Full scientific resource admission and frozen12-run protocol still required.',test_opened=False)
    put(OUT/'completion-review.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':review()
