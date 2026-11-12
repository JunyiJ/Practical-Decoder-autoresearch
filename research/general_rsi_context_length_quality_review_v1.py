"""Audit saved context-panel evidence without model execution or scientific reselection."""
import argparse,datetime,fcntl,hashlib,json,math,shutil
from pathlib import Path
import numpy as np
from research import general_rsi_native_horizon_review_v1 as inherited
from research.general_rsi_training_v2 import sha,read_checkpoint
from research.general_rsi_training_clock_v1 import TrainingClock
from research.general_rsi_generation_metrics_v1 import score as generation_metrics
from research.general_rsi_context_length_worker_v1 import put
ROOT=Path(__file__).resolve().parents[1]
PANEL=ROOT/'research/tokenizer-evidence/general-rsi-context-length-quality-001'
def read(p):return json.loads(p.read_text())
def audit_scores(path,indices,context,documents,values):
    s=read(path);assert s['context']==context
    expected=[]
    for i in indices:
        d=documents[i]
        for offset in range(0,d['tokens']-1,context):
            start=d['offset']+offset+1;count=min(context,d['tokens']-1-offset)
            expected.append((i,d['source'],start,count))
    assert len(expected)==len(s['chunks'])
    for r,(i,source,start,count) in zip(s['chunks'],expected):
        assert (r['document'],r['source'],r['target_start'],r['targets'])==(i,source,start,count)
        assert r['target_ids']==values[start:start+count].tolist()
        assert r['input_sha256']==hashlib.sha256(values[start-1:start+count-1].tobytes()).hexdigest()
        assert len(r['token_losses'])==count and all(math.isfinite(v) and v>=0 for v in r['token_losses'])
        assert r['loss_sum']==math.fsum(r['token_losses'])
    assert [r['document'] for r in s['documents']]==indices
    for d in s['documents']:
        rows=[r for r in s['chunks'] if r['document']==d['document']]
        assert d['source']==documents[d['document']]['source']
        assert d['count']==sum(r['targets'] for r in rows)==documents[d['document']]['tokens']-1
        assert d['loss_sum']==math.fsum(r['loss_sum'] for r in rows)
    for source,aggregate in s['sources'].items():
        docs=[d for d in s['documents'] if d['source']==source]
        assert aggregate['count']==sum(d['count'] for d in docs)
        assert aggregate['loss_sum']==math.fsum(d['loss_sum'] for d in docs)
        assert aggregate['ce']==aggregate['loss_sum']/aggregate['count']
    assert set(s['sources'])=={d['source'] for d in s['documents']}
    assert s['count']==sum(d['count'] for d in s['documents'])
    assert s['loss_sum']==math.fsum(d['loss_sum'] for d in s['documents'])
    assert s['ce']==s['loss_sum']/s['count']
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),targets=s['count'],chunks=len(expected),context=context)
def review(slot):
    import torch,tiktoken
    plan=read(PANEL/'plan.json');out=PANEL/f'slot-{slot:02d}';result=read(out/'result.json');binding=sha(PANEL/'plan.json')
    # Reuse the unchanged clock, receipt, paired-stream and first-journal audit.
    old=inherited.PANEL
    try:
        inherited.PANEL=PANEL;r=inherited.review(slot)
    finally:inherited.PANEL=old
    assert result['test_opened'] is False
    steps={}
    for p in sorted(out.glob('steps-*.jsonl')):
        for line in p.read_text().splitlines():
            row=json.loads(line);steps[row['step']]=row
    for row in steps.values():
        assert row['lr_ratio']==TrainingClock(result['slot']['budget_ns'],row['begin_ns']).lr_ratio()
        assert row['clip']['gradient_tensors']==88 and row['clip']['threshold']==1.0
        assert math.isfinite(row['clip']['pre_norm']) and math.isfinite(row['clip']['post_norm'])
    assert result['snapshots']==sum(row['begin_ns']>=(2*result['slot']['budget_ns']+2)//3 for row in steps.values())
    data=ROOT/'data/raw/rsi-general-10m-v1';manifest=read(data/'manifest.json')
    assert sha(data/'manifest.json')=='b1849a5fed1fb648f507e407e5794946bba64328f519e6b0d6c360fa4e2e4fa8'
    for name in ('val.bin','val.index.jsonl'):assert sha(data/name)==manifest['files'][name]['sha256']
    docs=[json.loads(x) for x in (data/'val.index.jsonl').read_text().splitlines()]
    values=np.fromfile(data/'val.bin',dtype='<u2').astype(np.int64);length=result['slot']['context'];scores=[]
    for e in result['endpoints']:
        payload=read_checkpoint(e['weights'])
        assert payload['kind']=='context_length_inference_v1' and payload['endpoint']==e['endpoint']
        assert payload['config']['binding']==binding and payload['config']['roles']==result['slot']['roles']
        assert payload['config']['context']==length and payload['config']['budget_ns']==result['slot']['budget_ns']
        assert payload['identity']==('general_rsi_context_length_v1','native',256,50257,length,'uncached')
        assert payload['step']==result['steps'] and payload['clock']['spent_ns']==result['training_ns']
        assert len(payload['parameters'])==88 and sum(t.numel() for t in payload['parameters'].values())==28492416
        assert all(t.device.type=='cpu' and t.dtype==torch.float32 and torch.isfinite(t).all() for t in payload['parameters'].values())
        del payload
        scores.append(audit_scores(out/f"{e['endpoint']}-validation.json",list(range(108)),length,docs,values))
        if length==256:scores.append(audit_scores(out/f"{e['endpoint']}-common128-validation.json",list(range(108)),128,docs,values))
    for fraction in (1,2):scores.append(audit_scores(out/f'probe-{fraction}.json',plan['probe_documents'],length,docs,values))
    gen=read(out/'generation/result.json');protocol=plan['generation_protocol'];enc=tiktoken.get_encoding('gpt2')
    assert gen['binding']==binding
    expected=[(d,p) for d in protocol['decoders'] for p in protocol['prompts']]
    for row,(decoder,prompt) in zip(gen['rows'],expected):
        assert row['id']==prompt['id']+'-'+decoder and row['prompt']==prompt and row['decoder']==decoder and row['seed']==42
        assert row['checkpoint_sha256']==gen['checkpoint']['sha256']
        assert row['metrics']==generation_metrics(row['generated_token_ids']) and not row['metrics']['unexpected_early_stop']
        assert row['continuation']==enc.decode(row['generated_token_ids'])
        assert row['stop_reason']==('eos' if row['metrics']['stopped_eos'] else 'length_limit')
        first=json.loads((out/'generation'/f"{row['id']}.jsonl").read_text().splitlines()[0])
        assert first['binding']==binding and first['context']==length and first['inference']=='uncached_full_prefix'
        assert first['generator_source_sha256']==sha(ROOT/'research/general_rsi_context_length_generation_v1.py')
    journal=[json.loads(x) for x in (out/'deletion-journal.jsonl').read_text().splitlines()]
    assert len(journal)%2==0
    for before,after in zip(journal[::2],journal[1::2]):
        assert before['event']=='before_unlink' and after['event']=='unlinked'
        assert {k:v for k,v in before.items() if k!='event'}=={k:v for k,v in after.items() if k!='event'}
        assert before['no_open_consumers'] and not Path(before['receipt']['path']).exists()
    assert read(out/'latest.json')['status']=='retired_after_completion'
    assert not list((ROOT/'research_runs/general-rsi-context-length-quality-001'/f'slot-{slot:02d}').glob('restart-*'))
    r.update(review_source_sha256=sha(__file__),reviewed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),score_audits=scores,finite_cpu_endpoints=2,first_outputs=24,exact_transient_retirements=len(journal)//2,schedule_and_ema_count_verified=True,source_bindings_unchanged=len(plan['sources']),free_bytes=shutil.disk_usage(ROOT).free,quality_decision='Pending complete three-pair panel; no adoption from this slot.')
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('slot',type=int);a=p.parse_args()
    with open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        r=review(a.slot);put(PANEL/f'slot-{a.slot:02d}'/'review.json',r)
    print(json.dumps({k:v for k,v in r.items() if k!='batch_prefix'},indent=2))
if __name__=='__main__':main()
