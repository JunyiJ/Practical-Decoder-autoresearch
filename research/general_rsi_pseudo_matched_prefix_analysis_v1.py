"""Post-hoc natural-batch comparison using immutable completed logs only."""
import json,hashlib,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/tokenizer-evidence/general-rsi-pseudo-repetition-001'
OUT=ROOT/'research/tokenizer-evidence/general-rsi-pseudo-mechanism-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def stats(rows):
    d=[x['delta'] for x in rows]
    return dict(count=len(d),mean=statistics.mean(d),median=statistics.median(d),positive_count=sum(x>0 for x in d),min=min(d),max=max(d))
def main():
    started=time.monotonic();plan=read(P/'plan.json');pairs=[];bindings={}
    for i in range(3):
        logs=[];allsteps=[]
        for slot in [2*i,2*i+1]:
            d=P/f'slot-{slot:02d}';v=read(d/'review.json');assert v['accepted_execution'] and v['result_sha256']==sha(d/'result.json')
            assert v['binding']==sha(P/'plan.json')
            f=d/'steps-01.jsonl';steps=[json.loads(x) for x in f.read_text().splitlines()];allsteps.append(steps);logs.append([x for x in steps if x['objective']=='ce'])
            for n in ['result.json','review.json','steps-01.jsonl']:bindings[str((d/n).relative_to(ROOT))]=sha(d/n)
        a,b=logs;aligned=[]
        for k,(x,y) in enumerate(zip(a,b),1):
            assert x['batch_sha256']==y['batch_sha256'] and x['valid_loss_targets']==y['valid_loss_targets']==1024
            aligned.append(dict(natural_ordinal=k,control_step=x['step'],candidate_step=y['step'],control_loss=x['loss'],candidate_loss=y['loss'],delta=y['loss']-x['loss'],control_lr_ratio=x['lr_ratio'],candidate_lr_ratio=y['lr_ratio'],same_constant_lr=x['lr_ratio']==y['lr_ratio']==1.0))
        constant=[x for x in aligned if x['same_constant_lr']];assert [x['natural_ordinal'] for x in constant]==list(range(1,len(constant)+1))
        blocks=[]
        for lo in range(0,len(constant),100):
            z=constant[lo:lo+100];blocks.append(dict(first=z[0]['natural_ordinal'],last=z[-1]['natural_ordinal'],**stats(z)))
        gradient={}
        for objective in ['ce','pseudo']:
            z=[x for x in allsteps[1] if x['objective']==objective]
            gradient[objective]=dict(count=len(z),pre_clip_norm_median=statistics.median(x['clip']['pre_norm'] for x in z),post_clip_norm_median=statistics.median(x['clip']['post_norm'] for x in z),clipped_count=sum(x['clip']['coefficient']<1 for x in z))
        pairs.append(dict(bundle=i+1,shared_prefix_count=len(aligned),constant_lr=stats(constant),constant_lr_blocks=blocks,first_natural_loss_delta=aligned[0]['delta'],all_shared_prefix=stats(aligned),candidate_gradient_summary=gradient,aligned=aligned))
    OUT.mkdir(exist_ok=True)
    r=dict(source_sha256=sha(Path(__file__)),plan_sha256=sha(P/'plan.json'),input_sha256=bindings,pairs=pairs,elapsed_seconds=time.monotonic()-started,scope='Post-hoc training-loss diagnostic on identical natural batches at equal natural-update ordinal. Constant-LR window includes every shared-prefix row with both ratios exactly1, before either cooldown. Full-prefix summaries include mismatched schedules and are descriptive only. No model loaded, optimizer executed, validation rescore, fresh confirmation or gate change.',test_opened=False)
    with (OUT/'matched-prefix.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps([{k:v for k,v in x.items() if k not in ['aligned','constant_lr_blocks']} for x in pairs],indent=2))
if __name__=='__main__':main()
