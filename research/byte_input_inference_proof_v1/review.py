"""Independent saved tensor/CE/coverage review; imports no candidate model."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
SOURCE=BASE/'inference-math001'
OUT=BASE/'inference-review001'

def read(p):return json.loads(p.read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while x:=f.read(1048576):h.update(x)
    return h.hexdigest()
def check(r):
    p=Path(r['path']);assert p.is_file() and not p.is_symlink() and p.stat().st_nlink==1
    assert p.stat().st_size==r['size'] and sha(p)==r['sha256']
def publish(n,x):
    with (OUT/n).open('x') as f:json.dump(x,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def run():
    import numpy as np
    import torch
    torch.set_num_threads(4)
    result=read(SOURCE/'result.json');plan=read(SOURCE/'plan.json')
    assert result['accepted'] and result['updates_executed']==0 and len(result['cases'])==6
    assert result['invalid_transactions']==78 and result['full_validation_targets']==205668
    assert read(SOURCE/'STOP')['status']=='complete_original_inference_consumers'
    assert read(SOURCE/'guard.json')['status']=='finished'
    common=None;cases=[];coordinates=0
    for row in result['cases']:
        prefix=row['backend']+'-'+row['variant'];loaded=read(SOURCE/(prefix+'-loaded.json'))
        score=read(SOURCE/(prefix+'-scores.json'));report=score['report']
        assert loaded['exact_parent_ema'] and loaded['exact_loaded_logits'] and loaded['caller_rng_preserved']
        assert len(loaded['invalid_rejections'])==len(set(loaded['invalid_rejections']))==13
        assert row['weights_unchanged'] and row['caller_rng_preserved'] and row['no_training_updates']
        parent=loaded['parent'];export=loaded['export'];check(parent);check(export);check(row['oracle'])
        raw=torch.load(parent['path'],map_location='cpu',weights_only=True)
        saved=torch.load(export['path'],map_location='cpu',weights_only=True)
        assert saved['identity']==raw['identity'] and saved['parent']==parent and saved['completed_steps']==4
        assert saved['averaging']==dict(half_life=200,spec=raw['averages']['spec'],snapshots=3)
        panel=raw['averages']['averages']['200'];assert list(saved['parameters'])==list(panel)
        for n,t in panel.items():
            x=saved['parameters'][n]
            assert t.dtype==x.dtype==torch.float32 and t.shape==x.shape and torch.equal(t,x)
            assert hashlib.sha256(x.numpy().tobytes()).hexdigest()==loaded['weights'][n]
            coordinates+=t.numel()
        del raw,saved,panel,t,x
        target=report['target'];assert target['targets']==34278 and target['validation_tokens']==34279
        assert target['window_counts']==[128]*267+[102] and target['block_counts']==[2048]*16+[1510]
        assert target['context']==128 and target['first_target_offset']==1 and target['tail_retained'] and not target['holdout_scored']
        assert common is None or target==common
        common=target
        original=plan['training_plan']['assets']
        original_data=read(Path(next(r['path'] for r in original if r['path'].endswith('quality002-plan.json'))))['data']
        assert target['ids_sha256']==original_data['validation_ids_sha256'] and target['target_ids_sha256']==original_data['validation_target_ids_sha256']
        assert target['tokenizer']==original_data['tokenizer'] and report['binding']==export
        losses=np.asarray(score['per_target_losses'],dtype='float64');assert losses.shape==(34278,) and np.isfinite(losses).all() and (losses>=0).all()
        for key,width in [('window_loss_sums',128),('block_loss_sums',2048)]:
            sums=[math.fsum(losses[i:i+width]) for i in range(0,len(losses),width)]
            assert len(sums)==len(report[key]) and max(abs(a-b) for a,b in zip(sums,report[key]))<1e-8
        total=math.fsum(losses);assert abs(total-report['total_loss'])<1e-8
        assert abs(total/34278-report['ce'])<1e-12
        assert abs(total/math.log(2)/target['target_bytes']-report['bits_per_byte'])<1e-12
        errors=[]
        with np.load(row['oracle']['path']) as arrays:
            for start,count in [(0,128),(2048,128),(34176,102)]:
                logits=arrays[str(start)];labels=arrays['labels'+str(start)]
                assert logits.shape==(1,count,50257) and logits.dtype==np.float32 and labels.shape==(count,)
                v=logits[0].astype('float64');h=v.max(axis=1)
                wanted=h+np.log(np.exp(v-h[:,None]).sum(axis=1))-v[np.arange(count),labels]
                error=np.abs(losses[start:start+count]-wanted)
                assert np.all(error<=4e-6+4e-5*np.abs(wanted))
                errors.append(dict(start=start,count=count,max_absolute_error=float(error.max())))
        cases.append(dict(backend=row['backend'],variant=row['variant'],exact_parent_ema=True,independent_ce=errors,
                          full_target_aggregation=True))
    # Byte accounting is independently reproduced from the original validation
    # text, not inferred from the scorer's reported denominator.
    import tiktoken
    corpus=next(r for r in plan['assets'] if r['path'].endswith('data/raw/shakespeare.txt'));check(corpus)
    text=Path(corpus['path']).read_text();enc=tiktoken.get_encoding('gpt2')
    ids=enc.encode(text[int(.8*len(text)):int((.8+.1)*len(text))],disallowed_special=())
    assert len(ids)==34279 and sum(len(enc.decode_single_token_bytes(i)) for i in ids[1:])==common['target_bytes']
    for row in result['cases']:
        with np.load(row['oracle']['path']) as arrays:
            for start,count in [(0,128),(2048,128),(34176,102)]:
                assert arrays['labels'+str(start)].tolist()==ids[start+1:start+count+1]
    for r in result['exports']+result['oracles']+plan['parents']:check(r)
    publish('review.json',dict(accepted=True,cases=cases,exact_export_coordinates=coordinates,
        independently_checked_ce_targets=6*358,full_validation_targets=205668,invalid_transactions=78,
        exports=result['exports'],oracles=result['oracles'],updates_executed=0,
        scope='Exact parent EMA export/load evidence; full validation target coverage and aggregation; FP64 CE oracle on first, block-boundary and tail windows. No quality-selection, cached-inference or worker qualification.'))

def main():
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(OUT.iterdir()),'Original once-only saved review'
        publish('attempt.json',dict(pid=os.getpid(),no_replay=True))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=300,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3):
                assert sys.flags.optimize==0
                plan=read(SOURCE/'plan.json')
                for r in plan['sources']:
                    p=Path(r['path']);assert p.stat().st_size==r['size'] and sha(p)==r['sha256']
                run()
            publish('STOP',dict(status='complete_original_inference_saved_review',no_replay=True))
        except BaseException as exc:
            publish('STOP',dict(status='closed_failed_original_inference_saved_review',error=repr(exc),no_replay=True));raise

if __name__=='__main__':main()
