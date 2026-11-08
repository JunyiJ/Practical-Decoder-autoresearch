"""Saved resource-pilot evidence acceptance; no training or quality ranking."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
SOURCE=BASE/'pilot-math001'
OUT=BASE/'pilot-review001'
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while x:=f.read(1048576):h.update(x)
    return h.hexdigest()
def publish(n,x):
    with (OUT/n).open('x') as f:json.dump(x,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def check(r):
    p=Path(r['path']);assert not p.is_symlink() and p.stat().st_nlink==1 and p.stat().st_size==r['size'] and sha(p)==r['sha256']

def run(plan):
    import torch
    torch.set_num_threads(4)
    result=read(SOURCE/'result.json');guard=read(SOURCE/'guard.json')
    assert result['accepted'] and result['updates_executed']==60 and guard['status']=='finished'
    assert read(SOURCE/'STOP')['status']=='complete_original_resource_pilot'
    assert [r['variant'] for r in result['arms']]==['native','raw_bytes','nonlexical_codes']
    costs=[];artifacts=[]
    for row in result['arms']:
        arm=row['variant'];native=arm=='native';count=88 if native else 105
        parent=row['parent'];h200=row['h200'];check(parent);check(h200);artifacts.extend((parent,h200))
        raw=torch.load(parent['path'],map_location='cpu',weights_only=True)
        avg=torch.load(h200['path'],map_location='cpu',weights_only=True)
        identity=raw['identity'];numerical=raw['numerical'];names=list(numerical['parameters'])
        assert raw['kind']=='byte_input_training_context_v2' and avg['kind']=='byte_input_h200_inference_v1'
        assert identity['roles']==plan['roles'] and identity['binding']==plan['binding']
        assert identity['recipe']['max_iters']==20 and identity['recipe']['learning_rate']==.00125
        assert numerical['completed_steps']==20 and numerical['backend']=='mps'
        assert len(names)==count and len(raw['history'])==20 and len(row['rows'])==20
        for i,(history,observed) in enumerate(zip(raw['history'],row['rows']),1):
            assert history=={k:observed[k] for k in history}
            assert history['step']==i and history['active_parameter_names']==names and history['gradient_tensors']==count
            assert history['threshold']==1. and history['post_norm']<=1.+1e-5
            assert 0<=history['coefficient']<=1. and math.isfinite(history['ce_loss']) and history['ce_loss']>=0
            assert observed['elapsed_seconds']>0 and observed['mps_current_bytes']<=observed['mps_driver_bytes']<6*1024**3
            assert read(SOURCE/(arm+'-step'+str(i).zfill(2)+'.json'))==observed
        for child in numerical['children']:
            assert list(child['state'])==[g['name'] for g in child['groups']]
            for fields in child['state'].values():
                for name,t in fields.items():
                    assert bool(torch.isfinite(t).all())
                    if name=='step':assert t.item()==20
        saved=raw['averages'];assert saved['completed_steps']==20 and saved['snapshots']==8
        assert saved['spec']['start_step']==13 and saved['spec']['interval']==1 and saved['spec']['half_lives']==[200]
        assert avg['parent']==parent and avg['identity']==identity and avg['completed_steps']==20
        assert avg['averaging']==dict(half_life=200,spec=saved['spec'],snapshots=8)
        assert list(avg['parameters'])==names
        for n,t in avg['parameters'].items():
            assert t.dtype==torch.float32 and torch.equal(t,saved['averages']['200'][n])
        for name,gstate in raw['samplers'].items():
            assert torch.Generator(device='cpu').set_state(gstate).initial_seed()==plan['roles']['train_sampler' if name=='train' else 'validation_sampler']
        scores=read(SOURCE/(arm+'-validation.json'));report=scores['report'];loss=scores['per_target_losses']
        assert len(loss)==34278 and report['target']['targets']==34278 and report['target']['window_counts']==[128]*267+[102]
        assert report['target']['ids_sha256']==identity['data']['validation_ids_sha256']
        assert report['target']['target_ids_sha256']==identity['data']['validation_target_ids_sha256']
        assert report['binding']==h200 and not report['target']['holdout_scored']
        assert all(type(x) is float and math.isfinite(x) and x>=0 for x in loss)
        assert abs(math.fsum(loss)-report['total_loss'])<1e-8
        for field,width in [('window_loss_sums',128),('block_loss_sums',2048)]:
            expected=[math.fsum(loss[i:i+width]) for i in range(0,len(loss),width)]
            assert len(expected)==len(report[field]) and all(abs(a-b)<1e-8 for a,b in zip(expected,report[field]))
        assert report['ce']==report['total_loss']/34278
        assert report['bits_per_byte']==report['total_loss']/math.log(2)/report['target']['target_bytes']
        assert row['completed_steps']==20 and row['targets_seen']==20480 and not row['quality_selected']
        assert [r['batch'] for r in row['cached_correctness']]==[1,8]
        for r in row['cached_correctness']:
            assert r['cache_bytes']==r['batch']*196608 and 0<=r['max_gate_ratio']<=1 and 0<=r['max_absolute_error']<math.inf
        times=[r['elapsed_seconds'] for r in row['rows']]
        assert row['max_update_seconds']==max(times) and row['total_update_seconds']==sum(times) and row['mean_update_seconds']==sum(times)/20
        costs.append(dict(variant=arm,mean_update_seconds=row['mean_update_seconds'],max_update_seconds=max(times),
            total_update_seconds=sum(times),validation_seconds=row['validation_seconds'],
            maximum_observed_step_mps_driver_bytes=row['max_mps_driver_bytes']))
        del raw,avg,numerical,saved
    for r in artifacts:check(r)
    publish('review.json',dict(accepted=True,updates=60,costs=costs,artifacts=artifacts,
        guard=guard,quality_selected=False,no_1200_step_launch=True,
        scope='Actual twenty-update resource pilots with complete RAW/H200 ancestry, clipping/exposure, saved full-validation aggregation and cached-correctness receipts. Scores are diagnostic only; no fresh quality evidence.'))

def main():
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not list(OUT.iterdir())
        publish('attempt.json',dict(pid=os.getpid(),no_replay=True))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=300,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3):
                plan=read(SOURCE/'plan.json')
                for r in plan['sources']:
                    p=Path(r['path']);assert p.stat().st_size==r['size'] and sha(p)==r['sha256']
                run(plan)
            publish('STOP',dict(status='complete_original_resource_pilot_review',no_replay=True))
        except BaseException as exc:
            publish('STOP',dict(status='closed_failed_original_resource_pilot_review',error=repr(exc),no_replay=True));raise

if __name__=='__main__':main()
