"""New downstream audit and metadata mirror; never reruns training or acceptance."""
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.tokenizer_moe_worker_v1.execution import sources,idle
from research.tokenizer_moe_resume_v1.contract import runtime
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

E=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def ref(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=sha(p),size=p.stat().st_size)


def read(p):return json.loads(Path(p).read_text())


def checked(v):
    p=Path(v['path']);require(sha(p)==v['sha256'],'Original evidence hash differs')
    return read(p)


def publish(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<1024**2,'OneMiB audit receipt cap')
    with (E/name).open('xb') as f:f.write(raw)


def main():
    torch.set_num_threads(4)
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);idle();env=runtime()
        require(not (E/'prerequisite-close001.json').exists() and not (E/'complete-metadata-mirror001').exists(),'New single closure/mirror only')
        pins=sources();require(len(pins)==1025,'Exact passed six-file worker binding required')
        with CpuPrerequisiteGuard(E/'prerequisite-close001-guard.json',ROOT):
            weights=[];metadata=[];acceptances=[];routing=[]
            roots=[('worker',ROOT/'research_runs/tokenizer-moe-worker-v1-001','worker001-review.json','three_arm_prerequisite_complete',4,3),
                   ('pilot',ROOT/'research_runs/tokenizer-moe-pilot-v1-001','pilot001-review.json','three_arm_resource_pilot_complete',20,8)]
            for stage,root,review_name,stop_status,steps,snapshots in roots:
                stop=read(root/'STOP');require(stop['status']==stop_status and stop['never_rerun'] is True,'Original terminal STOP required')
                review=checked(stop['review']);require(stop['review']['path']==str(E/review_name) and len(review['rows'])==3 and checked(review['plan'])['sources']==pins,'Original complete review/source plan differs')
                retained=0;metadata.append((stage,root,root/'STOP'))
                for i,arm in enumerate(['dense','wide','moe']):
                    slot=root/f'evidence/pilots/{i:02d}';accepted=read(slot/'accepted.json');require(accepted==review['rows'][i] and accepted['arm']==arm,'Original accepted row differs')
                    result=accepted['acceptance'];require(result['status']=='accepted' and result['step']==steps and result['targets']==steps*1024 and result['ema_snapshots']==snapshots and result['complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked'],'Complete original acceptance required')
                    for name in ['worker_guard','parent_guard']:
                        g=checked(accepted[name]);require(g['status']=='finished' and g['reason'] is None,'Original resource guard incomplete')
                    process=checked(accepted['process']);require(process['status']=='completed' and process['returncode']==0,'Original child process incomplete')
                    summary=checked(accepted['worker'])
                    for endpoint in ['parent','h200']:
                        v=summary[endpoint];p=Path(v['path']);before=p.stat()
                        require(p.is_file() and not p.is_symlink() and before.st_nlink==1 and before.st_size==v['size'] and sha(p)==v['sha256'],'Exact retained tensor identity differs')
                        after=p.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'Retained tensor changed during hashing')
                        weights.append(dict(stage=stage,arm=arm,kind=endpoint,path=str(p.resolve()),sha256=v['sha256'],size=v['size']));retained+=v['size']
                    for endpoint in ['raw','h200']:
                        reports=result['full_validation'][endpoint]
                        require(reports['cpu_mps_ce_difference']<=3e-5 and reports['cpu']['target']['targets']==reports['mps']['target']['targets']==34278
                                and reports['cpu']['aux_loss_included'] is reports['mps']['aux_loss_included'] is False,'Complete native CE-only scoring differs')
                    acceptances.append(ref(slot/'accepted.json'))
                    metadata.extend((stage,root,p) for p in slot.rglob('*') if p.is_file() and 'checkpoints' not in p.parts)
                    metadata.extend((stage,root,root/'control'/f'{name}-pilots-{i:02d}{suffix}') for name,suffix in [('worker','-guard.json'),('parent','-guard.json'),('process','.json')])
                    if stage=='pilot' and arm=='moe':
                        require(all(sum(c)==40960 for c in summary['route_counts']),'All20updates cumulative top2 assignment coverage required')
                        for layer,counts in enumerate(summary['route_counts']):
                            fractions=[n/sum(counts) for n in counts]
                            routing.append(dict(layer=layer,counts=counts,fractions=fractions,unused_experts=sum(n==0 for n in counts),
                                                cumulative_assignment_entropy_nats=-math.fsum(p*math.log(p) for p in fractions if p),
                                                scope='Entropy of cumulative hard-assignment fractions,not mean softmax entropy or semantic specialization.'))
                        require(all(v==20 for v in summary['parameter_gradient_steps'].values()),'All pilot parameters actually updated each step in this fixture')
                require(retained==review['retained_weight_bytes'],'Actual stage retained-byte total differs')
            require(len(weights)==12 and len({v['sha256'] for v in weights})==12,'Exact twelve distinct retained prerequisites required')
            mirror=E/'complete-metadata-mirror001';mirror.mkdir();copies=[]
            for stage,root,p in metadata:
                require(p.is_file() and not p.is_symlink() and p.suffix not in ('.pt','.tmp'),'Metadata-only mirror required')
                target=mirror/stage/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:f.write(p.read_bytes())
                require(sha(p)==sha(target),'Metadata mirror changed bytes');copies.append(dict(source=ref(p),copy=ref(target)))
            require(len(copies)==68,'Exactly six closed endpoints×11metadata plus twoSTOPs expected')
            publish('complete-metadata-mirror001.json',dict(files=copies,count=len(copies),weight_copies=0))
            pilot=read(E/'pilot001-review.json');require(len(pilot['projections'])==3 and all(v['conservative1200_seconds']<7200 for v in pilot['projections']),'Original complete resource projections required')
            # All eighteen future outputs retained; no future loser retirement.
            new_weights=6*(560+640+640)*1024**2
            floor=new_weights+256*1024**2+64*1024**2+8*1024**3+1267904512+1458176
            draft=dict(status='prospective_all18output_forecast_not_scientific_freeze',maximum_new_weight_bytes=new_weights,complete_floor=floor,cleanup_target=floor+2*1024**3,
                       free_bytes=shutil.disk_usage(ROOT).free,metadata=256*1024**2,sidecars=64*1024**2,reserve=8*1024**3,external_headroom=1267904512,entry_headroom=1458176,
                       keep_ALL9screens_and9fresh=True,prior_prerequisite_or_future_deletion_credit=0,
                       condition='Actual full1200history serialization/source/state/selection/statistical contracts and scientificseedcensus still required. Atomic temporary must remain within560/640/640MiB endpoint caps.')
            require(draft['free_bytes']>=draft['cleanup_target'],'New complete prospective budget does not fit')
            publish('quality-draft-capacity001.json',draft)
            require(sources()==pins,'Immutable worker/source predecessor changed')
        result=dict(status='closed_worker_and_resource_pilot_dependencies_verified_no_replay',source=ref(__file__),guard=ref(E/'prerequisite-close001-guard.json'),sources=pins,total_source_bindings=1025,
                    acceptances=acceptances,protected_payloads=weights,retained_weight_bytes=sum(v['size'] for v in weights),metadata_mirror=ref(E/'complete-metadata-mirror001.json'),
                    pilot_projections=pilot['projections'],cumulative_moe_routing=routing,draft_quality_capacity=ref(E/'quality-draft-capacity001.json'),
                    no_training_or_model_scoring=True,scientific_quality=False,new_scientific_seed=False,deletion=False,weight_copies=0,
                    next_action='Additive full1200scientific worker/controller and independent paired-statistics/selection contracts;actual pilot-layout serialization and full LIVE budget/unusedseedcensus before science. No automatic grid/rescue/old score pooling.')
        publish('prerequisite-close001.json',result)
        print(json.dumps(dict(status=result['status'],protected_payloads=len(weights),retained_weight_bytes=result['retained_weight_bytes'],metadata_files=len(copies),weight_copies=0,draft_quality_floor=floor)))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if not (E/'STOP-prerequisite-close001').exists():publish('STOP-prerequisite-close001',dict(error=repr(exc),closed=True,never_retry=True))
        raise
