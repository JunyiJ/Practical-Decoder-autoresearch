"""One-time original finite FFN chain metadata/source/stream-hash closure; no Torch."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

HERE=Path(__file__).resolve().parent
E=HERE.parent
ROOT=HERE.parents[3]
CAP=8*1024**2

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024**2),b''):h.update(part)
    return h.hexdigest()

def reference(path):return dict(path=str(path),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def checked(ref):
    require(reference(ref['path'])==ref,'Original metadata reference changed');return read(ref['path'])

def publish(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())+len(raw)<=CAP,'Closure metadata cap')
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    require(sha(HERE/'plan.json')==arg.plan_sha and set(p.name for p in HERE.iterdir())=={'plan.json','source-review.json','review-source001.py'},'One pristine exact closure plan required')
    plan=read(HERE/'plan.json');require('torch' not in sys.modules,'No Torch imported')
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        row=line.strip().split(None,1)
        if len(row)==2 and Path(row[1].split()[0]).name.lower().startswith('python'):
            require(not any(x in row[1] for x in ('relu-squared-worker001.py','research.relu_squared_worker_v1.execution')),'Original numerical process still active')
    roots={phase:ROOT/f'research_runs/tokenizer-relu-squared-{phase}-v1-001' for phase in ('worker','pilot')}
    with ExitStack() as stack:
        for p in [ROOT/'research_state/tokenizer-worker-lock/.lock',*[root/'.lock' for root in roots.values()]]:
            f=stack.enter_context(p.open('a+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        publish('attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha,source_sha256=sha(__file__)))
        try:
            with CpuPrerequisiteGuard(HERE/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
                require(shutil.disk_usage(ROOT).free>=8*1024**3+CAP,'Closure metadata/reserve fits')
                for p,h in plan['sources'].items():require(sha(ROOT/p)==h,'Pinned source differs: '+p)
                require(len(plan['sources'])==1201,'1200 accepted runtime predecessors plus this closure source')
                completed=checked(plan['completed']);require(completed['status']=='both_finite_phases_independently_accepted' and len(completed['retained'])==12,'Complete original six-run chain required')
                retained=completed['retained'];weights=[];seen=set();inodes=set()
                for v in retained:
                    p=Path(v['path']);st=p.lstat()
                    require(p.is_file() and not p.is_symlink() and st.st_nlink==1 and st.st_size==v['size'] and p==p.resolve()
                        and any(p.is_relative_to(root) for root in roots.values()) and st.st_dev==v['device'] and st.st_ino==v['inode'],'Original retained identity differs')
                    require(str(p) not in seen and (st.st_dev,st.st_ino) not in inodes and sha(p)==v['sha256'],'Original retained hash/unique allocation differs')
                    seen.add(str(p));inodes.add((st.st_dev,st.st_ino));weights.append(dict(v,links=st.st_nlink,allocated_bytes=st.st_blocks*512))
                require(sum(v['size'] for v in weights)==completed['retained_weight_bytes'],'Complete retained bytes differ')
                guards=[];rows=[];metadata=[];ce_gaps=[];cache_gaps=[];all_steps=0
                for phase,steps in [('worker',4),('pilot',20)]:
                    root=roots[phase];stop=read(root/'STOP');require(stop['status']=='three_arm_ffn_'+phase+'_complete','Original root not successfully closed')
                    review=checked(stop['review']);phase_plan=checked(review['plan'])
                    require(phase_plan['sources']=={p:h for p,h in plan['sources'].items() if p!=str(Path(__file__).resolve().relative_to(ROOT))}
                        and phase_plan['seconds']==600 and phase_plan['compare_cpu'] is True and phase_plan['scientific_quality'] is False,'Original phase source/scope differs')
                    require(len(review['rows'])==3,'All original arms required')
                    for row in review['rows']:
                        worker=checked(row['worker']);accept=row['acceptance'];checked(row['request']);checked(row['process'])
                        for name in ('worker_guard','parent_guard'):checked(row[name])
                        require(accept['status']=='accepted' and accept['step']==steps and accept['targets']==steps*1024
                            and accept['ema_snapshots']==(3 if steps==4 else 8) and accept['parameter_tensors']==(88 if row['arm']=='native_swiglu' else 84)
                            and all(v==steps for v in accept['parameter_gradient_steps'].values())
                            and len(accept['parameter_gradient_steps'])==accept['parameter_tensors']
                            and accept['complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked'] is True and accept['holdout_scored'] is False,'Independent complete state/exposure acceptance differs')
                        require(worker['step']==steps and worker['serialized_bytes']['pair']==sum(worker[n]['size'] for n in ('parent','h200')),'Original measured pair differs')
                        for name in ('parent','h200'):
                            require(worker[name]['path'] in seen,'Every original endpoint retained')
                        for kind,pair in accept['full_validation'].items():
                            require(kind in ('raw','h200'),'Own RAW/H200 reports only');record=checked(pair['pair_record'])
                            require(record['accepted'] is True and record['ce_pass'] is True and record['target_equal'] is True
                                and record['canonical_coverage_pass'] is True and record['absolute_ce_tolerance']==3e-5
                                and record['provenance']['physical_reports_validated'] is True,'Durable caller-validated own score pair required')
                            for device in ('cpu','mps'):
                                report=pair[device];require(record[device]==report,'Original durable report content differs');target=report['target']
                                require(target['policy']=='relu_squared_native_distribution_validation_v1' and target['targets']==34278
                                    and target['validation_tokens']==34279 and target['target_bytes']==111536 and target['context']==128
                                    and target['first_target_offset']==1 and target['tail_retained'] is True and target['holdout_scored'] is False
                                    and target['window_counts']==[128]*267+[102] and target['block_counts']==[2048]*16+[1510]
                                    and target['target_ids_sha256']=='dad00eaab7c2076736432b83eb9f1cc404c87aa5e4820b5f7abebf384c7fcdaf','Complete physical GPT2 target contract differs')
                                require(report['holdout_scored'] is False and report['aux_loss_included'] is False
                                    and math.isfinite(report['ce']) and report['ce']==report['total_loss']/34278
                                    and report['bits_per_byte']==report['total_loss']/math.log(2)/111536
                                    and abs(math.fsum(report['window_loss_sums'])-report['total_loss'])<1e-8
                                    and abs(math.fsum(report['block_loss_sums'])-report['total_loss'])<1e-8,'Original full report arithmetic differs')
                            gap=abs(pair['cpu']['ce']-pair['mps']['ce']);require(gap==pair['cpu_mps_ce_difference']<=3e-5,'Original fixed CE predicate differs');ce_gaps.append(gap)
                        require(set(accept['full_validation'])=={'raw','h200'} and set(accept['cache'])=={'cpu','mps'},'Both states/backends required')
                        for device,cache in accept['cache'].items():
                            require([v['batch'] for v in cache]==[1,8],'Both cache batches required')
                            for v in cache:
                                require(v['tokens']==128 and v['prefill']==41 and v['decode']==87 and v['cache_bytes_per_item']==196608,'Prespecified cache coverage differs')
                                require(all(math.isfinite(v[n]) and v[n]>=0 for n in ('maximum_log_probability_difference','reset_maximum_log_probability_difference')),'Finite original cache reports required')
                                cache_gaps.extend(v[n] for n in ('maximum_log_probability_difference','reset_maximum_log_probability_difference'))
                        all_steps+=steps;rows.append(dict(phase=phase,arm=row['arm'],worker=row['worker'],accepted=reference(Path(worker['parent']['path']).parents[1]/'accepted.json'),step=steps,
                            worker_seconds=worker['training_seconds'],clipped_updates=accept['clipped_updates']))
                    for p in root.rglob('*guard.json'):
                        g=read(p);require(g['status']=='finished' and g['reason'] is None and g['elapsed_seconds']<g['limits']['timeout_seconds']
                            and g['observation']['peak_rss_bytes']<4*1024**3 and g['observation']['free_bytes']>=8*1024**3,'Original resource guard differs')
                        if 'peak_observed_mps_driver_bytes' in g['observation']:require(g['observation']['peak_observed_mps_driver_bytes']<6*1024**3,'Original MPS guard differs')
                        guards.append(dict(path=str(p),**g))
                    for p in sorted(root.rglob('*')):
                        if p.is_file() and str(p) not in seen and not p.name.endswith('.lock'):
                            require(not p.is_symlink() and p.stat().st_size<=4*1024**2,'Ordinary bounded original metadata required')
                            raw=p.read_bytes();require(b'\0' not in raw,'No unknown binary payload in original metadata')
                            dest=Path('originals')/phase/p.relative_to(root);out=HERE/dest;out.parent.mkdir(parents=True,exist_ok=True)
                            require(sum(v['bytes'] for v in metadata)+len(raw)<2*1024**2,'Original metadata mirror cap')
                            with out.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
                            metadata.append(dict(original=str(p),copy=str(out),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
                require(len(guards)==36 and all_steps==72,'All24CPU+6worker+6parent guards/72updates required')
                for p,digest in plan['sources'].items():require(sha(ROOT/p)==digest,'Source changed during closure: '+p)
                require('torch' not in sys.modules,'Closure imported Torch')
                publish('review.json',dict(status='closed_accepted_original_real_ffn_four_twenty_chain',sources=plan['sources'],runtime_source_bindings=1200,
                    rows=rows,guards=guards,optimizer_updates=all_steps,targets=all_steps*1024,max_cpu_mps_ce_gap=max(ce_gaps),max_cache_logprob_gap=max(cache_gaps),
                    protected_weights=weights,protected_weight_bytes=sum(v['size'] for v in weights),original_metadata=metadata,mirrored_metadata_bytes=sum(v['bytes'] for v in metadata),
                    zero_tensor_model_corpus_scoring_training_rng_calls=True,zero_weight_copies_deletions=True,no_retirement_eligibility=True,holdout_closed=True,
                    no_quality_selection_adoption_or_failed_old_panel_rootcause=True,free_bytes=shutil.disk_usage(ROOT).free))
            publish('STOP',dict(status='complete_one_time_original_ffn_prerequisite_chain_closure',no_replay=True,no_retry=True))
        except BaseException as exc:
            if not (HERE/'STOP').exists():publish('STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
            raise

if __name__=='__main__':main()
