"""Descriptive analysis of accepted saved records only; no models or bootstrap."""
import hashlib,json,math,statistics,shutil
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.tokenizer_moe_worker_v1.execution import idle
E=Path(__file__).resolve().parent;ROOT=E.parents[2];OLD=E.parent/'post-prefix-boundary-001'
def ref(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(r):
 p=Path(r['path']);assert ref(p)=={k:r[k] for k in ['path','sha256']};return json.loads(p.read_text())
def publish(n,d):
 with (E/n).open('x') as f:json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def mean(v):return math.fsum(v)/len(v)
with session_lock(GLOBAL_LOCK.parent):
 idle();assert not(E/'existing-analysis001-plan.json').exists()
 floor=8*1024**3+16*1024**2+1267904512+1458176;assert shutil.disk_usage(ROOT).free>=floor+2*1024**3
 closure=checked(ref(OLD/'quality001-closure001.json'));assert closure['status']=='CLOSED_NOT_ADOPTED' and closure['accepted_endpoints']==18
 publish('existing-analysis001-plan.json',dict(source=ref(Path(__file__)),closure=ref(OLD/'quality001-closure001.json'),budget=dict(metadata_cap=16*1024**2,full_floor=floor,target=floor+2*1024**3,free=shutil.disk_usage(ROOT).free),training=False,model_loaded=False,scoring=False,bootstrap=False,holdout=False,new_scientific_seed=False,questions=['Where is observed MoE loss relative to both controls?','Does saved cumulative routing show unserved experts or dormant update exposure?','What do sparse curves,clipping,RAW-H200 differences and measured costs establish or leave unresolved?']))
 try:
  with CpuPrerequisiteGuard(E/'existing-analysis001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
   final=checked(closure['final_review']);primary=checked(checked(final['independent_review'])['primary']);by_arm={a:[] for a in ['dense','wide','moe']};by_seed={};routes=[];differences=[];controller_seconds=0.
   for slot,reference in enumerate(closure['accepted_references']):
    row=checked(reference);w=checked(row['worker']);guard=checked(row['parent_guard']);process=checked(row['process']);controller_seconds+=process['elapsed_seconds']+guard['elapsed_seconds'] if 'elapsed_seconds' in process else guard['elapsed_seconds']
    if slot<9:continue
    arm=row['arm'];seed=w['config']['training']['seed'];a=row['acceptance'];by_seed.setdefault(seed,{})[arm]=a['full_validation']['h200']['mps']
    by_arm[arm].append(dict(seed=seed,training_seconds=w['training_seconds'],clipped_updates=w['clipped_updates'],raw_ce=a['full_validation']['raw']['mps']['ce'],h200_ce=a['full_validation']['h200']['mps']['ce'],raw_minus_h200_ce=a['full_validation']['raw']['mps']['ce']-a['full_validation']['h200']['mps']['ce'],final_online_train_ce=w['final_online_train_ce'],final_online_aux_loss=w['final_online_aux_loss'],final_online_total_loss=w['final_online_total_loss'],sampled_curves=w['metrics'],min_gradient_steps=min(w['parameter_gradient_steps'].values()),max_gradient_steps=max(w['parameter_gradient_steps'].values()),parameters_ever_dormant=sum(v<1200 for v in w['parameter_gradient_steps'].values())))
    for kind in ['raw','h200']:differences.append(dict(seed=seed,arm=arm,kind=kind,cpu_mps_ce_difference=a['full_validation'][kind]['cpu_mps_ce_difference']))
    if arm=='moe':
     for layer,counts in enumerate(w['route_counts']):
      assert len(counts)==4 and sum(counts)==2457600
      fractions=[v/sum(counts) for v in counts];routes.append(dict(seed=seed,layer=layer,counts=counts,fractions=fractions,unused_experts=sum(v==0 for v in counts),cumulative_assignment_entropy_nats=-math.fsum(v*math.log(v) for v in fractions if v>0)))
   blocks={}
   for control in ['dense','wide']:
    records={}
    for width in [2048,4096,8192]:
     counts=[min(width,34278-i) for i in range(0,34278,width)];stride=width//2048;per_seed=[]
     for seed,arms in sorted(by_seed.items()):
      aa=arms[control]['block_loss_sums'];bb=arms['moe']['block_loss_sums'];values=[math.fsum(bb[i:i+stride])-math.fsum(aa[i:i+stride]) for i in range(0,len(aa),stride)];per_seed.append(values)
     perblock=[mean([s[i] for s in per_seed])/count for i,count in enumerate(counts)]
     reconstructed=math.fsum(delta*n for delta,n in zip(perblock,counts))/34278;assert abs(reconstructed-final['statistics']['contrasts'][control]['mean_ce_delta'])<1e-12
     records[str(width)]=dict(counts=counts,mean_loss_delta_per_block=perblock,positive_blocks=sum(v>0 for v in perblock),negative_blocks=sum(v<0 for v in perblock),zero_blocks=sum(v==0 for v in perblock),weighted_mean=reconstructed,scope='Descriptive mean loss on saved reused-text blocks;no new bootstrap or uncertainty claim.')
    blocks[control]=records
   summaries={a:dict(mean_training_seconds=mean([r['training_seconds'] for r in rows]),mean_raw_minus_h200_ce=mean([r['raw_minus_h200_ce'] for r in rows]),mean_h200_ce=mean([r['h200_ce'] for r in rows]),clipped_updates=[r['clipped_updates'] for r in rows],parameters_ever_dormant=[r['parameters_ever_dormant'] for r in rows]) for a,rows in by_arm.items()}
   assert len(by_seed)==3 and all(set(v)=={'dense','wide','moe'} for v in by_seed.values())
   assert max(v['cpu_mps_ce_difference'] for v in differences)<=3e-5
   result=dict(status='accepted_existing_metadata_analysis_complete_no_numerical_replay',source=ref(Path(__file__)),closure=ref(OLD/'quality001-closure001.json'),fresh_summaries=summaries,complete_fresh_records=by_arm,descriptive_loss_blocks=blocks,routing=routes,cpu_mps_agreement=differences,max_cpu_mps_ce_difference=max(v['cpu_mps_ce_difference'] for v in differences),adopted=False,incumbent_changed=False,training=False,model_loaded=False,scoring=False,bootstrap=False,holdout=False,new_scientific_seed=False,limits=['All3freshMoE losses versus both controls falsify useful improvement for this frozen recipe;not all possible MoE designs.','Cumulative hard assignment balance is not semantic specialization or softmax mean entropy.','Sparse online curves and final RAW-H200 gaps do not establish overfit,optimal horizon,EMA choice or an isolated cause.','Different architecture/auxiliary objective changes joint recipe;not a pure routing effect.','No wide fallback,cost-only adoption or automatic router/expert/horizon rescue.'],next_action='Evidence-grounded boundary review of remaining untested mechanism,including actual prior results and primary literature. Do not infer next panel from quota or from this single negative recipe.')
   assert len(json.dumps(result))<16*1024**2
  publish('existing-analysis001.json',result)
  print(json.dumps(dict(status=result['status'],summaries=summaries,positive_blocks={a:{w:r['positive_blocks'] for w,r in widths.items()} for a,widths in blocks.items()},routing_min=min(v for r in routes for v in r['fractions']),routing_max=max(v for r in routes for v in r['fractions']))))
 except BaseException as exc:publish('STOP-existing-analysis001',dict(error=repr(exc),closed=True,no_retry=True));raise
