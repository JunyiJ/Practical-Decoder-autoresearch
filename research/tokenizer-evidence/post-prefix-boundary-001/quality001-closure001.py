"""Terminal artifact/acceptance review only. No closed controller,model or statistics replay."""
import hashlib,json,shutil,datetime,time
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2];R=ROOT/'research_runs/tokenizer-moe-quality-v1-001';start=time.monotonic()
def ref(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(r):
 p=Path(r['path']);assert ref(p)=={k:r[k] for k in ['path','sha256']};return json.loads(p.read_text())
def publish(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
assert not(E/'quality001-closure001.json').exists()
stop_ref=ref(R/'STOP');stop=checked(stop_ref);assert stop['status']=='complete_finite_scientific_panel' and stop['never_rerun_training_inference_review'] is True
report=checked(stop['review']);assert report['status']=='complete_fresh_panel_reviewed' and report['adopted'] is False
seal=checked(report['freeze']);plan=checked(seal['plan']);selection=checked(report['selection']);confirmation=checked(report['confirmation'])
for p,h in plan['sources'].items():assert ref(ROOT/p)['sha256']==h
assert len(plan['sources'])==1035
ind=checked(report['independent_review']);primary=checked(ind['primary']);inference=checked(report['inference'])
assert primary['statistics']==report['statistics'] and ind['result']['status']=='independent_complete_panel_statistics_agreed' and ind['result']['atol']==1e-12
assert ind['result']['quality_gates_pass']==report['statistics']['quality_gates_pass']==False and ind['result']['gates']==report['statistics']['gates'] and not any(report['statistics']['gates'].values())
for arm in ['dense','wide']:
 c=report['statistics']['contrasts'][arm];i=ind['result']['contrasts'][arm]
 assert c['candidate_minus_control_ce']==i['deltas'] and abs(c['mean_ce_delta']-i['mean'])<1e-12
 for width in ['2048','4096','8192']:assert abs(c['intervals'][width]['upper']-i['upper'][width])<1e-12
assert inference['all_nine_fresh'] is True and inference['holdout_scored'] is False and inference['selection'] is False and len(inference['rows'])==9
for row in inference['rows']:
 assert [x['batch'] for x in row['cache_reset_checks']]==[1,8]
 assert all(x['cache_bytes_per_item']==196608 for x in row['cache_reset_checks'])
 assert [x['batch'] for x in row['measurements']]==[1,8] and all(len(x['full128_seconds'])==len(x['decode32_after_prefill96_seconds'])==5 for x in row['measurements'])
for n in ['quality001-fresh-inference-guard.json','quality001-independent-review-guard.json']:
 g=checked(ref(E/n));assert g['status']=='finished' and g['reason'] is None
mirror=E/'quality001-fresh-metadata-mirror001';mirror.mkdir();copies=[];payloads=[];acceptances=[];maxdiff=0.;peaks=[]
for slot in range(18):
 b=R/f'evidence/quality/{slot:02d}';anchor=checked(ref(E/f'quality001-accepted-{slot:02d}.json'));row=checked(anchor['accepted']);w=checked(row['worker']);request=checked(row['request']);a=row['acceptance']
 expected=plan['specs'][slot] if slot<9 else confirmation['specs'][slot-9]
 assert anchor['original_freeze']==report['freeze'] and row['slot']==slot and request['spec']==expected and w['config']==expected['config'] and w['request']==row['request']
 assert w['kind']=='tokenizer_moe_quality_worker_v1' and a['status']=='accepted' and a['complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked']
 assert w['step']==a['step']==1200 and w['targets']==a['targets']==1228800 and w['ema_snapshots']==a['ema_snapshots']==41
 assert a['parameter_tensors']==w['parameter_tensors']==(128 if row['arm']=='moe' else 88)
 proc=checked(row['process']);assert proc['status']=='completed' and proc['returncode']==0 and proc['timeout_seconds']==7200
 for k in ['worker_guard','parent_guard']:
  g=checked(row[k]);assert g['status']=='finished' and g['reason'] is None and g['limits']==dict(mps_driver_limit_bytes=6*1024**3,reserve_bytes=8*1024**3,rss_limit_bytes=4*1024**3,timeout_seconds=7200);peaks.append(g['observation'])
 for kind in ['parent','h200']:
  v=w[kind];p=Path(v['path']);s=p.stat();assert p==b/'checkpoints'/(kind+'.pt') and not p.is_symlink() and s.st_nlink==1 and s.st_size==v['size'] and ref(p)['sha256']==v['sha256'];payloads.append(dict(slot=slot,arm=row['arm'],kind=kind,**v))
 for kind in ['raw','h200']:
  scores=a['full_validation'][kind];maxdiff=max(maxdiff,scores['cpu_mps_ce_difference']);assert scores['cpu_mps_ce_difference']<=3e-5
  for device in ['cpu','mps']:assert scores[device]['target']['targets']==34278 and scores[device]['target']['target_bytes']==111536 and scores[device]['aux_loss_included'] is False
 for device in ['cpu','mps']:assert [x['batch'] for x in a['cache'][device]]==[1,8]
 acceptances.append(anchor['accepted'])
 if slot>=9:
  files=[b/n for n in ['request.json','accepted.json','config.json','worker.json','metrics.jsonl','process/worker.log','process/launch.json','process/process.json']]+[Path(row[k]['path']) for k in ['worker_guard','parent_guard']]+[E/f'quality001-accepted-{slot:02d}.json']
  for index,p in enumerate(files):
   before=ref(p);dest=mirror/f'{slot:02d}-{index:02d}-{p.name}';shutil.copyfile(p,dest);assert ref(p)==before and ref(dest)['sha256']==before['sha256'];copies.append(dict(source=before,destination=ref(dest),bytes=p.stat().st_size))
 assert time.monotonic()-start<120
# Closed final records are copied in addition to the99fresh endpoint metadata files.
for p in [R/'STOP']+[E/n for n in ['quality001-final-review.json','quality001-independent-final-review.json','quality001-primary-statistics.json','quality001-fresh-inference.json','quality001-fresh-inference-guard.json','quality001-independent-review-guard.json']]:
 before=ref(p);dest=mirror/('final-'+p.name);shutil.copyfile(p,dest);assert ref(p)==before and ref(dest)['sha256']==before['sha256'];copies.append(dict(source=before,destination=ref(dest),bytes=p.stat().st_size))
assert len(payloads)==36 and len(copies)==106 and ref(R/'STOP')==stop_ref
publish(mirror/'completion.json',dict(status='complete_closed_fresh_and_final_metadata_mirror',files=copies,weight_copies=0))
result=dict(status='CLOSED_NOT_ADOPTED',source=ref(Path(__file__)),observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),terminal=stop_ref,final_review=stop['review'],independent_review=report['independent_review'],fresh_inference=report['inference'],freeze=report['freeze'],selection=report['selection'],confirmation=report['confirmation'],sources=len(plan['sources']),accepted_endpoints=18,accepted_references=acceptances,protected_payloads=payloads,retained_weight_bytes=sum(v['size'] for v in payloads),statistics=report['statistics'],max_cpu_mps_ce_difference=maxdiff,peak_rss_bytes=max(v['peak_rss_bytes'] for v in peaks),peak_mps_driver_bytes=max(v['peak_observed_mps_driver_bytes'] for v in peaks),metadata_mirror=ref(mirror/'completion.json'),new_closed_metadata_files=106,weight_copies=0,training_scoring_or_bootstrap_replayed=False,adopted=False,incumbent_changed=False,all_scientific_seed_roles_used=True,free_bytes=shutil.disk_usage(ROOT).free,next_action='Keep native dense incumbent. Analyze existing accepted loss blocks/curves/routing/exposure/cost only,without model rescoring/training/bootstrap/holdout,then use evidence-grounded boundary review to qualify next hypothesis. No MoE router/expert/horizon rescue or wide fallback.')
publish(E/'quality001-closure001.json',result);print(json.dumps({k:result[k] for k in ['status','accepted_endpoints','retained_weight_bytes','new_closed_metadata_files','max_cpu_mps_ce_difference','peak_rss_bytes','peak_mps_driver_bytes','free_bytes']}))
