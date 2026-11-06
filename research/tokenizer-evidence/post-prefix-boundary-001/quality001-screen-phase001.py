"""Read-only screen/confirmation audit and exact closed metadata mirror; no controller calls."""
import copy,hashlib,json,shutil,time,datetime
from pathlib import Path
E=Path(__file__).resolve().parent;ROOT=E.parents[2];R=ROOT/'research_runs/tokenizer-moe-quality-v1-001';BEGAN=time.monotonic()
def ref(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(r):
 p=Path(r['path']);assert ref(p)=={k:r[k] for k in ['path','sha256']};return json.loads(p.read_text())
def publish(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
assert not(E/'quality001-screen-phase001.json').exists() and not(R/'STOP').exists()
seal=checked(ref(E/'quality001-freeze.json'));plan=checked(seal['plan']);assert seal['plan']==ref(E/'quality001-plan.json')
for p,h in plan['sources'].items():assert ref(ROOT/p)['sha256']==h
assert len(plan['sources'])==1035
chosen=checked(ref(E/'quality001-selection.json'));confirm=checked(ref(E/'quality001-confirmation-plan.json'));rows=[];weights=[];copies=[]
mirror=E/'quality001-screen-metadata-mirror001';mirror.mkdir()
for slot,spec in enumerate(plan['specs']):
 assert slot==spec['slot'] and slot<9
 b=R/f'evidence/quality/{slot:02d}';anchor=checked(ref(E/f'quality001-accepted-{slot:02d}.json'));assert anchor['original_freeze']==ref(E/'quality001-freeze.json') and anchor['accepted']==ref(b/'accepted.json')
 row=checked(anchor['accepted']);w=checked(row['worker']);rq=checked(row['request']);a=row['acceptance']
 assert row['slot']==slot and row['arm']==spec['arm'] and rq['spec']==spec and rq['plan']==seal['plan'] and w['config']==spec['config'] and w['request']==row['request']
 assert w['kind']=='tokenizer_moe_quality_worker_v1' and a['status']=='accepted' and w['step']==a['step']==1200 and w['targets']==a['targets']==1228800 and w['ema_snapshots']==a['ema_snapshots']==41
 assert a['complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked'] and a['holdout_scored'] is False
 proc=checked(row['process']);assert proc['status']=='completed' and proc['returncode']==0 and proc['timeout_seconds']==7200
 for field in ['worker_guard','parent_guard']:
  g=checked(row[field]);assert g['status']=='finished' and g['reason'] is None and g['limits']==dict(mps_driver_limit_bytes=6*1024**3,reserve_bytes=8*1024**3,rss_limit_bytes=4*1024**3,timeout_seconds=7200)
 for kind in ['parent','h200']:
  f=Path(w[kind]['path']);s=f.stat();assert f==b/'checkpoints'/(kind+'.pt') and not f.is_symlink() and s.st_nlink==1 and s.st_size==w[kind]['size'] and ref(f)['sha256']==w[kind]['sha256'];weights.append(w[kind])
 for name in ['raw','h200']:
  assert a['full_validation'][name]['cpu_mps_ce_difference']<=3e-5
  for device in ['cpu','mps']:assert a['full_validation'][name][device]['target']['targets']==34278
 for device in ['cpu','mps']:assert [v['batch'] for v in a['cache'][device]]==[1,8]
 rows.append(dict(slot=slot,status='accepted',arm=spec['arm'],peak_lr=spec['config']['training']['learning_rate'],h200=a['full_validation']['h200']['mps'],accepted=ref(b/'accepted.json')))
 files=[b/n for n in ['request.json','accepted.json','config.json','worker.json','metrics.jsonl','process/worker.log','process/launch.json','process/process.json']]+[Path(row[k]['path']) for k in ['worker_guard','parent_guard']]+[E/f'quality001-accepted-{slot:02d}.json']
 for index,f in enumerate(files):
  before=ref(f);destination=mirror/f'{slot:02d}-{index:02d}-{f.name}';shutil.copyfile(f,destination);assert ref(f)==before and ref(destination)['sha256']==before['sha256'];copies.append(dict(source=before,destination=ref(destination),bytes=f.stat().st_size))
 assert time.monotonic()-BEGAN<120
choices={arm:min((r for r in rows if r['arm']==arm),key=lambda r:(r['h200']['ce'],r['peak_lr']))['peak_lr'] for arm in ['dense','wide','moe']}
assert chosen==dict(kind='immutable_per_arm_native_CE_LR_selection',original_freeze=ref(E/'quality001-freeze.json'),rows=rows,choices=choices)
assert confirm['selection']==ref(E/'quality001-selection.json') and confirm['original_freeze']==ref(E/'quality001-freeze.json') and confirm['sources']==plan['sources'] and confirm['design']==plan['design'] and confirm['seeds']==plan['seeds'] and confirm['environment']==plan['environment']
for p,seed in enumerate(plan['seeds']['fresh']):
 order=['dense','wide','moe'];order=order[p:]+order[:p]
 for i,arm in enumerate(order):
  slot=9+3*p+i;spec=confirm['specs'][3*p+i];expected=copy.deepcopy(next(s for s in plan['specs'] if s['arm']==arm and s['config']['training']['learning_rate']==choices[arm]));expected['slot']=slot;t=expected['config']['training'];t['seed']=seed;base=R/f'evidence/quality/{slot:02d}'
  t.update(checkpoint_dir=str(base/'checkpoints'),metrics_file=str(base/'metrics.jsonl'),summary_file=str(base/'worker.json'));assert spec==expected
assert len(confirm['specs'])==9 and len(weights)==18 and len(copies)==99
publish(mirror/'completion.json',dict(status='lossless_closed_screen_metadata_mirror_complete',files=copies,weight_copies=0,protected_original_weight_files=weights))
active=[]
for b in sorted((R/'evidence/quality').glob('[0-9][0-9]')):
 if (b/'accepted.json').exists():continue
 g=R/f'control/worker-quality-{b.name}-guard.json'
 if g.exists():active.append(json.loads(g.read_text()))
result=dict(status='all_nine_screens_accepted_immutable_selection_and_fresh_resolution_verified',source=ref(Path(__file__)),observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze=ref(E/'quality001-freeze.json'),selection=ref(E/'quality001-selection.json'),confirmation=ref(E/'quality001-confirmation-plan.json'),choices=choices,screens=9,source_bindings=1035,protected_weight_files=weights,screen_weight_bytes=sum(v['size'] for v in weights),metadata_mirror=ref(mirror/'completion.json'),closed_metadata_files=99,weight_copies=0,original_acceptance_or_training_or_scoring_replayed=False,controller_state_commands=False,active_worker_guard_observations=active,free_bytes=shutil.disk_usage(ROOT).free,scientific_quality_result=False,adopted=False,next_action='Let original serial controller finish all9fresh endpoints and all9serving/independent final review. Keep all18screen payloads and every fresh;no reselect,bootstrap early,interim winner or adoption.')
publish(E/'quality001-screen-phase001.json',result)
print(json.dumps({k:result[k] for k in ['status','choices','screen_weight_bytes','closed_metadata_files','weight_copies','free_bytes']}))
