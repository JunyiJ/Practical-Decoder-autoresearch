"""Read-only archive reclassification from original acceptance and selection records."""
import collections,hashlib,json,os,stat,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
refs={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p,expected=None):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 assert p.is_relative_to(ROOT) and not p.is_symlink()
 h=sha(p)
 if expected:assert h==expected,str(p)
 refs[str(p)]=dict(path=str(p),size=p.stat().st_size,sha256=h)
 return json.loads(p.read_text())
def bound(r,base=ROOT):
 p=Path(r['path']);return read(p if p.is_absolute() else base/p,r['sha256'])
def selection(campaign,spec):
 name=campaign.name;key=spec.get('arm',spec.get('family'));lr=spec.get('lr',spec.get('learning_rate'))
 if 'wide-hybrid-' in name:
  p=campaign/'evidence/meta/rank-selection.json';d=read(p);assert d[key]['lr']==lr and d[key]['divisor']==spec['divisor'];return p,'selected_final_rank'
 if name=='tokenizer-activation-001':
  p=campaign/'evidence/meta/activation-selection.json';d=read(p);assert d['choices'][key]==lr;return p,'selected_screen'
 specials={'tokenizer-block-norm-v2-quality-001':'block-norm-preparation-001','tokenizer-ffn-quality-v1-001':'ffn-structure-preparation-001'}
 if name in specials:p=ROOT/'research/tokenizer-evidence'/specials[name]/'quality001-selection.json';d=read(p);assert d['value'][key]==lr;return p,'selected_screen'
 if name in ['tokenizer-ffn-norm-quality-v1-001','tokenizer-warmup-quality-v1-001','tokenizer-local-mix-quality-v3-001','tokenizer-mla-score-quality-v1-001']:
  p=campaign/'evidence/meta/selection.json';d=read(p);assert d['value'][key]==lr;return p,'selected_screen'
 p=ROOT/'research/tokenizer-evidence'/name.removeprefix('tokenizer-')/'selection.json';d=read(p)
 if name=='tokenizer-quality-001':assert d['selected_lr']==lr
 elif name=='tokenizer-horizon-001':assert d['selected_steps']==spec['steps']
 elif name=='tokenizer-hybrid-001':
  assert d['rates'][key]==lr
  return p,'obsolete_losing_control_screen' if key not in ['hybrid',d['control']] else 'selected_screen'
 else:assert d[key]==lr
 return p,'selected_screen'

def main():
 rows=[]
 for n in [6,7,8,9,10,11,12,13,14,15,16,17,19,21]:
  folder=ROOT/f'research/tokenizer-evidence/storage-recovery-{n:03d}'
  complete=read(folder/'raw-completed.json');plan=bound(complete['plan']);by_original={r['original']['path']:r for r in complete['files']}
  for row in plan['files']:
   original=Path(row['original']['path']);campaign=ROOT/'research_runs'/original.relative_to(ROOT/'research_runs').parts[0]
   done=by_original[str(original)];archive=Path(done['archive']['path']);assert archive==Path(row['archive'])
   s=archive.lstat();assert stat.S_ISREG(s.st_mode) and s.st_size==done['archive']['size'] and s.st_uid==os.getuid() and s.st_nlink==1
   a=bound(row['acceptance']);terminal=bound(row['terminal']);extra=[]
   rq=a.get('request',{});spec=rq.get('trial',{}).get('spec') if isinstance(rq,dict) else None;seeds=rq.get('trial',{}).get('seeds') if isinstance(rq,dict) else None
   if campaign.name=='tokenizer-activation-001':
    rq=bound(a['request'],campaign);result=bound(a['result'],campaign);spec=rq['spec'];assert result['status']=='completed' and result['record']['status']=='completed';assert all(result['record'][k]==v for k,v in spec.items());extra=[refs[str(campaign/a['request']['path'])],refs[str(campaign/a['result']['path'])]]
   elif campaign.name=='tokenizer-ema-horizon-v1-001':
    worker=bound(a['request']);request=worker['acceptance_request'];seeds=request['seeds'];seed=request['config']['training']['seed'];assert request['stage']=='quality' and seed in seeds['fresh'];spec=dict(stage='confirm',seed=seed,slot=a['slot'],lr=request['config']['training']['learning_rate']);extra=[refs[a['request']['path']]]
   elif spec is None:
    assert a['status']=='completed';spec={k:a[k] for k in ['stage','family','seed','lr','steps','width','divisor'] if k in a}
   if 'stage' not in spec:
    assert seeds and spec['seed'] in seeds['fresh'];spec=dict(spec,stage='confirm')
   stage=spec['stage'];sel=None
   if stage=='confirm':
    if seeds:assert spec['seed'] in seeds['fresh']
    classification='fresh_confirmation'
   else:
    assert stage in ['screen','lr','rank'];sel,classification=selection(campaign,spec)
   # Acceptance must bind original RAW identity, directly or in its referenced request/result.
   texts=[json.dumps(a),json.dumps(rq)]
   if campaign.name=='tokenizer-activation-001':texts.append(json.dumps(result))
   if campaign.name=='tokenizer-ema-horizon-v1-001':texts.append(json.dumps(request))
   assert any(row['original']['sha256'] in x for x in texts),str(original)
   rows.append(dict(archive=done['archive'],allocated_bytes=s.st_blocks*512,original=row['original'],h200=row.get('h200'),acceptance=row['acceptance'],terminal=row['terminal'],stage=stage,spec=spec,classification=classification,selection=refs[str(sel)] if sel else None,additional_evidence=extra,archive_plan=refs[str(folder/'raw-plan.json')],archive_receipt=refs[str(folder/'raw-completed.json')],classification_uses_old_protected_label=False))
 assert len(rows)==197
 report=dict(scope='197 completed historical RAW archives, independent classification using original acceptance stage/seed and final selection values. Binary archive contents not read in this classification pass; eligible deletion requires separate exact hash/member verification.',observed_unix=time.time(),counts=dict(collections.Counter(r['classification'] for r in rows)),rows=rows,evidence=list(refs.values()),candidate_archives=[r['archive']['path'] for r in rows if r['classification'].startswith('obsolete')],not_in_scope='Concurrent fresh-epsilon archival027/028 and mixed historical temporary-diagnostics archives; no absence or old protection label grants deletion permission.')
 with (HERE/'archive-audit.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(dict(counts=report['counts'],candidates=report['candidate_archives'],evidence_files=len(refs))))
if __name__=='__main__':main()
