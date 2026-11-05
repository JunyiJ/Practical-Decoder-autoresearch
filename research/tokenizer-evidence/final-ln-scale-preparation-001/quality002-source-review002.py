"""Exercise same-panel imported-prefix binding/state through metadata-only fixtures.

AST-extracted functions preserve verifier logic; only fixture path locators are
injected. Real module globals, sources and all retained weights stay unchanged.
No execution continuation freeze, worker, new scientific bootstrap or training.
"""
import ast,copy,json,shutil,datetime
from pathlib import Path
import torch
from research.final_ln_scale_quality_v3 import quality_execution as q,campaign_contract as c
from research.final_ln_scale_quality_v3.support import exclusive,idle
from research.block_norm_v1.fixture_dispatch import read,reference
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import budgets
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
B=q.EVIDENCE

def rejects(fn):
 try:fn()
 except (ValueError,AssertionError,TypeError):return
 raise AssertionError('Forged imported-prefix metadata accepted')

torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g:
 idle()
 with CpuPrerequisiteGuard(B/'quality002-source-review002-guard.json',c.ROOT,timeout_seconds=120):
  integration=read(reference(B/'quality002-integration001.json'));checked=read(integration['plan']);original=read(checked['original_plan']);sources=c.source_inventory();assert len(sources)==918 and sources==checked['sources']
  assert not q.ROOT.exists() and not q.FREEZE.exists()
  names=['quality002-freeze-build002.py','quality002-fresh-inference001.py','quality002-final-review001.py'];refs={n:reference(B/n) for n in names}
  for n in names:ast.parse((B/n).read_text())
  fixture=B/'quality002-state-fixture002';fixture.mkdir();campaign=fixture/'campaign';campaign.mkdir();anchors=fixture/'anchors';anchors.mkdir()
  plan=copy.deepcopy(original);plan.update(kind=q.KIND,status='frozen_quality_only',root=str(campaign),sources=sources,added_sources=sorted(set(sources)-set(read(original['pilot_plan'])['sources'])),environment=c.environment(),data=c.bind_data(),limits=dict(c.LIMITS),allowances=q.ALLOWANCES,required_free_bytes=q.FULL_FREE,original_plan=checked['original_plan'],entry_failure=checked['entry_failure'],original_STOP=checked['original_STOP'],original_root=checked['original_root'],original_anchors=checked['original_anchors'],imported_count=3,entry_headroom_bytes=c.ENTRY_HEADROOM,prerequisites=dict(integration=reference(B/'quality002-integration001.json'),entry_failure=checked['entry_failure']),final_sources={n:refs[n] for n in names[1:]})
  exclusive(fixture/'plan.json',plan);exclusive(fixture/'freeze.json',dict(plan=reference(fixture/'plan.json')));freeze=reference(fixture/'freeze.json');previous=freeze
  for item in integration['imports']:
   i=item['slot'];prior=read(item['original_acceptance']);receipt=dict(kind=q.KIND,slot=i,freeze=freeze,imported=True,original_acceptance=item['original_acceptance'],worker=item['original_worker'],request_sha256=item['request_sha256'],artifact_acceptance=item['acceptance'],target_contract=item['acceptance']['validation']['target'],cache=item['cache'],process=prior['process'],selection=None,cleanup=None,adopted=False)
   p=fixture/f'receipt-{i:02d}.json';exclusive(p,receipt);anchor=dict(kind=q.KIND,slot=i,previous=previous,acceptance=reference(p),parent_guard=item['original_parent_guard'],files=item['original_files']);exclusive(anchors/f'{i:02d}.json',anchor);previous=reference(anchors/f'{i:02d}.json')
  tree=ast.parse(Path(q.__file__).read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('bind','state')];assert len(functions)==2
  ns=dict(vars(q));ns.update(ROOT=campaign,FREEZE=fixture/'freeze.json',anchor_path=lambda slot:anchors/f'{slot:02d}.json')
  exec(compile(ast.Module(body=functions,type_ignores=[]),'unchanged_verifier_ast_with_fixture_paths','exec'),ns)
  with session_lock(campaign) as f:
   bound=ns['bind'](MetadataWriter(campaign,(g,f)),freeze['sha256']);assert bound==plan
   state=ns['state'](plan,freeze['sha256']);assert len(state['records'])==3 and state['used_weights']==1677774994 and q.next_spec(plan,state)['slot']==3
  # Missing imported slot, foreign request fingerprint, and extra weight-credit
  # metadata must fail before any numerical or filesystem mutation.
  broken=dict(ns);broken['anchor_path']=lambda slot:fixture/'missing-anchor.json' if slot==1 else anchors/f'{slot:02d}.json'
  exec(compile(ast.Module(body=[next(n for n in functions if n.name=='state')],type_ignores=[]),'missing_prefix_fixture','exec'),broken);rejects(lambda:broken['state'](plan,freeze['sha256']))
  for label,mutate in [('request',lambda r:r.update(request_sha256='0'*64)),('kind',lambda r:r.update(kind='foreign'))]:
   receipt=read(reference(fixture/'receipt-00.json'));mutate(receipt);p=fixture/f'forged-{label}.json';exclusive(p,receipt);anchor=read(reference(anchors/'00.json'));anchor['acceptance']=reference(p);fake=fixture/f'forged-{label}-anchor.json';exclusive(fake,anchor)
   bad=dict(ns);bad['anchor_path']=lambda slot,fake=fake:fake if slot==0 else anchors/f'{slot:02d}.json';exec(compile(ast.Module(body=[next(n for n in functions if n.name=='state')],type_ignores=[]),label+'_fixture','exec'),bad);rejects(lambda:bad['state'](plan,freeze['sha256']))
  anchor=read(reference(anchors/'00.json'));anchor['files']=anchor['files']+[dict(path=str(fixture/'nonexistent-extra.pt'),sha256='0'*64)];fake=fixture/'forged-weight-credit-anchor.json';exclusive(fake,anchor)
  bad=dict(ns);bad['anchor_path']=lambda slot:fake if slot==0 else anchors/f'{slot:02d}.json';exec(compile(ast.Module(body=[next(n for n in functions if n.name=='state')],type_ignores=[]),'credit_fixture','exec'),bad);rejects(lambda:bad['state'](plan,freeze['sha256']))
  active={k:v for k,v in budgets().items() if not k.startswith(('evidence/pilots/','evidence/quality/','evidence/.retention/')) or any(k.startswith(f'evidence/quality/{i:02d}/') for i in range(6))};legacy={k:v for k,v in budgets().items() if k.startswith(('evidence/meta','control','emergency')) or any(k.startswith(f'evidence/quality/{i:02d}/') for i in range(3))}
  assert sum(active.values())+sum(legacy.values())+32*1024**2<=256*1024**2
  assert q.FULL_FREE==12383043584 and q.FULL_FREE-state['used_weights']==10705268590
  assert not q.ROOT.exists() and not q.FREEZE.exists() and not any(q.anchor_path(i).exists() for i in range(6));c.verify_sources(sources)
  free=shutil.disk_usage(c.ROOT).free
  value=dict(status='passed_prepared_same_panel_continuation_capacity_blocked' if free<10705268590 else 'passed_prepared_same_panel_continuation_capacity_eligible',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,source_count=918,predecessors_preserved=912,integration=reference(B/'quality002-integration001.json'),prepared_sources=refs,fixture=reference(fixture/'plan.json'),actual_imported_prefix_verified=3,actual_imported_weight_bytes=1677774994,binding_and_state_verifier_ast_unchanged=True,fixture_path_locators_only=True,forgeries_rejected=['missing_prefix','request_fingerprint','foreign_kind','extra_weight_credit'],combined_metadata_group_caps=sum(active.values())+sum(legacy.values())+32*1024**2,metadata_cap=256*1024**2,remaining_requirement=10705268590,original_remaining_requirement=10703810414,measured_entry_headroom=1458176,free_bytes=free,remaining_deficit=max(0,10705268590-free),cleanup_target=12852752238,cleanup_target_deficit=max(0,12852752238-free),original_run_STOP_unchanged=True,new_run_created=False,new_execution_freeze_created=False,training_updates=0,weight_copies=0,holdout_scored=False,quality_adopted=False,scientific_panel='Original d00b8e... six-trajectory panel;no new seeds or old-study score pooling',next_action='Keep original001/slot03 entry CLOSED. On revised capacity wake verify918/prepared refs and original accepted-prefix provenance,then execute prepared quality002-freeze-build002.py ONCE to import3 endpoints metadata-only. Only after its full preflight may v3 dispatch the scientifically never-started learned012 trajectory in the new execution root. No original STOP removal,source edit,weight copy,retraining or relaxed guard.')
 exclusive(B/'quality002-source-review002.json',value);print(json.dumps({k:value[k] for k in ('status','source_count','actual_imported_prefix_verified','remaining_requirement','free_bytes','remaining_deficit')}),flush=True)
