"""Independent score aggregation and evidence checks for posthoc attribution."""
import datetime,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-quality-001'
A=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-attribution-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def run():
 j=read(A/'analysis.json');plan=read(P/'plan.json');decision=read(P/'decision.json')
 assert j['source_sha256']==sha(ROOT/'research/general_rsi_cross_size_attribution_v1.py')
 for p,h in j['input_sha256'].items():assert sha(ROOT/p)==h,p
 by={(s['depth'],s['bundle'],s['arm']):P/f"slot-{s['slot']:02d}" for s in plan['slots']}
 deltas={}
 for depth in (4,8,16):
  paired=[]
  for b in (1,2,3):
   n=read(by[depth,b,'native']/'h200-validation.json');x=read(by[depth,b,'raw_bytes']/'h200-validation.json')
   paired.append((math.fsum(r['loss_sum'] for r in x['documents'])-math.fsum(r['loss_sum'] for r in n['documents']))/n['count'])
  deltas[depth]=paired
  target=next(r for r in decision['contrasts'] if r['contrast']==f'delta{depth}')
  assert max(abs(a-b) for a,b in zip(paired,target['paired_deltas']))<2e-14
 interactions=[a-b for a,b in zip(deltas[16],deltas[4])]
 target=next(r for r in decision['contrasts'] if r['contrast']=='delta16_minus_delta4')
 assert max(abs(a-b) for a,b in zip(interactions,target['paired_deltas']))<2e-14
 final_probe={}
 for endpoint in ('raw','h200'):
  rows=[r for r in j['documents'] if r['endpoint']==endpoint];assert len(rows)==108
  group=next(r for r in j['source_summary'] if r['source']=='all' and r['endpoint']==endpoint)
  assert abs(math.fsum(r['mean_interaction']*r['targets'] for r in rows)/100416-group['mean_interaction'])<1e-14
  assert sum(r['mean_interaction']>0 for r in rows)==group['positive_mean_documents']
  loo=[]
  for excluded in rows:
   remaining=[r for r in rows if r['document']!=excluded['document']]
   loo.append(math.fsum(r['mean_interaction']*r['targets'] for r in remaining)/sum(r['targets'] for r in remaining))
  sen=next(r for r in j['sensitivity'] if r['endpoint']==endpoint)
  assert max(abs(a-b) for a,b in zip([min(loo),max(loo)],sen['leave_one_out_range']))<1e-14
  for r in sen['remove_largest_positive_contributors']:
   remain=[x for x in rows if x['document'] not in r['removed_ids']]
   value=math.fsum(x['mean_interaction']*x['targets'] for x in remain)/sum(x['targets'] for x in remain)
   assert abs(value-r['remaining_interaction'])<1e-14
  selected=[r for r in rows if r['document'] in (0,1)]
  final_probe[endpoint]=dict(targets=sum(r['targets'] for r in selected),interaction=math.fsum(r['mean_interaction']*r['targets'] for r in selected)/4503)
 cleanup=read(ROOT/'research/storage-cleanup-2026-11-14-context-transfer-001/completed.json')
 assert cleanup['files_removed']==22 and not cleanup['skipped'] and cleanup['margin_bytes']>0
 cost=read(A/'context-transfer-planning-cost-v2.json');assert cost['margin_bytes']>0 and cost['components_bytes']['metadata']==2**30
 result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted=True,input_hashes_verified=len(j['input_sha256']),frozen_delta_recomputation_max_tolerance=2e-14,all_document_counts_and_leave_out_values_verified=True,final_same_document_probe=final_probe,cleanup=cleanup,test_opened=False,scientific_updates=0,source_sha256=sha(Path(__file__)),analysis_sha256=sha(A/'analysis.json'),report_sha256=sha(ROOT/'research/GENERAL_RSI_CROSS_SIZE_ATTRIBUTION_RESULTS.md'),planning_cost_sha256=sha(A/'context-transfer-planning-cost-v2.json'),revised_capacity=cost)
 with (A/'verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':run()
