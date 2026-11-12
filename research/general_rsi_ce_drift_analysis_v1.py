"""Read-only audit of retained CE continuation curves, scores and weight geometry."""
import fcntl,gc,json,math,os,shutil,statistics,time,traceback
from pathlib import Path
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,check_stop
OUT=ROOT/'research/tokenizer-evidence/general-rsi-ce-drift-001'
P=ROOT/'research/tokenizer-evidence/general-rsi-pseudo-repetition-001'
N=ROOT/'research/tokenizer-evidence/general-rsi-native-horizon-001'

def main():
    os.chdir(ROOT);tick=time.monotonic();check_stop()
    lock=open('research_state/tokenizer-worker-lock/.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists();OUT.mkdir();G=1024**3
    assert shutil.disk_usage(ROOT).free>=10*G+64*1024**2
    bindings={}
    def read(path):
        bindings[str(Path(path).relative_to(ROOT))]=sha(path);return json.loads(Path(path).read_text())
    panel=read(P/'plan.json');assert len(panel['slots'])==6
    configs=[c for c in panel['slots'] if c['recipe']=='native_ce'];assert [c['slot'] for c in configs]==[0,2,4]
    for path in [Path(__file__).resolve(),ROOT/'research/general_rsi_pseudo_training_v1.py',ROOT/'research/general_rsi_pseudo_worker_v1.py',ROOT/'research/byte_input_model_v6/optimizer.py']:
        bindings[str(path.relative_to(ROOT))]=sha(path)
    put(OUT/'scope.json',dict(created_unix=time.time(),kind='existing_evidence_audit',model_scores=0,test_opened=False,
        subjects=[c['slot'] for c in configs],online_step_bins=[[1,100],[101,200],[201,500]],online_time_bins_seconds=[[0,400],[400,800],[800,1201]],
        forecast_seconds=90,ceiling_seconds=180,rss_limit_bytes=2*G,reserve_bytes=10*G,metadata_cap_bytes=32*1024**2,
        serialization_bytes=0,overlap_bytes=32*1024**2,free_bytes=shutil.disk_usage(ROOT).free,lock_acquired=True,
        interpretation='Descriptive audit. Separate fixed validation subset/full validation, sampled training probes and online minibatch losses. Weight geometry does not identify causes. No old quality decision changes.'))
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    try:
        with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=180,rss_limit_bytes=2*G,reserve_bytes=10*G):
            import torch
            assert torch.__version__==panel['torch_version'];torch.set_num_threads(4)
            def subset(score):
                rows=[r for r in score['documents'] if r['document'] in panel['probe_documents']]
                assert [r['document'] for r in rows]==panel['probe_documents'] and sum(r['count'] for r in rows)==8586
                return math.fsum(r['loss_sum'] for r in rows)/8586
            def weights(rec):
                path=Path(rec['path']);assert path.stat().st_size==rec['size'] and sha(path)==rec['sha256']
                bindings[str(path.relative_to(ROOT))]=rec['sha256'];return torch.load(path,map_location='cpu',weights_only=True)['parameters']
            def geometry(parent,current):
                assert list(parent)==list(current);s={g:dict(parent_sq=0.,current_sq=0.,difference_sq=0.,dot=0.) for g in ['Muon','AdamW']}
                for name,a in parent.items():
                    b=current[name];assert a.shape==b.shape and a.dtype==b.dtype and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
                    group='Muon' if a.ndim==2 and name.startswith('transformer.h.') else 'AdamW'
                    x=a.double();y=b.double();v=s[group];v['parent_sq']+=float(x.square().sum());v['current_sq']+=float(y.square().sum());v['difference_sq']+=float((y-x).square().sum());v['dot']+=float((x*y).sum())
                return {g:dict(parent_norm=math.sqrt(v['parent_sq']),current_norm=math.sqrt(v['current_sq']),norm_ratio=math.sqrt(v['current_sq']/v['parent_sq']),relative_displacement=math.sqrt(v['difference_sq']/v['parent_sq']),cosine=v['dot']/math.sqrt(v['current_sq']*v['parent_sq'])) for g,v in s.items()}
            rows=[]
            for c in configs:
                check_stop();slot=c['slot'];parent=c['parent']['slot'];folder=P/f'slot-{slot:02d}'
                review=read(folder/'review.json');assert review['accepted_execution']
                result=read(folder/'result.json');assert sha(folder/'result.json')==review['result_sha256']
                base=read(N/f'slot-{parent:02d}/time_ema-validation.json');assert base['count']==100416
                log=folder/'steps-01.jsonl';bindings[str(log.relative_to(ROOT))]=sha(log);history=[json.loads(l) for l in log.read_text().splitlines()]
                assert len(history)==result['steps'] and all(r['objective']=='ce' and r['step']==i for i,r in enumerate(history,1))
                assert not result['recovery_charges']
                endpoint=[];parent_weights=weights(c['parent'])
                for e in result['endpoints']:
                    score=read(folder/(e['endpoint']+'-validation.json'));assert score['ce']==e['ce'] and score['count']==100416
                    assert [(r['document'],r['source'],r['count']) for r in score['documents']]==[(r['document'],r['source'],r['count']) for r in base['documents']]
                    current=weights(e['weights']);geo=geometry(parent_weights,current);del current;gc.collect()
                    endpoint.append(dict(endpoint=e['endpoint'],ce=score['ce'],delta_from_parent=score['ce']-base['ce'],subset_ce=subset(score),subset_delta=subset(score)-subset(base),
                        source_deltas={k:score['sources'][k]['ce']-base['sources'][k]['ce'] for k in base['sources']},geometry=geo))
                del parent_weights;gc.collect()
                probes=[]
                for fraction in [1,2]:
                    score=read(folder/f'probe-{fraction}.json');boundary=next(r for r in history if r['end_ns']>=c['budget_ns']*fraction/3)
                    assert score['count']==8586 and subset(score)==score['ce']
                    probes.append(dict(fraction=fraction,step=boundary['step'],seconds=boundary['end_ns']/1e9,ce=score['ce'],delta_from_parent_subset=score['ce']-subset(base)))
                def summarize(selected):
                    return dict(updates=len(selected),mean_online_ce=statistics.mean(r['loss'] for r in selected),median_gradient_norm=statistics.median(r['clip']['pre_norm'] for r in selected),clipped=sum(r['clip']['coefficient']<1 for r in selected),first_lr_ratio=selected[0]['lr_ratio'],last_lr_ratio=selected[-1]['lr_ratio'])
                bins=[dict(step_range=[a,b],**summarize(history[a-1:b])) for a,b in [(1,100),(101,200),(201,500)]]
                times=[dict(start_seconds=a,end_seconds=b,**summarize([r for r in history if a<=r['begin_ns']/1e9<b])) for a,b in [(0,400),(400,800),(800,1201)]]
                rows.append(dict(continuation_slot=slot,parent_slot=parent,parent_ce=base['ce'],parent_subset_ce=subset(base),steps=result['steps'],training_seconds=result['training_ns']/1e9,
                    endpoints=endpoint,probes=probes,online_step_bins=bins,online_time_bins=times,
                    independent_decay_only_factor=(1-.00006)**result['steps'],decay_factor_scope='Algebraic no-gradient shrinkage factor only; actual updates also include learned gradient terms.'))
            short=[]
            for campaign,arm in [('general-rsi-pseudo-accumulated-001','ce'),('general-rsi-joint-diagnostic-001','ce_full')]:
                for parent in [1,3,5]:
                    folder=ROOT/'research/tokenizer-evidence'/campaign/'attempt-01';initial=read(folder/f'p{parent:02d}-initial.json');base=initial.get('evaluation',initial.get('scores',{}).get('natural'))
                    points=[]
                    for n in [1,10,25,50,100]:
                        v=read(folder/f'p{parent:02d}-{arm}-n{n:03d}.json');score=v.get('evaluation',v.get('scores',{}).get('natural'));assert score['targets']==8192
                        points.append(dict(n=n,ce=score['ce'],delta_from_parent=score['ce']-base['ce']))
                    short.append(dict(campaign=campaign,parent_slot=parent,initial_ce=base['ce'],points=points))
            means={endpoint:statistics.mean(next(e for e in r['endpoints'] if e['endpoint']==endpoint)['delta_from_parent'] for r in rows) for endpoint in ['raw','time_ema']}
            put(OUT/'analysis.json',dict(rows=rows,short_training_probes=short,mean_full_validation_delta=means,source_bindings=bindings,
                total_seconds=time.monotonic()-tick,model_scores_computed=0,test_opened=False,quality_decision_changed=False))
            assert all(sha(ROOT/p)==h for p,h in bindings.items())
            print(json.dumps(dict(mean_full_validation_delta=means,rows=[{k:r[k] for k in ['parent_slot','probes','independent_decay_only_factor']} for r in rows]),indent=2))
    except BaseException:
        put(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
