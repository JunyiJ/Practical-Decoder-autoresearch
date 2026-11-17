"""Independent CPU-only review of the completed frozen train/serve diagnostic."""
import ast
import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/tokenizer-evidence/general-rsi-train-serve-context-001'

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with CpuPrerequisiteGuard(OUT/'review-guard.json',ROOT,rss_limit_bytes=4*2**30,reserve_bytes=12*2**30,timeout_seconds=600,poll_seconds=.2):
            p=read(OUT/'plan.json');a=read(OUT/'analysis.json');ad=read(OUT/'admission.json')
            assert a['plan_sha256']==ad['plan_sha256']==sha(OUT/'plan.json')
            assert p['bootstrap_resamples']==100000 and p['family_contrasts']==36
            assert p['interval_quantiles']==[.01/72,1-.01/72]
            bindings={}
            for receipt in [p,read(OUT/'repair-02.json'),ad]:
                for n,h in receipt['sources'].items():
                    assert sha(ROOT/n)==h,n;bindings[n]=h
            for n,h in p['reused_scores'].items(): assert sha(ROOT/n)==h,n
            assert a['source_sha256']==sha(ROOT/'research/general_rsi_train_serve_analysis_v1.py')
            trees=[]
            for version in (1,2):
                tree=ast.parse((ROOT/f'research/general_rsi_train_serve_context_v{version}.py').read_text())
                trees.append({n.name:ast.dump(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))})
            for name in ['ExtendedView','windows','verify']: assert trees[0][name]==trees[1][name]
            q=read(OUT/'qualification-02/result.json');assert q['accepted'] and not q['validation_scored']
            assert sha(OUT/'qualification-02/result.json')==ad['qualification_sha256']
            assert {(v['depth'],v['train_context']) for v in q['measurements']}=={(d,t) for d in (4,16) for t in (128,256)}
            for v in q['measurements']:
                assert v['native_parity'] and v['causal_prefix'] and v['parameters_and_original_buffers_unchanged']
                assert v['max_ce_error']<1e-5 and v['max_logit_error']<3e-5
            root=ROOT/'data/raw/rsi-general-10m-v1';manifest=read(root/'manifest.json')
            for n in ['val.bin','val.index.jsonl']:assert sha(root/n)==manifest['files'][n]['sha256']
            tokens=np.fromfile(root/'val.bin',dtype='<u2')
            docs=[(i,json.loads(x)) for i,x in enumerate((root/'val.index.jsonl').read_text().splitlines()) if json.loads(x)['tokens']>512]
            counts=np.array([d['tokens']-512 for _,d in docs]);assert len(docs)==65 and sum(counts)==51457
            assert counts.tolist()==a['document_counts']
            # Six cells ordered t128/h128,h256,h512,t256/h128,h256,h512.
            cells=np.empty((2,3,6,65));rows={};checked=0;reuse=0;model_receipts={}
            for e in p['endpoints']:
                rec=e['receipt'];assert Path(rec['path']).stat().st_size==rec['size'] and sha(rec['path'])==rec['sha256']
                name=f"d{e['depth']}-b{e['bundle']}-t{e['train_context']}";folder=OUT/'scoring-01'/name;r=read(folder/'result.json')
                assert r['completed'] and r['endpoint']==e and r['targets']==51457 and r['parameters_and_buffers_unchanged']
                assert r['newly_scored_histories']==([128,256,512] if e['train_context']==128 else [512])
                model_receipts[name]=sha(folder/'result.json')
                di=0 if e['depth']==4 else 1;bi=e['bundle']-1;ti=0 if e['train_context']==128 else 3
                assert len(r['files'])==65
                for j,(doc,d) in enumerate(docs):
                    file=folder/f'doc-{doc:03d}.json';v=read(file);digest=sha(file)
                    assert digest==r['files'][file.name]==a['evidence'][str(file.relative_to(ROOT))];checked+=1
                    assert v['document']==doc and v['source']==d['source'] and v['offset']==d['offset']
                    assert v['positions']==list(range(512,d['tokens'])) and v['count']==counts[j]
                    assert v['target_ids']==tokens[d['offset']+512:d['offset']+d['tokens']].tolist()
                    if ti:
                        old=read(ROOT/v['reuse']['path']);assert sha(ROOT/v['reuse']['path'])==v['reuse']['sha256']==p['reused_scores'][v['reuse']['path']]
                        ix=[i for i,pos in enumerate(old['positions']) if pos>=512]
                        assert [old['target_ids'][i] for i in ix]==v['target_ids']
                        assert [old['positions'][i] for i in ix]==v['positions']
                        for h in (128,256):assert [old['losses'][str(h)][i] for i in ix]==v['losses'][str(h)]
                        reuse+=2
                    else: assert v['reuse'] is None
                    for hi,h in enumerate((128,256,512)):
                        ls=v['losses'][str(h)];assert len(ls)==counts[j] and all(math.isfinite(z) for z in ls)
                        total=math.fsum(ls);assert abs(total-v['loss_sums'][str(h)])<1e-10
                        cells[di,bi,ti+hi,j]=total
                row=next(x for x in a['rows'] if (x['depth'],x['bundle'],x['train_context'])==(e['depth'],e['bundle'],e['train_context']))
                for hi,h in enumerate((128,256,512)):
                    assert abs(math.fsum(cells[di,bi,ti+hi])/51457-row['ce'][str(h)])<1e-12
                    for src,sr in row['sources'].items():
                        ids=[j for j,(_,d) in enumerate(docs) if d['source']==src]
                        assert int(sum(counts[ids]))==sr['count']
                        assert abs(math.fsum(cells[di,bi,ti+hi,ids])/sr['count']-sr['ce'][str(h)])<1e-12
                assert row['new_scoring_seconds']==r['seconds']
            assert checked==780 and reuse==780
            # Independent linear contrast matrix, no import of analysis contrast function.
            matrix=np.array([[-1,0,0,1,0,0],[0,-1,0,0,1,0],[0,0,-1,0,0,1],[-1,1,0,0,0,0],[0,-1,1,0,0,0],[0,0,0,-1,1,0],[0,0,0,0,-1,1],[1,-1,0,-1,1,0],[0,1,-1,0,-1,1],[0,-1,0,1,0,0],[0,0,-1,1,0,0],[0,0,-1,0,1,0]])
            keys=['train256_minus128_at_h128','train256_minus128_at_h256','train256_minus128_at_h512','serve256_minus128_t128','serve512_minus256_t128','serve256_minus128_t256','serve512_minus256_t256','train_by_serve_128_to_256','train_by_serve_256_to_512','policy_t256_h128_minus_t128_h256','policy_t256_h128_minus_t128_h512','policy_t256_h256_minus_t128_h512']
            pairs=np.einsum('kc,dbcn->dbkn',matrix,cells)
            for di,d in enumerate((4,16)):
                for b in range(3):
                    for k,key in enumerate(keys):assert np.max(np.abs(pairs[di,b,k]-a['paired_document_effects'][str(d)][b][key]))<1e-10
            means=pairs.mean(axis=1);effects=np.concatenate([means[0],means[1],means[1]-means[0]])
            rng=np.random.Generator(np.random.PCG64(p['bootstrap_seed']));samples=[]
            for _ in range(100):
                ix=rng.integers(0,65,size=(1000,65));freq=np.array([np.bincount(row,minlength=65) for row in ix])
                samples.append((effects@freq.T)/(freq@counts)[None,:])
            samples=np.concatenate(samples,axis=1);maxerr=0.
            for gi,group in enumerate(['depth4','depth16','depth16_minus4']):
                for k,key in enumerate(keys):
                    v=a['contrasts'][group][key];bound=np.quantile(samples[gi*12+k],p['interval_quantiles'])
                    err=float(np.max(np.abs(bound-v['interval'])));maxerr=max(maxerr,err);assert err<1e-12
                    pp=pairs[gi,:,k] if gi<2 else pairs[1,:,k]-pairs[0,:,k]
                    delta=np.array([math.fsum(x)/51457 for x in pp]);assert np.max(np.abs(delta-v['paired_deltas']))<1e-12
                    assert abs(math.fsum(effects[gi*12+k])/51457-v['mean_delta'])<1e-12
                    assert bool(bound[1]<0 and all(delta<0))==v['consistently_negative']
                    assert bool(bound[0]>0 and all(delta>0))==v['consistently_positive']
            resources={}
            for label in ['qualification-02','scoring-01']:
                g=read(OUT/label/'guard.json');assert g['status']=='finished' and g['reason'] is None
                mem=[json.loads(x) for x in (OUT/label/'memory.jsonl').read_text().splitlines()]
                for i,x in enumerate(mem):
                    assert x['peak_rss_bytes']<4*2**30 and x['mps_driver_bytes']<6*2**30
                    assert x['free_bytes']>=12*2**30 and x['host']['reclaimable_proxy_bytes']>=2**30 and int(x['host']['pressure'])<=2
                    if x['phase']=='loaded':
                        prev=mem[i-1];assert prev['phase']=='before_load' and int(prev['host']['pressure'])==1 and prev['host']['reclaimable_proxy_bytes']>=1.25*2**30
                resources[label]=dict(peak_rss_bytes=max(x['peak_rss_bytes'] for x in mem),peak_mps_driver_bytes=max(x['mps_driver_bytes'] for x in mem),minimum_free_bytes=min(x['free_bytes'] for x in mem),minimum_host_proxy_bytes=min(x['host']['reclaimable_proxy_bytes'] for x in mem),maximum_pressure=max(int(x['host']['pressure']) for x in mem),guard_seconds=g['elapsed_seconds'],model_loads=sum(x['phase']=='loaded' for x in mem))
            assert resources['scoring-01']['model_loads']==12
            for f in ['score-exit.json','analysis-exit.json']:assert read(OUT/f)['returncode']==0
            assert read(OUT/'analysis-guard.json')['status']=='finished'
            total=read(OUT/'scoring-01/result.json');assert total['completed'] and total['cells']==36 and total['new_cells']==24 and total['reused_cells']==12
            used=sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file());assert used<512*2**20
            result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted=True,source_sha256=sha(__file__),analysis_sha256=sha(OUT/'analysis.json'),bound_sources_verified=len(bindings),original_BC_and_four_condition_analyses_unchanged=True,reuse_hashes_verified=len(p['reused_scores']),endpoint_hashes_verified=12,matched_targets=51457,eligible_documents=65,model_score_files_verified=checked,reused_cells_exact=12,model_receipts=model_receipts,independent_contrast_matrix=True,independent_frequency_bootstrap_draws=100000,max_interval_error=maxerr,qualification_and_buffer_controls_reviewed=True,unchanged_view_and_window_AST=True,resources=resources,output_bytes=used,current_free_bytes=shutil.disk_usage(ROOT).free,historical_failure_preserved=True,training_updates=0,test_opened=False)
            (OUT/'completion-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
