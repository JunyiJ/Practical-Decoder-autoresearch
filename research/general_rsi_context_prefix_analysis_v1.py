"""Audit every common-target score and apply the frozen context-use criterion."""
import datetime,fcntl,json,math,shutil,statistics
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put
from research.general_rsi_context_prefix_diagnostic_v1 import OUT

def read(p):return json.loads(p.read_text())
def main():
    with open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+') as l:
        fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=read(OUT/'plan.json');a=OUT/'attempt-01';r=read(a/'result.json');g=read(a/'guard.json');binding=sha(OUT/'plan.json')
        assert g['status']=='finished' and not list(a.glob('*failure*')) and binding==r['plan_sha256']==read(OUT/'admission.json')['plan_sha256']
        assert all(sha(ROOT/f)==h for f,h in p['sources'].items())
        rows=[];sequences=0;tokens=0;targets=0;parents=[];source_rows=[]
        assert len({w['document'] for w in p['windows']})==128
        for parent in p['parents']:
            slot=parent['parent']['slot'];scores=[json.loads(x) for x in (a/f'p{slot:02d}-scores.jsonl').read_text().splitlines()];assert len(scores)==384
            lookup={(x['window'],x['context']):x for x in scores};assert len(lookup)==384
            review=read(a/f'p{slot:02d}-review.json');assert review['parameters_unchanged'] and review['checkpoint_unchanged'] and review in r['reports']
            assert sha(parent['parent']['path'])==parent['parent']['sha256']
            for w in p['windows']:
                ce={};losses={}
                for c in p['contexts']:
                    s=lookup[(w['window'],c)];assert s['parent']==slot and s['document']==w['document'] and s['source']==w['source'] and s['target_start']==w['target_start']
                    assert s['target_ids']==w['target_ids'] and s['input_sha256']==w['input_sha256'][str(c)] and s['input_tokens']==c+31
                    assert s['targets']==32 and len(s['token_losses'])==32 and all(math.isfinite(x) for x in s['token_losses'])
                    assert math.fsum(s['token_losses'])==s['loss_sum'] and s['loss_sum']/32==s['ce']
                    ce[str(c)]=s['ce'];losses[c]=s['token_losses'];sequences+=1;tokens+=s['input_tokens'];targets+=32
                rows.append(dict(parent=slot,window=w['window'],document=w['document'],source=w['source'],ce=ce,delta96_minus48=ce['96']-ce['48'],delta48_minus16=ce['48']-ce['16'],target_quartile_delta96_minus48=[math.fsum(losses[96][i]-losses[48][i] for i in range(j,j+8))/8 for j in range(0,32,8)]))
            subset=[x for x in rows if x['parent']==slot]
            parents.append(dict(parent=slot,ce={str(c):statistics.mean(x['ce'][str(c)] for x in subset) for c in p['contexts']},delta96_minus48=statistics.mean(x['delta96_minus48'] for x in subset),delta48_minus16=statistics.mean(x['delta48_minus16'] for x in subset),windows_favorable96=sum(x['delta96_minus48']<0 for x in subset)))
            for src in p['source_document_counts']:
                ss=[x for x in subset if x['source']==src];assert len(ss)==64
                source_rows.append(dict(parent=slot,source=src,delta96_minus48=statistics.mean(x['delta96_minus48'] for x in ss),delta48_minus16=statistics.mean(x['delta48_minus16'] for x in ss)))
        assert sequences==r['sequences']==1152 and targets==r['scored_targets']==36864 and tokens==r['input_tokens']==97152
        sources=[dict(source=src,delta96_minus48=statistics.mean(x['delta96_minus48'] for x in source_rows if x['source']==src),delta48_minus16=statistics.mean(x['delta48_minus16'] for x in source_rows if x['source']==src)) for src in p['source_document_counts']]
        overall={k:statistics.mean(x[k] for x in parents) for k in ['delta96_minus48','delta48_minus16']}
        support=all(x['delta96_minus48']<0 for x in parents+sources)
        put(OUT/'analysis.json',dict(plan_sha256=binding,rows=rows,parent_means=parents,parent_source_means=source_rows,source_means=sources,overall=overall,primary_support=support,quality_adopted=False,scope='Source-balanced training-only common targets, context within127 input tokens; no beyond128 benefit or solution to generation loops established.'))
        review=dict(accepted_execution=True,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=binding,result_sha256=sha(a/'result.json'),analysis_sha256=sha(OUT/'analysis.json'),analysis_source_sha256=sha(__file__),source_bindings_unchanged=len(p['sources']),sequences=sequences,scored_targets=targets,input_tokens=tokens,unique_targets=4096,guard_seconds=g['elapsed_seconds'],peak_rss_bytes=g['observation']['peak_rss_bytes'],peak_mps_bytes=r['peak_mps_bytes'],checkpoints_unchanged=3,failures=0,lock_reacquired=True,metadata_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),free_bytes=shutil.disk_usage(ROOT).free,optimizer_updates=0,new_checkpoints=0,test_opened=False,validation_scored=False,quality_adopted=False)
        put(OUT/'completion-review.json',review);print(json.dumps(dict(parents=parents,sources=sources,overall=overall,primary_support=support,review=review),indent=2))
if __name__=='__main__':main()
