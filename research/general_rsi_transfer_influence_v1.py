"""Existing-score influence and literal format markers. Exploratory, no new CE."""
import datetime
import fcntl
import re
import time
import numpy as np
import tiktoken
from research.general_rsi_transfer_data_v1 import ROOT,P,D,read,put,sha

OUT=ROOT/'research/tokenizer-evidence/general-rsi-transfer-diagnosis-001'

def fraction_spans(text,pattern):
    return sum(m.end()-m.start() for m in re.finditer(pattern,text))/max(1,len(text))

def features(text):
    lines=text.splitlines(keepends=True)
    # Literal marker descriptors, deliberately not a reStructuredText parser.
    grid=[s for s in lines if (s.lstrip().startswith('|') and s.rstrip().endswith('|')) or re.fullmatch(r'\s*\+(?:[-=]+\+)+\s*',s)]
    return dict(chars=len(text),lines=len(lines),grid_line_character_fraction=sum(map(len,grid))/max(1,len(text)),long_space_character_fraction=fraction_spans(text,r' {4,}'),greek_placeholder_character_fraction=fraction_spans(text,r'\{~GREEK[^\n]*?~\}'),backtick_character_fraction=text.count('`')/max(1,len(text)),doctest_prompt_lines=sum(bool(re.match(r'^\s*>>>',s)) for s in lines),replacement_characters=text.count('\ufffd'))

def pearson(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None

def main():
    start=time.monotonic()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not read(ROOT/'research_state/general-rsi-state.json').get('stop_requested',False)
        import shutil
        free=shutil.disk_usage(ROOT).free
        assert free>=10*1024**3+16*1024**2
        dataset=read(P/'dataset.json');prior=read(P/'document-diagnostic.json');decision=read(P/'decision.json')
        assert sha(P/'dataset.json')==prior['dataset_sha256'] and sha(P/'decision.json')==prior['decision_sha256']
        assert sha(D/'excerpts.bin')==dataset['payloads'][0]['sha256']
        for file,h in decision['evidence_sha256'].items():assert sha(ROOT/file)==h
        ids=np.fromfile(D/'excerpts.bin',dtype='<u2');enc=tiktoken.get_encoding('gpt2');rows=[]
        for doc,previous in zip(dataset['documents'],prior['documents'],strict=True):
            assert doc['document']==previous['document'] and doc['name']==previous['name']
            text=enc.decode(ids[doc['offset']:doc['offset']+doc['tokens']].tolist())
            rows.append(dict(**previous,format=features(text)))
        summary={}
        for domain in ['pg19_books','python_library_rst']:
            rr=[r for r in rows if r['source']==domain];values=np.array([r['native_minus_byte'] for r in rr]);n=len(rr)
            assert n==32 and all(r['count']==3200 for r in rr)
            assert abs(values.mean()-decision['directions']['native']['domains'][domain]['mean_delta'])<1e-12
            order=sorted(rr,key=lambda r:(r['native_minus_byte'],r['document']))
            loo=(values.sum()-values)/(n-1)
            prefix=[]
            for k in [1,2,3,4]:
                removed=order[:k];total=sum(r['native_minus_byte'] for r in removed)
                prefix.append(dict(k=k,documents=[r['document'] for r in removed],contribution_to_original_mean=total/n,remaining_mean_for_sensitivity_only=(float(values.sum())-total)/(n-k)))
            summary[domain]=dict(original_mean=float(values.mean()),median_parent_delta=float(np.median(values)),leave_one_parent_out_mean_min=float(loo.min()),leave_one_parent_out_mean_max=float(loo.max()),native_favoring_prefix_sensitivity=prefix,format_correlations={k:pearson([r['format'][k] for r in rr],values) for k in ['grid_line_character_fraction','long_space_character_fraction','greek_placeholder_character_fraction','backtick_character_fraction','doctest_prompt_lines']})
        elapsed=time.monotonic()-start;assert elapsed<60
        result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),input_sha256={str(f.relative_to(ROOT)):sha(f) for f in [P/'dataset.json',P/'decision.json',P/'document-diagnostic.json',D/'excerpts.bin']},summary=summary,documents=rows,elapsed_seconds=elapsed,resource=dict(projected_complete_outputs_bytes=16*1024**2,reserve_bytes=10*1024**3,free_bytes_before=free,new_checkpoint_bytes=0,new_model_evaluations=0),scope='Post-result exploratory influence and literal format descriptors on all64 fixed excerpts. No acceptance changes, exclusions, bootstrap reruns or causal claims. Prefix removal is deliberately outcome-selected sensitivity, not a revised estimate or evidence to discard parents. Character descriptors include the first context token; loss excludes it. Markers are not parsed code/prose labels. Text may begin/end mid-line or mid-character. Shared source dependencies and multiple descriptive correlations are not controlled.',test_opened=False)
        put(OUT/'influence.json',result)
        assert sum(f.stat().st_size for f in OUT.iterdir() if f.is_file())<16*1024**2
        print(result['summary'])
if __name__=='__main__':main()
