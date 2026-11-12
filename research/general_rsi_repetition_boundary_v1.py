"""Post-hoc existing-data diagnostic. No model loading, generation or test access."""
import fcntl
import hashlib
import json
import statistics
import time
from pathlib import Path
import numpy as np
import tiktoken
from research.general_rsi_generation_metrics_v1 import score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/tokenizer-evidence/general-rsi-repetition-boundary-001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_duplicate(tokens):
    """1-based endpoint of the first repeated generated trigram, overlaps allowed."""
    seen = set()
    for i in range(len(tokens)-2):
        tri = tuple(tokens[i:i+3])
        if tri in seen:
            return i+3
        seen.add(tri)
    return None


def summarize(rows):
    onsets = [r['first_duplicate_trigram_endpoint'] for r in rows
              if r['first_duplicate_trigram_endpoint'] is not None]
    return dict(n=len(rows), repeated_trigram_mean=statistics.mean(
        r['metrics']['repeated_trigram_fraction'] for r in rows),
        terminal_loops=sum(r['metrics']['terminal_loop'] for r in rows),
        any_repeated_trigram=len(onsets), first_repeat_endpoint_median=(statistics.median(onsets) if onsets else None),
        repeats_by_token32=sum(x<=32 for x in onsets),
        eos=sum(r['metrics']['stopped_eos'] for r in rows),
        cap=sum(r['metrics']['hit_token_cap'] for r in rows))


def main():
    start=time.monotonic()
    assert first_duplicate([1,2,3,1,2,3]) == 6
    assert first_duplicate([1,1,1,1]) == 4
    assert first_duplicate([1,2,3,4]) is None
    lock=(ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not OUT.exists()
    inputs={}
    def read(path):
        path=ROOT/path; inputs[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text())
    protocol=read('research/general-rsi-generation-eval-v1.json')
    enc=tiktoken.get_encoding('gpt2')
    lengths={p['id']:len(enc.encode(p['text'])) for p in protocol['prompts']}
    assert max(lengths.values())+64<=128
    rows=[]
    base='research/tokenizer-evidence/general-rsi-native-horizon-001'
    for slot in range(6):
        review=read(f'{base}/slot-{slot:02d}/review.json')
        assert review['accepted_execution'] and not review['test_opened']
        gen=read(f'{base}/slot-{slot:02d}/generation/result.json')
        assert len(gen['rows'])==24
        for r in gen['rows']:
            tokens=r['generated_token_ids']
            metrics=score(tokens)
            assert metrics==r['metrics']
            body=tokens[:-1] if metrics['stopped_eos'] else tokens
            rows.append(dict(slot=slot,horizon='long' if slot%2 else 'short',
                id=r['id'],decoder=r['decoder'],prompt_tokens=lengths[r['prompt']['id']],
                total_tokens=lengths[r['prompt']['id']]+len(tokens),
                first_duplicate_trigram_endpoint=first_duplicate(body),metrics=metrics))
    data=Path('data/raw/rsi-general-10m-v1')
    manifest=read(data/'manifest.json')
    for name in ['train.bin','train.index.jsonl']:
        p=ROOT/data/name; inputs[str(data/name)]=sha(p)
        assert inputs[str(data/name)]==manifest['files'][name]['sha256']
    tokens=np.fromfile(ROOT/data/'train.bin',dtype='<u2')
    docs=[json.loads(x) for x in (ROOT/data/'train.index.jsonl').read_text().splitlines()]
    natural=[];skipped=0
    for i,d in enumerate(docs):
        n=d['tokens']-1;assert int(tokens[d['offset']+n])==50256
        if n<64:
            skipped+=1;continue
        offset=d['offset']+(n-64)//2
        window=tokens[offset:offset+64].tolist();assert 50256 not in window
        natural.append(dict(document_index=i,source=d['source'],offset=offset,
            first_duplicate_trigram_endpoint=first_duplicate(window),metrics=score(window)))
    # A window boundary is not natural EOS. Only compare repetition metrics.
    natural_summary={s:summarize([r for r in natural if r['source']==s]) for s in sorted({r['source'] for r in natural})}
    for v in natural_summary.values():
        del v['eos'];del v['cap']
    groups={f'{h}/{d}':summarize([r for r in rows if r['horizon']==h and r['decoder']==d])
            for h in ['short','long'] for d in protocol['decoders']}
    by_slot={f'{s}/{d}':summarize([r for r in rows if r['slot']==s and r['decoder']==d])
            for s in range(6) for d in protocol['decoders']}
    inputs['research/general_rsi_generation_v2.py']=sha(ROOT/'research/general_rsi_generation_v2.py')
    inputs['research/general_rsi_generation_metrics_v1.py']=sha(ROOT/'research/general_rsi_generation_metrics_v1.py')
    result=dict(kind='posthoc_repetition_context_and_training_window_diagnostic',
        inputs=inputs,source_sha256=sha(Path(__file__)),test_opened=False,new_model_outputs=0,
        synthetic_checks_passed=3,lock='nonblocking acquired',prompt_token_lengths=lengths,
        max_total_tokens=max(r['total_tokens'] for r in rows),groups=groups,by_slot=by_slot,
        training_window_policy='one centered64-token non-EOS window per eligible training document; all documents; equal-document descriptive weights',
        training_windows=len(natural),training_documents_skipped=skipped,training=natural_summary,
        limits=['Post-hoc diagnostic, no changed frozen decision or generation rating.',
                'Training windows are not matched to prompts or a population-random sample.',
                'Token trigrams include punctuation/subwords; repetitions can be legitimate.',
                'No probabilities measured: self-reinforcement is a hypothesis, not established.',
                'No EOS conclusion from fixed training windows or64-token caps.'],
        generation_rows=rows,training_window_rows=natural,seconds=time.monotonic()-start)
    OUT.mkdir(parents=True)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['generation_rows','training_window_rows','inputs','by_slot']},indent=2))


if __name__=='__main__':main()
