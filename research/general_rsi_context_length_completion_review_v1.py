"""Audit the completed context comparison without training or rerating outputs."""
import datetime
import fcntl
import json
import shutil
from pathlib import Path
import numpy as np
from research.general_rsi_context_length_analysis_v1 import ROOT, PANEL, PAIRS, read, sha, analyze, decide
from research.general_rsi_context_length_generation_analysis_v1 import aggregate


def main():
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = read(PANEL/'decision.json')
        replay = json.loads(json.dumps(analyze()))  # Rechecks frozen sources, reviews and all12 weight hashes.
        assert {k:v for k,v in saved.items() if k!='at'} == {k:v for k,v in replay.items() if k!='at'}
        plan = read(PANEL/'plan.json')
        contract = plan['acceptance']
        pairs = []
        for a,b in PAIRS:
            x=read(PANEL/f'slot-{a:02d}/time_ema-validation.json')['documents']
            y=read(PANEL/f'slot-{b:02d}/time_ema-validation.json')['documents']
            pairs.append([q['loss_sum']-p['loss_sum'] for p,q in zip(x,y)])
        counts=np.array([p['count'] for p in x]); meanloss=np.mean(pairs,axis=0)
        rng=np.random.Generator(np.random.PCG64(contract['bootstrap']['seed']))
        # Independent per-draw frequency-weighted implementation, no bootstrap helper.
        draws=[]
        for _ in range(contract['bootstrap']['replicates']):
            frequencies=np.bincount(rng.integers(0,len(counts),len(counts)),minlength=len(counts))
            draws.append(float(np.dot(frequencies,meanloss)/np.dot(frequencies,counts)))
        upper=float(np.quantile(draws,.99,method='linear'))
        assert abs(upper-saved['decision']['bootstrap_upper_99'])<1e-12
        # Synthetic decision counterexamples are execution checks, not research runs.
        sources=['fineweb_edu','wikipedia']; tiny=dict(contract,bootstrap=dict(contract['bootstrap'],replicates=50))
        cases=[('benefit',[[-.03,-.03]]*3,True),('too_small',[[-.01,-.01]]*3,False),('one_bad_pair',[[-.2,-.2],[-.2,-.2],[.001,.001]],False),('source_regression',[[-.3,.06]]*3,False),('zero',[[0,0]]*3,False)]
        for name,delta,expected in cases:
            assert decide(delta,[1,1],sources,tiny)['context256_accepted']==expected,name
        import research.general_rsi_context_length_generation_analysis_v1 as generation
        outputs={}
        oldput=generation.put
        generation.put=lambda path,value: outputs.update({path.name:value})
        try: aggregate()
        finally: generation.put=oldput
        assert outputs['summary.json']==read(PANEL/'generation-evaluation/summary.json')
        bound=[PANEL/'decision.json',PANEL/'generation-evaluation/summary.json',PANEL/'generation-evaluation/annotations.json',PANEL/'generation-evaluation/preparation.json']
        bound += [PANEL/f'slot-{i:02d}/review.json' for i in range(6)]
        receipt=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_six_execution_reviews_accepted=True,
            decision_replay_exact=True,independent_frequency_bootstrap_upper_99=upper,gate_counterexamples_passed=[x[0] for x in cases],
            generation_aggregate_replay_exact=True,all_144_first_outputs_rated=True,annotator='Codex agent; identity-hidden exploratory text review, not independent human validation',
            all_12_final_weights_hash_verified=True,retained_weight_bytes=sum(e['weights']['size'] for i in range(6) for e in read(PANEL/f'slot-{i:02d}/result.json')['endpoints']),
            shared_lock_available=True,process_inspection='ps inspection immediately before audit found no research worker; completed slot05 supervisor exit0 retained',
            free_bytes=shutil.disk_usage(ROOT).free,evidence_sha256={str(p.relative_to(ROOT)):sha(p) for p in bound},review_source_sha256=sha(Path(__file__)),test_opened=False)
        with (PANEL/'completion-review.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
        print(json.dumps(receipt,indent=2))
if __name__=='__main__': main()
