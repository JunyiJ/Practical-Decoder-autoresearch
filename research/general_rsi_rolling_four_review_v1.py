"""Read-only numerical and artifact review of the completed four-condition study."""
import ast
import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/tokenizer-evidence/general-rsi-rolling-four-001'
BC = ROOT / 'research/tokenizer-evidence/general-rsi-rolling-context-001'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, bp, result = read(OUT/'plan.json'), read(BC/'plan.json'), read(OUT/'analysis.json')
        assert result['plan_sha256'] == sha(OUT/'plan.json')
        assert result['source_sha256'] == sha(ROOT/'research/general_rsi_rolling_four_analysis_v3.py')
        bound = {}
        for receipt in [bp, plan, read(OUT/'recovery-02.json'), read(OUT/'analysis-path-repair-03.json')]:
            for name, digest in receipt['sources'].items():
                assert sha(ROOT/name) == digest, name
                bound[name] = digest
        for name, digest in result['evidence'].items():
            assert sha(ROOT/name) == digest, name
        for e in plan['endpoints']:
            receipt = e['receipt']; p = Path(receipt['path'])
            assert p.stat().st_size == receipt['size'] and sha(p) == receipt['sha256']
        original = ROOT/'research/general_rsi_rolling_four_analysis_v1.py'
        repaired = ROOT/'research/general_rsi_rolling_four_analysis_v3.py'
        x = ast.parse(original.read_text().replace("OUT/'scoring-01", "OUT/'scoring-02"))
        y = ast.parse(repaired.read_text())
        assert ast.dump(ast.Module(body=x.body[1:], type_ignores=[])) == ast.dump(ast.Module(body=y.body[1:], type_ignores=[]))
        old_bc = subprocess.check_output(['git','show','44c209d8d:research/tokenizer-evidence/general-rsi-rolling-context-001/analysis.json'], cwd=ROOT)
        assert hashlib.sha256(old_bc).hexdigest() == sha(BC/'analysis.json')
        root = ROOT/'data/raw/rsi-general-10m-v1'
        manifest = read(root/'manifest.json')
        for n in ['val.bin','val.index.jsonl']:
            assert sha(root/n) == manifest['files'][n]['sha256']
        tokens = np.fromfile(root/'val.bin', dtype='<u2').astype(np.int64)
        docs = [json.loads(x) for x in (root/'val.index.jsonl').read_text().splitlines()]
        counts = np.array([x['tokens']-256 for x in docs])
        assert len(docs) == 108 and sum(counts) == 72876
        digests = []
        for doc,d in enumerate(docs):
            ph, ih = hashlib.sha256(), hashlib.sha256()
            for pos in range(256,d['tokens']):
                key = f'rolling-older-shuffle-v1|621116220001|val|{doc}|{pos}'
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:16],'big')
                order = np.random.Generator(np.random.PCG64(seed)).permutation(128)
                start = d['offset']+pos
                old, recent = tokens[start-256:start-128], tokens[start-128:start]
                window = np.r_[old[order], recent]
                assert np.array_equal(np.sort(window[:128]),np.sort(old))
                assert np.array_equal(window[128:],recent)
                ph.update(order.astype(np.uint8).tobytes()); ih.update(window.tobytes())
            digests.append((ph.hexdigest(),ih.hexdigest()))
        effects = {4:[],16:[]}; control_count = 0; control_error = 0.
        for e,ar in zip(plan['endpoints'],plan['saved_a']):
            name = f"d{e['depth']}-b{e['bundle']}"
            lookup = {}
            for c in read(ROOT/ar['path'])['chunks']:
                for j,(t,l) in enumerate(zip(c['target_ids'],c['token_losses'])):
                    lookup[c['document'],c['target_start']+j] = (t,l)
            sums = []
            for doc,d in enumerate(docs):
                b = read(BC/'scoring-01'/name/f'doc-{doc:03d}.json')
                sh = read(OUT/'scoring-02'/name/f'doc-{doc:03d}.json')
                pos = list(range(256,d['tokens'])); ids = tokens[d['offset']+256:d['offset']+d['tokens']].tolist()
                assert b['positions'] == sh['positions'] == pos
                assert b['target_ids'] == sh['target_ids'] == ids
                assert (sh['permutations_sha256'],sh['inputs_sha256']) == digests[doc]
                aa = []
                for j,p in enumerate(pos):
                    tid,loss = lookup[doc,d['offset']+p]; assert tid == ids[j]; aa.append(loss)
                    if p%128 == 0:
                        control_count += 1; control_error = max(control_error,abs(loss-b['losses']['128'][j]))
                arr = np.array([aa,b['losses']['128'],b['losses']['256'],sh['losses']],dtype=np.float64)
                assert arr.shape == (4,counts[doc]) and np.isfinite(arr).all()
                sums.append([math.fsum(row) for row in arr])
            sums = np.array(sums).T
            delta = np.array([sums[1]-sums[0],sums[2]-sums[1],sums[2]-sums[3],sums[3]-sums[1]])
            row = result['rows'][str(e['depth'])][e['bundle']-1]
            assert np.max(np.abs(delta-np.array(row['document_delta_loss_sums']))) < 1e-10
            for j,k in enumerate('ABCD'): assert abs(math.fsum(sums[j])/72876-row['ce'][k]) < 1e-12
            effects[e['depth']].append(delta)
        assert control_error < 1e-5 and control_count == 3762
        assert control_error == result['A_B_same_input_control']['max_abs_loss_difference']
        means = {d:np.mean(effects[d],axis=0) for d in (4,16)}
        effect = np.concatenate([means[4],means[16],means[16]-means[4]])
        rng = np.random.Generator(np.random.PCG64(plan['bootstrap_seed'])); samples = []
        for _ in range(100):
            ix = rng.integers(0,108,size=(1000,108))
            frequency = np.array([np.bincount(row,minlength=108) for row in ix])
            samples.append((effect @ frequency.T)/(frequency @ counts)[None,:])
        samples = np.concatenate(samples,axis=1); max_interval_error = 0.
        for gi,group in enumerate(['depth4','depth16','depth16_minus_depth4']):
            for j,key in enumerate(plan['contrasts']):
                bounds = np.quantile(samples[gi*4+j],plan['interval_quantiles'])
                err = float(np.max(np.abs(bounds-result['contrasts'][group][key]['interval'])))
                max_interval_error = max(max_interval_error,err); assert err < 1e-12
        resources = {}
        for label,path in [('BC',BC/'scoring-01'),('D',OUT/'scoring-02'),('D_qualification',OUT/'qualification-01')]:
            guard = read(path/'guard.json'); assert guard['status'] == 'finished'
            rows = [json.loads(x) for x in (path/'memory.jsonl').read_text().splitlines()]
            summary = dict(peak_rss_bytes=max(x['peak_rss_bytes'] for x in rows),peak_mps_driver_bytes=max(x['mps_driver_bytes'] for x in rows),minimum_free_bytes=min(x['free_bytes'] for x in rows),minimum_host_proxy_bytes=min(x['host']['reclaimable_proxy_bytes'] for x in rows),maximum_pressure=max(int(x['host']['pressure']) for x in rows),guard_seconds=guard['elapsed_seconds'])
            assert summary['peak_rss_bytes'] < 4*2**30 and summary['peak_mps_driver_bytes'] < 6*2**30
            assert summary['minimum_free_bytes'] > 12*2**30 and summary['minimum_host_proxy_bytes'] >= 2**30 and summary['maximum_pressure'] <= 2
            resources[label] = summary
        assert read(OUT/'analysis-03-guard.json')['status'] == 'finished'
        assert read(OUT/'analysis-03-exit.json')['returncode'] == 0
        assert sha(OUT/'scoring-02/recovery-failure.json') == read(OUT/'analysis-03-exit.json')['original_path_failure_sha256']
        assert 'scoring-01/result.json' in read(OUT/'scoring-02/recovery-failure.json')['traceback']
        review = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted=True,source_sha256=sha(__file__),analysis_sha256=sha(OUT/'analysis.json'),bound_source_count=len(bound),score_and_saved_A_hash_count=len(result['evidence']),endpoint_hashes_verified=6,matched_targets_per_model_condition=72876,models=6,conditions=4,independent_shuffle_targets=72876,A_B_control_targets=control_count,A_B_control_max_error=control_error,independent_frequency_bootstrap_draws=100000,maximum_interval_error=max_interval_error,original_BC_analysis_unchanged=True,path_repair_AST_verified=True,resources=resources,analysis_guard=read(OUT/'analysis-03-guard.json'),failures_preserved=True,test_opened=False,training_updates=0,current_free_bytes=shutil.disk_usage(ROOT).free)
        (OUT/'completion-review.json').write_text(json.dumps(review,indent=2)+'\n')
        print(json.dumps(review,indent=2))


if __name__ == '__main__':
    main()
