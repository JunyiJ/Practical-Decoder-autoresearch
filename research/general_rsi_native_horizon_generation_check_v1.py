"""Synthetic six-model mapping/aggregation check; never generates model text."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import time

from research import general_rsi_native_horizon_generation_analysis_v1 as analysis


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    start = time.monotonic()
    root, panel, output, binding = analysis.ROOT, analysis.P, analysis.OUT, analysis.BINDING
    plan = analysis.read(panel/'plan.json')
    assert analysis.sha(panel/'plan.json') == binding
    def summary_ast(path):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'summary'))
    assert summary_ast(root/'research/general_rsi_equal_time_generation_analysis_v1.py') == summary_ast(Path(analysis.__file__))
    try: analysis.prepare()
    except AssertionError as error: assert str(error) == 'Six accepted endpoints required'
    else: raise AssertionError('Incomplete scientific panel was accepted')
    captured = io.StringIO()
    try:
        with tempfile.TemporaryDirectory(prefix='rsi-horizon-generation-check-') as tmp:
            analysis.ROOT = Path(tmp)
            analysis.P = analysis.ROOT/'synthetic-panel'
            analysis.P.mkdir()
            analysis.OUT = analysis.P/'generation-evaluation'
            fixture = {k:plan[k] for k in ('slots', 'generation_protocol')}
            dump(analysis.P/'plan.json', fixture)
            analysis.BINDING = analysis.sha(analysis.P/'plan.json')
            for slot, config in enumerate(plan['slots']):
                d = analysis.P/f'slot-{slot:02d}'
                (d/'generation').mkdir(parents=True)
                checkpoint = hashlib.sha256(f'synthetic-{slot}'.encode()).hexdigest()
                r = dict(binding=analysis.BINDING, slot=config, endpoints=[{}, dict(weights=dict(sha256=checkpoint))])
                dump(d/'result.json', r)
                dump(d/'review.json', dict(accepted_execution=True, binding=analysis.BINDING, result_sha256=analysis.sha(d/'result.json')))
                tokens = [3,4]*6 if config['horizon'] == 'short' else [5,6,7,50256]
                rows = [dict(id=f"{p['id']}-{decoder}", prompt=p, decoder=decoder,
                    checkpoint_sha256=checkpoint, generated_token_ids=tokens,
                    continuation='Synthetic fixture only.', metrics=analysis.score(tokens))
                    for decoder in plan['generation_protocol']['decoders'] for p in plan['generation_protocol']['prompts']]
                dump(d/'generation/result.json', dict(binding=analysis.BINDING, rows=rows, checkpoint=dict(sha256=checkpoint)))
            with contextlib.redirect_stdout(captured): analysis.prepare()
            blind = analysis.read(analysis.OUT/'blind.json')
            mapping = analysis.read(analysis.OUT/'mapping.json')
            assert len(blind) == len(mapping) == 144
            assert all(set(row) == {'id','prompt','continuation'} for row in blind)
            assert sorted((r['slot'],r['horizon'],r['bundle']) for r in mapping) == sorted(
                [(slot,horizon,bundle) for slot,horizon,bundle in
                 [(0,'short',1),(1,'long',1),(2,'short',2),(3,'long',2),(4,'short',3),(5,'long',3)] for _ in range(24)])
            annotations = []
            for b,m in zip(blind,mapping):
                assert b['id'] == m['id']
                unassessable = b['prompt']['id'] == 'p00'
                annotations.append(dict(id=b['id'], relevance='unassessable' if unassessable else (0 if m['horizon']=='short' else 2),
                    consistency='unassessable' if unassessable else (1 if m['horizon']=='short' else 2),
                    evidence_quote='Synthetic fixture only.', reason='Synthetic control.', annotator_identity='synthetic fixture'))
            dump(analysis.OUT/'annotations.json', annotations)
            metrics_path = analysis.OUT/'metrics.json'
            original = metrics_path.read_bytes()
            metrics_path.write_bytes(original+b' ')
            try: analysis.aggregate()
            except AssertionError: pass
            else: raise AssertionError('Modified metrics were accepted')
            metrics_path.write_bytes(original)
            with contextlib.redirect_stdout(captured): analysis.aggregate()
            result = analysis.read(analysis.OUT/'summary.json')
            for horizon, relevance, consistency, loop in [('short',0,1,1),('long',2,2,0)]:
                for decoder in plan['generation_protocol']['decoders']:
                    group = result['groups'][horizon][decoder]
                    assert group['all_seeds']['count'] == 36
                    assert group['all_seeds']['relevance']['assessable_mean'] == relevance
                    assert group['all_seeds']['consistency']['equal_weight_prompt_mean'] == consistency
                    assert group['all_seeds']['relevance']['unassessable_fraction'] == 1/12
                    assert group['all_seeds']['relevance']['assessable_prompt_count'] == 11
                    assert group['all_seeds']['terminal_loop']['mean'] == loop
                    assert all(x['count'] == 12 for x in group['per_seed'].values())
    finally:
        analysis.ROOT, analysis.P, analysis.OUT, analysis.BINDING = root, panel, output, binding
    paths = [Path(__file__), Path(analysis.__file__), root/'research/general_rsi_equal_time_generation_analysis_v1.py']
    analysis.put(panel/'generation-analysis-check.json', dict(qualified=True, plan_sha256=binding,
        source_sha256={str(p.relative_to(root)):analysis.sha(p) for p in paths},
        seconds=time.monotonic()-start, input_kind='144 synthetic rows in temporary fixture; no model generation',
        checks=dict(incomplete_panel_rejected=True, inherited_summary_ast_unchanged=True,
            blinded_fields_only=True, six_slot_horizon_and_bundle_mapping=True, tampered_metrics_rejected=True,
            separate_decoder_groups=True, per_seed_counts=True, unassessable_excluded=True,
            prompt_equal_weighting=True, distinct_horizon_repetition_controls=True),
        scientific_outputs_read=False, test_opened=False))
    print('Synthetic generation aggregation checks passed; no scientific outputs read.')


if __name__ == '__main__': main()
