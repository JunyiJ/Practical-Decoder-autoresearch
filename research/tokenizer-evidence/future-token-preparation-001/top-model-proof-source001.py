"""NEW independent AST/literal reviewer of TOP finite proof SOURCE only.

Never import that proof, model, target, objective, Torch or any extracted function.
"""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name, payload):
    data = (json.dumps(payload, sort_keys=True, allow_nan=False) + '\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file()) + len(data) + 65536 < 2097152
    with (R / name).open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def review(plan):
    assert plan['kind'] == 'NEW_TOP_model_math_proof_SOURCE_only'
    assert plan['numerical_execution_admitted'] is False and plan['initializer'] is None
    assert plan['exclusive_initializer_census'] is None and plan['runtime_plan'] is None
    assert plan['predecessor_source_bindings'] == 1260 and plan['future_source_bindings'] == 1262
    for path, sha in plan['sources'].items(): assert digest(ROOT / path) == sha
    text = (ROOT / plan['proof_source']).read_text()
    tree = ast.parse(text)
    constants = {}
    for node in tree.body:
        assert isinstance(node, (ast.Expr, ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef, ast.If))
        if isinstance(node, ast.Expr): assert isinstance(node.value, ast.Constant) and type(node.value.value) is str
        if isinstance(node, ast.Assign):
            key = node.targets[0].id
            if key != 'ROOT': constants[key] = ast.literal_eval(node.value)
        if isinstance(node, ast.If):
            assert ast.unparse(node.test) == "__name__ == '__main__'"
    assert constants['CASES'] == (
        ('unit_nearest_001', 1, 1, ((40001, 1117),)),
        ('repeated_tail_001', 3, 4, ((2107, 41, 2107, 913, 41, 49999, 2107), (87, 50003, 87, 111, 50003, 111, 209))),
        ('changed_tail_001', 3, 4, ((2107, 41, 2107, 112, 41, 112, 41), (87, 50003, 87, 208, 209, 208, 209))))
    case_ids = [case[0] for case in constants['CASES']]
    assert len(set(case_ids)) == 3
    for _, context, window, rows in constants['CASES']:
        assert 1 <= len(rows) <= 2 and 1 <= context <= 3 and window in (1, 4)
        assert all(len(row) == context + window and all(type(v) is int and 0 <= v < 50257 for v in row) for row in rows)
    before, after = constants['CASES'][1:]
    assert all(a[:3] == b[:3] and a[3:] != b[3:] for a, b in zip(before[3], after[3]))
    arms = constants['ARMS']; backends = constants['BACKENDS']
    assert len(arms) == 5 and len(set(arms)) == 5 and backends == ('cpu', 'mps')
    ids = [device + '-' + variant + '-' + case for device in backends for variant, case in arms]
    negatives = constants['NEGATIVE_IDS']; cpu_only = constants['CPU_ONLY_NEGATIVE_IDS']
    assert len(negatives) == len(set(negatives)) == 44 and cpu_only == ('gradient_sparse_cpu',)
    negative_ids = [device + '-negative-' + name for device in backends for name in negatives + (cpu_only if device == 'cpu' else ())]
    assert len(ids + negative_ids) == len(set(ids + negative_ids)) == 99
    assert constants['EXPECTED_REPORTS'] == dict(arms=10, transactions=89, tail_pairs=4, updates=4)
    assert constants['TOLERANCES'] == {'loss': (3e-5, 0.), 'logits': (4e-5, 4e-4), 'logit_gradient': (4e-6, 4e-5),
        'parameter_gradient': (4e-5, 4e-4), 'hidden_gradient': (4e-5, 4e-4), 'head_update': (4e-5, 4e-4), 'exact': (0., 0.)}
    assert constants['RESOURCE_CONTRACT'] == plan['numerical_design']['resources']
    assert constants['RESOURCE_CONTRACT']['output_bytes'] == 536870912
    assert plan['numerical_design']['entry_target_bytes'] == 11274289152
    assert plan['numerical_design']['call_counts'] == dict(
        own_model_constructions=10, independent_native_constructions=10, total_native_initializations=20,
        positive_training_forwards=10, uncached_primary_calls=10, independent_native_primary_forwards=10,
        independent_body_graphs=8, independent_scalar_calls=18, autograd_grad_calls=32,
        own_optimizer_constructions=10, invalid_optimizer_constructions=10,
        full_clips=4, own_steps=4, child_steps=8, EMA_calls=0, serialization_calls=0,
        numerical_model_transactions=89, CPU_phase_guards=10, whole_MPS_guards=1)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    def source(name): return ast.get_source_segment(text, functions[name])
    assert set(functions) == {'sha_file','json_safe','publish','independent_targets','independent_scalar','cpu_values',
        'dense_evidence','comparison','digest_tensor','snapshot','all_ok','graph_identity','reference_hidden',
        'negative_transaction','negatives','run_arm','numerical_body','main'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {'eval','exec','compile','__import__'}
    # Torch and candidate imports occur only in unexecuted function bodies.
    assert all(not (isinstance(n, ast.Import) and any(a.name.startswith(('torch','research')) for a in n.names))
               and not (isinstance(n, ast.ImportFrom) and (n.module or '').startswith(('torch','research')))
               for n in tree.body)
    for forbidden in ('future_token_model_v1', '62089790001', '62090710001', '62090290001', 'register_forward_hook', '.backward(', 'torch.save(', 'torch.load('):
        assert forbidden not in text
    assert "struct.unpack('<f', struct.pack('<f', mass / total))" in source('independent_targets')
    assert 'if token not in first:' in source('independent_targets') and 'range(1, window + 1)' in source('independent_targets')
    oracle = source('independent_scalar')
    assert all(term in oracle for term in ('math.fsum(exponentials)', 'math.fsum(weight for _, weight in weights)',
           'mass * exponential / denominator - by_id.get(token, 0.)', '/ len(rows)', 'math.log(denominator)'))
    assert 'sparse_proximity_ce' not in oracle and 'cross_entropy' not in oracle and 'softmax' not in oracle
    assert "value.detach().to('cpu').contiguous()" in source('cpu_values')
    compare = source('comparison')
    assert 'errors <= atol + rtol * ed.abs()' in compare and 'all_coordinate_mask' in compare
    assert 'mask.numpy().tobytes()' in compare and 'dense_evidence(e)' in compare and 'first_64_failures' in compare
    assert 'max_abs' in compare and 'bool(mask.all())' in compare
    graph = source('graph_identity')
    assert 'norm_first & norm_second' in graph and '== 1' in graph and 'next_functions' in graph
    snap = source('snapshot')
    assert all(term in snap for term in ('optimizer._items','own_inventory','foreign_parameters','buffers','extra_input_payload',
        'parameters_and_populated_gradslots','CPU_RNG','MPS_RNG','optimizer_state_sizes','OmegaConf.to_container'))
    negative = source('negative_transaction')
    assert negative.index('publish(run, ident') < negative.index("if not evidence['passed']")
    assert 'before == after' in negative and 'except ValueError as error' in negative and 'finally:' in negative
    payloads = source('negatives')
    assert all(term in payloads for term in ('raw.float()', 'raw.bool()', "device='meta'", 'raw[:0]', 'repeat(9, 1)',
          "raw_bad['raw_negative'][0, 0] = -1", "raw_bad['raw_oov'][0, 0] = 50257", "torch.autocast('cpu', dtype=torch.bfloat16)",
          "if device == 'cpu':", 'original_grad.to_sparse()', 'extra=value', "set(cases) != set(expected)"))
    arm = source('run_arm')
    assert 'torch.random.default_generator.manual_seed(initializer)' in arm
    assert arm.index('reference = Native(') < arm.index('model = TopModel(')
    assert 'torch.full_like(parameter, .125)' in arm and 'caller_preservation' in arm
    assert 'true_primary, _ = reference(inputs)' in arm and 'independent_scalar(primary_cpu, primary_rows)' in arm
    assert 'independent_scalar(cpu_values(ref_aux), target_rows)' in arm
    assert 'torch.autograd.grad(output.total_loss, tuple(p for _, p in own_items))' in arm
    assert 'torch.autograd.grad((ref_primary, ref_aux), tuple(p for _, p in ref_items)' in arm
    assert 'hidden_expected = (.5 * gp @' in arm and 'graph_identity(output.primary.values, output.auxiliary_logits)' in arm
    assert arm.index("publish(run, ident + '-math.json', checks)") < arm.index("raise AssertionError(ident + ' mathematical predicates')")
    assert arm.index('negatives(run, model, optimizer') < arm.index('clip_grad_norm_') < arm.index('optimizer.step()')
    assert 'before * (1. - .00006) - .00125 * clipped / (clipped.abs() + 1e-6)' in arm
    assert 'gradient_exposure' in arm and "publish(run, ident + '-update.json'" in arm
    main = source('main')
    assert main.index("plan.get('execution_admitted') is not True") < main.index('from research.block_norm_v1.mps_guard')
    assert 'census[\'prior_matches\'] == 0' in main and 'initializer_census_sha256' in main
    assert 'LOCK_EX | fcntl.LOCK_NB' in main and 'attempt.json' in main and "publish(run, 'STOP'" in main
    assert 'torch.mps.set_rng_state(caller_mps)' in source('numerical_body')
    assert 'counts == EXPECTED_REPORTS' in source('numerical_body') and 'timeout_seconds=120' in source('numerical_body')
    publish('review.json', {'status':'complete_NEW_TOP_model_math_proof_SOURCE_review_only',
        'accepted_source_preparation_only':True,'numerical_execution_admitted':False,
        'initializer_role_assigned':False,'future_source_bindings':1262,'proof_source_bytes':len(text.encode()),
        'positive_arm_case_ids':ids,'negative_transaction_ids':negative_ids,'sources':plan['sources'],
        'no_Torch_candidate_oracle_or_extracted_function_execution':True})


def main():
    assert len(sys.argv) == 2 and digest(R / 'plan.json') == sys.argv[1]
    plan = json.loads((R / 'plan.json').read_text())
    assert not any((R / name).exists() for name in ('attempt.json','STOP','review.json'))
    assert shutil.disk_usage(ROOT).free >= 10739515392
    for path, sha in plan['sources'].items(): assert digest(ROOT / path) == sha
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish('attempt.json', {'pid':os.getpid(),'plan_sha256':sys.argv[1],'no_replay':True,'no_retry':True})
        try:
            with CpuPrerequisiteGuard(R / 'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25): review(plan)
        except BaseException as error:
            publish('STOP',{'status':'closed_failed_NEW_TOP_proof_SOURCE_review','error':repr(error),'no_replay':True,'no_retry':True})
            raise
        else:
            publish('STOP',{'status':'complete_NEW_TOP_proof_SOURCE_review_only','no_replay':True,'no_retry':True})


if __name__ == '__main__': main()
