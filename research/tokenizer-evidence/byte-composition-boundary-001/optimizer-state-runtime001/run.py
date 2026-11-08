"""Read-only validation comparison on the accepted RAW optimizer state."""
import ast
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(1024*1024): h.update(chunk)
    return h.hexdigest()


def read(p): return json.loads(p.read_text())


def publish(name, value):
    with (OUT/name).open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def method(source, class_name):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    n = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_validate_state')
    return '\n'.join(source.splitlines()[n.lineno-1:n.end_lineno])+'\n'


def run():
    assert not (OUT/'guard.json').exists()
    plan = read(OUT/'plan.json'); assert sha(Path(__file__)) == plan['runner_sha256']
    for r in plan['bindings']:
        p = ROOT/r['path']; assert p.stat().st_size == r['size'] and sha(p) == r['sha256']
    mapping = read(OUT.parent/'optimizer-state-source001/source-map.json')
    for r in (mapping['original'], mapping['candidate']):
        p = ROOT/r['path']; assert p.stat().st_size == r['size'] and sha(p) == r['sha256']
    restored = method((ROOT/mapping['candidate']['path']).read_text(), 'BatchedStateOptimizer')
    for old, new in reversed(mapping['exact_replacements']): restored = restored.replace(new, old)
    assert restored == method((ROOT/mapping['original']['path']).read_text(), 'ByteInputOptimizer')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from research.byte_input_quality_recovery_v1.guard import Guard
        with Guard(OUT/'guard.json', ROOT, timeout_seconds=180, rss_limit_bytes=4*1024**3,
                   reserve_bytes=8*1024**3, weight_roots=[ROOT/'research_runs/byte-input-quality-v1-001',
                       ROOT/'research_runs/byte-input-quality-recovery-v1-001']) as guard:
            import torch
            from research import byte_input_checkpoint_consumer_v3 as consumer
            from research import byte_input_training_v4 as training
            from research import byte_input_state_v4 as numerical
            from research.byte_input_model_v5 import build_optimizer
            from research.byte_input_batched_optimizer_state_v1 import BatchedStateOptimizer
            torch.set_num_threads(4); assert torch.backends.mps.is_available(); guard.torch = torch
            cpu_rng = torch.get_rng_state().clone(); mps_rng = torch.mps.get_rng_state().clone()
            recipe = read(OUT.parent/'finite-batch-adoption001/recipe.json'); loader = recipe['loader']
            raw = training.read_bound(recipe['representative']['parent'])
            cases = []
            for backend in ('cpu', 'mps'):
                model, identity, receipt = consumer.load(loader['implementation_record'], loader['plan_record'],
                                                         loader['acceptance_record'], backend)
                reference = build_optimizer(model, .0025)
                candidate = BatchedStateOptimizer(model, .0025)
                optimizers = dict(reference=reference, batched=candidate)
                # Test lazy-state acceptance before restoring the accepted endpoint.
                for optimizer in optimizers.values():
                    optimizer.validate(); assert all(not c.state for c in optimizer._optimizers)
                # Only transport backend metadata; every saved numerical value is unchanged.
                payload = dict(raw['numerical'], backend=backend)
                try:
                    numerical.restore_pristine(model, reference, payload, identity['binding'])
                finally:
                    torch.set_rng_state(cpu_rng); torch.mps.set_rng_state(mps_rng)
                assert reference._steps == 1200
                # Copy only the independently validated saved state into the isolated prototype.
                parameters = dict(model.ordered_parameters())
                for child, saved in zip(candidate._optimizers, payload['children'], strict=True):
                    for group, record in zip(child.param_groups, saved['groups'], strict=True):
                        group['lr'] = record['options']['lr']
                    for name, fields in saved['state'].items():
                        child.state[parameters[name]] = {k: v.to('cpu' if k == 'step' else backend).clone()
                                                        for k, v in fields.items()}
                candidate._steps = reference._steps
                for optimizer in optimizers.values(): optimizer.validate()
                def tensors(optimizer):
                    rows = [('parameter:'+n, p) for n, p in model.ordered_parameters()]
                    rows += [('buffer:'+n, p) for n, p in model.named_buffers()]
                    rows += [('gradient:'+n, p.grad) for n, p in model.ordered_parameters() if p.grad is not None]
                    for index, child in enumerate(optimizer._optimizers):
                        for parameter, fields in child.state.items():
                            name = next((n for n, p in model.ordered_parameters() if p is parameter), 'foreign')
                            rows += [(f'state:{index}:{name}:{k}', v) for k, v in fields.items()]
                    return rows
                def snapshot(optimizer):
                    return dict(steps=optimizer._steps,
                        child_keys=[[id(p) for p in c.state] for c in optimizer._optimizers],
                        rows=[(n, id(t), t.untyped_storage().data_ptr(), t._version, str(t.dtype),
                               str(t.device), list(t.shape), t.requires_grad,
                               hashlib.sha256(t.detach().to('cpu').numpy().tobytes()).hexdigest())
                              for n,t in tensors(optimizer)],
                        groups=[{k:v for k,v in g.items() if k != 'params'} for g in optimizer.param_groups])
                # Compare the numerical values, independent of allocation addresses.
                assert [(n, str(t.dtype), list(t.shape), hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest())
                        for n,t in tensors(reference) if n.startswith('state:')] == [
                       (n, str(t.dtype), list(t.shape), hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest())
                        for n,t in tensors(candidate) if n.startswith('state:')]
                timings = {}
                for scope in ('state_only', 'complete_validate'):
                    samples = {k:[] for k in optimizers}
                    def invoke(optimizer):
                        if scope == 'state_only': optimizer._validate_state(torch.device(backend))
                        else: optimizer.validate()
                    for optimizer in optimizers.values():
                        for _ in range(2): invoke(optimizer)
                    for repeat in range(7):
                        for label in (list(optimizers) if repeat % 2 == 0 else list(reversed(optimizers))):
                            if backend == 'mps': torch.mps.synchronize()
                            started = time.perf_counter(); invoke(optimizers[label])
                            if backend == 'mps': torch.mps.synchronize()
                            samples[label].append(time.perf_counter()-started)
                    medians = {k:statistics.median(v) for k,v in samples.items()}
                    timings[scope] = dict(samples_seconds=samples, medians_seconds=medians,
                        median_ratio=medians['batched']/medians['reference'],
                        paired_faster=sum(b<a for a,b in zip(samples['reference'],samples['batched'])))
                # Valid gradients ensure step() reaches state validation before mutation.
                for _, p in model.ordered_parameters(): p.grad = torch.zeros_like(p)
                rejected = []
                for label, optimizer in optimizers.items():
                    original_snapshot = snapshot(optimizer)
                    muon, adam = optimizer._optimizers
                    mp = muon.param_groups[0]['params'][0]; ap = adam.param_groups[0]['params'][0]
                    ms, ads = muon.state[mp], adam.state[ap]
                    for case in plan['invalid_cases']:
                        undo = None
                        try:
                            if case in ('nan_momentum', 'inf_exp_avg', 'minus_inf_exp_avg_sq',
                                        'negative_exp_avg_sq', 'nan_cpu_step', 'wrong_cpu_step'):
                                table, key, value = {
                                    'nan_momentum':(ms,'momentum_buffer',float('nan')),
                                    'inf_exp_avg':(ads,'exp_avg',float('inf')),
                                    'minus_inf_exp_avg_sq':(ads,'exp_avg_sq',-float('inf')),
                                    'negative_exp_avg_sq':(ads,'exp_avg_sq',-1.),
                                    'nan_cpu_step':(ads,'step',float('nan')),
                                    'wrong_cpu_step':(ads,'step',1201.)}[case]
                                original = table[key]; replacement = original.clone()
                                replacement.reshape(-1)[0] = value; table[key] = replacement
                                undo = lambda table=table,key=key,original=original: table.__setitem__(key,original)
                            elif case == 'wrong_shape':
                                original = ms['momentum_buffer']; ms['momentum_buffer'] = original.reshape(-1)
                                assert ms['momentum_buffer'].shape != mp.shape
                                undo = lambda: ms.__setitem__('momentum_buffer',original)
                            elif case == 'missing_field':
                                original = ms.pop('momentum_buffer')
                                undo = lambda: ms.__setitem__('momentum_buffer',original)
                            elif case == 'parameter_alias':
                                original = ms['momentum_buffer']; ms['momentum_buffer'] = mp.detach()
                                undo = lambda: ms.__setitem__('momentum_buffer',original)
                            elif case == 'foreign_key':
                                foreign = torch.nn.Parameter(torch.zeros(1)); muon.state[foreign] = {}
                                undo = lambda: muon.state.pop(foreign)
                            else: raise AssertionError(case)
                            before = snapshot(optimizer)
                            try: optimizer.step()
                            except ValueError as error: error_text = str(error)
                            else: raise AssertionError('Invalid state accepted: '+case)
                            assert before == snapshot(optimizer), 'Rejected step mutated state'
                            rejected.append(dict(implementation=label, case=case, error=error_text,
                                                 failure_before_mutation=True))
                        finally:
                            if undo is not None: undo()
                        optimizer.validate(require_gradients=True)
                        assert original_snapshot == snapshot(optimizer), 'Restoration differs'
                assert [(r['case'],r['error']) for r in rejected if r['implementation']=='reference'] == [
                       (r['case'],r['error']) for r in rejected if r['implementation']=='batched']
                for _, p in model.ordered_parameters(): p.grad = None
                row = dict(backend=backend, timings=timings, rejections=rejected,
                           accepted_saved_state_steps=1200, lazy_state_preserved=True,
                           restored_state_exact=True, prototype_shares_model_only=True)
                publish(backend+'.json', row); cases.append(row)
                del tensors, snapshot, optimizers, reference, candidate, model, parameters, payload
                import gc
                gc.collect()
                torch.mps.empty_cache()
            assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(mps_rng, torch.mps.get_rng_state())
            gpu = next(c for c in cases if c['backend']=='mps')['timings']
            useful = (gpu['state_only']['median_ratio'] <= .8 and
                      gpu['complete_validate']['median_ratio'] <= .95 and
                      gpu['complete_validate']['paired_faster'] >= 5)
            result = dict(correctness_passed=True, useful_mps_improvement=useful, adopted=False,
                          cases=cases, plan_sha256=sha(OUT/'plan.json'), runner_sha256=sha(Path(__file__)),
                          training_updates=0, quality_evaluations=0, rng_unchanged=True,
                          limits=plan['interpretation_limits'])
        assert read(OUT/'guard.json')['status']=='finished'
        result['guard_sha256'] = sha(OUT/'guard.json'); publish('result.json',result)
        publish('STOP',dict(status='complete_candidate_pending_integration' if useful else 'complete_no_useful_gain',no_replay=True))
    print(json.dumps({k:result[k] for k in ('correctness_passed','useful_mps_improvement','training_updates')}))
    for c in cases: print(json.dumps(dict(backend=c['backend'],timings=c['timings'],rejections=len(c['rejections']))))


if __name__ == '__main__':
    try: run()
    except Exception:
        publish('failure.json',dict(traceback=traceback.format_exc(),no_automatic_replay=True))
        raise
