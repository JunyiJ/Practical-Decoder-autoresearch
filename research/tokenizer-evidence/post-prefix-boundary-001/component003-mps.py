"""New MPS component acceptance; independent generated CPU FP64 oracle only."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import traceback
import torch
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.tokenizer_moe_component_v1.component import RoutedSwiGLU

BASE = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent
ATOL, RTOL = 4e-5, 4e-4


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pins():
    old = json.loads((BASE / 'research/tokenizer-evidence/tokenizer-capacity-boundary-001/quality001-plan.json').read_text())['sources']
    component = json.loads((E / 'continuation-completed001.json').read_text())['source_bindings']
    assert len(old) == 1004 and len(component) == 4
    result = {**old, **component}
    assert len(result) == 1008
    for name, digest in result.items():
        assert sha(BASE / name) == digest, name
    return result


def main():
    for n in ['component003-mps-result.json', 'component003-mps-failure.json', 'component003-mps-guard.json']:
        assert not (E / n).exists(), 'This new proof is single-attempt only'
    assert all(os.environ.get(k) == v for k,v in {
        'PYTORCH_ENABLE_MPS_FALLBACK':'0', 'PYTORCH_MPS_FAST_MATH':'0',
        'OMP_NUM_THREADS':'4', 'OPENBLAS_NUM_THREADS':'4', 'MKL_NUM_THREADS':'4'}.items())
    torch.set_num_threads(4)
    before_pins = pins()
    # Import the proved independent oracle function, never its closed main().
    spec = importlib.util.spec_from_file_location('closed_cpu_equation_reference', E / 'component002.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    oracle = module.oracle
    cpu_rng = torch.get_rng_state().clone()
    rows = []
    with (BASE / 'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with StrictMpsGuard(E / 'component003-mps-guard.json', BASE), torch.random.fork_rng(devices=[]):
            assert torch.backends.mps.is_available()
            mps_rng = torch.mps.get_rng_state().clone()
            torch.random.default_generator.manual_seed(600451)  # new generated device fixture, not scientific
            for dim, hidden, batch, tokens, scale in [(7,5,1,1,1.), (7,5,2,17,1.), (256,336,1,8,4.)]:
                for mode in ['ordinary', 'exact_tie_zero_route']:
                    model = RoutedSwiGLU(dim, hidden, scale)
                    if mode == 'exact_tie_zero_route':
                        with torch.no_grad(): model.router.weight.zero_()
                    original = {n:p.detach().double().clone().requires_grad_() for n,p in model.named_parameters()}
                    x_cpu = torch.randn(batch, tokens, dim).double().requires_grad_()
                    probe = torch.randn(batch, tokens, dim)
                    model = model.to('mps')
                    x = x_cpu.detach().float().to('mps').requires_grad_()
                    params = dict(model.named_parameters())
                    got, aux, counts = model(x, record_routes=True)
                    expected, expected_aux, expected_counts = oracle(x_cpu, original, scale)
                    torch.mps.synchronize()
                    errors = []
                    def compare(a,b):
                        a = a.detach().cpu().double(); b = b.detach().cpu().double()
                        assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
                        error = float((a-b).abs().max()); errors.append(error)
                        assert torch.allclose(a,b,atol=ATOL,rtol=RTOL), (dim,mode,error)
                    compare(got, expected); compare(aux, expected_aux)
                    assert torch.equal(counts.cpu(),expected_counts)
                    assert counts.device.type == model.assignment_counts.device.type == model.routed_tokens.device.type == 'mps'
                    assert torch.equal(model.assignment_counts.cpu(), expected_counts)
                    assert int(model.routed_tokens.cpu()) == batch*tokens
                    if mode == 'exact_tie_zero_route':
                        assert counts.cpu().tolist() == [batch*tokens,batch*tokens,0,0]
                    for coefficient in [0., .01]:
                        loss = (got * probe.to('mps')).sum() + coefficient*aux
                        reference_loss = (expected * probe.double()).sum() + coefficient*expected_aux
                        grads = torch.autograd.grad(loss,[x,*params.values()],allow_unused=True,retain_graph=True)
                        refs = torch.autograd.grad(reference_loss,[x_cpu,*original.values()],retain_graph=True)
                        for a,b in zip(grads,refs):
                            if a is None: assert torch.count_nonzero(b) == 0
                            else: compare(a,b)
                    state_before = model.assignment_counts.clone()
                    with torch.no_grad():
                        model(x,record_routes=False)
                        assert torch.equal(state_before.cpu(),model.assignment_counts.cpu())
                        model(x,record_routes=True)
                    assert torch.equal(model.assignment_counts.cpu(),2*expected_counts)
                    assert int(model.routed_tokens.cpu()) == 2*batch*tokens
                    rows.append(dict(dim=dim,hidden=hidden,batch=batch,tokens=tokens,mode=mode,
                                     maximum_absolute_error=max(errors),counts=counts.cpu().tolist()))
                    del model,x,got,aux,counts,params,grads,refs
                    torch.mps.synchronize(); torch.mps.empty_cache()
            assert torch.equal(mps_rng,torch.mps.get_rng_state()), 'Deterministic component consumed MPS RNG'
    assert torch.equal(cpu_rng,torch.get_rng_state()) and before_pins == pins()
    result = dict(status='mps_component_independent_cpu_fp64_oracle_passed',cases=rows,atol=ATOL,rtol=RTOL,
                  predecessor_sources_unchanged=1008,source={str(Path(__file__).relative_to(BASE)):sha(__file__)},
                  fixture_seed=600451,scientific_seed=False,cpu_rng_unchanged=True,mps_rng_unchanged=True,
                  corpus_loaded=False,weights_written=0,full_model=False,optimizer_resume=False,quality_result=False)
    assert sum(p.stat().st_size for p in E.rglob('*') if p.is_file()) < 16*1024**2
    with (E / 'component003-mps-result.json').open('x') as f:
        json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(dict(status=result['status'],cases=len(rows),maximum_absolute_error=max(r['maximum_absolute_error'] for r in rows))))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        path = E / 'component003-mps-failure.json'
        if not path.exists():
            with path.open('x') as f:
                json.dump(dict(status='CLOSED_FAILED_MPS_COMPONENT_PROOF',error=repr(exc),traceback=traceback.format_exc(),
                               source_sha256=sha(__file__),no_training_or_quality_result=True,never_rerun_this_proof=True),f,indent=2)
                f.write('\n')
        raise
