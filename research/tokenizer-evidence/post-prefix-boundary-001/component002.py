"""New CPU equation/gradient oracle; no corpus, model scoring or weight files."""
from pathlib import Path
import fcntl
import hashlib
import json
import torch
from research.tokenizer_moe_component_v1.component import RoutedSwiGLU
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

BASE = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def oracle(x, p, scale):
    z = x.reshape(-1, x.shape[-1])
    logits = z @ p['router.weight'].T / scale
    probs = (logits - torch.logsumexp(logits, dim=1, keepdim=True)).exp()
    # Python stable ordering is independent of the component argsort/dispatch.
    selected = [sorted(range(4), key=lambda i: (-v[i], i))[:2] for v in logits.detach().tolist()]
    mask = torch.zeros_like(probs)
    for n, experts in enumerate(selected):
        for i in experts:
            mask[n, i] = 1
    weights = mask * probs
    weights = weights / weights.sum(dim=1, keepdim=True)
    terms = []
    for i in range(4):
        u = z @ p[f'experts.{i}.gate.weight'].T
        v = z @ p[f'experts.{i}.up.weight'].T
        terms.append(((u * u.sigmoid()) * v) @ p[f'experts.{i}.down.weight'].T)
    out = sum(v * weights[:, i, None] for i, v in enumerate(terms))
    counts = mask.sum(dim=0).to(torch.int64)
    aux = sum(4 * int(counts[i]) / (2 * z.shape[0]) * probs[:, i].mean() for i in range(4))
    return out.reshape_as(x), aux, counts


def main():
    torch.set_num_threads(4)
    pins = json.loads((BASE / 'research/tokenizer-evidence/tokenizer-capacity-boundary-001/quality001-plan.json').read_text())['sources']
    for p, h in pins.items():
        assert sha(BASE / p) == h, p
    assert not (E / 'component002-result.json').exists()
    original_rng = torch.get_rng_state().clone()
    rows = []
    with (BASE / 'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with CpuPrerequisiteGuard(E / 'component002-guard.json', BASE), torch.random.fork_rng(devices=[]):
            torch.manual_seed(600450)  # generated fixture only; never a quality seed
            for dim, hidden, tokens, scale in [(7, 5, 1, 1.), (7, 5, 17, 1.), (256, 336, 8, 4.)]:
                for mode in ['ordinary', 'tie_and_zero_route']:
                    model = RoutedSwiGLU(dim, hidden, scale).double()
                    if mode == 'tie_and_zero_route':
                        with torch.no_grad():
                            model.router.weight.zero_()
                    x = torch.randn(1, tokens, dim, dtype=torch.float64, requires_grad=True)
                    other = x.detach().clone().requires_grad_()
                    params = dict(model.named_parameters())
                    copies = {n: p.detach().clone().requires_grad_() for n, p in params.items()}
                    y, aux, counts = model(x, record_routes=True)
                    expected, expected_aux, expected_counts = oracle(other, copies, scale)
                    assert torch.equal(counts, expected_counts) and int(counts.sum()) == 2 * tokens
                    assert torch.equal(model.assignment_counts, counts) and int(model.routed_tokens) == tokens
                    values = [torch.max((y - expected).abs()).item(), abs(float(aux.detach() - expected_aux.detach()))]
                    probe = torch.randn_like(y)
                    for coefficient in [0., .01]:
                        loss = (y * probe).sum() + coefficient * aux
                        expected_loss = (expected * probe).sum() + coefficient * expected_aux
                        grads = torch.autograd.grad(loss, [x, *params.values()], allow_unused=True, retain_graph=True)
                        expected_grads = torch.autograd.grad(expected_loss, [other, *copies.values()], retain_graph=True)
                        for got, want in zip(grads, expected_grads):
                            if got is None:
                                assert torch.count_nonzero(want) == 0
                            else:
                                values.append(float((got - want).abs().max()))
                    assert max(values) < 2e-10, (dim, mode, max(values))
                    if mode == 'tie_and_zero_route':
                        assert counts.tolist() == [tokens, tokens, 0, 0]
                    before = model.assignment_counts.clone()
                    with torch.no_grad():
                        model(x.detach(), record_routes=False)
                        assert torch.equal(before, model.assignment_counts)
                        model(x.detach(), record_routes=True)
                    assert torch.equal(model.assignment_counts, 2 * counts) and int(model.routed_tokens) == 2*tokens
                    rows.append(dict(dim=dim, hidden=hidden, tokens=tokens, router_scale=scale, mode=mode,
                                     maximum_error=max(values), assignment_counts=counts.tolist()))
    assert torch.equal(original_rng, torch.get_rng_state())
    for p, h in pins.items():
        assert sha(BASE / p) == h, p
    sources = [Path(__file__), BASE / 'research/tokenizer_moe_component_v1/__init__.py', BASE / 'research/tokenizer_moe_component_v1/component.py']
    assert sum(p.stat().st_size for p in E.rglob('*') if p.is_file()) < 16*1024**2
    result = dict(status='cpu_component_equations_gradients_and_route_accounting_passed', cases=rows,
                  source_bindings={str(p.relative_to(BASE)):sha(p) for p in sources}, predecessor_sources_unchanged=len(pins),
                  fixture_seed=600450, scientific_seed=False, oracle_predecessor='component001-failure.json', global_cpu_rng_unchanged=True,
                  full_model=False, optimizer_or_resume_proved=False, mps_proved=False, quality_panel=False,
                  corpus_loaded=False, holdout_opened=False, weight_files_written=0,
                  limitations='Component oracle only;not a full model,parameter grouping,resource pilot or quality result. Dormant expert None gradients match mathematical zero;optimizer skip/decay semantics remain unproved.')
    with (E / 'component002-result.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(dict(status=result['status'], cases=len(rows), maximum_error=max(r['maximum_error'] for r in rows))))


if __name__ == '__main__':
    main()
