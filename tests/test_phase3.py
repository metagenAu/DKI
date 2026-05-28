"""Phase-3 tests: deep-equilibrium solver correctness vs the ODE."""

from __future__ import annotations

import torch

from dki.deq import solve_equilibrium
from dki.losses import bray_curtis
from dki.model import ReplicatorODEFunc, integrate


def _simplex_batch(B: int, N: int, n_present: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(B, N)
    for b in range(B):
        idx = torch.randperm(N, generator=g)[:n_present]
        z[b, idx] = 1.0
    return z / z.sum(dim=-1, keepdim=True)


def _symmetric_model(N: int, seed: int = 0) -> ReplicatorODEFunc:
    """Linear model with symmetric, strongly self-limiting interactions.

    A dominant negative diagonal makes the replicator potential strongly
    concave with a unique interior equilibrium, so both the ODE flow and the
    mirror-descent DEQ map are strong contractions onto the same point —
    letting us assert DEQ ≈ ODE within a tight tolerance.
    """
    torch.manual_seed(seed)
    func = ReplicatorODEFunc(N, nonlinear=False)
    S = torch.randn(N, N)
    S = (S + S.t()) / 2.0
    A = -(torch.eye(N) + 0.15 * S)   # symmetric, dominant self-competition
    with torch.no_grad():
        func.fcc1.weight.copy_(A)
        func.fcc1.bias.copy_(0.1 * torch.randn(N))
        func.fcc2.weight.copy_(torch.eye(N))
        func.fcc2.bias.zero_()
    return func


def test_deq_output_on_simplex_and_support_preserved():
    N = 10
    z = _simplex_batch(4, N, n_present=6, seed=1)
    func = _symmetric_model(N, seed=2)
    out = solve_equilibrium(func, z, ode_fallback=False)
    sums = out.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), sums
    # Absent species stay absent.
    assert (out[z == 0] < 1e-6).all()
    assert (out >= -1e-6).all()


def test_deq_is_a_fixed_point():
    """g(z*) ≈ z* — applying the mirror step leaves the solution unchanged."""
    N = 10
    z = _simplex_batch(3, N, n_present=5, seed=3)
    func = _symmetric_model(N, seed=4)
    z_star = solve_equilibrium(func, z, ode_fallback=False).detach()

    step, support = 0.5, z > 0
    f = func.fitness(z_star)
    logits = step * f + z_star.clamp_min(1e-30).log()
    logits = logits.masked_fill(~support, float("-inf"))
    g_z = torch.softmax(logits, dim=-1)
    assert bray_curtis(g_z, z_star).item() < 2e-3


def test_deq_matches_ode_on_potential_game():
    N = 8
    z = _simplex_batch(4, N, n_present=5, seed=5)
    func = _symmetric_model(N, seed=6)
    deq = solve_equilibrium(func, z, ode_fallback=False, t_final=40.0).detach()
    ode = integrate(func, z, t_final=40.0).detach()
    assert bray_curtis(deq, ode).item() < 0.02


def test_deq_gradient_flows():
    N = 8
    z = _simplex_batch(4, N, n_present=5, seed=7)
    func = _symmetric_model(N, seed=8)
    target = _simplex_batch(4, N, n_present=5, seed=9)
    out = solve_equilibrium(func, z, ode_fallback=False)
    loss = bray_curtis(out, target)
    loss.backward()
    grads = [p.grad for p in func.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
