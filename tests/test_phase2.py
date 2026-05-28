"""Phase-2 tests: fitness expressivity + composite CLR loss."""

from __future__ import annotations

import torch

from dki.losses import bray_curtis, clr, clr_mse, composite_loss
from dki.model import ReplicatorODEFunc


def _best_linear_residual(func: ReplicatorODEFunc, N: int, n: int = 1000) -> float:
    """Max residual of the best affine fit to ``func.fitness`` over n inputs.

    For two stacked Linear layers (no activation) the fitness IS an affine map,
    so the residual is numerically zero. The SiLU version cannot be fit by any
    single affine map, so the residual is large.
    """
    torch.manual_seed(1)
    y = 2.0 * torch.randn(n, N)
    f = func.fitness(y).detach()
    y_aug = torch.cat([y, torch.ones(n, 1)], dim=1)
    sol = torch.linalg.lstsq(y_aug, f).solution
    pred = y_aug @ sol
    return (pred - f).abs().max().item()


def test_linear_fitness_collapses_to_single_affine():
    torch.manual_seed(0)
    N = 8
    func = ReplicatorODEFunc(N, nonlinear=False)
    assert _best_linear_residual(func, N) < 1e-3


def test_silu_fitness_is_genuinely_nonlinear():
    torch.manual_seed(0)
    N = 8
    func = ReplicatorODEFunc(N, nonlinear=True, hidden_mult=2)
    assert func.fcc1.out_features == 2 * N
    assert _best_linear_residual(func, N) > 1e-2


def test_clr_of_uniform_is_zero():
    x = torch.full((3, 5), 0.2)
    assert torch.allclose(clr(x), torch.zeros_like(x), atol=1e-6)


def test_clr_mse_self_is_zero():
    p = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
    assert torch.isclose(clr_mse(p, p), torch.tensor(0.0), atol=1e-7)


def test_composite_endpoints():
    p = torch.tensor([[0.5, 0.3, 0.2]])
    q = torch.tensor([[0.2, 0.3, 0.5]])
    assert torch.isclose(composite_loss(p, q, alpha=1.0), bray_curtis(p, q))
    assert torch.isclose(composite_loss(p, q, alpha=0.0), clr_mse(p, q))
    mid = composite_loss(p, q, alpha=0.3)
    expected = 0.3 * bray_curtis(p, q) + 0.7 * clr_mse(p, q)
    assert torch.isclose(mid, expected)
