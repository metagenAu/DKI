"""Phase-6 tests: self-consistency regulariser."""

from __future__ import annotations

import torch

from dki.losses import _mask_one_present, self_consistency_loss
from dki.model import ReplicatorODEFunc, integrate


def _simplex_batch(B: int, N: int, n_present: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(B, N)
    for b in range(B):
        idx = torch.randperm(N, generator=g)[:n_present]
        z[b, idx] = 1.0
    return z / z.sum(dim=-1, keepdim=True)


def test_mask_one_present_drops_exactly_one():
    N = 10
    z = _simplex_batch(5, N, n_present=6, seed=1)
    masked = _mask_one_present(z, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(masked.sum(dim=-1), torch.ones(5), atol=1e-6)
    before = (z > 0).sum(dim=-1)
    after = (masked > 0).sum(dim=-1)
    assert torch.equal(after, before - 1)
    # Masked support is a subset of the original support.
    assert ((masked > 0) <= (z > 0)).all()


def test_mask_one_present_keeps_singletons():
    N = 6
    z = torch.zeros(2, N)
    z[0, 0] = 1.0
    z[1, 3] = 1.0
    masked = _mask_one_present(z, generator=torch.Generator().manual_seed(0))
    assert torch.equal(masked > 0, z > 0)


def test_self_consistency_loss_nonnegative_and_differentiable():
    N = 8
    z = _simplex_batch(4, N, n_present=5, seed=2)
    torch.manual_seed(0)
    func = ReplicatorODEFunc(N)

    def solve(f, zz):
        return integrate(f, zz, t_final=20.0)

    loss = self_consistency_loss(func, z, solve, generator=torch.Generator().manual_seed(0))
    assert loss.item() >= 0
    loss.backward()
    grads = [p.grad for p in func.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
