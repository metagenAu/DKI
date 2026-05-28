"""Batched inference utilities."""

from __future__ import annotations

import torch
from torch import nn

from .model import integrate


@torch.no_grad()
def predict(
    func: nn.Module,
    z: torch.Tensor,
    t_final: float = 100.0,
    batch_size: int = 256,
    method: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> torch.Tensor:
    """Predict equilibrium composition for each row of ``z``.

    Splits into mini-batches to bound memory; each mini-batch is integrated
    in one ``odeint`` call.
    """
    func.eval()
    outs = []
    for start in range(0, z.shape[0], batch_size):
        chunk = z[start : start + batch_size]
        outs.append(integrate(func, chunk, t_final=t_final, method=method, rtol=rtol, atol=atol))
    return torch.cat(outs, dim=0)
