"""ODE-based DKI model with batched replicator dynamics."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torchdiffeq import odeint


class ReplicatorODEFunc(nn.Module):
    """Per-capita-fitness network wrapped in replicator dynamics.

    Phase-1 architecture matches the legacy DKI model: two stacked linear
    layers with no activation (i.e. an affine map ``A y + b``). The replicator
    wrapper ``y * (out - <out, y>)`` keeps the trajectory on the simplex.

    The forward pass accepts state of shape ``(B, N)`` where B is the batch
    dimension. The legacy code integrated each sample in a Python loop with
    shape ``(1, N)``; this class handles both transparently.
    """

    def __init__(self, n_species: int):
        super().__init__()
        self.n_species = n_species
        self.fcc1 = nn.Linear(n_species, n_species)
        self.fcc2 = nn.Linear(n_species, n_species)

    def fitness(self, y: torch.Tensor) -> torch.Tensor:
        return self.fcc2(self.fcc1(y))

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        out = self.fitness(y)
        mean_fitness = (y * out).sum(dim=-1, keepdim=True)
        return y * (out - mean_fitness)


def integrate(
    func: nn.Module,
    z0: torch.Tensor,
    t_final: float = 100.0,
    method: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-7,
    options: Optional[dict] = None,
) -> torch.Tensor:
    """Batched ODE integration. ``z0`` shape ``(B, N)`` → returns ``(B, N)``."""
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0)
    t = torch.tensor([0.0, float(t_final)], dtype=z0.dtype, device=z0.device)
    traj = odeint(func, z0, t, method=method, rtol=rtol, atol=atol, options=options)
    # traj shape: (len(t), B, N)
    return traj[-1]
