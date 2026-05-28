"""ODE-based DKI model with batched replicator dynamics."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torchdiffeq import odeint


class ReplicatorODEFunc(nn.Module):
    """Per-capita-fitness network wrapped in replicator dynamics.

    Two regimes, selected by ``nonlinear``:

    * ``nonlinear=False`` (Phase-1, default) — fitness is two stacked linear
      layers with no activation, ``fcc2(fcc1(y))``. This is the legacy DKI
      ``cNODE2`` model. Note ``W₂(W₁ y) = (W₂ W₁) y`` is a single linear map,
      so the *fitness* function is linear; the dynamics are nonlinear only
      through the replicator wrapping.
    * ``nonlinear=True`` (Phase-2) — fitness is ``fcc2(SiLU(fcc1(y)))`` with a
      hidden dimension of ``hidden_mult * n_species``. The SiLU makes the
      per-capita fitness itself nonlinear, which two stacked ``Linear`` layers
      alone cannot achieve.

    The replicator wrapper ``y * (out - <out, y>)`` keeps the trajectory on the
    simplex. The forward pass accepts state of shape ``(B, N)``; the legacy
    code used ``(1, N)`` and this class handles both transparently.
    """

    def __init__(self, n_species: int, nonlinear: bool = False, hidden_mult: int = 2):
        super().__init__()
        self.n_species = n_species
        self.nonlinear = nonlinear
        self.hidden_mult = hidden_mult
        if nonlinear:
            hidden = hidden_mult * n_species
            self.fcc1 = nn.Linear(n_species, hidden)
            self.fcc2 = nn.Linear(hidden, n_species)
            self.activation: Optional[nn.Module] = nn.SiLU()
        else:
            self.fcc1 = nn.Linear(n_species, n_species)
            self.fcc2 = nn.Linear(n_species, n_species)
            self.activation = None

    def fitness(self, y: torch.Tensor) -> torch.Tensor:
        h = self.fcc1(y)
        if self.activation is not None:
            h = self.activation(h)
        return self.fcc2(h)

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
    fallback_method: str = "rk4",
    fallback_steps: int = 200,
) -> torch.Tensor:
    """Batched ODE integration. ``z0`` shape ``(B, N)`` → returns ``(B, N)``.

    Robustness: a nonlinear fitness can make the learned vector field stiff mid
    training, at which point the adaptive solver (``dopri5``) shrinks its step
    until it underflows machine precision and torchdiffeq raises
    ``AssertionError: underflow in dt ...`` (or asserts on a non-finite state),
    killing the whole run over a single bad batch. When that happens — or when
    the adaptive solve returns a non-finite result — we fall back to a
    fixed-step solver (``fallback_method`` over ``fallback_steps`` uniform
    steps), which has no adaptive step to underflow and so always returns. The
    common case (adaptive solve succeeds) is unchanged. ``fallback_steps <= 0``
    disables the fallback and re-raises the original error.
    """
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0)
    t = torch.tensor([0.0, float(t_final)], dtype=z0.dtype, device=z0.device)
    try:
        traj = odeint(func, z0, t, method=method, rtol=rtol, atol=atol, options=options)
        out = traj[-1]
        if torch.isfinite(out).all():
            return out
    except AssertionError:
        if fallback_steps <= 0:
            raise
    if fallback_steps <= 0:
        raise RuntimeError(
            f"Adaptive solver '{method}' returned non-finite values and the "
            "fixed-step fallback is disabled (fallback_steps <= 0)."
        )
    step = float(t_final) / fallback_steps
    traj = odeint(func, z0, t, method=fallback_method, options={"step_size": step})
    # traj shape: (len(t), B, N)
    return traj[-1]
