"""ODE-based DKI model with batched replicator dynamics."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint


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


def _projected_integrate(
    func: nn.Module, z0: torch.Tensor, t_final: float, steps: int
) -> torch.Tensor:
    """Last-resort fixed-step integrator that can never return non-finite values.

    Explicit Euler with a simplex projection (clamp to ``>= 0`` then renormalise
    to sum 1) applied after every step. Replicator dynamics already preserve the
    simplex, so the projection only corrects numerical drift — but crucially it
    keeps the state a valid composition, where the fitness network is bounded.
    A bounded field plus a finite state means ``y + dt·f(y)`` stays finite, so
    unlike ``dopri5``/``rk4`` this integrator cannot overflow to ``inf``/``nan``.
    Lower accuracy, but it always returns a finite estimate for rows the
    adaptive and fixed-step solvers blow up on.
    """
    dt = float(t_final) / max(steps, 1)
    t0 = torch.zeros((), dtype=z0.dtype, device=z0.device)
    y = z0.clamp_min(0.0)
    y = y / y.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    for _ in range(max(steps, 1)):
        y = y + dt * func(t0, y)
        y = y.clamp_min(0.0)
        y = y / y.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return y


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
    steps), which has no adaptive step to underflow.

    The fixed-step solver still has no projection back onto the simplex, so on a
    sufficiently stiff/explosive field it can *itself* overflow to a non-finite
    state for some rows. Returning those rows unchecked is exactly what makes a
    whole-batch validation solve report ``val_bc=nan`` (one ``nan`` row poisons
    the batch mean), even though the weights stayed finite. So we detect any
    remaining non-finite rows and re-solve just those with a simplex-projected
    integrator that is guaranteed to return finite values. The common case
    (adaptive solve succeeds) is unchanged. ``fallback_steps <= 0`` disables the
    fallbacks and re-raises the original error.
    """
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0)
    t = torch.tensor([0.0, float(t_final)], dtype=z0.dtype, device=z0.device)
    # Adjoint backprop is O(1) in the number of solver steps, which keeps the
    # nonlinear, large-batch training forward from OOMing on a 16 GB card. Only
    # pay that overhead when a backward is actually coming — validation and
    # inference run under no_grad and stay on the cheaper direct odeint.
    use_adjoint = torch.is_grad_enabled() and any(p.requires_grad for p in func.parameters())
    solver = odeint_adjoint if use_adjoint else odeint
    solver_kwargs: dict = {"method": method, "rtol": rtol, "atol": atol, "options": options}
    if use_adjoint:
        solver_kwargs["adjoint_params"] = tuple(func.parameters())
    try:
        traj = solver(func, z0, t, **solver_kwargs)
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
    fb_kwargs: dict = {"method": fallback_method, "options": {"step_size": step}}
    if use_adjoint:
        fb_kwargs["adjoint_params"] = tuple(func.parameters())
    traj = solver(func, z0, t, **fb_kwargs)
    out = traj[-1]  # traj shape: (len(t), B, N)

    # The fixed-step fallback has no simplex projection and can still overflow on
    # an explosive field. Re-solve only the non-finite rows with the projected
    # integrator so the result is always finite (no more val_bc=nan).
    bad = ~torch.isfinite(out).all(dim=-1)
    if bad.any():
        out = out.clone()
        out[bad] = _projected_integrate(func, z0[bad], t_final, fallback_steps)
    return out
