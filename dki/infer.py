"""Batched inference utilities."""

from __future__ import annotations

import torch
from torch import nn

from .model import integrate


def equilibrium(
    func: nn.Module,
    z0: torch.Tensor,
    mode: str = "ode",
    t_final: float = 100.0,
    method: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-7,
    deq_step: float = 0.5,
    deq_max_iter: int = 50,
    deq_tol: float = 1e-6,
    deq_fallback: bool = True,
) -> torch.Tensor:
    """Compute the replicator equilibrium for ``z0`` via the chosen solver.

    ``mode="ode"`` integrates the replicator ODE (Phase 1); ``mode="deq"``
    solves the fixed point directly with Anderson acceleration (Phase 3),
    falling back to the ODE on non-convergence.
    """
    if mode == "ode":
        return integrate(func, z0, t_final=t_final, method=method, rtol=rtol, atol=atol)
    if mode == "deq":
        from .deq import solve_equilibrium

        return solve_equilibrium(
            func,
            z0,
            step=deq_step,
            max_iter=deq_max_iter,
            tol=deq_tol,
            ode_fallback=deq_fallback,
            t_final=t_final,
            method=method,
            rtol=rtol,
            atol=atol,
        )
    raise ValueError(f"Unknown mode {mode!r}; expected 'ode' or 'deq'.")


@torch.no_grad()
def predict(
    func: nn.Module,
    z: torch.Tensor,
    t_final: float = 100.0,
    batch_size: int = 256,
    method: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-7,
    mode: str = "ode",
    deq_step: float = 0.5,
    deq_max_iter: int = 50,
    deq_tol: float = 1e-6,
    deq_fallback: bool = True,
) -> torch.Tensor:
    """Predict equilibrium composition for each row of ``z``.

    Splits into mini-batches to bound memory; each mini-batch is solved in one
    ``odeint`` (or DEQ) call.
    """
    func.eval()
    model_device = next(func.parameters()).device
    outs = []
    for start in range(0, z.shape[0], batch_size):
        chunk = z[start : start + batch_size].to(model_device)
        out = equilibrium(
            func,
            chunk,
            mode=mode,
            t_final=t_final,
            method=method,
            rtol=rtol,
            atol=atol,
            deq_step=deq_step,
            deq_max_iter=deq_max_iter,
            deq_tol=deq_tol,
            deq_fallback=deq_fallback,
        )
        # Offload each batch back to z's device immediately. This keeps only a
        # single batch resident on the model's (GPU) device at a time: without
        # it, every batch output stays on the GPU until the final cat, which
        # OOMs on large inputs such as the leave-one-out assemblages in the
        # keystoneness workflow. Moving the chunk onto the model device first
        # also lets callers pass a CPU `z` while the model lives on the GPU.
        outs.append(out.to(z.device))
    return torch.cat(outs, dim=0)
