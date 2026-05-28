"""Phase-3 deep-equilibrium reformulation of the replicator model.

Instead of integrating the replicator ODE out to ``t=100`` we solve for its
fixed point directly. The fixed-point map is a multiplicative-weights
(mirror-descent) replicator step on the simplex::

    g(y)_i = softmax_i( log y_i + step * f(y)_i )   over the present support

whose fixed points satisfy ``f(y)_i = <f, y>`` on the support (``y_i = 0``
off it) — exactly the replicator equilibrium condition. The map keeps the
iterate on the simplex and never resurrects an absent species, matching the
ODE flow.

The fixed point is found with Anderson acceleration (no autograd), and the
gradient is obtained via the implicit function theorem: at ``z* = g(z*)`` we
have ``dz*/dθ`` from solving ``(I − J_g) v = ∂g/∂θ``, implemented with the
standard backward-hook trick that solves the adjoint linear system with the
same Anderson solver. On non-convergence we fall back to the ODE integrator.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch
from torch import nn

from .model import integrate

_TINY = 1e-30


def anderson(
    g: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    m: int = 5,
    max_iter: int = 50,
    tol: float = 1e-6,
    beta: float = 1.0,
    lam: float = 1e-4,
    safeguard: bool = True,
) -> Tuple[torch.Tensor, List[float]]:
    """Batched, safeguarded Anderson acceleration for ``x = g(x)``.

    ``x0`` has shape ``(B, N)``. Returns the fixed-point estimate (same shape)
    and the list of relative residuals per iteration.

    With ``safeguard`` (default), each row keeps the Anderson extrapolation
    only when it does not increase that row's fixed-point residual relative to
    a plain map step; otherwise it takes the plain step. This prevents the
    accelerator from jumping onto *unstable* fixed points, so the iteration
    converges to the same stable equilibrium the ODE flow would reach.
    """
    B, N = x0.shape
    dtype, device = x0.dtype, x0.device
    X = torch.zeros(B, m, N, dtype=dtype, device=device)
    F = torch.zeros(B, m, N, dtype=dtype, device=device)
    X[:, 0] = x0
    F[:, 0] = g(x0)
    X[:, 1] = F[:, 0]
    F[:, 1] = g(F[:, 0])

    H = torch.zeros(B, m + 1, m + 1, dtype=dtype, device=device)
    H[:, 0, 1:] = 1.0
    H[:, 1:, 0] = 1.0
    rhs = torch.zeros(B, m + 1, 1, dtype=dtype, device=device)
    rhs[:, 0] = 1.0

    residuals: List[float] = []
    k = 1
    slot = 1
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1 : n + 1, 1 : n + 1] = (
            torch.bmm(G, G.transpose(1, 2))
            + lam * torch.eye(n, dtype=dtype, device=device)[None]
        )
        alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], rhs[:, : n + 1])[:, 1 : n + 1, 0]
        slot = k % m
        x_acc = (
            beta * (alpha[:, None] @ F[:, :n])[:, 0]
            + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        )
        if safeguard:
            prev = (k - 1) % m
            x_pic = F[:, prev]                 # plain map step g(x_prev)
            g_acc, g_pic = g(x_acc), g(x_pic)
            r_acc = (g_acc - x_acc).norm(dim=-1)
            r_pic = (g_pic - x_pic).norm(dim=-1)
            use_acc = (r_acc <= r_pic).unsqueeze(-1)
            X[:, slot] = torch.where(use_acc, x_acc, x_pic)
            F[:, slot] = torch.where(use_acc, g_acc, g_pic)
        else:
            X[:, slot] = x_acc
            F[:, slot] = g(X[:, slot])
        residual = (
            (F[:, slot] - X[:, slot]).norm().item()
            / (1e-5 + F[:, slot].norm().item())
        )
        residuals.append(residual)
        if residual < tol:
            break
    return X[:, slot], residuals


def _support_softmax_step(
    func: nn.Module, support: torch.Tensor, step: float
) -> Callable[[torch.Tensor], torch.Tensor]:
    def g(y: torch.Tensor) -> torch.Tensor:
        f = func.fitness(y)
        logits = step * f + y.clamp_min(_TINY).log()
        logits = logits.masked_fill(~support, float("-inf"))
        return torch.softmax(logits, dim=-1)

    return g


def solve_equilibrium(
    func: nn.Module,
    z0: torch.Tensor,
    step: float = 0.5,
    m: int = 5,
    max_iter: int = 50,
    tol: float = 1e-6,
    ode_fallback: bool = True,
    t_final: float = 100.0,
    method: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> torch.Tensor:
    """Solve the replicator fixed point for each row of ``z0`` (shape ``(B, N)``).

    Differentiable w.r.t. ``func`` parameters via the implicit function
    theorem. Falls back to ODE integration when Anderson does not reach
    ``tol`` within ``max_iter`` iterations (and ``ode_fallback`` is set).
    """
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0)
    support = z0 > 0
    g = _support_softmax_step(func, support, step)

    with torch.no_grad():
        z_star, residuals = anderson(g, z0, m=m, max_iter=max_iter, tol=tol)
    converged = len(residuals) > 0 and residuals[-1] < tol
    if not converged:
        if ode_fallback:
            return integrate(
                func, z0, t_final=t_final, method=method, rtol=rtol, atol=atol
            )
        # No fallback: return the best estimate so downstream code still runs.

    # Re-engage autograd through one application of the map, and attach an
    # IFT backward hook that solves the adjoint fixed point.
    z = g(z_star)
    if torch.is_grad_enabled() and any(p.requires_grad for p in func.parameters()):
        z0_param = z.clone().detach().requires_grad_()
        g0 = g(z0_param)

        def backward_hook(grad: torch.Tensor) -> torch.Tensor:
            def adjoint(u: torch.Tensor) -> torch.Tensor:
                Ju = torch.autograd.grad(g0, z0_param, u, retain_graph=True)[0]
                return Ju + grad

            new_grad, _ = anderson(adjoint, torch.zeros_like(grad), m=m, max_iter=max_iter, tol=tol)
            return new_grad

        z.register_hook(backward_hook)
    return z
