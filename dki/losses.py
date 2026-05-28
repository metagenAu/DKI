"""Loss functions for DKI."""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn


def bray_curtis(p_pred: torch.Tensor, p_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Bray-Curtis dissimilarity averaged over the batch.

    Matches the legacy formulation: ``sum |a-b| / sum |a+b|`` per sample,
    then averaged over samples. Both inputs have shape ``(B, N)``.
    """
    num = (p_pred - p_true).abs().sum(dim=-1)
    den = (p_pred + p_true).abs().sum(dim=-1).clamp_min(eps)
    return (num / den).mean()


def clr(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Centered-log-ratio transform along the last axis.

    ``clr(x)_i = log(x_i) - mean_j log(x_j)`` after flooring at ``eps`` to keep
    structural zeros finite. CLR works in ratio space, so it weights rare
    species far more heavily than Bray-Curtis (which is dominated by the
    abundant species). That is the point of Phase 2's composite loss.

    The floor doubles as gradient regularisation: ``d log(x)/dx = 1/x`` so a
    too-small ``eps`` lets a single near-zero prediction inject a ~1/eps spike
    into the backward pass, which then destabilises the SiLU fitness and the
    downstream ODE solve (``dopri5`` shrinks dt until it underflows). ``1e-4``
    caps the per-element gradient at ~10^4 while leaving abundant-species
    behaviour intact.
    """
    log_x = x.clamp_min(eps).log()
    return log_x - log_x.mean(dim=-1, keepdim=True)


def clr_mse(p_pred: torch.Tensor, p_true: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Mean-squared error between CLR-transformed predictions and targets."""
    return ((clr(p_pred, eps) - clr(p_true, eps)) ** 2).mean()


def composite_loss(
    p_pred: torch.Tensor,
    p_true: torch.Tensor,
    alpha: float = 0.3,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Phase-2 composite loss ``α·BC + (1−α)·CLR-MSE``.

    ``alpha`` weights the Bray-Curtis term (abundant-species accuracy); the
    remaining ``1−alpha`` weights the CLR-MSE term (rare-species / ratio
    accuracy). Default ``alpha=0.3``.
    """
    return alpha * bray_curtis(p_pred, p_true) + (1.0 - alpha) * clr_mse(p_pred, p_true, eps)


def _mask_one_present(z: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Drop one present species per row and renormalise the support uniformly.

    ``z`` rows are presence-normalised assemblages (uniform over the present
    species). For each row we pick one present species at random, zero it, and
    renormalise the remaining support to sum to 1. Rows with a single present
    species are returned unchanged.
    """
    present = z > 0
    counts = present.sum(dim=-1, keepdim=True)
    # Uniform random scores over present entries; pick the arg-max to drop.
    noise = torch.rand(z.shape, generator=generator, device=z.device, dtype=z.dtype)
    scores = torch.where(present, noise, torch.full_like(noise, -1.0))
    drop_idx = scores.argmax(dim=-1, keepdim=True)
    keep = present.clone()
    keep.scatter_(-1, drop_idx, False)
    # Don't drop the last remaining species.
    keep = torch.where(counts > 1, keep, present)
    out = keep.to(z.dtype)
    out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return out


def self_consistency_loss(
    func: nn.Module,
    z: torch.Tensor,
    equilibrium_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Phase-6 self-consistency auxiliary loss.

    For each assemblage we mask one present species, predict the equilibrium
    ``q'`` of the reduced community, then re-feed ``q'`` as an initial
    condition and integrate again to ``q''``. A well-trained replicator should
    leave its own equilibria fixed, so we penalise ``BC(q'', q')``. ``q'`` is
    detached when used as the re-feed initial condition and as the target, so
    the gradient pushes the model to make ``q'`` an actual fixed point rather
    than collapsing both predictions.
    """
    z_masked = _mask_one_present(z, generator=generator)
    q1 = equilibrium_fn(func, z_masked)
    z2 = q1.detach()
    z2 = z2 / z2.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    q2 = equilibrium_fn(func, z2)
    return bray_curtis(q2, q1.detach())
