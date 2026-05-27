"""Loss functions for DKI."""

from __future__ import annotations

import torch


def bray_curtis(p_pred: torch.Tensor, p_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Bray-Curtis dissimilarity averaged over the batch.

    Matches the legacy formulation: ``sum |a-b| / sum |a+b|`` per sample,
    then averaged over samples. Both inputs have shape ``(B, N)``.
    """
    num = (p_pred - p_true).abs().sum(dim=-1)
    den = (p_pred + p_true).abs().sum(dim=-1).clamp_min(eps)
    return (num / den).mean()
