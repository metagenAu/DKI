"""Phase-4 bootstrap ensembles + uncertainty.

Trains ``K`` replicator models, each on a bootstrap resample (sample with
replacement) of the training assemblages, with a distinct seed so the network
initialisations differ too. Predictions become ``(mean, std)`` across the
ensemble, giving a per-(sample, species) uncertainty estimate.

The ensemble mean also serves as the ``predict_fn`` consumed by the
null-model z-score keystoneness (this module's companion in
``keystoneness.py``) and by the Shapley extension.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .data import DKIData, load_dataset
from .infer import predict
from .train import TrainConfig, TrainResult, train


def train_ensemble(
    cfg: TrainConfig,
    k: int = 5,
    seed: int = 0,
    data: Optional[DKIData] = None,
) -> Tuple[List[nn.Module], List[TrainResult], DKIData]:
    """Train ``k`` bootstrap-resampled models sharing one val split.

    Each member resamples the training columns with replacement and uses seed
    ``seed + i``. Per-member prediction saving is disabled; predictions come
    from :class:`EnsemblePredictor`.
    """
    if data is None:
        data = load_dataset(cfg.data_dir, val_fraction=cfg.val_fraction, seed=seed)

    rng = np.random.default_rng(seed)
    n_train = data.z_train.shape[0]

    models: List[nn.Module] = []
    results: List[TrainResult] = []
    for i in range(k):
        idx = torch.as_tensor(rng.integers(0, n_train, size=n_train), dtype=torch.long)
        member = replace(
            data,
            z_train=data.z_train[idx],
            p_train=data.p_train[idx],
        )
        member_cfg = replace(cfg, seed=seed + i, save_predictions=False)
        model, result, _ = train(member_cfg, data=member)
        models.append(model)
        results.append(result)
    return models, results, data


class EnsemblePredictor:
    """Wraps trained ensemble members to return ``(mean, std)`` predictions."""

    def __init__(
        self,
        models: List[nn.Module],
        mode: str = "ode",
        t_final: float = 100.0,
        method: str = "dopri5",
        rtol: float = 1e-5,
        atol: float = 1e-7,
        deq_step: float = 0.5,
        deq_max_iter: int = 50,
        deq_tol: float = 1e-6,
        deq_fallback: bool = True,
        batch_size: int = 256,
    ):
        if not models:
            raise ValueError("EnsemblePredictor needs at least one model.")
        self.models = models
        self._kwargs = dict(
            mode=mode, t_final=t_final, method=method, rtol=rtol, atol=atol,
            deq_step=deq_step, deq_max_iter=deq_max_iter, deq_tol=deq_tol,
            deq_fallback=deq_fallback, batch_size=batch_size,
        )

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` over members, each shape ``(B, N)``."""
        preds = torch.stack([predict(m, z, **self._kwargs) for m in self.models], dim=0)
        std = preds.std(dim=0, unbiased=False) if len(self.models) > 1 else torch.zeros_like(preds[0])
        return preds.mean(dim=0), std

    def mean_predict_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """A ``predict_fn(z) -> mean`` closure for keystoneness/Shapley."""
        def fn(z: torch.Tensor) -> torch.Tensor:
            return self.predict(z)[0]
        return fn
