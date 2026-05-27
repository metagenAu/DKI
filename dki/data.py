"""Data loading for DKI.

Files are stored as (n_species, n_samples) CSVs (matching the original
DKI.py / R conventions). Internally we work with (n_samples, n_species)
tensors after normalising each sample to the simplex.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def _normalize_columns(P: np.ndarray) -> np.ndarray:
    """Normalize each column (sample) to sum to 1. Empty columns are left as 0."""
    col_sum = P.sum(axis=0, keepdims=True)
    col_sum = np.where(col_sum == 0, 1.0, col_sum)
    return P / col_sum


def process_data(P: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match legacy ``process_data``: returns (p, z) of shape (n_samples, n_species).

    z is the presence-pattern normalised over present species (uniform on the
    support of the sample).
    """
    Z = (P > 0).astype(P.dtype)
    P_norm = _normalize_columns(P).astype(np.float32)
    Z_norm = _normalize_columns(Z).astype(np.float32)
    p = torch.from_numpy(P_norm.T).contiguous()
    z = torch.from_numpy(Z_norm.T).contiguous()
    return p, z


def _resolve_path(data_dir: str, name: str) -> Path:
    p = Path(data_dir) / name
    return p


@dataclass
class DKIData:
    p_train: torch.Tensor
    z_train: torch.Tensor
    p_val: torch.Tensor
    z_val: torch.Tensor
    p_all: torch.Tensor  # train + val combined (the original "P")
    z_all: torch.Tensor
    p_test: Optional[torch.Tensor]
    z_test: Optional[torch.Tensor]
    n_species: int

    def to(self, device: torch.device) -> "DKIData":
        return DKIData(
            p_train=self.p_train.to(device),
            z_train=self.z_train.to(device),
            p_val=self.p_val.to(device),
            z_val=self.z_val.to(device),
            p_all=self.p_all.to(device),
            z_all=self.z_all.to(device),
            p_test=None if self.p_test is None else self.p_test.to(device),
            z_test=None if self.z_test is None else self.z_test.to(device),
            n_species=self.n_species,
        )


def load_dataset(
    data_dir: str,
    val_fraction: float = 0.2,
    seed: int = 0,
    test_uses_z: bool = True,
) -> DKIData:
    """Load Ptrain/Ptest/Ztest CSVs from ``data_dir`` and split off a val set.

    Parameters
    ----------
    data_dir
        Directory containing ``Ptrain.csv`` (and optionally ``Ptest.csv``,
        ``Ztest.csv``).
    val_fraction
        Fraction of *columns* (samples) from Ptrain to hold out as validation.
    test_uses_z
        If True and ``Ztest.csv`` exists, use it as the initial-condition
        source for test predictions (matches real-data setting in the paper).
        If False, derive z from Ptest.
    """
    rng = np.random.default_rng(seed)
    ptrain_path = _resolve_path(data_dir, "Ptrain.csv")
    if not ptrain_path.exists():
        raise FileNotFoundError(f"Missing {ptrain_path}")

    P = np.loadtxt(ptrain_path, delimiter=",")
    n_species, n_cols = P.shape
    n_val = max(1, int(val_fraction * n_cols))
    val_idx = rng.choice(n_cols, size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(n_cols), val_idx)

    p_train, z_train = process_data(P[:, train_idx])
    p_val, z_val = process_data(P[:, val_idx])
    p_all, z_all = process_data(P)

    p_test: Optional[torch.Tensor] = None
    z_test: Optional[torch.Tensor] = None
    ptest_path = _resolve_path(data_dir, "Ptest.csv")
    ztest_path = _resolve_path(data_dir, "Ztest.csv")
    if ptest_path.exists():
        P_test = np.loadtxt(ptest_path, delimiter=",")
        p_test, z_from_p = process_data(P_test)
        if test_uses_z and ztest_path.exists():
            Z_test = np.loadtxt(ztest_path, delimiter=",")
            _, z_test = process_data(Z_test)
        else:
            z_test = z_from_p

    return DKIData(
        p_train=p_train,
        z_train=z_train,
        p_val=p_val,
        z_val=z_val,
        p_all=p_all,
        z_all=z_all,
        p_test=p_test,
        z_test=z_test,
        n_species=n_species,
    )
