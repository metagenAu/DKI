"""Data loading for DKI.

Files are stored as (n_species, n_samples) CSVs (matching the original
DKI.py / R conventions). Internally we work with (n_samples, n_species)
tensors after normalising each sample to the simplex.
"""

from __future__ import annotations

import os
import warnings
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


def filter_low_depth(
    P: np.ndarray, min_reads: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop shallow samples, then taxa that vanish as a result.

    Operates on the **raw count** matrix ``P`` of shape ``(n_taxa, n_samples)``
    *before* any normalisation:

    1. drop every sample (column) whose total reads ``< min_reads``;
    2. drop every taxon (row) with zero detections across the surviving samples.

    Returns ``(P_filtered, sample_mask, taxa_mask)`` where the masks are boolean
    arrays over the **original** columns / rows of ``P`` (so callers can apply the
    same taxa mask to ``Ptest``/``Ztest`` and remap species/sample indices).
    """
    col_sums = P.sum(axis=0)
    sample_mask = col_sums >= min_reads
    if not sample_mask.any():
        raise ValueError(
            f"min_reads={min_reads:g} removed all {P.shape[1]} samples "
            f"(deepest sample had {col_sums.max():.0f} reads). If the data is "
            "already relative abundance, leave min_reads at 0; otherwise pass "
            "raw counts or lower the threshold."
        )
    P_s = P[:, sample_mask]
    taxa_mask = (P_s > 0).any(axis=1)
    return P_s[taxa_mask, :], sample_mask, taxa_mask


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
    # 0-indexed positions into the ORIGINAL Ptrain that survived ``filter_low_depth``.
    # ``None`` when no read-depth filter was applied (all columns/rows kept).
    kept_samples: Optional[np.ndarray] = None
    kept_taxa: Optional[np.ndarray] = None

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
            kept_samples=self.kept_samples,
            kept_taxa=self.kept_taxa,
        )


def load_dataset(
    data_dir: str,
    val_fraction: float = 0.2,
    seed: int = 0,
    test_uses_z: bool = True,
    min_reads: float = 0.0,
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
    min_reads
        Read-depth quality filter applied to the **raw** ``Ptrain`` counts
        *before* normalisation and the train/val split: samples whose total
        reads ``< min_reads`` are dropped, then taxa with zero detections among
        the survivors are dropped. ``0`` (default) disables it. The surviving
        taxa rows are also applied to ``Ptest``/``Ztest`` so model dimensions
        stay aligned. ``DKIData.kept_samples`` / ``kept_taxa`` record the
        original indices that survived.
    """
    rng = np.random.default_rng(seed)
    ptrain_path = _resolve_path(data_dir, "Ptrain.csv")
    if not ptrain_path.exists():
        raise FileNotFoundError(f"Missing {ptrain_path}")

    P = np.loadtxt(ptrain_path, delimiter=",")

    kept_samples: Optional[np.ndarray] = None
    kept_taxa: Optional[np.ndarray] = None
    if min_reads and min_reads > 0:
        if np.allclose(P.sum(axis=0), 1.0, atol=1e-3):
            warnings.warn(
                "min_reads is set but every Ptrain column already sums to ~1, so "
                "the data looks like relative abundance, not raw counts. The "
                "filter expects counts; skipping it would normally drop every "
                "sample. Proceeding anyway — pass raw counts if this is wrong.",
                stacklevel=2,
            )
        n0_taxa, n0_samples = P.shape
        P, sample_mask, taxa_mask = filter_low_depth(P, min_reads)
        kept_samples = np.nonzero(sample_mask)[0]
        kept_taxa = np.nonzero(taxa_mask)[0]
        print(
            f"[load_dataset] read-depth filter (min_reads={min_reads:g}): "
            f"kept {kept_samples.size}/{n0_samples} samples, "
            f"{kept_taxa.size}/{n0_taxa} taxa (dropped "
            f"{n0_samples - kept_samples.size} shallow samples, "
            f"{n0_taxa - kept_taxa.size} now-empty taxa)."
        )

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
        if kept_taxa is not None:
            if P_test.shape[0] != len(taxa_mask):
                raise ValueError(
                    f"Ptest has {P_test.shape[0]} taxa rows but Ptrain had "
                    f"{len(taxa_mask)}; cannot apply the read-depth taxa filter "
                    "consistently. Ensure Ptest uses the same taxa order as Ptrain."
                )
            P_test = P_test[taxa_mask, :]
        p_test, z_from_p = process_data(P_test)
        if test_uses_z and ztest_path.exists():
            Z_test = np.loadtxt(ztest_path, delimiter=",")
            if kept_taxa is not None:
                Z_test = Z_test[taxa_mask, :]
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
        kept_samples=kept_samples,
        kept_taxa=kept_taxa,
    )
