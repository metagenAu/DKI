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
from typing import List, Optional, Sequence, Tuple, Union

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
    # Multi-community provenance (set by ``load_multi_community_dataset``; ``None``
    # for the single-matrix ``load_dataset``). ``community_index`` is a length
    # ``n_species`` int array giving, for each row of the combined species axis,
    # the 0-based community it came from; ``community_names`` labels those
    # communities. They let downstream code (e.g. keystoneness) map a combined
    # species index back to its originating community.
    community_index: Optional[np.ndarray] = None
    community_names: Optional[list] = None
    # The per-community mass weights actually applied when fusing separately-
    # normalised communities (e.g. amplicons from different marker libraries).
    # ``None`` means the communities were stacked on a single shared scale
    # (raw concatenation), appropriate only when they came from one library.
    community_weights: Optional[np.ndarray] = None

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
            community_index=self.community_index,
            community_names=self.community_names,
            community_weights=self.community_weights,
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
    ptrain_path = _resolve_path(data_dir, "Ptrain.csv")
    if not ptrain_path.exists():
        raise FileNotFoundError(f"Missing {ptrain_path}")

    P = np.loadtxt(ptrain_path, delimiter=",")

    P_test: Optional[np.ndarray] = None
    Z_test: Optional[np.ndarray] = None
    ptest_path = _resolve_path(data_dir, "Ptest.csv")
    ztest_path = _resolve_path(data_dir, "Ztest.csv")
    if ptest_path.exists():
        P_test = np.loadtxt(ptest_path, delimiter=",")
        if test_uses_z and ztest_path.exists():
            Z_test = np.loadtxt(ztest_path, delimiter=",")

    return _assemble_dataset(
        P,
        P_test,
        Z_test,
        val_fraction=val_fraction,
        seed=seed,
        test_uses_z=test_uses_z,
        min_reads=min_reads,
        log_prefix="load_dataset",
    )


def _assemble_dataset(
    P: np.ndarray,
    P_test: Optional[np.ndarray],
    Z_test: Optional[np.ndarray],
    *,
    val_fraction: float,
    seed: int,
    test_uses_z: bool,
    min_reads: float,
    community_index: Optional[np.ndarray] = None,
    community_names: Optional[List[str]] = None,
    community_weights: Optional[np.ndarray] = None,
    log_prefix: str = "load_dataset",
) -> DKIData:
    """Build a :class:`DKIData` from already-loaded raw count matrices.

    Shared by :func:`load_dataset` (single matrix) and
    :func:`load_multi_community_dataset` (several communities row-stacked into
    one matrix). All matrices are ``(n_species, n_samples)`` raw counts /
    relative abundances; normalisation and the train/val split happen here so
    both entry points behave identically once the species axis is assembled.

    ``community_index`` (length ``n_species``) is carried through the read-depth
    taxa filter so it always lines up with the model's species dimension.
    """
    rng = np.random.default_rng(seed)

    kept_samples: Optional[np.ndarray] = None
    kept_taxa: Optional[np.ndarray] = None
    taxa_mask: Optional[np.ndarray] = None
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
        if community_index is not None:
            community_index = community_index[taxa_mask]
        print(
            f"[{log_prefix}] read-depth filter (min_reads={min_reads:g}): "
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
    if P_test is not None:
        if taxa_mask is not None:
            if P_test.shape[0] != len(taxa_mask):
                raise ValueError(
                    f"Ptest has {P_test.shape[0]} taxa rows but Ptrain had "
                    f"{len(taxa_mask)}; cannot apply the read-depth taxa filter "
                    "consistently. Ensure Ptest uses the same taxa order as Ptrain."
                )
            P_test = P_test[taxa_mask, :]
        p_test, z_from_p = process_data(P_test)
        if test_uses_z and Z_test is not None:
            if taxa_mask is not None:
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
        community_index=community_index,
        community_names=community_names,
        community_weights=community_weights,
    )


def _load_matrix(path: Union[str, Path]) -> np.ndarray:
    """Load a CSV with ``np.loadtxt``; a single row/column comes back 1-D."""
    return np.loadtxt(path, delimiter=",")


def stack_communities(
    matrices: Sequence[np.ndarray], what: str = "matrices"
) -> Tuple[np.ndarray, np.ndarray]:
    """Row-stack per-community ``(n_species, n_samples)`` matrices on a shared sample axis.

    Each input matrix is one community (e.g. bacteria / fungi / archaea) profiled
    on the **same** samples, so every matrix must have the same number of columns
    in the same order. The species rows are concatenated, fusing the communities
    into a single ``(Σ n_speciesᵢ, n_samples)`` matrix that DKI treats as one
    community under a single shared assembly rule.

    A single-species community loads from CSV as a 1-D array; it is interpreted
    as one ``(1, n_samples)`` row by matching the shared sample-column count
    (which also disambiguates it from a single-sample column vector).

    Returns ``(combined, community_index)`` where ``community_index[r]`` is the
    0-based community that contributed row ``r`` of ``combined``.
    """
    matrices = list(matrices)
    if not matrices:
        raise ValueError(f"need at least one community matrix, got 0 ({what})")

    # The shared sample-column count comes from any genuinely 2-D community; a
    # 1-D community (single species) is then read as one row over those columns.
    n_cols: Optional[int] = next(
        (m.shape[1] for m in matrices if m.ndim == 2), None
    )
    if n_cols is None:
        n_cols = matrices[0].shape[0]   # all single-species: length is the sample count

    fixed: List[np.ndarray] = []
    for i, m in enumerate(matrices):
        if m.ndim == 1:
            if m.shape[0] != n_cols:
                raise ValueError(
                    f"{what}[{i}] is 1-D with {m.shape[0]} entries, which does not "
                    f"match the {n_cols} shared samples; a single-species community "
                    "must have one value per shared sample."
                )
            m = m.reshape(1, n_cols)
        elif m.shape[1] != n_cols:
            raise ValueError(
                f"all communities must share the same samples (columns): {what} has a "
                f"community with {n_cols} sample columns but {what}[{i}] has "
                f"{m.shape[1]}. Multi-community DKI stacks species rows on a shared "
                "sample axis, so every community must be profiled on the same samples "
                "in the same order."
            )
        fixed.append(m)
    combined = np.vstack(fixed)
    community_index = np.concatenate(
        [np.full(m.shape[0], i, dtype=int) for i, m in enumerate(fixed)]
    )
    return combined, community_index


def _resolve_community_weights(
    community_weights: Optional[Union[str, Sequence[float]]], n_communities: int
) -> Optional[np.ndarray]:
    """Turn the ``community_weights`` argument into a length-N array summing to 1.

    ``None`` -> ``None`` (no weighting; raw shared-scale stacking).
    ``"equal"`` -> uniform ``1/N``.
    A sequence -> normalised to sum 1 (must be length N, non-negative, not all 0).
    """
    if community_weights is None:
        return None
    if isinstance(community_weights, str):
        if community_weights != "equal":
            raise ValueError(
                f"community_weights string must be 'equal', got {community_weights!r}."
            )
        return np.full(n_communities, 1.0 / n_communities)
    w = np.asarray(list(community_weights), dtype=float)
    if w.shape != (n_communities,):
        raise ValueError(
            f"community_weights has {w.shape[0]} entries but there are "
            f"{n_communities} communities."
        )
    if np.any(w < 0):
        raise ValueError("community_weights must be non-negative.")
    total = w.sum()
    if total <= 0:
        raise ValueError("community_weights must not be all zero.")
    return w / total


def _apply_community_weights(
    M: np.ndarray, community_index: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Re-scale each community block to a declared per-sample mass share.

    For separately-amplified markers there is no shared scale across
    communities, so the only honest fusion is an explicit one: normalise each
    community to a within-marker composition (columns sum to 1) and give it a
    fixed mass ``weights[c]`` of the joint per-sample composition. ``M`` has the
    combined row layout described by ``community_index``.

    A community absent from a given sample (all-zero block) contributes no mass;
    the joint column then sums to the weights of the *present* communities only,
    which the downstream per-sample normalisation rescales — mass cannot be
    assigned to a community that was not observed in that sample.
    """
    out = M.astype(np.float64, copy=True)
    for c, w in enumerate(weights):
        rows = community_index == c
        block = out[rows]
        col_sum = block.sum(axis=0, keepdims=True)
        col_sum = np.where(col_sum == 0, 1.0, col_sum)
        out[rows] = block / col_sum * w
    return out


def _stack_test_matrices(
    paths: Optional[Union[str, Path, Sequence[Union[str, Path]]]],
    n_species: int,
    n_communities: int,
    what: str,
) -> Optional[np.ndarray]:
    """Resolve a Ptest/Ztest argument to a combined ``(n_species, n_pairs)`` matrix.

    Accepts either a single already-combined CSV path, or a sequence of one
    per-community CSV (parallel to ``train_paths``) that is row-stacked the same
    way as the training matrices. ``None`` means "no test set".
    """
    if paths is None:
        return None
    if isinstance(paths, (str, Path)):
        mat = _load_matrix(paths)
        if mat.ndim == 1:
            # A single perturbation column loads 1-D; it must span all species.
            if mat.shape[0] != n_species:
                raise ValueError(
                    f"{what} is 1-D with {mat.shape[0]} entries but the combined "
                    f"training matrix has {n_species} species; a single combined "
                    f"{what} column must cover all communities' species."
                )
            mat = mat.reshape(n_species, 1)
        if mat.shape[0] != n_species:
            raise ValueError(
                f"{what} has {mat.shape[0]} species rows but the combined training "
                f"matrix has {n_species}; a single combined {what} must cover all "
                "communities' species in the same row order."
            )
        return mat
    paths = list(paths)
    if len(paths) != n_communities:
        raise ValueError(
            f"{what} must have one entry per community (parallel to train_paths): "
            f"expected {n_communities}, got {len(paths)}."
        )
    combined, _ = stack_communities([_load_matrix(p) for p in paths], what=what)
    if combined.shape[0] != n_species:
        raise ValueError(
            f"{what} stacks to {combined.shape[0]} species rows but the combined "
            f"training matrix has {n_species}; each community's {what} must have the "
            "same number of species rows as its Ptrain."
        )
    return combined


def load_multi_community_dataset(
    train_paths: Sequence[Union[str, Path]],
    test_paths: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    ztest_paths: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    community_names: Optional[Sequence[str]] = None,
    community_weights: Optional[Union[str, Sequence[float]]] = None,
    val_fraction: float = 0.2,
    seed: int = 0,
    test_uses_z: bool = True,
    min_reads: float = 0.0,
) -> DKIData:
    """Load N community matrices sharing the same samples and fuse them into one model.

    Each path in ``train_paths`` is a ``(n_species, n_samples)`` CSV for one
    community (e.g. bacteria, fungi, archaea) measured on the **same** samples.
    The species rows are stacked into a single ``(Σ n_speciesᵢ, n_samples)``
    matrix and handed to the ordinary DKI pipeline, so a single shared assembly
    rule ``f`` is learned across all communities — letting the model capture
    cross-community interactions and rank keystoneness on the combined species set.

    Parameters
    ----------
    train_paths
        One Ptrain-style CSV per community. All must have the same number of
        sample columns, in the same order.
    test_paths, ztest_paths
        Optional Ptest / Ztest leave-one-species-out matrices. Each may be a
        single already-combined CSV (covering all communities' species in the
        same row order as the stacked training matrix) **or** a sequence of one
        CSV per community (parallel to ``train_paths``), which is row-stacked the
        same way. ``ztest_paths`` is only used when ``test_uses_z`` is True.
    community_names
        Optional labels for the communities (parallel to ``train_paths``);
        defaults to the file stems. Recorded on ``DKIData.community_names`` and,
        together with ``DKIData.community_index``, lets downstream code map a
        combined species index back to its originating community.
    community_weights
        How to set the relative *mass* of each community in the fused
        composition. This matters when the communities are **separately
        amplified markers** (e.g. 16S + ITS + 18S): each marker is closed to its
        own sequencing total, so the data carries **no shared scale** between
        them and the true between-community proportions are not identifiable.

        * ``None`` (default) — concatenate the raw matrices and renormalise
          jointly. Correct only when all communities came from **one** library
          (one PCR split by taxonomy). For separate libraries this lets
          sequencing depth decide the between-community ratio, which is
          meaningless; a warning is emitted when ``N > 1``.
        * ``"equal"`` — normalise each community to a within-marker composition,
          then give every community an equal ``1/N`` share of each sample's
          joint composition. The honest default for separate markers: an
          explicit, depth-independent assumption.
        * a sequence of ``N`` non-negative weights — as ``"equal"`` but with a
          chosen per-community mass (normalised to sum 1).

        Either way, **within-community keystoneness rankings are unaffected by
        the choice; only cross-community comparisons depend on it** — report the
        weights you used. The applied weights are stored on
        ``DKIData.community_weights``. Weighting requires relative input, so it
        is incompatible with ``min_reads`` (raise); do read-depth QC upstream.
    val_fraction, seed, test_uses_z, min_reads
        Same meaning as in :func:`load_dataset`. The read-depth filter, when
        enabled, runs on the **combined** matrix (a sample is judged by its total
        reads across all communities) and its taxa mask is applied to
        ``community_index`` so provenance stays aligned with the model dimension.
    """
    train_paths = list(train_paths)
    if not train_paths:
        raise FileNotFoundError(
            "train_paths is empty; pass one Ptrain-style CSV per community."
        )
    for p in train_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing community training matrix: {p}")

    P, community_index = stack_communities(
        [_load_matrix(p) for p in train_paths], what="train_paths"
    )
    n_species = P.shape[0]
    n_communities = len(train_paths)

    if community_names is None:
        community_names = [Path(p).stem for p in train_paths]
    else:
        community_names = list(community_names)
        if len(community_names) != n_communities:
            raise ValueError(
                f"community_names has {len(community_names)} entries but there are "
                f"{n_communities} communities (train_paths)."
            )

    P_test = _stack_test_matrices(test_paths, n_species, n_communities, "Ptest")
    Z_test = (
        _stack_test_matrices(ztest_paths, n_species, n_communities, "Ztest")
        if test_uses_z
        else None
    )

    weights = _resolve_community_weights(community_weights, n_communities)
    if weights is not None:
        if min_reads and min_reads > 0:
            raise ValueError(
                "community_weights operates on relative compositions and is "
                "incompatible with min_reads (read-depth QC needs raw counts). "
                "Filter each community's counts upstream, then fuse with weights."
            )
        P = _apply_community_weights(P, community_index, weights)
        # Ptest is the observed after-removal composition that predictions are
        # scored against, so it must live on the same weighted simplex. Ztest is
        # presence-only (binarised downstream), so weighting it is a no-op — skip.
        if P_test is not None:
            P_test = _apply_community_weights(P_test, community_index, weights)
    elif n_communities > 1:
        warnings.warn(
            f"Fusing {n_communities} communities by raw concatenation "
            "(community_weights=None). This is only valid if they came from ONE "
            "amplicon library (one marker split by taxonomy). For SEPARATELY "
            "amplified markers (e.g. 16S + ITS) there is no shared scale, so the "
            "between-community ratio would be driven by sequencing depth. Pass "
            "community_weights='equal' (or explicit weights) to make the "
            "assumption explicit.",
            stacklevel=2,
        )

    return _assemble_dataset(
        P,
        P_test,
        Z_test,
        val_fraction=val_fraction,
        seed=seed,
        test_uses_z=test_uses_z,
        min_reads=min_reads,
        community_index=community_index,
        community_names=community_names,
        community_weights=weights,
        log_prefix="load_multi_community_dataset",
    )
