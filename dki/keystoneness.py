"""Structural keystoneness — Python port of ``Keystoneness_computing.R``.

For each (sample, species) pair listed in ``sample_id`` / ``species_id``
this computes the *classical structural keystoneness* of the paper:

    k = BC(q_with_renorm, q_without)  *  (1 - p_species_in_sample)

where:
  * ``q_with``      — predicted composition of the full training sample.
  * ``q_with_renorm`` — ``q_with`` with the target species masked to zero
    and renormalised to the simplex.
  * ``q_without``   — predicted composition of the leave-one-species-out
    assemblage (i.e. row of ``qtst``).

Predicted (``k_pred``) and ground-truth (``k_true``) variants are returned.

Notes
-----
This is Paine's single-species notion of keystoneness as operationalised
in Wang et al. (bioRxiv 2023.03.15.532858). The ``(1 - p)`` factor encodes
the "disproportionate effect relative to abundance" clause. Phase 4 will
add an alternative null-model z-score calibration; Phase 5 will add a
distinct synergy-aware (Shapley) extension. Both will live alongside
``k_classical`` rather than replacing it.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch


def _bc(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    num = np.abs(a - b).sum()
    den = max(np.abs(a + b).sum(), eps)
    return float(num / den)


def _renormalise_without(q: np.ndarray, idx: int) -> np.ndarray:
    out = q.copy()
    out[idx] = 0.0
    s = out.sum()
    if s > 0:
        out /= s
    return out


def _project_function(q: np.ndarray, gcn: np.ndarray) -> np.ndarray:
    """Project a composition onto function space and renormalise to the simplex.

    Mirrors the R ``q %*% GCN`` step followed by ``f / sum(f)``. ``gcn`` is
    ``(n_species, n_functions)``.
    """
    f = q @ gcn
    s = f.sum()
    if s > 0:
        f = f / s
    return f


def _orient_gcn(gcn: np.ndarray, n_species: int) -> np.ndarray:
    """Return ``gcn`` as ``(n_species, n_functions)``, transposing if needed."""
    gcn = np.asarray(gcn, dtype=float)
    if gcn.ndim != 2:
        raise ValueError(f"GCN must be 2-D, got shape {gcn.shape}")
    if gcn.shape[0] == n_species:
        return gcn
    if gcn.shape[1] == n_species:
        return gcn.T
    raise ValueError(
        f"GCN shape {gcn.shape} has no axis matching n_species={n_species}; "
        "expected (n_species, n_functions) or its transpose."
    )


def classical_structural_keystoneness(
    qtrn: np.ndarray,
    qtst: np.ndarray,
    ptrn: np.ndarray,
    ptst: np.ndarray,
    sample_id: np.ndarray,
    species_id: np.ndarray,
) -> pd.DataFrame:
    """Classical structural keystoneness.

    Parameters
    ----------
    qtrn
        Predicted full-community composition, shape ``(n_samples, n_species)``.
    qtst
        Predicted leave-one-out composition, shape ``(n_pairs, n_species)``.
        Row ``i`` corresponds to ``(sample_id[i], species_id[i])``.
    ptrn
        Observed full-community composition, shape ``(n_species, n_samples)``
        (legacy orientation — matches the on-disk ``Ptrain.csv``).
    ptst
        Observed leave-one-out composition, shape ``(n_species, n_pairs)``
        (legacy orientation — matches ``Ptest.csv``).
    sample_id
        1-indexed sample indices into ``Ptrain`` (matches the R script's CSV).
    species_id
        1-indexed species indices.

    Returns
    -------
    DataFrame with columns ``sample``, ``species`` (1-indexed),
    ``p_species`` (abundance of the focal species in the focal sample),
    ``k_pred`` and ``k_true``.
    """
    if qtst.shape[0] != len(sample_id) or qtst.shape[0] != len(species_id):
        raise ValueError("qtst rows must match len(sample_id) == len(species_id)")

    n_pairs = qtst.shape[0]
    rows = []
    for i in range(n_pairs):
        s = int(sample_id[i]) - 1   # 1-indexed -> 0-indexed
        sp = int(species_id[i]) - 1

        q_with = qtrn[s]
        q_without_pred = qtst[i]
        q_renorm_pred = _renormalise_without(q_with, sp)

        p_with = ptrn[:, s]
        p_without_true = ptst[:, i]
        p_renorm_true = _renormalise_without(p_with, sp)

        bc_pred = _bc(q_renorm_pred, q_without_pred)
        bc_true = _bc(p_renorm_true, p_without_true)
        p_s = float(p_with[sp])

        rows.append({
            "sample": s + 1,
            "species": sp + 1,
            "p_species": p_s,
            "k_pred": bc_pred * (1.0 - p_s),
            "k_true": bc_true * (1.0 - p_s),
        })
    return pd.DataFrame(rows)


def functional_keystoneness(
    qtrn: np.ndarray,
    qtst: np.ndarray,
    ptrn: np.ndarray,
    ptst: np.ndarray,
    gcn: np.ndarray,
    sample_id: np.ndarray,
    species_id: np.ndarray,
) -> pd.DataFrame:
    """Functional keystoneness — Python port of the ``GCN`` half of the R script.

    Where structural keystoneness measures the Bray–Curtis shift in *composition*
    after removing a species, functional keystoneness measures the shift in the
    community's *function* profile. Each composition ``q`` is projected onto
    function space via the gene-copy-number matrix (``f = q @ GCN``), renormalised
    to the simplex, and the BC shift is taken there::

        kf = BC(f(q_with_renorm), f(q_without)) * (1 - p_species_in_sample)

    Parameters mirror :func:`classical_structural_keystoneness`, with one addition:

    gcn
        Gene-copy-number / trait matrix. Accepted as ``(n_species, n_functions)``
        or its transpose (auto-detected); any other shape raises ``ValueError``.

    Returns
    -------
    DataFrame with ``sample``, ``species`` (1-indexed), ``p_species``,
    ``kf_pred`` and ``kf_true``.
    """
    if qtst.shape[0] != len(sample_id) or qtst.shape[0] != len(species_id):
        raise ValueError("qtst rows must match len(sample_id) == len(species_id)")

    n_species = qtrn.shape[1]
    G = _orient_gcn(gcn, n_species)

    rows = []
    for i in range(qtst.shape[0]):
        s = int(sample_id[i]) - 1
        sp = int(species_id[i]) - 1

        q_with = qtrn[s]
        q_renorm_pred = _renormalise_without(q_with, sp)
        f_before_pred = _project_function(q_renorm_pred, G)
        f_after_pred = _project_function(qtst[i], G)
        bc_pred = _bc(f_before_pred, f_after_pred)

        p_with = ptrn[:, s]
        p_renorm_true = _renormalise_without(p_with, sp)
        f_before_true = _project_function(p_renorm_true, G)
        f_after_true = _project_function(ptst[:, i], G)
        bc_true = _bc(f_before_true, f_after_true)

        p_s = float(p_with[sp])
        rows.append({
            "sample": s + 1,
            "species": sp + 1,
            "p_species": p_s,
            "kf_pred": bc_pred * (1.0 - p_s),
            "kf_true": bc_true * (1.0 - p_s),
        })
    return pd.DataFrame(rows)


def keystoneness_table(
    qtrn: np.ndarray,
    qtst: np.ndarray,
    ptrn: np.ndarray,
    ptst: np.ndarray,
    sample_id: np.ndarray,
    species_id: np.ndarray,
    gcn: np.ndarray | None = None,
) -> pd.DataFrame:
    """Combined structural + functional keystoneness, matching the R output.

    Returns a single DataFrame with ``sample``, ``species``, ``p_species`` and the
    R script's four keystone columns ``str_pred``, ``func_pred``, ``str_true``,
    ``func_true``. When ``gcn`` is ``None`` the functional columns are omitted
    (structural keystoneness needs no trait matrix).
    """
    s = classical_structural_keystoneness(
        qtrn, qtst, ptrn, ptst, sample_id, species_id
    ).rename(columns={"k_pred": "str_pred", "k_true": "str_true"})
    if gcn is None:
        return s
    f = functional_keystoneness(qtrn, qtst, ptrn, ptst, gcn, sample_id, species_id)
    s["func_pred"] = f["kf_pred"].to_numpy()
    s["func_true"] = f["kf_true"].to_numpy()
    return s[["sample", "species", "p_species",
              "str_pred", "func_pred", "str_true", "func_true"]]


def null_model_keystoneness(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    z_all: torch.Tensor,
    ptrn: np.ndarray,
    sample_id: np.ndarray,
    species_id: np.ndarray,
    n_null: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Phase-4 null-model z-score calibration of structural keystoneness.

    For each (sample, focal species) pair we draw up to ``n_null`` *abundance-
    matched* null species — present species in the same sample whose relative
    abundance is closest to the focal species — and compute the classical
    structural keystoneness for each. The focal score is then reported as a
    z-score against this per-sample null distribution::

        k_zscore = (k_classical - mean_null) / std_null

    This is an **alternative** calibration that answers "is this species more
    impactful than equally-abundant species in the same community?", reported
    alongside (never replacing) ``k_classical``.

    Parameters
    ----------
    predict_fn
        ``predict_fn(z) -> q`` mapping a batch of presence-normalised
        assemblages (shape ``(B, N)``) to predicted equilibria. Typically an
        ensemble mean (see :class:`dki.ensemble.EnsemblePredictor`).
    z_all
        Presence-normalised assemblages (uniform over the present support),
        shape ``(n_samples, N)``. The leave-one-out assemblages are derived
        from these.
    ptrn
        Observed abundances, shape ``(n_species, n_samples)`` (legacy
        orientation). Columns are renormalised internally to the simplex.
    sample_id, species_id
        1-indexed (sample, species) pairs, as in the classical port.
    n_null
        Maximum number of abundance-matched null species per pair.
    seed
        Seeds tie-breaking when more candidates than ``n_null`` share the same
        abundance distance.

    Returns
    -------
    DataFrame with ``sample``, ``species``, ``p_species``, ``k_classical``,
    ``k_null_mean``, ``k_null_std``, ``k_zscore`` and ``n_null_used``.
    """
    if len(sample_id) != len(species_id):
        raise ValueError("sample_id and species_id must be the same length")

    rng = np.random.default_rng(seed)
    device = z_all.device
    P = ptrn / ptrn.sum(axis=0, keepdims=True).clip(min=1e-12)

    n_pairs = len(sample_id)
    samples0 = np.asarray(sample_id, dtype=int) - 1
    species0 = np.asarray(species_id, dtype=int) - 1

    # Predicted full-community composition for each sample that appears.
    unique_samples = np.unique(samples0)
    q_with_all = predict_fn(z_all[torch.as_tensor(unique_samples, device=device)])
    q_with_map = {
        int(s): q_with_all[i].detach().cpu().numpy()
        for i, s in enumerate(unique_samples)
    }

    # Build every leave-one-out assemblage (focal + nulls) up front, then run
    # a single batched prediction.
    loo_rows = []                 # tensors (N,)
    pair_meta = []                # per-pair: (focal_offset, null_offsets, null_species)
    for i in range(n_pairs):
        s, sp = int(samples0[i]), int(species0[i])
        z_s = z_all[s]
        present = torch.nonzero(z_s > 0, as_tuple=False).flatten().cpu().numpy()
        candidates = present[present != sp]
        if candidates.size > n_null:
            a_focal = P[sp, s]
            dist = np.abs(P[candidates, s] - a_focal)
            jitter = rng.random(candidates.size) * 1e-12
            order = np.argsort(dist + jitter)
            chosen = candidates[order[:n_null]]
        else:
            chosen = candidates

        focal_offset = len(loo_rows)
        loo_rows.append(_loo_assemblage(z_s, sp))
        null_offsets = []
        for c in chosen:
            null_offsets.append(len(loo_rows))
            loo_rows.append(_loo_assemblage(z_s, int(c)))
        pair_meta.append((focal_offset, null_offsets, chosen))

    q_loo = predict_fn(torch.stack(loo_rows, dim=0)).detach().cpu().numpy()

    rows = []
    for i in range(n_pairs):
        s, sp = int(samples0[i]), int(species0[i])
        focal_offset, null_offsets, chosen = pair_meta[i]
        q_with = q_with_map[s]

        k_focal = _structural_k(q_with, q_loo[focal_offset], sp, P[sp, s])
        null_ks = np.array(
            [_structural_k(q_with, q_loo[o], int(c), P[int(c), s])
             for o, c in zip(null_offsets, chosen)]
        )

        if null_ks.size >= 2:
            mean_null = float(null_ks.mean())
            std_null = float(null_ks.std(ddof=1))
            z = (k_focal - mean_null) / std_null if std_null > 0 else np.nan
        else:
            mean_null = float(null_ks.mean()) if null_ks.size else np.nan
            std_null = np.nan
            z = np.nan

        rows.append({
            "sample": s + 1,
            "species": sp + 1,
            "p_species": float(P[sp, s]),
            "k_classical": k_focal,
            "k_null_mean": mean_null,
            "k_null_std": std_null,
            "k_zscore": z,
            "n_null_used": int(null_ks.size),
        })
    return pd.DataFrame(rows)


def _loo_assemblage(z_s: torch.Tensor, drop: int) -> torch.Tensor:
    """Uniform assemblage over ``z_s``'s support with species ``drop`` removed."""
    v = z_s.clone()
    v[drop] = 0.0
    return v / v.sum().clamp_min(1e-12)


def _structural_k(q_with: np.ndarray, q_without: np.ndarray, sp: int, p_s: float) -> float:
    return _bc(_renormalise_without(q_with, sp), q_without) * (1.0 - float(p_s))
