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

import numpy as np
import pandas as pd


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
