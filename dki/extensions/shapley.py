"""Phase-5 Monte-Carlo Shapley keystoneness (synergy-aware extension).

The classical structural keystoneness measures a *single* species' removal
effect. The Shapley value instead distributes each species' contribution to
the community while accounting for **synergy and redundancy**: a species that
is only impactful in the presence of certain partners, or one whose role is
covered by redundant partners, is scored fairly by averaging over coalitions.

Value function. For a sample with present-species set ``Ω`` and predicted
full community ``q_Ω``, the value of a coalition ``S ⊆ Ω`` is::

    v(S) = 1 - BC(q_Ω, q_S)

where ``q_S`` is the model's prediction from the assemblage uniform over ``S``
(``v(∅) = 0`` and ``v(Ω) = 1`` by construction). The Shapley value of species
``s`` is its average marginal contribution ``v(S ∪ {s}) - v(S)`` over uniformly
random join orders, estimated by Monte-Carlo over ``n_perm`` permutations.

This is reported as ``k_shapley_synergistic`` and is a **different question**
from ``k_classical`` / ``k_zscore`` — it is never a replacement.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch


def _bc(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = (a - b).abs().sum(dim=-1)
    den = (a + b).abs().sum(dim=-1).clamp_min(eps)
    return num / den


def _uniform_over(indices: np.ndarray, n: int, device, dtype) -> torch.Tensor:
    v = torch.zeros(n, device=device, dtype=dtype)
    if indices.size:
        v[torch.as_tensor(indices, device=device)] = 1.0 / indices.size
    return v


def shapley_values_for_sample(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    z_s: torch.Tensor,
    n_perm: int = 200,
    seed: int = 0,
) -> Dict[int, float]:
    """Monte-Carlo Shapley value of every present species in one sample.

    ``z_s`` is the sample's presence-normalised assemblage (shape ``(N,)``).
    Returns ``{species_index: shapley_value}`` for present species.
    """
    device, dtype = z_s.device, z_s.dtype
    n = z_s.shape[0]
    present = torch.nonzero(z_s > 0, as_tuple=False).flatten().cpu().numpy()
    K = present.size
    if K == 0:
        return {}

    q_full = predict_fn(_uniform_over(present, n, device, dtype).unsqueeze(0))[0]

    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(present) for _ in range(n_perm)])  # (n_perm, K)

    if K == 1:
        return {int(present[0]): 1.0}

    # Build all interior-prefix coalitions (sizes 1..K-1) for every permutation,
    # predict in one batched call, then accumulate marginals with no model use.
    coalition_vecs = []
    for p in range(n_perm):
        for t in range(1, K):
            coalition_vecs.append(_uniform_over(perms[p, :t], n, device, dtype))
    q_coal = predict_fn(torch.stack(coalition_vecs, dim=0))
    v_interior = (1.0 - _bc(q_full.unsqueeze(0), q_coal)).view(n_perm, K - 1).cpu().numpy()

    marginals = np.zeros(n)
    for p in range(n_perm):
        # v at prefix sizes 0..K: 0 (empty), interior values, 1 (full).
        vs = np.empty(K + 1)
        vs[0] = 0.0
        vs[1:K] = v_interior[p]
        vs[K] = 1.0
        for t in range(1, K + 1):
            sp = int(perms[p, t - 1])
            marginals[sp] += vs[t] - vs[t - 1]
    marginals /= n_perm
    return {int(sp): float(marginals[sp]) for sp in present}


def shapley_keystoneness(
    predict_fn: Callable[[torch.Tensor], torch.Tensor],
    z_all: torch.Tensor,
    sample_id: np.ndarray,
    species_id: np.ndarray,
    n_perm: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Shapley keystoneness for each (sample, species) pair.

    Computes Shapley values once per unique sample (covering all its present
    species) and reports the requested pairs.

    Returns a DataFrame with ``sample``, ``species`` (1-indexed) and
    ``k_shapley_synergistic``.
    """
    if len(sample_id) != len(species_id):
        raise ValueError("sample_id and species_id must be the same length")

    samples0 = np.asarray(sample_id, dtype=int) - 1
    species0 = np.asarray(species_id, dtype=int) - 1

    cache: Dict[int, Dict[int, float]] = {}
    rows = []
    for s, sp in zip(samples0, species0):
        s, sp = int(s), int(sp)
        if s not in cache:
            cache[s] = shapley_values_for_sample(
                predict_fn, z_all[s], n_perm=n_perm, seed=seed + s
            )
        rows.append({
            "sample": s + 1,
            "species": sp + 1,
            "k_shapley_synergistic": cache[s].get(sp, np.nan),
        })
    return pd.DataFrame(rows)
