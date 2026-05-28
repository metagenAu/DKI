"""Phase-5 tests: Monte-Carlo Shapley keystoneness."""

from __future__ import annotations

import numpy as np
import torch

from dki.extensions.shapley import shapley_keystoneness, shapley_values_for_sample


def _make_predict_fn(N: int, seed: int = 0):
    """Deterministic, support-respecting predict_fn for testing."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(N, N, generator=g)

    def predict_fn(z: torch.Tensor) -> torch.Tensor:
        support = z > 0
        logits = z @ W
        logits = logits.masked_fill(~support, float("-inf"))
        return torch.softmax(logits, dim=-1)

    return predict_fn


def test_shapley_efficiency_sums_to_one():
    """Shapley marginals telescope to v(Ω) - v(∅) = 1 for every permutation."""
    N = 9
    z = torch.zeros(N)
    z[[0, 2, 4, 5, 7]] = 1.0
    z /= z.sum()
    predict_fn = _make_predict_fn(N, seed=1)
    vals = shapley_values_for_sample(predict_fn, z, n_perm=50, seed=0)
    assert len(vals) == 5
    assert np.isclose(sum(vals.values()), 1.0, atol=1e-6)


def test_shapley_single_species():
    N = 5
    z = torch.zeros(N)
    z[3] = 1.0
    predict_fn = _make_predict_fn(N, seed=2)
    vals = shapley_values_for_sample(predict_fn, z, n_perm=10, seed=0)
    assert vals == {3: 1.0}


def test_shapley_keystoneness_dataframe():
    N = 8
    z_all = torch.zeros(3, N)
    z_all[0, [0, 1, 2, 3]] = 1.0
    z_all[1, [2, 3, 4, 5]] = 1.0
    z_all[2, [1, 4, 6, 7]] = 1.0
    z_all = z_all / z_all.sum(dim=-1, keepdim=True)
    predict_fn = _make_predict_fn(N, seed=3)

    sample_id = np.array([1, 1, 2, 3])
    species_id = np.array([1, 3, 5, 7])
    df = shapley_keystoneness(predict_fn, z_all, sample_id, species_id, n_perm=30, seed=0)
    assert list(df.columns) == ["sample", "species", "k_shapley_synergistic"]
    assert len(df) == 4
    assert np.isfinite(df["k_shapley_synergistic"]).all()
