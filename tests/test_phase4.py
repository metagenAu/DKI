"""Phase-4 tests: ensemble uncertainty + null-model z-score keystoneness."""

from __future__ import annotations

import numpy as np
import torch

from dki.ensemble import EnsemblePredictor
from dki.infer import predict
from dki.keystoneness import null_model_keystoneness
from dki.model import ReplicatorODEFunc


def _simplex_batch(B: int, N: int, n_present: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(B, N)
    for b in range(B):
        idx = torch.randperm(N, generator=g)[:n_present]
        z[b, idx] = 1.0
    return z / z.sum(dim=-1, keepdim=True)


def test_ensemble_mean_std_shapes_and_simplex():
    N = 8
    models = [ReplicatorODEFunc(N) for _ in range(5)]
    for i, m in enumerate(models):
        torch.manual_seed(100 + i)
        for p in m.parameters():
            torch.nn.init.normal_(p, std=0.1)
    ens = EnsemblePredictor(models, t_final=20.0)
    z = _simplex_batch(4, N, n_present=5, seed=1)
    mean, std = ens.predict(z)
    assert mean.shape == (4, N) and std.shape == (4, N)
    assert torch.allclose(mean.sum(dim=-1), torch.ones(4), atol=1e-4)
    assert (std >= 0).all()
    # Distinct members ⇒ some spread.
    assert std.sum() > 0


def test_null_model_keystoneness_columns_and_nulls():
    N = 10
    n_samples = 5
    z_all = _simplex_batch(n_samples, N, n_present=7, seed=2)
    # Observed abundances (legacy orientation N x n_samples); only present
    # species carry mass.
    rng = np.random.default_rng(0)
    ptrn = np.zeros((N, n_samples))
    for s in range(n_samples):
        present = (z_all[s] > 0).numpy()
        ptrn[present, s] = rng.random(present.sum()) + 0.1

    torch.manual_seed(3)
    model = ReplicatorODEFunc(N)

    def predict_fn(z):
        return predict(model, z, t_final=20.0)

    # Build a few (sample, species) pairs over present species.
    sample_id, species_id = [], []
    for s in range(n_samples):
        present = np.nonzero((z_all[s] > 0).numpy())[0]
        for sp in present[:2]:
            sample_id.append(s + 1)
            species_id.append(int(sp) + 1)

    df = null_model_keystoneness(
        predict_fn, z_all, ptrn,
        np.array(sample_id), np.array(species_id),
        n_null=3, seed=0,
    )
    assert set(df.columns) == {
        "sample", "species", "p_species", "k_classical",
        "k_null_mean", "k_null_std", "k_zscore", "n_null_used",
    }
    assert (df["k_classical"] >= 0).all()
    # 7 present species ⇒ up to 3 abundance-matched nulls available.
    assert (df["n_null_used"] <= 3).all()
    assert (df["n_null_used"] >= 1).all()
    assert np.isfinite(df["k_classical"]).all()
