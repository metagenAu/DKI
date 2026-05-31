"""Tests for the classical-structural-keystoneness Python port."""

from __future__ import annotations

import numpy as np
import pytest

from dki.keystoneness import (
    classical_structural_keystoneness,
    functional_keystoneness,
    keystoneness_table,
)


def test_zero_keystoneness_when_renormalize_matches_dynamics():
    """If q_without equals the renormalised q_with, BC is 0 and so is k."""
    rng = np.random.default_rng(0)
    N = 6
    q_with = rng.random(N).astype(float)
    q_with /= q_with.sum()

    # Build qtst so q_without == renormalise_without(q_with, sp)
    sp_idx = 2
    q_renorm = q_with.copy()
    q_renorm[sp_idx] = 0
    q_renorm /= q_renorm.sum()

    qtrn = np.stack([q_with])                # 1 training sample
    qtst = np.stack([q_renorm])              # 1 leave-one-out row
    ptrn = q_with[:, None]
    ptst = q_renorm[:, None]
    sample_id = np.array([1])
    species_id = np.array([sp_idx + 1])

    df = classical_structural_keystoneness(qtrn, qtst, ptrn, ptst, sample_id, species_id)
    assert df.shape == (1, 5)
    assert np.isclose(df["k_pred"].iloc[0], 0.0)
    assert np.isclose(df["k_true"].iloc[0], 0.0)


def test_keystoneness_scales_with_one_minus_p():
    """k = BC * (1-p_s); halving (1-p_s) halves k for a fixed BC."""
    N = 4
    sp = 0
    q_with_low = np.array([0.1, 0.4, 0.3, 0.2])
    q_with_high = np.array([0.55, 0.2, 0.15, 0.1])  # same other-species ratios
    # other species kept in same proportions so renormalised vectors match
    q_renorm = np.array([0.0, 0.4/0.9, 0.3/0.9, 0.2/0.9])

    # Make q_without a fixed deviation from q_renorm
    q_without = q_renorm + np.array([0.0, 0.1, -0.05, -0.05])

    def run(q_with):
        qtrn = q_with[None, :]
        qtst = q_without[None, :]
        ptrn = q_with[:, None]
        ptst = q_without[:, None]
        return classical_structural_keystoneness(
            qtrn, qtst, ptrn, ptst, np.array([1]), np.array([sp + 1])
        ).iloc[0]

    low = run(q_with_low)
    high = run(q_with_high)
    # Same BC numerator since q_renorm coincides; ratio should equal (1-p_low)/(1-p_high)
    assert np.isclose(
        low["k_pred"] / high["k_pred"], (1 - 0.1) / (1 - 0.55), atol=1e-9
    )


def _toy_inputs(seed: int = 1):
    """Small valid (qtrn, qtst, ptrn, ptst, sample_id, species_id) bundle."""
    rng = np.random.default_rng(seed)
    N, n_samples = 5, 3
    qtrn = rng.random((n_samples, N)); qtrn /= qtrn.sum(1, keepdims=True)
    sample_id = np.array([1, 2, 3])
    species_id = np.array([1, 3, 5])
    qtst = rng.random((3, N)); qtst /= qtst.sum(1, keepdims=True)
    ptrn = qtrn.T.copy()
    ptst = rng.random((N, 3)); ptst /= ptst.sum(0, keepdims=True)
    return qtrn, qtst, ptrn, ptst, sample_id, species_id


def test_functional_equals_structural_for_identity_gcn():
    """With GCN = I, f = q (already a simplex), so Kf must equal Ks exactly."""
    qtrn, qtst, ptrn, ptst, sid, spid = _toy_inputs()
    N = qtrn.shape[1]
    ks = classical_structural_keystoneness(qtrn, qtst, ptrn, ptst, sid, spid)
    kf = functional_keystoneness(qtrn, qtst, ptrn, ptst, np.eye(N), sid, spid)
    assert np.allclose(kf["kf_pred"], ks["k_pred"])
    assert np.allclose(kf["kf_true"], ks["k_true"])


def test_functional_gcn_orientation_autodetect():
    """Passing GCN transposed (n_functions, n_species) gives the same result."""
    qtrn, qtst, ptrn, ptst, sid, spid = _toy_inputs()
    N = qtrn.shape[1]
    rng = np.random.default_rng(2)
    gcn = rng.random((N, 7))                      # (n_species, n_functions)
    a = functional_keystoneness(qtrn, qtst, ptrn, ptst, gcn, sid, spid)
    b = functional_keystoneness(qtrn, qtst, ptrn, ptst, gcn.T, sid, spid)
    assert np.allclose(a["kf_pred"], b["kf_pred"])


def test_functional_rejects_mismatched_gcn():
    qtrn, qtst, ptrn, ptst, sid, spid = _toy_inputs()
    bad = np.ones((4, 4))                          # neither axis == N (=5)
    with pytest.raises(ValueError):
        functional_keystoneness(qtrn, qtst, ptrn, ptst, bad, sid, spid)


def test_keystoneness_table_columns():
    qtrn, qtst, ptrn, ptst, sid, spid = _toy_inputs()
    N = qtrn.shape[1]
    no_gcn = keystoneness_table(qtrn, qtst, ptrn, ptst, sid, spid)
    assert list(no_gcn.columns) == ["sample", "species", "p_species", "str_pred", "str_true"]
    with_gcn = keystoneness_table(qtrn, qtst, ptrn, ptst, sid, spid, gcn=np.eye(N))
    assert list(with_gcn.columns) == [
        "sample", "species", "p_species", "str_pred", "func_pred", "str_true", "func_true"
    ]
    # Identity GCN => functional matches structural inside the combined table too.
    assert np.allclose(with_gcn["func_pred"], with_gcn["str_pred"])


def test_compositional_input_gives_nonnegative_keystoneness():
    """k = BC * (1-p) with p in [0,1] can never be negative."""
    qtrn, qtst, ptrn, ptst, sid, spid = _toy_inputs()
    ks = classical_structural_keystoneness(qtrn, qtst, ptrn, ptst, sid, spid)
    assert (ks["k_pred"] >= 0).all()


def test_warns_when_ptrn_not_compositional():
    """Counts/percentages (entries > 1) trip the (1-p) sign and must warn."""
    qtrn, qtst, ptrn, ptst, sid, spid = _toy_inputs()
    ptrn_counts = ptrn * 1000.0          # raw counts, columns no longer sum to 1
    with pytest.warns(UserWarning, match="not per-sample relative abundance"):
        ks = classical_structural_keystoneness(qtrn, qtst, ptrn_counts, ptst, sid, spid)
    # And the symptom the warning is about: negative keystoneness.
    assert (ks["k_pred"] < 0).any()
