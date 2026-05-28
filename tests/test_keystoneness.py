"""Tests for the classical-structural-keystoneness Python port."""

from __future__ import annotations

import numpy as np

from dki.keystoneness import classical_structural_keystoneness


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
