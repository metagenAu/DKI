"""Tests for the read-depth QC filter in dki.data."""

from __future__ import annotations

import numpy as np
import pytest

from dki.data import filter_low_depth, load_dataset


# 4 taxa x 3 samples of raw counts.
#  - sample 0 is shallow (total 100 < 1000) -> dropped
#  - taxon 3 only appears in sample 0 -> empty after the drop -> removed
_COUNTS = np.array(
    [
        [10, 1000, 2000],   # taxon 0
        [ 5, 2000, 3000],   # taxon 1
        [ 5, 2000, 3000],   # taxon 2
        [80,    0,    0],   # taxon 3 (only in the shallow sample)
    ],
    dtype=float,
)


def test_filter_low_depth_drops_shallow_samples_and_empty_taxa():
    P_f, sample_mask, taxa_mask = filter_low_depth(_COUNTS, min_reads=1000)
    assert sample_mask.tolist() == [False, True, True]
    assert taxa_mask.tolist() == [True, True, True, False]
    assert P_f.shape == (3, 2)
    assert (P_f > 0).any(axis=1).all()      # no all-zero taxa remain


def test_filter_low_depth_raises_when_everything_dropped():
    rel = _COUNTS / _COUNTS.sum(axis=0, keepdims=True)   # columns sum to 1
    with pytest.raises(ValueError, match="removed all"):
        filter_low_depth(rel, min_reads=1000)


def test_load_dataset_applies_filter_and_masks_ptest(tmp_path):
    np.savetxt(tmp_path / "Ptrain.csv", _COUNTS, delimiter=",")
    # Ptest carries the same 4 taxa rows (2 perturbed columns).
    ptest = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
    np.savetxt(tmp_path / "Ptest.csv", ptest, delimiter=",")

    data = load_dataset(str(tmp_path), val_fraction=0.2, seed=0, min_reads=1000)

    assert data.n_species == 3
    assert data.kept_samples.tolist() == [1, 2]
    assert data.kept_taxa.tolist() == [0, 1, 2]
    # train + val cover the 2 surviving samples; every tensor has 3 species.
    assert data.p_all.shape == (2, 3)
    assert data.p_test is not None and data.p_test.shape[1] == 3   # row-masked to 3 taxa


def test_load_dataset_no_filter_by_default(tmp_path):
    np.savetxt(tmp_path / "Ptrain.csv", _COUNTS, delimiter=",")
    data = load_dataset(str(tmp_path), val_fraction=0.2, seed=0)   # min_reads=0
    assert data.n_species == 4
    assert data.kept_samples is None and data.kept_taxa is None
