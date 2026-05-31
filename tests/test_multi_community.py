"""Tests for multi-community loading (N matrices, same samples, stacked species).

These cover :func:`stack_communities` and :func:`load_multi_community_dataset`,
which fuse several per-community ``(n_species, n_samples)`` matrices that share
the same sample columns into one combined matrix under a single shared model.
"""

from __future__ import annotations

import numpy as np
import pytest

from dki.data import (
    load_dataset,
    load_multi_community_dataset,
    stack_communities,
)


# Three communities (e.g. bacteria / fungi / archaea) on the SAME 4 samples.
#  - community A: 3 taxa, community B: 2 taxa, community C: 1 taxon  -> 6 combined.
_A = np.array(
    [[5.0, 0.0, 2.0, 1.0],
     [1.0, 3.0, 0.0, 4.0],
     [0.0, 2.0, 1.0, 1.0]]
)
_B = np.array(
    [[2.0, 1.0, 0.0, 3.0],
     [0.0, 4.0, 2.0, 1.0]]
)
_C = np.array(
    [[1.0, 1.0, 5.0, 0.0]]
)


def _write(tmp_path, name, mat):
    p = tmp_path / name
    np.savetxt(p, mat, delimiter=",")
    return str(p)


def test_stack_communities_concatenates_rows_and_labels_provenance():
    combined, community_index = stack_communities([_A, _B, _C])
    assert combined.shape == (6, 4)              # 3+2+1 species, 4 shared samples
    assert community_index.tolist() == [0, 0, 0, 1, 1, 2]
    np.testing.assert_array_equal(combined[:3], _A)
    np.testing.assert_array_equal(combined[3:5], _B)
    np.testing.assert_array_equal(combined[5:], _C)


def test_stack_communities_rejects_mismatched_samples():
    bad = _B[:, :3]   # only 3 sample columns
    with pytest.raises(ValueError, match="same samples"):
        stack_communities([_A, bad])


def test_stack_communities_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        stack_communities([])


def test_load_multi_community_combines_and_normalises(tmp_path):
    paths = [
        _write(tmp_path, "A.csv", _A),
        _write(tmp_path, "B.csv", _B),
        _write(tmp_path, "C.csv", _C),
    ]
    data = load_multi_community_dataset(paths, val_fraction=0.25, seed=0)

    assert data.n_species == 6
    # every tensor lives on the 6-species combined axis
    assert data.p_all.shape == (4, 6)
    assert data.p_train.shape[1] == 6 and data.z_train.shape[1] == 6
    # per-sample (row) simplex normalisation across the whole combined community
    sums = data.p_all.sum(dim=1).numpy()
    np.testing.assert_allclose(sums, np.ones(4), atol=1e-5)
    # provenance is carried through and labelled by file stem
    assert data.community_index.tolist() == [0, 0, 0, 1, 1, 2]
    assert data.community_names == ["A", "B", "C"]


def test_multi_community_matches_a_manually_stacked_single_load(tmp_path):
    """Fusing N files must equal pre-stacking them into one Ptrain.csv."""
    paths = [
        _write(tmp_path, "A.csv", _A),
        _write(tmp_path, "B.csv", _B),
        _write(tmp_path, "C.csv", _C),
    ]
    multi = load_multi_community_dataset(paths, val_fraction=0.25, seed=3)

    np.savetxt(tmp_path / "Ptrain.csv", np.vstack([_A, _B, _C]), delimiter=",")
    single = load_dataset(str(tmp_path), val_fraction=0.25, seed=3)

    np.testing.assert_allclose(multi.p_all.numpy(), single.p_all.numpy(), atol=1e-6)
    np.testing.assert_allclose(multi.z_all.numpy(), single.z_all.numpy(), atol=1e-6)


def test_custom_community_names(tmp_path):
    paths = [_write(tmp_path, "A.csv", _A), _write(tmp_path, "B.csv", _B)]
    data = load_multi_community_dataset(
        paths, community_names=["bacteria", "fungi"], seed=0
    )
    assert data.community_names == ["bacteria", "fungi"]
    with pytest.raises(ValueError, match="community_names has"):
        load_multi_community_dataset(paths, community_names=["only-one"])


def test_per_community_test_matrices_are_stacked(tmp_path):
    train = [_write(tmp_path, "A.csv", _A), _write(tmp_path, "B.csv", _B)]
    # two leave-one-out perturbation columns per community
    ptest_a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ptest_b = np.array([[7.0, 8.0], [9.0, 1.0]])
    test = [_write(tmp_path, "Pa.csv", ptest_a), _write(tmp_path, "Pb.csv", ptest_b)]

    data = load_multi_community_dataset(train, test_paths=test, seed=0)
    assert data.p_test is not None
    assert data.p_test.shape == (2, 5)            # 2 pairs, 5 combined species
    # with no ztest given, z_test is derived from the stacked Ptest
    assert data.z_test is not None and data.z_test.shape == (2, 5)


def test_combined_test_matrix_path_is_accepted(tmp_path):
    train = [_write(tmp_path, "A.csv", _A), _write(tmp_path, "B.csv", _B)]
    combined_ptest = np.vstack(
        [np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
         np.array([[7.0, 8.0], [9.0, 1.0]])]
    )
    ptest_path = _write(tmp_path, "Ptest.csv", combined_ptest)
    data = load_multi_community_dataset(train, test_paths=ptest_path, seed=0)
    assert data.p_test is not None and data.p_test.shape == (2, 5)


def test_test_matrix_species_mismatch_raises(tmp_path):
    train = [_write(tmp_path, "A.csv", _A), _write(tmp_path, "B.csv", _B)]
    wrong = _write(tmp_path, "bad.csv", np.ones((4, 2)))   # 4 != 5 species
    with pytest.raises(ValueError, match="species rows"):
        load_multi_community_dataset(train, test_paths=wrong, seed=0)


def test_read_depth_filter_remaps_community_index(tmp_path):
    # Raw counts so the filter is meaningful. Sample 0 is shallow (total small).
    a = np.array([[1.0, 2000.0, 3000.0, 4000.0],
                  [0.0, 1000.0, 2000.0, 1000.0]])
    b = np.array([[2.0, 5000.0, 1000.0, 2000.0],
                  [0.0, 0.0, 0.0, 0.0]])          # taxon empty among deep samples
    paths = [_write(tmp_path, "A.csv", a), _write(tmp_path, "B.csv", b)]
    data = load_multi_community_dataset(paths, val_fraction=0.5, seed=0, min_reads=1000)

    # community B's second taxon vanishes -> 3 species survive, index stays aligned
    assert data.n_species == 3
    assert data.community_index.tolist() == [0, 0, 1]
    assert data.kept_taxa.tolist() == [0, 1, 2]
