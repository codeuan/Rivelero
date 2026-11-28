"""Regression tests for the plant placement model used by Experiment 2."""
import numpy as np

from rivelero.experiments.plants import sample_plants


def draw(seed, contrast=5):
    return sample_plants((31, 47), 100, (5, 5), 40, contrast, np.random.default_rng(seed))


def test_reproducible_fresh_placements_without_replacement():
    first = draw(47)
    np.testing.assert_array_equal(first, draw(47))
    assert not np.array_equal(first, draw(48))
    assert len(np.unique(first, axis=0)) == 100
    assert np.all(first >= 0) and np.all(first < (31, 47))


def test_contrast_changes_sampling_and_full_grid_is_supported():
    assert not np.array_equal(draw(47, 0), draw(47, 8))
    cells = sample_plants((3, 2), 6, (5, 5), 2, 5, np.random.default_rng(0))
    assert set(map(tuple, cells)) == set(np.ndindex((3, 2)))
