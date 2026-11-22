"""Correction energies, local descent and cycle entropy on hand-checkable graphs."""

import math

import numpy as np
import pytest
import stim

from src.decoders.free_energy import CorrectionScorer


def _w(p):
    return math.log((1 - p) / p)


def _score(scorer, *corrections):
    pairs = [np.asarray(c, np.int64).reshape(-1, 2) for c in corrections]
    offsets = np.cumsum([0] + [len(p) for p in pairs])
    return scorer.score(np.concatenate(pairs), offsets)


def test_energy_counts_edges_and_hyperedge_savings():
    dem = stim.DetectorErrorModel("""
        error(.01) D0 D1
        error(.02) D2 D3
        error(.005) D0 D1 ^ D2 D3
    """)
    sc = CorrectionScorer(dem, descent_steps=0)
    e, h, _ = _score(sc, [[0, 1]], [[0, 1], [2, 3]])
    # Edge weights use only the faults that produce that edge on its own.
    assert e[0] == pytest.approx(_w(.01))
    # Both edges together are cheaper as the single correlated fault.
    assert e[1] == pytest.approx(_w(.005))
    np.testing.assert_allclose(h, 0)


def test_descent_and_entropy_on_a_square():
    # Two equal routes from D0 to D2 around a square; a direct diagonal is absent.
    dem = stim.DetectorErrorModel("""
        error(.01) D0 D1
        error(.01) D1 D2
        error(.01) D2 D3
        error(.01) D3 D0
    """)
    sc = CorrectionScorer(dem)
    e, h, ident = _score(sc, [[0, 1], [1, 2]], [[0, 3], [3, 2]])
    w = _w(.01)
    np.testing.assert_allclose(e, 2 * w)
    # Toggling the square swaps the two equal routes (dE = 0), worth log 2,
    # minus the vacuum term log(1 + exp(-4w)).
    np.testing.assert_allclose(h, math.log(2) - math.log1p(math.exp(-4 * w)))
    assert ident[0] != ident[1]


def test_descent_moves_to_a_cheaper_equivalent_correction():
    dem = stim.DetectorErrorModel("""
        error(.001) D0 D1
        error(.001) D1 D2
        error(.05) D0 D2
    """)
    sc = CorrectionScorer(dem)
    long, short = _score(sc, [[0, 1], [1, 2]], [[0, 2]])[0]
    assert long == pytest.approx(short) == pytest.approx(_w(.05))


def test_boundary_pairs_and_unknown_edges():
    dem = stim.DetectorErrorModel("error(.02) D0\nerror(.03) D0 D1")
    sc = CorrectionScorer(dem)
    e, _, _ = _score(sc, [[0, -1]], [[1, 0], [0, -1]], [[1, -1]])
    assert e[0] == pytest.approx(_w(.02))
    assert e[1] == pytest.approx(_w(.02) + _w(.03))
    assert np.isinf(e[2])


def test_scratch_state_is_clean_between_calls():
    dem = stim.DetectorErrorModel("""
        error(.01) D0 D1
        error(.01) D1 D2
        error(.01) D2 D3
        error(.01) D3 D0
        error(.004) D0 D1 ^ D2 D3
    """)
    sc = CorrectionScorer(dem)
    first = _score(sc, [[0, 1], [2, 3]], [[1, 2]])
    again = _score(sc, [[0, 1], [2, 3]], [[1, 2]])
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)
    for scratch in sc.state.arrays[:3]:
        assert not scratch.any()
