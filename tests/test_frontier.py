"""Independent enumeration of covers and logical parities for the confidence screen."""

from itertools import product

import numpy as np
import pytest
import stim

from src.decoders.faults import read_model
from src.decoders.frontier import MAX_HITS, frontier_screen, path_model


def _enumerate(vertices, even, odd, boundary, accepted):
    """Yield the individual weights of every completed cover and logical assignment."""
    if not vertices:
        yield 0, 1.
        return
    u, *rest = vertices
    options = [(None, boundary[u])]
    for v in rest:
        if accepted[u, v]:
            options.append((v, (even[u, v], odd[u, v])))
    for partner, weights in options:
        remaining = [v for v in rest if v != partner]
        for bit, (frame, mass) in product(range(2), _enumerate(remaining, even, odd, boundary, accepted)):
            yield bit ^ frame, weights[bit] * mass


@pytest.mark.parametrize("cutoff", [0., .35])
def test_frontier_matches_exhaustive_random_cover_odds(cutoff):
    rng = np.random.default_rng(8502)
    n = 7
    even, odd = (rng.uniform(.001, .5, (n, n)) for _ in range(2))
    for weights in (even, odd):
        weights[:] = (weights + weights.T) / 2
        np.fill_diagonal(weights, 0.)
    boundary = rng.uniform(.03, .6, (n, 2))
    keep = np.arange(n, dtype=np.int32)
    for _ in range(14):
        row = rng.random(n) < .62
        live = np.flatnonzero(row)
        if not len(live):
            continue
        total = even + odd
        scale = np.maximum(boundary.sum(axis=1), np.sqrt(total[:, live].max(axis=1)))
        accepted = total > cutoff * np.outer(scale, scale)
        mass = np.zeros(2)
        for frame, weight in _enumerate(list(live), even, odd, boundary, accepted):
            mass[frame] += weight
        ratio = max(mass) / min(mass)
        best, uncertain = frontier_screen(row[None, :], keep, even, odd, boundary, 7, cutoff, max(1., ratio*(1-1e-9)))
        assert not uncertain[0] and best[0] == np.argmax(mass)
        _, uncertain = frontier_screen(row[None, :], keep, even, odd, boundary, 7, cutoff, ratio*(1+1e-9))
        assert uncertain[0]


def test_frontier_defers_on_limits_or_no_valid_cover():
    n = MAX_HITS + 1
    keep = np.arange(n, dtype=np.int32)
    zeros = np.zeros((n,n))
    rows = np.zeros((3,n), bool)
    rows[0, :] = True
    rows[1, :5] = True
    rows[2, 0] = True
    # All pairs cross the frontier in row 1. Row 2 has no boundary solution.
    boundary = np.zeros((n,2))
    _, pending = frontier_screen(rows, keep, np.ones_like(zeros), zeros, boundary, 1, 0., 1.)
    assert pending.tolist() == [True, True, True]
    with pytest.raises(ValueError, match="width"):
        frontier_screen(rows, keep, zeros, zeros, boundary, 99, 0., 1.)


def test_frontier_preserves_tiny_cover_odds_across_many_steps():
    n = 24
    zeros = np.zeros((n,n))
    boundary = np.zeros((n,2))
    boundary[:,0] = 1e-50
    boundary[-1] = [0., 2e-50]
    pred, pending = frontier_screen(np.ones((1,n), bool), np.arange(n,dtype=np.int32),
                                     zeros, zeros, boundary, 2, 0., 100.)
    assert pred[0] and not pending[0]


def test_walk_projection_declines_invisible_and_multiple_logicals():
    for spec in ("error(.1) L0\nerror(.15) D0 L0", "error(.1) D0 L1"):
        dem = stim.DetectorErrorModel(spec)
        assert path_model(read_model(dem), dem.num_detectors) is None
