"""Compare exact-table predictions against the independent Bernoulli fault law."""
from itertools import product

import numpy as np
import stim

from src.decoders.exact import ExactFaultTable
from src.decoders.faults import read_model


def _enumerate(faults, detectors, observables):
    mass = np.zeros((1 << detectors, 1 << observables))
    for state in product((False, True), repeat=len(faults)):
        syndrome = logical = 0
        probability = 1.0
        for present, (sites, mask, rate) in zip(state, faults):
            if present:
                syndrome ^= sites
                logical ^= mask
            probability *= rate if present else 1.0 - rate
        mass[syndrome, logical] += probability
    return mass


def _lookup_all(dem, max_rank=24):
    table = ExactFaultTable.from_model(read_model(dem), dem.num_detectors, dem.num_observables, max_rank)
    inputs = (np.arange(1 << dem.num_detectors)[:, None] >> np.arange(dem.num_detectors)) & 1
    predictions, pending = table.lookup(inputs)
    assert predictions.shape == (len(inputs), dem.num_observables)
    assert predictions.dtype == pending.dtype == np.bool_
    return predictions @ (1 << np.arange(dem.num_observables)), pending


def test_binary_map_with_offsets_fair_fault_and_dependent_detectors():
    dem = stim.DetectorErrorModel("""
        error(0.17) D0 D2 L0
        error(0.5) D1
        error(0.78) D1 ^ D0 D2
        error(0.07) L0
        error(1) D3 L0
    """)
    faults = [(5, 1, .17), (2, 0, .5), (7, 0, .78), (0, 1, .07), (8, 1, 1.)]
    mass = _enumerate(faults, 4, 1)
    # The affine support has just two independent coordinates in four detector columns.
    pred, pending = _lookup_all(dem, max_rank=2)
    live = mass.sum(axis=1) > 0
    np.testing.assert_array_equal(pending, ~live)
    np.testing.assert_array_equal(pred[live], mass[live].argmax(axis=1))


def test_joint_map_can_differ_from_individual_observable_votes():
    dem = stim.DetectorErrorModel("""
        error(0.39) D1 L0 L1
        error(0.35) D0 D1 L0
        error(0.25) L0
        error(0.58) D0 L0
        error(0.82) D0 D1 L0 L1
        error(0.06) D1 L1
    """)
    faults = [(2, 3, .39), (3, 1, .35), (0, 1, .25),
              (1, 1, .58), (3, 3, .82), (2, 2, .06)]
    mass = _enumerate(faults, 2, 2)
    pred, pending = _lookup_all(dem)
    assert not pending.any()
    np.testing.assert_array_equal(pred, mass.argmax(axis=1))
    # D0 alone favors logical word 3 even though marginal voting gives word 1.
    row = mass[1]
    assert pred[1] == 3 and row[1] + row[3] > row[0] + row[2]
    assert row[2] + row[3] < row[0] + row[1]


def test_unresolved_and_unattainable_syndromes_fall_back():
    dem = stim.DetectorErrorModel("error(1e-20) D0 L0\ndetector D1")
    pred, pending = _lookup_all(dem)
    assert pred[0] == 0
    np.testing.assert_array_equal(pending, [False, True, True, True])
