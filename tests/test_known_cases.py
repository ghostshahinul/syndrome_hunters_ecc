"""Hand-checkable syndromes on the d=3 code.

Our decoders must recover the observable flip for each individual d=3 DEM
mechanism. The optional correlated library reference follows its upstream result.
"""

import numpy as np
import pymatching
import pytest

from src.decoders import DECODERS


def single_fault_cases(dem):
    """(syndrome, expected_observables) for each error mechanism in the DEM."""
    cases = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        syn = np.zeros(dem.num_detectors, dtype=bool)
        obs = np.zeros(dem.num_observables, dtype=bool)
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                syn[t.val] ^= True
            elif t.is_logical_observable_id():
                obs[t.val] ^= True
        cases.append((syn, obs))
    return cases


def test_no_detections_means_no_flip(decoder):
    out = decoder.decode(np.zeros(decoder.num_detectors, dtype=bool))
    assert not out.any()


@pytest.mark.parametrize("name", ["fault_join", "union_find", "pymatching"])
def test_every_single_fault_is_corrected(name, small_dem):
    decoder = DECODERS[name](small_dem)
    cases = single_fault_cases(small_dem)
    assert cases, "DEM has no error mechanisms?"
    syndromes = np.array([c[0] for c in cases])
    expected = np.array([c[1] for c in cases])
    predicted = decoder.decode_batch(syndromes)
    wrong = np.flatnonzero(np.any(predicted != expected, axis=1))
    assert wrong.size == 0, f"{wrong.size}/{len(cases)} single faults mis-decoded, e.g. #{wrong[:5]}"


def test_correlated_reference_matches_upstream_on_all_single_faults(small_dem):
    # The optional library comparator is a two-pass heuristic. At d=3 it can
    # reweight a correct first pass into a logical error on a single fault.
    hits = np.array([s for s, _ in single_fault_cases(small_dem)])
    reference = pymatching.Matching.from_detector_error_model(small_dem, enable_correlations=True)
    expected = reference.decode_batch(hits, enable_correlations=True)
    np.testing.assert_array_equal(DECODERS["pymatching_correlated"](small_dem).decode_batch(hits), expected)
