"""Hand-checkable syndromes on the d=3 code.

A distance-3 code corrects any single fault, so for every individual error mechanism
in the DEM, decoding its syndrome must recover exactly its observable flip.
"""

import numpy as np


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


def test_every_single_fault_is_corrected(decoder, small_dem, report):
    cases = single_fault_cases(small_dem)
    assert cases, "DEM has no error mechanisms?"
    syndromes = np.array([c[0] for c in cases])
    expected = np.array([c[1] for c in cases])
    predicted = decoder.decode_batch(syndromes)
    wrong = np.flatnonzero(np.any(predicted != expected, axis=1))
    report.add("single_fault_correction", decoder.name, f"d=3 ({len(cases)} faults)",
               1 - wrong.size / len(cases), ylabel="fraction of single faults corrected")
    assert wrong.size == 0, f"{wrong.size}/{len(cases)} single faults mis-decoded, e.g. #{wrong[:5]}"
