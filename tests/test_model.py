"""Model transforms: regrouping keeps the fault law, splitting forces logical classes."""

import numpy as np
import pymatching
import pytest
import stim

from src.circuits import detector_error_model, surface_code_circuit
from src.decoders.model import LogicalSplit, center, regroup, restrict


def _law(size, faults):
    mass = np.zeros(size)
    mass[0] = 1
    words = np.arange(size)
    for rate, mask in faults:
        mass = (1 - rate) * mass + rate * mass[words ^ mask]
    return mass


def _distribution(dem):
    """Joint law of detectors and observables, ignoring separators."""
    events = []
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        mask = 0
        for t in ins.targets_copy():
            if t.is_relative_detector_id():
                mask ^= 1 << t.val
            elif t.is_logical_observable_id():
                mask ^= 1 << (dem.num_detectors + t.val)
        events.append((ins.args_copy()[0], mask))
    return _law(1 << (dem.num_detectors + dem.num_observables), events)


def test_regroup_merges_same_sector_parts_and_keeps_the_joint_law():
    dem = stim.DetectorErrorModel("""
        error(.08) D0 D1
        error(.11) D2 D3
        error(.19) D0 L0 ^ D2 ^ D1 L0 ^ D3 L1
        error(.21) D0 ^ L0
    """)
    out = regroup(dem)
    np.testing.assert_allclose(_distribution(out), _distribution(dem), rtol=1e-14, atol=1e-16)
    groups = next(ins.target_groups() for ins in out if ins.type == "error" and ins.args_copy() == [.19])
    assert sorted(sum(t.is_relative_detector_id() for t in g) for g in groups) == [2, 2]
    # The logical-only part rides on the fault's detector part.
    last = [ins for ins in out if ins.type == "error"][-1]
    assert len(last.target_groups()) == 1


def test_regroup_scales_odds_per_component():
    dem = stim.DetectorErrorModel("error(.2) D0 D1\nerror(.2) D0 ^ D2 D3")
    (a, b) = [ins.args_copy()[0] for ins in regroup(dem, .5) if ins.type == "error"]
    odds = .2 / .8
    assert a == pytest.approx(odds * .5 / (1 + odds * .5))
    assert b == pytest.approx(odds * .25 / (1 + odds * .25))


def test_center_complements_likely_faults_and_keeps_the_affine_law():
    dem = stim.DetectorErrorModel("""
        error(.12) D0 D1
        error(.8) D0 L0
        error(1) D2 L1
    """)
    out, offset, frame = center(dem)
    assert all(ins.args_copy()[0] <= .5 for ins in out if ins.type == "error")
    np.testing.assert_array_equal(offset, [True, False, True])
    np.testing.assert_array_equal(frame, [True, True])
    shift = 1 | 4 | 8 | 16  # D0, D2, L0, L1
    np.testing.assert_allclose(_distribution(out)[np.arange(32) ^ shift], _distribution(dem),
                               rtol=1e-14, atol=1e-16)


def test_regroup_rejects_likely_faults():
    with pytest.raises(ValueError):
        regroup(stim.DetectorErrorModel("error(.7) D0 L0"))


@pytest.mark.parametrize("basis", ["z", "x"])
def test_split_forces_each_logical_class(basis):
    dem = regroup(detector_error_model(surface_code_circuit(5, .01, rounds=3, basis=basis)))
    split = LogicalSplit.build(dem)
    assert split is not None and not split.gauge.any()
    circuit = surface_code_circuit(5, .01, rounds=3, basis=basis)
    dets = circuit.compile_detector_sampler(seed=3).sample(300)
    forced = pymatching.Matching.from_detector_error_model(split.dem)
    shift = split.shift(dets)
    for c in (False, True):
        out = forced.decode_batch(split.force(dets, np.full(len(dets), c)))[:, 0].astype(bool)
        np.testing.assert_array_equal(out, c ^ shift)


def test_split_gauges_an_observable_on_bulk_edges():
    # A repetition-like chain whose observable sits on the middle bulk edge.
    dem = stim.DetectorErrorModel("""
        error(.1) D0
        error(.1) D0 D1
        error(.1) D1 D2 L0
        error(.1) D2
    """)
    split = LogicalSplit.build(dem)
    assert split is not None
    np.testing.assert_array_equal(split.gauge, [False, False, True])
    forced = pymatching.Matching.from_detector_error_model(split.dem)
    rows = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 0]], bool)
    for c in (False, True):
        out = forced.decode_batch(split.force(rows, np.full(len(rows), c)))[:, 0].astype(bool)
        np.testing.assert_array_equal(out, c ^ split.shift(rows))


def test_split_declines_unsupported_models():
    assert LogicalSplit.build(stim.DetectorErrorModel("error(.1) D0 L0\nerror(.1) D1 L1")) is None
    assert LogicalSplit.build(stim.DetectorErrorModel("error(.1) D0 L0\nerror(.1) L0")) is None


def test_restricted_plain_matching_matches_the_full_graph_on_its_sector():
    circuit = surface_code_circuit(5, .01)
    dem = regroup(detector_error_model(circuit))
    split = LogicalSplit.build(dem)
    full = pymatching.Matching.from_detector_error_model(dem)
    part = pymatching.Matching.from_detector_error_model(restrict(dem, split.sector))
    dets = circuit.compile_detector_sampler(seed=9).sample(500)
    np.testing.assert_array_equal(part.decode_batch(dets & split.sector), full.decode_batch(dets))
