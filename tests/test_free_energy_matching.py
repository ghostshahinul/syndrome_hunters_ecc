"""Public behaviour of the free-energy matching decoder."""

import numpy as np
import pymatching
import pytest
import stim

from src.circuits import detector_error_model, surface_code_circuit
from src.decoders.free_energy_matching import FreeEnergyMatchingDecoder
from test_known_cases import single_fault_cases


@pytest.mark.parametrize("distance,basis", [(4, "z"), (5, "x"), (5, "z")])
def test_single_faults_with_nondefault_rounds(distance, basis):
    dem = detector_error_model(surface_code_circuit(distance, .002, rounds=3, basis=basis))
    hits, truth = map(np.asarray, zip(*single_fault_cases(dem)))
    np.testing.assert_array_equal(FreeEnergyMatchingDecoder(dem).decode_batch(hits), truth)


def test_vote_runs_and_only_changes_hard_shots():
    circuit = surface_code_circuit(5, .01)
    dem = detector_error_model(circuit)
    dets = circuit.compile_detector_sampler(seed=11).sample(2000)
    dec = FreeEnergyMatchingDecoder(dem)
    assert dec.split is not None
    base = dec.matching.decode_batch(dets, enable_correlations=True).astype(bool)
    hard = dec.hard_shots(dets, base[:, 0])
    assert 0 < len(hard) < len(dets)
    out = dec.decode_batch(dets)
    easy = np.setdiff1d(np.arange(len(dets)), hard)
    np.testing.assert_array_equal(out[easy], base[easy])


def test_streaming_single_call_matches_batch_on_strided_input():
    circuit = surface_code_circuit(5, .025)
    hits = circuit.compile_detector_sampler(seed=5754).sample(300)
    dec = FreeEnergyMatchingDecoder(detector_error_model(circuit))
    wide = np.zeros((len(hits), hits.shape[1] * 2), bool)
    wide[:, ::2] = hits
    np.testing.assert_array_equal([dec.decode(row[::2]) for row in wide], dec.decode_batch(hits))


def test_unsupported_models_fall_back_to_correlated_matching():
    dem = stim.DetectorErrorModel("""
        error(.01) D0 D1 L0
        error(.01) D2 D3 L1
        error(.01) D4 D5 L0
        error(.01) D6 D7
    """)
    dec = FreeEnergyMatchingDecoder(dem)
    assert dec.split is None
    np.testing.assert_array_equal(dec.decode(np.ones(8, bool)), [False, True])
    # A fault above probability 1/2 is the reference: D0 firing means it happened.
    likely = stim.DetectorErrorModel("error(.9) D0 L0\nerror(.01) D1 L0")
    rows = np.array([[1, 0], [0, 0], [1, 1]], bool)
    np.testing.assert_array_equal(FreeEnergyMatchingDecoder(likely).decode_batch(rows), [[1], [0], [0]])


def test_model_without_observables():
    out = FreeEnergyMatchingDecoder(stim.DetectorErrorModel("error(.1) D0")).decode_batch([[True], [False]])
    assert out.shape == (2, 0) and out.dtype == np.bool_


@pytest.mark.parametrize("distance", [3, 5])
def test_fewer_failures_than_correlated_matching(distance):
    circuit = surface_code_circuit(distance, .007)
    dem = detector_error_model(circuit)
    dets, obs = circuit.compile_detector_sampler(seed=2026).sample(20_000, separate_observables=True)
    ours = (FreeEnergyMatchingDecoder(dem).decode_batch(dets) != obs).any(axis=1)
    corr = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True)
    theirs = (corr.decode_batch(dets, enable_correlations=True) != obs).any(axis=1)
    # Paired comparison on identical shots.
    assert (ours & ~theirs).sum() < (theirs & ~ours).sum()
