"""Public FaultJoin entry points on known syndromes and circuit samples."""

import numpy as np
import pytest
import stim

from src.circuits import detector_error_model, surface_code_circuit
from src.decoders.experimental import FaultJoinDecoder
from src.decoders.union_find import UnionFindDecoder
from test_known_cases import single_fault_cases


def test_disconnected_faults_combine_their_logical_frames():
    dem = stim.DetectorErrorModel("""
        error(.01) D0 D1 L0
        error(.01) D2 D3 L1
        error(.01) D4 D5 L0
        error(.01) D6 D7
    """)
    np.testing.assert_array_equal(FaultJoinDecoder(dem).decode(np.ones(8, bool)), [False, True])


@pytest.mark.parametrize("cls", [FaultJoinDecoder, UnionFindDecoder])
def test_more_likely_than_not_fault_is_centered(cls):
    dem = stim.DetectorErrorModel("error(1) D0 L0\nerror(.01) D1 L1")
    np.testing.assert_array_equal(cls(dem).decode_batch([[1, 0], [1, 1]]), [[1, 0], [1, 1]])


@pytest.mark.parametrize("distance,basis", [(4, "z"), (5, "x"), (5, "z")])
def test_single_faults_with_nondefault_rounds(distance, basis):
    dem = detector_error_model(surface_code_circuit(distance, .002, rounds=3, basis=basis))
    cases = single_fault_cases(dem)
    hits, truth = map(np.asarray, zip(*cases))
    np.testing.assert_array_equal(FaultJoinDecoder(dem).decode_batch(hits), truth)


def test_streaming_single_call_matches_batch_on_dense_and_strided_input():
    circuit = surface_code_circuit(5, .025)
    hits = circuit.compile_detector_sampler(seed=5754).sample(300)
    dec = FaultJoinDecoder(detector_error_model(circuit))
    wide = np.zeros((len(hits), hits.shape[1] * 2), bool)
    wide[:, ::2] = hits
    np.testing.assert_array_equal([dec.decode(row[::2]) for row in wide], dec.decode_batch(hits))


def test_single_lookup_returns_an_independent_array(small_dem):
    dec = FaultJoinDecoder(small_dem)
    syndrome = np.zeros(dec.num_detectors, bool)
    first = dec.decode(syndrome)
    first[0] = True
    assert not dec.decode(syndrome)[0]


def test_detector_model_without_logical_observables():
    decoder = FaultJoinDecoder(stim.DetectorErrorModel("error(.1) D0"))
    predictions = decoder.decode_batch([[True], [False]])
    assert predictions.shape == (2, 0)
    assert predictions.dtype == np.bool_
