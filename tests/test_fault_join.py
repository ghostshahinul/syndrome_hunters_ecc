"""Independent oracles for the logical vote and conditional edge probabilities."""

from collections import defaultdict
from itertools import combinations, product
import math

import numpy as np
import pytest
import stim

from src.circuits import detector_error_model, surface_code_circuit
from src.decoders.faultjoin.condition import conditioned_graph
from src.decoders.faultjoin.decoder import FaultJoinDecoder
from src.decoders.faultjoin.faults import edge_arrays, read_model
from src.decoders.faultjoin.union_find import UnionFindDecoder, growth_unit
from test_known_cases import single_fault_cases


@pytest.mark.parametrize("width", [5, 72])
def test_two_fault_decisions_match_direct_enumeration(width):
    # Two independent ways to produce D0 without L0 jointly outweigh the
    # single most likely explanation for D0. Include a second observable.
    end = width - 1
    errors = [(.004, (0,), 1), (.05, (1,), 0), (.05, (0, 1), 0),
              (.05, (2,), 0), (.05, (0, 2), 0), (.018, (1, 2, 3, end), 2)]
    dem = stim.DetectorErrorModel(f"""
        error(.004) D0 L0
        error(.05) D1
        error(.05) D0 D1
        error(.05) D2
        error(.05) D0 D2
        error(.018) D1 D2 ^ D3 D{end} L1
    """)
    mass = defaultdict(lambda: defaultdict(float))
    for weight in range(3):
        for chosen in combinations(range(len(errors)), weight):
            hits, frame, probability = set(), 0, 1.0
            for j, (p, dets, logical) in enumerate(errors):
                if j in chosen:
                    hits.symmetric_difference_update(dets)
                    frame ^= logical
                    probability *= p
                else:
                    probability *= 1 - p
            mass[tuple(sorted(hits))][frame] += probability
    inputs = np.zeros((len(mass), width), bool)
    expected = []
    for row, (hits, votes) in enumerate(mass.items()):
        inputs[row, list(hits)] = True
        winner = max(votes, key=votes.get)
        expected.append([bool(winner & 1), bool(winner & 2)])
    assert mass[(0,)][0] > mass[(0,)][1]
    np.testing.assert_array_equal(FaultJoinDecoder(dem).decode_batch(inputs), expected)


def test_partner_cost_matches_exhaustive_conditional_probability():
    dem = stim.DetectorErrorModel("""
        error(.01) D0 D1 L0 ^ D2 D3
        error(.02) D0 D1 L0
        error(.03) D2 D3
        error(.11) D2
        error(.13) D3
    """)
    total = selected = 0.0
    for state in product((0, 1), repeat=5):
        shared, target, partner, left, right = state
        if (shared ^ partner ^ left) and (shared ^ partner ^ right):
            p = math.prod(q if on else 1 - q for on, q in zip(state, (.01, .02, .03, .11, .13)))
            total += p
            if shared ^ target:
                selected += p
    q = selected / total
    model = read_model(dem)
    unit = growth_unit(edge_arrays(model, 4)[2])
    keep, _, (starts, other, edge, ticks) = conditioned_graph(model, 4)
    np.testing.assert_array_equal(keep, [0, 1])
    idx = starts[2]
    assert (other[idx], edge[idx], ticks[idx]) == (3, 0, max(1, round(math.log((1 - q) / q) / unit)))


def test_dense_syndrome_requires_global_completion():
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
