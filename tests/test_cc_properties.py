"""Cube-Collapse end-to-end properties: every syndrome resolved, exhaustive low weights."""

import itertools

import numpy as np
import pytest

from src.circuits import NOISE_TIERS, tier_circuit
from src.decoders.cube_collapse import CubeCollapseConfig, CubeCollapseDecoder
from src.decoders.cube_collapse.baselines import decode_2d, pymatching_2d

from cc_helpers import cc_decoder

TIER_P = {"code_capacity": 0.08, "phenomenological": 0.02, "circuit": 0.008}


@pytest.mark.parametrize("include_final_layer", [True, False])
@pytest.mark.parametrize("d", [3, 5, 7])
@pytest.mark.parametrize("tier", NOISE_TIERS)
def test_correction_matches_flat_every_shot(tier, d, include_final_layer):
    circuit = tier_circuit(tier, d, TIER_P[tier])
    dec = CubeCollapseDecoder(circuit.detector_error_model(decompose_errors=True),
                              CubeCollapseConfig(include_final_layer=include_final_layer))
    dets, _ = circuit.compile_detector_sampler(seed=d).sample(2000, separate_observables=True)
    r = dec.decode_full(dets)
    np.testing.assert_array_equal(dec.graph.syndrome(r.correction), r.flat)
    total = sum(r.cleared[st] for st in ("s1", "s2", "s3", "s4"))
    np.testing.assert_array_equal(total, r.flat.sum(axis=1))
    assert r.filter.filtered.shape[1] == (d + 1 if include_final_layer else d) or tier == "code_capacity"


def test_decode_cube_accepts_any_T():
    dec = cc_decoder(5)
    rng = np.random.default_rng(1)
    for T in (1, 2, 5, 11):
        raw = rng.integers(0, 2, size=(200, T, dec.graph.n_nodes), dtype=np.uint8)
        r = dec.decode_cube(raw)
        np.testing.assert_array_equal(r.flat, raw[:, -1, :])
        np.testing.assert_array_equal(dec.graph.syndrome(r.correction), r.flat)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_every_single_edge_error(d):
    dec = cc_decoder(d)
    g = dec.graph
    corr = np.eye(g.n_edges, dtype=np.uint8)
    r = dec.decode_flat(g.syndrome(corr))
    np.testing.assert_array_equal(r.predictions, g.observables(corr))


def test_all_weight2_errors_d5(report):
    """Reported, not asserted for Cube-Collapse; PyMatching 2D must get all of them."""
    dec = cc_decoder(5)
    g = dec.graph
    pairs = list(itertools.combinations(range(g.n_edges), 2))
    corr = np.zeros((len(pairs), g.n_edges), dtype=np.uint8)
    for i, (a, b) in enumerate(pairs):
        corr[i, [a, b]] = 1
    flat, truth = g.syndrome(corr), g.observables(corr)
    cube_wrong = int((dec.decode_flat(flat).predictions != truth).any(axis=1).sum())
    pm_wrong = int((decode_2d(pymatching_2d(g), flat, g.n_obs) != truth).any(axis=1).sum())
    for name, wrong in (("cube_collapse", cube_wrong), ("pymatching_2d", pm_wrong)):
        report.add("weight2_errors_d5", name, "wrong", wrong,
                   ylabel=f"mis-decoded of {len(pairs)} weight-2 errors (d=5)")
    print(f"\nweight-2 d=5: {len(pairs)} cases, cube_collapse wrong={cube_wrong}, pymatching_2d wrong={pm_wrong}")
    assert pm_wrong == 0
