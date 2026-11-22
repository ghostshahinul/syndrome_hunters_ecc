"""Milestone 5 (B): MWPM3D (our own exact matching) and the three-tier decoder."""

import numpy as np
import pymatching
import pytest

from src.circuits import NOISE_TIERS, tier_circuit
from src.decoders import DECODERS
from src.decoders.cube3d.graph import detector_graph
from src.decoders.cube3d.hierarchical import Hierarchical
from src.decoders.cube3d.mwpm import MWPM3D
from src.decoders.cube3d.three_tier import ThreeTier, ThreeTierConfig

TIER_P = {"code_capacity": 0.05, "phenomenological": 0.01, "circuit": 0.005}


def sample(tier, d, shots, seed):
    c = tier_circuit(tier, d, TIER_P[tier])
    dets, obs = c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return c, dets, obs


@pytest.mark.parametrize("tier,d", [(t, d) for t in NOISE_TIERS for d in (3, 5)])
def test_every_single_component(tier, d):
    g = detector_graph(tier, d, TIER_P[tier])
    edges = np.arange(g.n_edges)
    dets = np.zeros((len(edges), g.n_det), dtype=np.uint8)
    for i, e in enumerate(edges):
        dets[i, g.edge_u[e]] = 1
        if g.edge_v[e] != g.boundary:
            dets[i, g.edge_v[e]] = 1
    np.testing.assert_array_equal(MWPM3D(g).run(dets), g.edge_obs[edges])


@pytest.mark.parametrize("tier,d", [(t, d) for t in NOISE_TIERS for d in (3, 5)])
def test_matches_pymatching_reference(tier, d, report):
    """Both are exact MWPM on the same DEM graph; only degenerate ties may differ."""
    c, dets, obs = sample(tier, d, 400, seed=31 + d)
    g = detector_graph(tier, d, TIER_P[tier])
    ours = MWPM3D(g).run(dets)
    pm = pymatching.Matching.from_detector_error_model(
        c.detector_error_model(decompose_errors=True)).decode_batch(dets).astype(bool)
    disagree = int((ours != pm).any(axis=1).sum())
    report.add("mwpm3d_disagree_with_pymatching", tier, f"d={d}", disagree, ylabel="shots (of 400)")
    assert disagree <= 4
    assert int((ours != obs).any(axis=1).sum()) <= int((pm != obs).any(axis=1).sum()) + 2


def test_empty_shot_and_bad_width():
    g = detector_graph("circuit", 3, 0.005)
    m = MWPM3D(g)
    assert not m.run(np.zeros((2, g.n_det), np.uint8)).any()
    with pytest.raises(ValueError):
        m.run(np.zeros((1, g.n_det + 1), np.uint8))


@pytest.mark.parametrize("tier", NOISE_TIERS)
def test_three_tier_routing(tier):
    d = 5
    g = detector_graph(tier, d, TIER_P[tier])
    _, dets, _ = sample(tier, d, 600, seed=9)
    tt = ThreeTier(g)
    h = Hierarchical(g).run(dets)
    # threshold above every cluster: exactly Hierarchical
    never = tt.run(dets, threshold=10**6)
    assert not never.routed.any()
    np.testing.assert_array_equal(never.predictions, h.predictions)
    # threshold 0: every shot UF touched goes to MWPM; fast-path shots keep their answer
    always = tt.run(dets, threshold=0)
    np.testing.assert_array_equal(always.routed, h.max_cluster > 0)
    np.testing.assert_array_equal(always.routed, h.uf_called)
    np.testing.assert_array_equal(always.predictions[always.routed], tt.mwpm.run(dets[always.routed]))
    np.testing.assert_array_equal(always.predictions[~always.routed], h.predictions[~always.routed])


def test_default_threshold_is_two_rounds():
    for d in (3, 5, 7):
        assert ThreeTier(detector_graph("circuit", d, 0.005)).threshold == 2 * d
    assert ThreeTier(detector_graph("circuit", 5, 0.005), ThreeTierConfig(threshold=3)).threshold == 3
    with pytest.raises(ValueError):
        ThreeTierConfig(threshold=-1)


def test_registered():
    assert "mwpm_3d" in DECODERS and "hierarchical_mwpm" in DECODERS
