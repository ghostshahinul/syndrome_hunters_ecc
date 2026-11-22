"""Item 2: ensemble UF (M variants per shot, cheapest correction under the original weights)."""

import numpy as np
import pytest

from src.circuits import NOISE_TIERS, tier_circuit
from src.decoders import DECODERS
from src.decoders.cube3d.ensemble import EnsembleConfig, UFEnsemble
from src.decoders.cube3d.graph import detector_graph
from src.decoders.cube3d.union_find import UnionFind3D, UnionFind3DConfig

TIER_P = {"code_capacity": 0.05, "phenomenological": 0.01, "circuit": 0.005}


def sample(tier, d, shots, seed):
    c = tier_circuit(tier, d, TIER_P[tier])
    return c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)


def dets_for_edges(g, edges):
    dets = np.zeros((len(edges), g.n_det), dtype=np.uint8)
    for i, e in enumerate(edges):
        dets[i, g.edge_u[e]] = 1
        if g.edge_v[e] != g.boundary:
            dets[i, g.edge_v[e]] = 1
    return dets


@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("tier,d", [(t, d) for t in NOISE_TIERS for d in (3, 5)])
def test_numba_both_modes_bit_identical_to_reference(tier, d, correlated):
    g = detector_graph(tier, d, TIER_P[tier])
    ens = UFEnsemble(g, EnsembleConfig(M=4, correlated=correlated))
    dets, _ = sample(tier, d, 150, seed=40 + d)
    ref = ens.run(dets, record_correction=True, backend="reference")
    for mode in ("shots", "variants"):
        fast = ens.run(dets, record_correction=True, backend="numba", parallel=mode)
        for f in ("predictions", "ok", "chosen", "correction"):
            np.testing.assert_array_equal(getattr(fast, f), getattr(ref, f), err_msg=f"{mode} {f}")


@pytest.mark.parametrize("correlated", [False, True])
def test_single_variant_equals_uf(correlated):
    g = detector_graph("circuit", 5, 0.005)
    dets, _ = sample("circuit", 5, 3000, seed=3)
    ens = UFEnsemble(g, EnsembleConfig(M=1, correlated=correlated)).run(dets)
    uf = UnionFind3D(g, UnionFind3DConfig(correlated=correlated)).run(dets)
    np.testing.assert_array_equal(ens.predictions, uf.predictions)
    assert (ens.chosen == 0).all()


@pytest.mark.parametrize("tier,d", [(t, d) for t in NOISE_TIERS for d in (3, 5, 7)])
def test_every_single_component(tier, d):
    g = detector_graph(tier, d, TIER_P[tier])
    edges = np.arange(g.n_edges)
    res = UFEnsemble(g).run(dets_for_edges(g, edges))
    np.testing.assert_array_equal(res.predictions, g.edge_obs[edges])


@pytest.mark.parametrize("tier,d", [(t, d) for t in NOISE_TIERS for d in (3, 5)])
def test_syndrome_consistency(tier, d):
    g = detector_graph(tier, d, TIER_P[tier])
    dets, _ = sample(tier, d, 1000, seed=60 + d)
    res = UFEnsemble(g, EnsembleConfig(M=8)).run(dets, record_correction=True)
    assert res.ok.all()
    np.testing.assert_array_equal(g.syndrome(res.correction), dets)


def test_more_variants_never_raise_the_score():
    """The winner's original-weight score can only drop as M grows (variant 0 is always there)."""
    from src.decoders.cube3d.union_find import correlated_score
    g = detector_graph("circuit", 5, 0.005)
    dets, _ = sample("circuit", 5, 400, seed=5)
    scores = []
    for M in (1, 4, 8):
        ens = UFEnsemble(g, EnsembleConfig(M=M))
        res = ens.run(dets, record_correction=True)
        scores.append(np.array([correlated_score(ens.tables, np.flatnonzero(c)) for c in res.correction]))
    assert (scores[1] <= scores[0]).all() and (scores[2] <= scores[1]).all()


def test_config_validation_and_registry():
    for bad in (dict(M=0), dict(sigma=-1), dict(parallel="both"), dict(backend="c")):
        with pytest.raises(ValueError):
            EnsembleConfig(**bad)
    assert "uf3d_ensemble" in DECODERS
