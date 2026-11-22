"""Step 3: UnionFind3D (reference + numba), on the shared 3D detector graph."""

import numpy as np
import pytest

from src.circuits import NOISE_TIERS, tier_circuit
from src.decoders.cube3d.graph import detector_graph
from src.decoders.cube3d.union_find import UnionFind3D, UnionFind3DConfig

TIER_P = {"code_capacity": 0.05, "phenomenological": 0.01, "circuit": 0.005}
CASES = [(tier, d) for tier in NOISE_TIERS for d in (3, 5, 7)]


def engine(tier, d):
    g = detector_graph(tier, d, TIER_P[tier])
    return g, UnionFind3D(g)


def dets_for_edges(g, edges):
    dets = np.zeros((len(edges), g.n_det), dtype=np.uint8)
    for i, e in enumerate(edges):
        dets[i, g.edge_u[e]] = 1
        if g.edge_v[e] != g.boundary:
            dets[i, g.edge_v[e]] = 1
    return dets


def sample(tier, d, shots, seed):
    c = tier_circuit(tier, d, TIER_P[tier])
    return c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)


@pytest.mark.parametrize("tier,d", CASES)
def test_every_single_component(tier, d):
    g, uf = engine(tier, d)
    edges = np.arange(g.n_edges)
    res = uf.run(dets_for_edges(g, edges), record_correction=True)
    assert res.ok.all()
    np.testing.assert_array_equal(res.predictions, g.edge_obs[edges])
    np.testing.assert_array_equal(g.syndrome(res.correction), dets_for_edges(g, edges).astype(bool))


@pytest.mark.parametrize("tier,d", CASES)
def test_every_full_dem_instruction(tier, d, report):
    """Whole DEM error instructions (all components of one fault) as single shots."""
    g, uf = engine(tier, d)
    dem = tier_circuit(tier, d, TIER_P[tier]).detector_error_model(decompose_errors=True)
    rows, truth = [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        r = np.zeros(g.n_det, dtype=np.uint8)
        o = np.zeros(g.n_obs, dtype=bool)
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                r[t.val] ^= 1
            elif t.is_logical_observable_id():
                o[t.val] ^= True
        rows.append(r)
        truth.append(o)
    res = uf.run(np.array(rows))
    wrong = int((res.predictions != np.array(truth)).any(axis=1).sum())
    report.add("uf3d_single_fault_wrong", tier, f"d={d}", wrong, ylabel="DEM instructions mis-decoded")
    assert res.ok.all() and wrong == 0


@pytest.mark.parametrize("tier,d", CASES)
def test_syndrome_consistency_random_shots(tier, d):
    g, uf = engine(tier, d)
    dets, _ = sample(tier, d, 2000, seed=d)
    res = uf.run(dets, record_correction=True)
    assert res.ok.all()
    np.testing.assert_array_equal(g.syndrome(res.correction), dets)


@pytest.mark.parametrize("tier,d", CASES)
def test_numba_bit_identical_to_reference(tier, d):
    g, uf = engine(tier, d)
    shots = 400 if d < 7 else 150
    dets, _ = sample(tier, d, shots, seed=100 + d)
    ref = uf.run(dets, record_correction=True, backend="reference")
    fast = uf.run(dets, record_correction=True, backend="numba")
    np.testing.assert_array_equal(fast.predictions, ref.predictions)
    np.testing.assert_array_equal(fast.correction, ref.correction)
    np.testing.assert_array_equal(fast.ok, ref.ok)


def test_lengths_are_small_integers():
    for d in (3, 5, 7):
        uf = UnionFind3D(detector_graph("circuit", d, 0.005), UnionFind3DConfig(scale=2))
        L = uf.tables.length
        assert L.min() == 2 and L.max() <= 8


@pytest.mark.parametrize("scale", [2, 4, 8])
def test_scale_sets_minimum_length(scale):
    uf = UnionFind3D(detector_graph("circuit", 5, 0.005), UnionFind3DConfig(scale=scale))
    assert uf.tables.length.min() == scale and uf.tables.length.max() <= 4 * scale


VARIANTS = [(S, gr) for gr in ("all", "smallest") for S in (2, 4, 8)]


@pytest.mark.parametrize("scale,growth", VARIANTS)
@pytest.mark.parametrize("tier", NOISE_TIERS)
def test_variants_bit_identical_and_consistent(tier, scale, growth):
    """Every (scale, growth) variant: numba == reference, and the correction reproduces
    the syndrome; single DEM components decode exactly."""
    d = 5
    g = detector_graph(tier, d, TIER_P[tier])
    uf = UnionFind3D(g, UnionFind3DConfig(scale=scale, growth=growth))
    dets, _ = sample(tier, d, 200, seed=7)
    ref = uf.run(dets, record_correction=True, backend="reference")
    fast = uf.run(dets, record_correction=True, backend="numba")
    np.testing.assert_array_equal(fast.predictions, ref.predictions)
    np.testing.assert_array_equal(fast.correction, ref.correction)
    assert fast.ok.all()
    np.testing.assert_array_equal(g.syndrome(fast.correction), dets)
    edges = np.arange(g.n_edges)
    single = uf.run(dets_for_edges(g, edges))
    np.testing.assert_array_equal(single.predictions, g.edge_obs[edges])


def test_config_validation():
    with pytest.raises(ValueError):
        UnionFind3DConfig(growth="largest")
    with pytest.raises(ValueError):
        UnionFind3DConfig(scale=0)
