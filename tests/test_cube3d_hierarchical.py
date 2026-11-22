"""Milestone 2: Cube3D-Local numba backend and the Hierarchical (fast + deep) decoder."""

import numpy as np
import pytest

from src.circuits import NOISE_TIERS, tier_circuit
from src.decoders import DECODERS
from src.decoders.cube3d.graph import detector_graph
from src.decoders.cube3d.hierarchical import GATES, Hierarchical, HierarchicalConfig
from src.decoders.cube3d.local import RULES, Cube3DLocal, Cube3DLocalConfig
from src.decoders.cube3d.union_find import UnionFind3D, UnionFind3DConfig

TIER_P = {"code_capacity": 0.05, "phenomenological": 0.01, "circuit": 0.005}


def sample(tier, d, shots, seed, p=None):
    c = tier_circuit(tier, d, p or TIER_P[tier])
    return c.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)


def dets_for_edges(g, edges):
    dets = np.zeros((len(edges), g.n_det), dtype=np.uint8)
    for i, e in enumerate(edges):
        dets[i, g.edge_u[e]] = 1
        if g.edge_v[e] != g.boundary:
            dets[i, g.edge_v[e]] = 1
    return dets


# ---- Cube3D-Local: numba == reference ----------------------------------------------------
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("K", [0, 2, 6, 10, 10_000])
@pytest.mark.parametrize("tier,d", [(t, d) for t in NOISE_TIERS for d in (3, 5, 7)])
def test_local_numba_bit_identical(tier, d, K, strict):
    g = detector_graph(tier, d, TIER_P[tier])
    loc = Cube3DLocal(g, Cube3DLocalConfig(strict_r1=strict))
    dets, _ = sample(tier, d, 1500, seed=11 + d)
    ref = loc.run(dets, K=K, record_residual=True, backend="reference")
    fast = loc.run(dets, K=K, record_residual=True, backend="numba")
    np.testing.assert_array_equal(fast.accepted, ref.accepted)
    np.testing.assert_array_equal(fast.predictions, ref.predictions)
    np.testing.assert_array_equal(fast.obs_bits, ref.obs_bits)
    np.testing.assert_array_equal(fast.residual, ref.residual)
    np.testing.assert_array_equal(fast.lit_count, ref.lit_count)
    for r in RULES:
        np.testing.assert_array_equal(fast.cleared[r], ref.cleared[r])


def test_local_numba_rejects_record_flips():
    g = detector_graph("circuit", 3, 0.005)
    with pytest.raises(ValueError):
        Cube3DLocal(g).run(np.zeros((1, g.n_det), bool), record_flips=True, backend="numba")


# ---- Hierarchical ------------------------------------------------------------------------
@pytest.mark.parametrize("correlated", [True, False])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("tier,d", [(t, d) for t in NOISE_TIERS for d in (3, 5)])
def test_hierarchical_numba_bit_identical(tier, d, gate, strict, correlated):
    g = detector_graph(tier, d, TIER_P[tier])
    h = Hierarchical(g, HierarchicalConfig(gate=gate, strict_r1=strict,
                                           uf=UnionFind3DConfig(correlated=correlated)))
    dets, _ = sample(tier, d, 300, seed=21 + d)
    ref = h.run(dets, backend="reference")
    fast = h.run(dets, backend="numba")
    for f in ("predictions", "fast", "uf_called", "lit_count", "cleared", "ok", "max_cluster"):
        np.testing.assert_array_equal(getattr(fast, f), getattr(ref, f), err_msg=f)


@pytest.mark.parametrize("gate", GATES)
def test_hierarchical_numba_bit_identical_d7(gate):
    g = detector_graph("circuit", 7, 0.005)
    h = Hierarchical(g, HierarchicalConfig(gate=gate))
    dets, _ = sample("circuit", 7, 80, seed=5)
    ref, fast = h.run(dets, backend="reference"), h.run(dets, backend="numba")
    np.testing.assert_array_equal(fast.predictions, ref.predictions)
    np.testing.assert_array_equal(fast.fast, ref.fast)


@pytest.mark.parametrize("tier", NOISE_TIERS)
def test_whole_shot_gate_routes_rejected_shots_to_uf_unchanged(tier):
    d = 5
    g = detector_graph(tier, d, TIER_P[tier])
    h = Hierarchical(g, HierarchicalConfig(gate="whole_shot", K=6))
    dets, _ = sample(tier, d, 3000, seed=3)
    res = h.run(dets)
    uf = UnionFind3D(g, h.config.uf).run(dets)
    loc = Cube3DLocal(g).run(dets, K=6, backend="numba")
    np.testing.assert_array_equal(res.fast, loc.accepted)
    np.testing.assert_array_equal(res.predictions[~res.fast], uf.predictions[~res.fast])
    np.testing.assert_array_equal(res.predictions[res.fast], loc.predictions[res.fast])
    assert not res.uf_called[res.fast].any() and res.uf_called[~res.fast].all()


@pytest.mark.parametrize("tier", NOISE_TIERS)
def test_partial_gate_is_local_xor_uf_on_residual(tier):
    d = 5
    g = detector_graph(tier, d, TIER_P[tier])
    h = Hierarchical(g, HierarchicalConfig(gate="partial"))
    dets, _ = sample(tier, d, 3000, seed=4)
    res = h.run(dets)
    loc = Cube3DLocal(g).run(dets, K=g.n_det, record_residual=True, backend="numba")
    uf = UnionFind3D(g, h.config.uf).run(loc.residual)
    from src.decoders.cube3d.union_find import bits_to_bool
    expect = bits_to_bool(loc.obs_bits, g.n_obs) ^ uf.predictions
    np.testing.assert_array_equal(res.predictions, expect)
    assert res.ok.all()


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("tier", NOISE_TIERS)
def test_K0_equals_uf_alone(tier, gate):
    g = detector_graph(tier, 5, TIER_P[tier])
    dets, _ = sample(tier, 5, 2000, seed=6)
    h = Hierarchical(g, HierarchicalConfig(gate=gate, K=0))
    res = h.run(dets)
    np.testing.assert_array_equal(res.predictions, UnionFind3D(g, h.config.uf).run(dets).predictions)
    assert res.cleared.sum() == 0


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("tier,d", [(t, d) for t in NOISE_TIERS for d in (3, 5, 7)])
def test_every_single_component(tier, d, gate):
    g = detector_graph(tier, d, TIER_P[tier])
    edges = np.arange(g.n_edges)
    res = Hierarchical(g, HierarchicalConfig(gate=gate)).run(dets_for_edges(g, edges))
    np.testing.assert_array_equal(res.predictions, g.edge_obs[edges])


@pytest.mark.parametrize("gate", GATES)
def test_fractions_and_counts_are_consistent(gate):
    g = detector_graph("circuit", 5, 0.002)
    dets, _ = sample("circuit", 5, 4000, seed=8, p=0.002)
    res = Hierarchical(g, HierarchicalConfig(gate=gate)).run(dets)
    assert np.array_equal(res.fast, ~res.uf_called)
    assert (res.cleared <= res.lit_count).all()
    by_rules = res.fast & ~res.table  # fast path proper: the local rules cleared every event
    assert (res.cleared[by_rules] == res.lit_count[by_rules]).all()
    assert (res.cleared[res.table] == 0).all() and (res.lit_count[res.table] <= 4).all()
    assert 0.0 < res.fast_fraction <= 1.0
    assert 0.0 <= res.cleared_fraction <= 1.0


def test_config_validation():
    with pytest.raises(ValueError):
        HierarchicalConfig(gate="sometimes")
    with pytest.raises(ValueError):
        HierarchicalConfig(K=-1)


def test_registered():
    assert "hierarchical" in DECODERS and "hierarchical_partial" in DECODERS


# ---- Phase C: single-fault table in Hierarchical -----------------------------------------
@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("tier,d", [(t, d) for t in NOISE_TIERS for d in (3, 5, 7)])
def test_table_every_single_dem_fault_exact(tier, d, gate):
    dem = tier_circuit(tier, d, TIER_P[tier]).detector_error_model(decompose_errors=True)
    rows, truth = [], []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        r = np.zeros(dem.num_detectors, dtype=np.uint8)
        o = np.zeros(dem.num_observables, dtype=bool)
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                r[t.val] ^= 1
            elif t.is_logical_observable_id():
                o[t.val] ^= True
        rows.append(r)
        truth.append(o)
    g = detector_graph(tier, d, TIER_P[tier])
    res = Hierarchical(g, HierarchicalConfig(gate=gate)).run(np.array(rows))
    np.testing.assert_array_equal(res.predictions, np.array(truth))
    assert res.table.all()


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("tier,d", [(t, d) for t in NOISE_TIERS for d in (3, 5)])
def test_table_numba_bit_identical(tier, d, gate):
    g = detector_graph(tier, d, TIER_P[tier])
    h = Hierarchical(g, HierarchicalConfig(gate=gate))
    dets, _ = sample(tier, d, 300, seed=500 + d)
    ref, fast = h.run(dets, backend="reference"), h.run(dets, backend="numba")
    for f in ("predictions", "fast", "uf_called", "lit_count", "cleared", "ok", "max_cluster", "table"):
        np.testing.assert_array_equal(getattr(fast, f), getattr(ref, f), err_msg=f)


@pytest.mark.parametrize("opts", [dict(float_score=True), dict(weighted_forest=True), dict(passes=3)],
                         ids=lambda o: "+".join(o))
@pytest.mark.parametrize("gate", GATES)
def test_deep_tier_options_numba_equals_reference(gate, opts):
    g = detector_graph("circuit", 5, 0.005)
    h = Hierarchical(g, HierarchicalConfig(gate=gate, uf=UnionFind3DConfig(correlated=True, **opts)))
    dets, _ = sample("circuit", 5, 200, seed=77)
    ref, fast = h.run(dets, backend="reference"), h.run(dets, backend="numba")
    for f in ("predictions", "fast", "uf_called", "ok", "max_cluster", "table"):
        np.testing.assert_array_equal(getattr(fast, f), getattr(ref, f), err_msg=f)



@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("tier,d", [(t, d) for t in NOISE_TIERS for d in (3, 5, 7)])
def test_pass2_skip_is_exact_in_hierarchical(tier, d, gate):
    g = detector_graph(tier, d, TIER_P[tier])
    dets, _ = sample(tier, d, 800, seed=950 + d)
    on = Hierarchical(g, HierarchicalConfig(gate=gate, uf=UnionFind3DConfig(correlated=True))).run(dets)
    off = Hierarchical(g, HierarchicalConfig(gate=gate, uf=UnionFind3DConfig(correlated=True,
                                                                             pass2_skip=False))).run(dets)
    for f in ("predictions", "fast", "uf_called", "ok", "max_cluster", "table"):
        np.testing.assert_array_equal(getattr(on, f), getattr(off, f), err_msg=f)



def test_default_deep_tier_uses_auto_pass2_gate():
    g = detector_graph("circuit", 7, 0.005)
    h = Hierarchical(g)
    assert h.config.uf.correlated and h.config.uf.pass2_gate == -1 and h.uf.tables.p2gate == 7
