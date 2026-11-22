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
    np.testing.assert_array_equal(fast.max_cluster, ref.max_cluster)
    assert (fast.max_cluster >= (dets.sum(axis=1) > 0)).all()


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


# ---- correlated two-pass UF -------------------------------------------------------------
def corr_engine(tier, d):
    g = detector_graph(tier, d, TIER_P[tier])
    return g, UnionFind3D(g, UnionFind3DConfig(correlated=True))


@pytest.mark.parametrize("tier,d", CASES)
def test_correlated_every_single_component(tier, d):
    g, uf = corr_engine(tier, d)
    edges = np.arange(g.n_edges)
    res = uf.run(dets_for_edges(g, edges), record_correction=True)
    assert res.ok.all()
    np.testing.assert_array_equal(res.predictions, g.edge_obs[edges])
    np.testing.assert_array_equal(g.syndrome(res.correction), dets_for_edges(g, edges).astype(bool))


# Correlated decoding is not exact on every single fault in general: correlated PyMatching
# mis-decodes 1 of the 286 circuit-level d=3 instructions. Ours mis-decodes 2 there: the
# three-component instructions "D5 D16 ^ D9 ^ D12" and "D7 D18 ^ D14 ^ D11" (p = 3.3e-4),
# where pass 2 finds a pairing of two likelier correlated faults (score 16 vs 17). Every
# other tier and distance (up to 6,602 instructions at circuit d=7) is exact.
CORRELATED_SINGLE_FAULT_ALLOWANCE = {("circuit", 3): 2}


@pytest.mark.parametrize("tier,d", CASES)
def test_correlated_every_full_dem_instruction(tier, d, report):
    g, uf = corr_engine(tier, d)
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
    report.add("uf3d_correlated_single_fault_wrong", tier, f"d={d}", wrong, ylabel="DEM instructions mis-decoded")
    assert res.ok.all() and wrong <= CORRELATED_SINGLE_FAULT_ALLOWANCE.get((tier, d), 0)


@pytest.mark.parametrize("tier,d", CASES)
def test_correlated_syndrome_consistency(tier, d):
    g, uf = corr_engine(tier, d)
    dets, _ = sample(tier, d, 2000, seed=d + 50)
    res = uf.run(dets, record_correction=True)
    assert res.ok.all()
    np.testing.assert_array_equal(g.syndrome(res.correction), dets)


@pytest.mark.parametrize("tier,d", CASES)
def test_correlated_numba_bit_identical_to_reference(tier, d):
    g, uf = corr_engine(tier, d)
    dets, _ = sample(tier, d, 300 if d < 7 else 100, seed=200 + d)
    ref = uf.run(dets, record_correction=True, backend="reference")
    fast = uf.run(dets, record_correction=True, backend="numba")
    np.testing.assert_array_equal(fast.predictions, ref.predictions)
    np.testing.assert_array_equal(fast.correction, ref.correction)
    np.testing.assert_array_equal(fast.ok, ref.ok)
    np.testing.assert_array_equal(fast.max_cluster, ref.max_cluster)


def test_correlated_lengths_never_exceed_base():
    g, uf = corr_engine("circuit", 5)
    t = uf.tables
    src = np.repeat(np.arange(g.n_edges), np.diff(t.corr_ptr))
    assert len(t.corr_idx) > 0 and (t.corr_len <= t.length[t.corr_idx]).all() and (t.corr_len >= 1).all()
    assert not np.any(t.corr_idx == src)


def test_correlated_registered():
    from src.decoders import DECODERS
    assert "uf3d_correlated" in DECODERS


@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("scale", [4, 16])
@pytest.mark.parametrize("tier,d", [(t, d) for t in NOISE_TIERS for d in (3, 5)])
def test_skip_ahead_growth_is_exact(tier, d, scale, correlated):
    """Skip-ahead growth (numba) == round-by-round growth (numba) == reference."""
    g = detector_graph(tier, d, TIER_P[tier])
    dets, _ = sample(tier, d, 300, seed=300 + d)
    base = UnionFind3D(g, UnionFind3DConfig(scale=scale, correlated=correlated))
    fast = UnionFind3D(g, UnionFind3DConfig(scale=scale, correlated=correlated, skip_ahead=True))
    a = base.run(dets, record_correction=True)
    b = fast.run(dets, record_correction=True)
    r = fast.run(dets, record_correction=True, backend="reference")
    for x in (b, r):
        np.testing.assert_array_equal(x.predictions, a.predictions)
        np.testing.assert_array_equal(x.correction, a.correction)
        np.testing.assert_array_equal(x.max_cluster, a.max_cluster)


# ---- Phase C: single-fault table -----------------------------------------------------
def dem_instruction_shots(tier, d):
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
    return np.array(rows), np.array(truth)


@pytest.mark.parametrize("tier,d", CASES)
def test_table_makes_every_single_fault_exact(tier, d):
    """With the table, correlated UF decodes EVERY single DEM fault exactly (no allowance)."""
    g = detector_graph(tier, d, TIER_P[tier])
    rows, truth = dem_instruction_shots(tier, d)
    uf = UnionFind3D(g, UnionFind3DConfig(correlated=True, table=True))
    res = uf.run(rows)
    assert res.table.all()  # every instruction lights 1-4 detectors and is in the table
    np.testing.assert_array_equal(res.predictions, truth)
    ref = uf.run(rows, backend="reference")
    np.testing.assert_array_equal(ref.predictions, truth)


@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("tier,d", [(t, d) for t in NOISE_TIERS for d in (3, 5)])
def test_table_numba_bit_identical_and_only_changes_table_shots(tier, d, correlated):
    g = detector_graph(tier, d, TIER_P[tier])
    dets, _ = sample(tier, d, 1500, seed=400 + d)
    with_t = UnionFind3D(g, UnionFind3DConfig(correlated=correlated, table=True))
    without = UnionFind3D(g, UnionFind3DConfig(correlated=correlated)).run(dets)
    a = with_t.run(dets)
    r = with_t.run(dets[:300], backend="reference")
    np.testing.assert_array_equal(a.predictions[:300], r.predictions)
    np.testing.assert_array_equal(a.table[:300], r.table)
    np.testing.assert_array_equal(a.predictions[~a.table], without.predictions[~a.table])
    assert not a.table[dets.sum(axis=1) > 4].any() and not a.table[dets.sum(axis=1) == 0].any()


def test_pack_syndrome_matches_numba_lookup():
    from src.decoders.cube3d.graph import pack_syndrome
    from src.decoders.cube3d.union_find import table_lookup
    keys = np.array(sorted([pack_syndrome([5]), pack_syndrome([3, 9]), pack_syndrome([1, 2, 7, 4])]),
                    dtype=np.uint64)
    vals = np.arange(3, dtype=np.uint64) + np.uint64(10)
    for lits, want in (([9, 3], True), ([4, 7, 2, 1], True), ([5], True), ([5, 6], False), ([], False)):
        hit, _ = table_lookup(np.array(lits + [0] * (4 - len(lits)), np.int32), len(lits), keys, vals)
        assert hit == want, lits
