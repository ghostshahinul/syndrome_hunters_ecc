"""Step 1: the shared 3D detector graph."""

from collections import Counter

import numpy as np
import pytest

from src.circuits import NOISE_TIERS, tier_circuit
from src.decoders.cube3d.graph import (
    EDGE_TYPES, T_BOUNDARY, T_DIAGONAL, T_SPACE, T_TIME, _components, build_detector_graph,
    detector_graph,
)

TIER_P = {"code_capacity": 0.05, "phenomenological": 0.01, "circuit": 0.005}
CASES = [(tier, d) for tier in NOISE_TIERS for d in (3, 5, 7)]


def dem_for(tier, d):
    return tier_circuit(tier, d, TIER_P[tier]).detector_error_model(decompose_errors=True)


@pytest.mark.parametrize("tier,d", CASES)
def test_counts_match_dem(tier, d, report):
    dem = dem_for(tier, d)
    g = build_detector_graph(dem)
    comps = list(_components(dem))
    detectable = [(dets, obs) for _, dets, obs in comps if dets]
    distinct = {(dets[0], dets[1] if len(dets) == 2 else g.boundary, obs) for dets, obs in detectable}

    assert g.n_det == dem.num_detectors
    assert g.n_obs == dem.num_observables
    assert g.n_components == len(comps)
    assert g.n_undetectable == len(comps) - len(detectable)
    assert g.n_edges == len(distinct)
    assert sum(g.type_counts().values()) == g.n_edges
    report.add("cube3d_graph_edges", tier, f"d={d}", g.n_edges, ylabel="3D graph edges", log=True)


@pytest.mark.parametrize("tier,d", CASES)
def test_every_dem_component_is_an_edge(tier, d):
    dem = dem_for(tier, d)
    g = build_detector_graph(dem)
    index = {(int(g.edge_u[e]), int(g.edge_v[e]), tuple(np.flatnonzero(g.edge_obs[e]).tolist())): e
             for e in range(g.n_edges)}
    for _, dets, obs in _components(dem):
        if not dets:
            continue
        key = (dets[0], dets[1] if len(dets) == 2 else g.boundary, obs)
        e = index[key]
        # the edge's syndrome is exactly the component's detectors
        corr = np.zeros((1, g.n_edges), dtype=np.uint8)
        corr[0, e] = 1
        assert np.flatnonzero(g.syndrome(corr)[0]).tolist() == list(dets)
        assert np.flatnonzero(g.observables(corr)[0]).tolist() == list(obs)


@pytest.mark.parametrize("tier,d", CASES)
def test_edge_types_from_coordinates(tier, d):
    g = build_detector_graph(dem_for(tier, d))
    t = g.coords[:, 2]
    xy = g.coords[:, :2]
    internal = g.edge_v != g.boundary
    u, v = g.edge_u[internal], g.edge_v[internal]
    types = g.edge_type[internal]
    np.testing.assert_array_equal(types == T_SPACE, t[u] == t[v])
    same_xy = (xy[u] == xy[v]).all(axis=1)
    np.testing.assert_array_equal(types == T_TIME, (t[u] != t[v]) & same_xy)
    np.testing.assert_array_equal(types == T_DIAGONAL, (t[u] != t[v]) & ~same_xy)
    assert (g.edge_type[~internal] == T_BOUNDARY).all()
    # time-like and diagonal edges only ever join consecutive layers
    assert (np.abs(t[u] - t[v])[types != T_SPACE] == 1).all()
    if tier == "code_capacity":
        assert g.type_counts()["time"] == g.type_counts()["diagonal"] == 0


@pytest.mark.parametrize("d", [3, 5, 7])
def test_neighbor_tables(d):
    g = detector_graph("circuit", d, 0.005)
    n = g.n_det
    for a in range(n):
        nb = g.nbr[a][g.nbr[a] >= 0]
        assert len(nb) == g.degree[a] == len(set(nb.tolist()))
        for b, e in zip(nb, g.nbr_edge[a][: len(nb)]):
            assert {int(g.edge_u[e]), int(g.edge_v[e])} == {a, int(b)}
            parallel = [x for x in g.inc_edge[a] if x >= 0 and {int(g.edge_u[x]), int(g.edge_v[x])} == {a, int(b)}]
            assert g.edge_prob[e] == max(g.edge_prob[parallel])   # highest-probability parallel edge
        be = g.boundary_edge[a]
        if be >= 0:
            assert g.edge_u[be] == a and g.edge_v[be] == n
    # symmetric adjacency
    pairs = {(a, int(b)) for a in range(n) for b in g.nbr[a] if b >= 0}
    assert all((b, a) in pairs for a, b in pairs)
    assert (g.edge_weight > 0).all() and np.isfinite(g.edge_weight).all()


def test_cache_returns_same_object():
    assert detector_graph("circuit", 3, 0.005) is detector_graph("circuit", 3, 0.005)
    assert detector_graph("circuit", 3, 0.005) is not detector_graph("circuit", 3, 0.004)
